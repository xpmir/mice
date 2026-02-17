import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple

from experimaestro import Param, Constant
from datamaestro_text.data.ir import TextItem
from xpmir.learning.context import TrainerContext
from xpmir.letor.records import BaseRecords
from xpmir.rankers import LearnableScorer
from xpmir.text.encoders import TokenizedTexts
from xpmir.distributed import DistributableModel

from pylate.models import ColBERT

class PyLateColBERT(LearnableScorer, DistributableModel):
    """Load a ColBERT model using PyLate"""

    hf_id: Param[str]
    """the id for the huggingface model"""

    max_length: Param[Optional[int]] = None
    """the max length for the transformer model"""

    attend_to_expansion_tokens: Param[bool] = False
    """whether to attend to expansion tokens in queries"""
    
    do_query_expansion: Param[bool] = True
    """whether to perform query expansion with mask tokens"""

    document_length: Param[Optional[int]] = None
    """the max length for documents (if different from max_length)"""

    _version: Constant[str] = "0.3"

    @property
    def device(self):
        # quicker than next(self.model.parameters()).device
        return self._dummy_param.device
    
    @property
    def projection(self) -> torch.nn.Module:
        return self.model[1]  # ColBERT's projection layer is the second module
    

    def __post_init__(self):
        self._dummy_param = torch.nn.Parameter(torch.Tensor())

        # 1. Load the Model via PyLate
        # PyLate automatically handles the linear projection layer for MiniLM here.
        self.model = ColBERT(model_name_or_path=self.hf_id,
                             attend_to_expansion_tokens=self.attend_to_expansion_tokens,
                             do_query_expansion=self.do_query_expansion,
                             document_length=self.document_length)
        
        # Apply max_length constraint if provided
        if self.max_length:
            self.model.max_seq_length = self.max_length

        # 2. Extract the Tokenizer
        # PyLate attaches the tokenizer directly to the model instance.
        self.tokenizer = self.model.tokenizer

        # 3. Extract the Hugging Face Config
        # PyLate models are `SentenceTransformer` objects. The core transformer 
        # is always the first module (index 0). We access its config here.
        self.config = self.model[0].auto_model.config
    

    def batch_tokenize_PyLate(
        self,
        input_records: BaseRecords,
        maxlen=None, # Note: ColBERT usually needs specific lengths for Q (32) and D (128-180)
        mask=False,
    ) -> dict:
        """
        Tokenize queries and documents separately for Late Interaction.
        Returns a dictionary containing the batch for both.
        
        Leverages PyLate's tokenize() method which handles ColBERTv2-specific features:
        - Prefix token insertion ([Q] for queries, [D] for documents)
        - Query expansion with mask tokens
        - Asymmetric lengths (32 for queries, 180 for documents)
        - Proper attention mask handling
        """
        # 1. Extract raw texts
        queries = [q[TextItem].text for q in input_records.queries]
        docs = [d[TextItem].text for d in input_records.documents]

        # 2. Tokenize Queries using PyLate's tokenize() method
        # This automatically handles:
        # - [Q] prefix insertion
        # - Query expansion to 32 tokens with mask tokens
        # - Attention mask for expansion tokens
        q_batch = self.model.tokenize(texts=queries, is_query=True, pad=True)
        # 3. Tokenize Documents using PyLate's tokenize() method
        # This automatically handles:
        # - [D] prefix insertion
        # - Truncation to document_length (180 by default)
        # - Skiplist mask will be applied during encoding
        d_batch = self.model.tokenize(texts=docs, is_query=False, pad=False)

        # print(q_batch["input_ids"])
        # print(q_batch["attention_mask"])

        # print(d_batch["input_ids"])
        # print(d_batch["attention_mask"])

        # assert False, "Debugging prints - remove after verification"
        # Return a dict with properly tokenized Q and D tensors
        return {
            "q_ids": q_batch["input_ids"].to(self.device),
            "q_mask": q_batch["attention_mask"].to(self.device),
            "d_ids": d_batch["input_ids"].to(self.device),
            "d_mask": d_batch["attention_mask"].to(self.device),
        }

    def batch_tokenize(
        self,
        input_records: BaseRecords,
        maxlen=None, # Note: ColBERT usually needs specific lengths for Q (32) and D (128-180)
        mask=False,
    ) -> dict:
        """
        Tokenize queries and documents separately for Late Interaction.
        Returns a dictionary containing the batch for both.
        """
        # 1. Extract raw texts
        queries = [q[TextItem].text for q in input_records.queries]
        docs = [d[TextItem].text for d in input_records.documents]

        # 2. Define lengths (ColBERT uses asymmetric lengths)
        # Defaults: 32 for queries, user-provided maxlen (or 180) for docs
        q_len = 32 
        d_len = maxlen if maxlen else self.document_length if self.document_length else 180 

        # 3. Tokenize Queries
        # PyLate models handle the [Q] marker internally or via the prompt if needed,
        # but standard tokenization is usually sufficient for the PyLate wrapper.
        q_batch = self.tokenizer(
            queries,
            max_length=q_len,
            truncation=True,
            padding="max_length", # Important for tensor aggregation
            return_tensors="pt",
        )

        # 4. Tokenize Documents
        d_batch = self.tokenizer(
            docs,
            max_length=d_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Return a dict (flexible container) instead of strict TokenizedTexts
        # so we can carry both Q and D tensors.
        return {
            "q_ids": q_batch["input_ids"].to(self.device),
            "q_mask": q_batch["attention_mask"].to(self.device),
            "d_ids": d_batch["input_ids"].to(self.device),
            "d_mask": d_batch["attention_mask"].to(self.device),
        }
    
    
    def forward(self, inputs: BaseRecords, 
                tokenized_queries: Optional[dict] = None,
                tokenized_docs: Optional[dict] = None,
                doc_hidden_states: Optional[torch.Tensor] = None,
                info: TrainerContext = None):
        """
        Forward pass of the Mid-Fusion Cross Encoder.
        inputs: BaseRecords containing 'topics' and 'documents' with TextItems.
        tokenized_queries: Optional pre-tokenized queries to skip tokenization step.
        tokenized_docs: Optional pre-tokenized documents to skip tokenization step.
        doc_hidden_states: Optional pre-computed document hidden states from bottom layers.
        info: TrainerContext for additional context (not used here).
        """

        if self._version == "0.2":
            batch = self.batch_tokenize_PyLate(inputs, maxlen=self.max_length)
        else:
            batch = self.batch_tokenize(inputs, maxlen=self.max_length)
        
        # 2. Encode Queries & Documents
        # PyLate (SentenceTransformer) models expect a dict of features.
        # This passes data through: Transformer -> Linear -> Normalize (if configured)
        q_out = self.model({"input_ids": batch["q_ids"], "attention_mask": batch["q_mask"]})
        # Shape: [batch_size, seq_len, dim]
        q_emb = q_out["token_embeddings"]


        if doc_hidden_states is not None:
            # If pre-computed document embeddings are provided, use them directly
            d_emb = doc_hidden_states
        else:
            d_out = self.model({"input_ids": batch["d_ids"], "attention_mask": batch["d_mask"]})
            d_emb = d_out["token_embeddings"]


        # 3. Normalize Embeddings (along embedding dimension)
        q_emb = F.normalize(q_emb, p=2, dim=2)
        d_emb = F.normalize(d_emb, p=2, dim=2)

        # 4. Compute Late Interaction (MaxSim) Score
        # Calculate similarity matrix: [batch, q_len, d_len]
        # We perform a batch matrix multiplication between Q and D^T
        sim_matrix = torch.bmm(q_emb, d_emb.transpose(1, 2))

        # Mask padding in documents (so padding tokens don't become the "max")
        # d_mask shape: [batch, d_len] -> unsqueeze to [batch, 1, d_len]
        mask_value = -1e9
        d_mask = batch["d_mask"].unsqueeze(1).float()
        sim_matrix = sim_matrix * d_mask + (1 - d_mask) * mask_value

        # Max over document tokens (dim=2), then Sum over query tokens (dim=1)
        # Result shape: [batch_size]
        scores = sim_matrix.max(dim=2).values.sum(dim=1)
        
        # Return as a tensor suitable for your loss function
        # Most HF classification losses expect (batch, num_labels), so we might need to unsqueeze
        return scores.unsqueeze(1) 

    def distribute_models(self, update):
        self.model = update(self.model)