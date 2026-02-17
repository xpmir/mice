from sentence_transformers import CrossEncoder
from xpmir.learning.context import TrainerContext
from xpmir.letor.records import BaseRecords
from xpmir.rankers import LearnableScorer
from experimaestro import Param
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from xpmir.text.encoders import TokenizedTexts
from typing import List, Tuple
from xpmir.distributed import DistributableModel
import torch
from datamaestro_text.data.ir import TextItem

class XPMIRCrossEncoder(LearnableScorer, DistributableModel):
    hf_id: Param[str]
    """the id for the huggingface model"""

    max_length: Param[int] = None
    """the max length for the transformer model"""

    @property
    def device(self):
        return self._dummy_param.device

    def __post_init__(self):
        self._dummy_param = torch.nn.Parameter(torch.Tensor())

        self.model = CrossEncoder(self.hf_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_id)

    def batch_tokenize(
        self,
        input_records: BaseRecords,
        maxlen=None,
        mask=False,
    ) -> TokenizedTexts:
        """Transform the text to tokens by using the tokenizer"""
        if maxlen is None:
            maxlen = self.tokenizer.model_max_length
        else:
            maxlen = min(maxlen, self.tokenizer.model_max_length)

        texts: List[Tuple[str, str]] = [
            (q[TextItem].text, d[TextItem].text)
            for q, d in zip(input_records.queries, input_records.documents)
        ]

        r = self.tokenizer(
            texts,
            max_length=maxlen,
            truncation=True,
            padding=True,
            return_tensors="pt",
            return_length=True,
            return_attention_mask=mask,
        )
        return TokenizedTexts(
            None,
            r["input_ids"].to(self.device),
            r["length"],
            r.get("attention_mask", None),
            r.get("token_type_ids", None),  # if r["token_type_ids"] else None
        )

    def forward(self, inputs: BaseRecords, info: TrainerContext = None):

        tokenized = self.batch_tokenize(inputs, maxlen=self.max_length, mask=True)
        # strange that some existing models on the huggingface don't use the token_type
        with torch.set_grad_enabled(torch.is_grad_enabled()):
            result = self.model(
                tokenized.ids, token_type_ids=tokenized.token_type_ids.to(self.device), attention_mask=tokenized.mask.to(self.device)
            ).logits  # Tensor[float] of length records size
        return result

    def distribute_models(self, update):
        self.model = update(self.model)