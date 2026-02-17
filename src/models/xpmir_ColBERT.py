"""Copied from https://github.com/yzong12138/MVDR_pruning/blob/master/neural/ColBERT.py"""
import json, os, string
from dataclasses import InitVar, dataclass
from typing import NamedTuple, Union, Optional, List

from huggingface_hub import hf_hub_download
from experimaestro import Param, LightweightTask

import torch
import torch.nn as nn
from transformers import AutoConfig, PreTrainedModel
from transformers.models.bert import BertModel
from transformers.models.distilbert import DistilBertModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.utils import cached_file

from xpmir.text.huggingface import HFTokenizer
from xpmir.learning.optim import ModuleInitOptions
from xpmir.text.huggingface.tokenizers import HFTokenizerAdapter

from xpmir.text.huggingface import HFModel
from xpmir.text.huggingface.encoders import HFTokensEncoder
from xpmir.utils.utils import easylog, foreach
from xpmir.learning.optim import ModuleInitMode
from xpmir.learning.context import TrainerContext
from xpmir.text import TokenizerOptions
from xpmir.text.encoders import (
    TokenizedTexts,
    TokensRepresentationOutput,
)
from xpmir.neural.dual import DualVectorListener
from xpmir.neural.interaction import InteractionScorer
from xpmir.neural.interaction.common import (
    SimilarityInput,
    SimilarityOutput,
)

logger = easylog()
HFConfigName = Union[str, os.PathLike]


@dataclass
class TransformersColBERTOutput(BaseModelOutputWithPoolingAndCrossAttentions):
    """A dataclass include the orginal bert model's last hidden state before
    projection"""

    bert_last_hidden_state: torch.FloatTensor = None


# --- Huggingface Configs
class ColBERTConfig(NamedTuple):
    """ColBERT configuration when loading a pre-trained ColBERT model"""

    dim: int
    query_maxlen: int
    similarity: str
    attend_to_mask_tokens: bool
    data: dict

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path: HFConfigName):
        resolved_config_file = cached_file(
            pretrained_model_name_or_path, "artifact.metadata"
        )
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        with open(resolved_config_file, "rt") as fp:
            data = json.load(fp)
            kwargs = {key: data[key] for key in ColBERTConfig._fields if key != "data"}
            config.colbert = ColBERTConfig(**kwargs, data=data)
        return config


class DistilColBERTConfig(NamedTuple):
    """ColBERT configuration when training from scratch, based on DistilColBERT"""

    dim: int

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path: HFConfigName):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        config.colbert = DistilColBERTConfig(dim=128)
        return config


# --- Huggingface Models
class ColBERTModel(PreTrainedModel):
    """ColBERT model"""

    DEFAULT_OUTPUT_SIZE = 128

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, config.colbert.dim, bias=False)

    def forward(self, ids, **kwargs):
        output = self.bert(ids, **kwargs)
        bert_last_hidden_state = output.last_hidden_state.clone()
        output.last_hidden_state = self.linear(output.last_hidden_state)
        # store also the bert's last hidden state
        return TransformersColBERTOutput(
            bert_last_hidden_state=bert_last_hidden_state,
            **output,
        )

    @classmethod
    def from_config(cls, config):
        return super(ColBERTModel, cls)._from_config(config)


class DistilColBERTModel(PreTrainedModel):
    """ColBERT model"""

    DEFAULT_OUTPUT_SIZE = 128

    def __init__(self, config):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.linear = nn.Linear(config.hidden_size, config.colbert.dim, bias=False)

    def forward(self, ids, **kwargs):
        output = self.distilbert(ids, **kwargs)
        bert_last_hidden_state = output.last_hidden_state.clone()
        output.last_hidden_state = self.linear(output.last_hidden_state)
        # store also the distilbert's last hidden state
        return TransformersColBERTOutput(
            bert_last_hidden_state=bert_last_hidden_state,
            **output,
        )

    @classmethod
    def from_config(cls, config):
        return super(DistilColBERTModel, cls)._from_config(config)


# --- Xpmir Encoders
class ColBERTEncoder(HFModel):
    model: InitVar[ColBERTModel]
    automodel = ColBERTModel
    autoconfig = ColBERTConfig


class ColBERTEncoderScratch(ColBERTEncoder):
    model: InitVar[ColBERTModel]
    automodel = DistilColBERTModel
    autoconfig = DistilColBERTConfig


# --- Xpmir Encoder with projector on top of it
class ColBERTProjectorAdapter(HFTokensEncoder):
    """The colbert projector, the encoded vector is pseudo-normalized,
    e.g. norm <= 1
    """

    model: Param[ColBERTEncoder]
    """The base colbert encoder"""

    ns_dim: Param[int] = 64
    """The additional dimension"""

    projector_norm: Param[float] = 1e-6
    """The norm for the projector vectors,
    will be buggy if setting to 0
    """

    def __initialize__(self, options):
        super().__initialize__(options)
        hidden_size = self.model.hf_config.hidden_size
        self.projector = nn.Linear(hidden_size, self.ns_dim, bias=False)
        self.zero_projector()
        if isinstance(self.model, ColBERTEncoderScratch):
            # initialize the projector's weight if training from distilbert
            self.model.model.linear.reset_parameters()

    def zero_projector(self):
        self.projector.weight.data = (
            torch.nn.functional.normalize(self.projector.weight.data)
            * self.projector_norm
        )

    def forward(self, tokenized: TokenizedTexts) -> TokensRepresentationOutput:
        tokenized = tokenized.to(self.model.contextual_model.device)
        if isinstance(self.model, ColBERTEncoderScratch):
            # distilbert doesn't have the token type ids
            y: TransformersColBERTOutput = self.model.contextual_model(
                tokenized.ids,
                attention_mask=tokenized.mask.to(self.device),
            )
        else:
            y: TransformersColBERTOutput = self.model.contextual_model(
                tokenized.ids,
                attention_mask=tokenized.mask.to(self.device),
                token_type_ids=tokenized.token_type_ids,
            )
        projected_last_hidden_state = self.projector(y.bert_last_hidden_state)
        ns_last_hidden_state = y.last_hidden_state / torch.sqrt(
            y.last_hidden_state.norm(dim=-1) ** 2
            + projected_last_hidden_state.norm(dim=-1) ** 2
        ).unsqueeze(-1)
        return TokensRepresentationOutput(
            tokenized=tokenized, value=ns_last_hidden_state
        )


# --- Initialization tasks: Modify the params of the model before learning
class ColBERTProjectionInitialization(LightweightTask):
    """Load the projector layer, and then adjust the additional embeddings related
    to the additional tokens"""

    hf_id: Param[str]
    model: Param[ColBERTEncoder]
    num_add_tokens: Param[int] = 1
    """with a number of the additional tokens>1, it will initialize the [unusedi]
    token to the embeddings of the token [unused1]
    the id to copy from is at id [2]

    tried, but doesn't work
    """

    def execute(self):
        self.model.initialize(ModuleInitMode.DEFAULT.to_options())
        cp_path = hf_hub_download(repo_id=self.hf_id, filename="pytorch_model.bin")
        loaded = torch.load(cp_path)
        linear = loaded["linear.weight"]
        self.model.model.linear.weight.data = linear
        if self.num_add_tokens == 1:
            return
        w_embeddings = loaded["bert.embeddings.word_embeddings.weight"]
        w_replacing = w_embeddings[2].unsqueeze(0).expand(self.num_add_tokens, -1)
        w_embeddings[2 : 2 + self.num_add_tokens] = w_replacing
        self.model.model.bert.embeddings.word_embeddings.weight.data = w_embeddings


class ColBERTSecondStageTrainingRandomProjector(LightweightTask):
    """During the DistilBERT second stage training, replace the 0 projector
    norm to a low value, otherwise will be buggy"""

    model: Param[ColBERTProjectorAdapter]

    def execute(self):
        self.model.initialize(ModuleInitMode.DEFAULT.to_options())
        # get the projector
        projector_in = self.model.projector.in_features
        projector_out = self.model.projector.out_features
        new_projector = nn.Linear(projector_in, projector_out, bias=False)
        new_projector.weight.data = (
            torch.nn.functional.normalize(new_projector.weight.data) * 1e-3
        )
        self.model.projector.weight.data = new_projector.weight.data
        logger.info("Replacing the previous 0 projector to a random initialization!")


# --- Model
# --------- Vanilla Version
class ColBERTWithProjector(InteractionScorer):
    """a colbert model with a projector"""

    mask_attend: Param[bool] = True
    """Whether the mask token of the query attend the
    final scoring"""

    mask_punctuation: Param[bool] = True
    """Whether we mask the punctuation for the document tokens during
    matching"""

    num_add_tokens: Param[int] = 1
    """The number of the additional tokens on the document side"""

    def __initialize__(self, options):
        super().__initialize__(options)
        # a trainable parameter to learn to scale the distillation scores
        self.alpha = nn.Parameter(torch.tensor(1.0))
        # a list of the punctuations
        self.skiplist = [
            self.encoder.tokenizer.tokenizer.tokenizer.encode(
                symbol, add_special_tokens=False
            )[0]
            for symbol in string.punctuation
        ]

    def prepare_vanilla_inputs(self, records):
        """return the value and mask after encoding
        with the post processing of punctuation"""
        pad_token_id = self.encoder.tokenizer.tokenizer.tokenizer.pad_token_id
        encoded = self.encoder(
            records,
            options=TokenizerOptions(self.dlen),
        )

        if self.mask_punctuation:
            # not only mask the pad tokens, but also mask all the
            # punctuation tokens
            # The mask will only apply on the scoring stage but during
            # the calculation of the self-attention
            mask = [
                [(x not in self.skiplist) and (x != pad_token_id) for x in d]
                for d in encoded.tokenized.ids.cpu().tolist()
            ]
            mask = (
                torch.tensor(mask)
                .to(dtype=encoded.tokenized.mask.dtype)
                .to(encoded.tokenized.mask.device)
            )
        else:
            mask = encoded.tokenized.mask

        value = encoded.value

        return value, mask

    def encode_documents(self, records):
        value, mask = self.prepare_vanilla_inputs(records)
        return self.similarity.preprocess(
            SimilarityInput(
                value=value,
                mask=mask,
            )
        )

    def merge(self, objects: List[SimilarityInput]):
        # Used to merge the query terms
        # As in colbert, all the queries are append to a fix length
        # using a simple concatenate is OK.
        mask = torch.cat([object.mask for object in objects], dim=0)
        value = torch.cat([object.value for object in objects], dim=0)

        return SimilarityInput(
            value=value,
            mask=mask,
        )

    def compute_scores(
        self,
        queries: SimilarityInput,
        documents: SimilarityInput,
        value: SimilarityOutput,
        info: Optional[TrainerContext] = None,
    ):
        # Similarity matrix B x Lq x Ld or Bq x Lq x Bd x Ld
        # In vanilla colbertv2 they don't mask the queries
        s = value.similarity.masked_fill(
            value.d_view(documents.mask).logical_not(), float("-inf")
        )
        if not self.mask_attend:
            s = s.masked_fill(value.q_view(queries.mask).logical_not(), 0)

        # call the hooks -- Loggers or Regularizations
        # Apply the dual vector hook
        if info is not None:
            foreach(
                info.hooks(DualVectorListener),
                lambda hook: hook(info, queries, documents),
            )

        return nn.functional.relu(s).max(-1).values.sum(1).flatten()


### Special Tokenizer for ColBERT (why ?? )

from typing import List, Tuple, Union
import torch
from experimaestro import Param
from xpmir.text.huggingface import HFTokenizer
from xpmir.learning.optim import ModuleInitOptions
from xpmir.text.tokenizers import TokenizedTexts, TokenizerOptions
from xpmir.text.huggingface.tokenizers import HFTokenizerAdapter

# Follow the work of ColBERTv2, we insert markers


def _insert_prefix_token(
    tensor: torch.Tensor, prefix_id: Union[int, List], num: int = 1
):
    if isinstance(prefix_id, int):
        prefix_tensor = torch.full(
            (tensor.size(0), num),
            prefix_id,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        assert len(prefix_id) == num
        prefix_tensor = (
            torch.tensor(prefix_id)
            .unsqueeze(0)
            .expand(tensor.size(0), -1)
            .to(dtype=tensor.dtype)
            .to(tensor.device)
        )
    return torch.cat([tensor[:, :1], prefix_tensor, tensor[:, 1:]], dim=1)


# In ColBERTv2, it always pad the query to a fix length whatever the given
# length
class HFTokenizerColBERTQuery(HFTokenizer):
    """The colbert's tokenzier, which pads the query with [MASK]
    to fix length"""

    q_marker: Param[str] = "[unused0]"

    def __initialize__(self, options: ModuleInitOptions):
        super().__initialize__(options)
        self.mask_token_id = self.tokenizer.mask_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.q_marker_id = self.tokenizer.convert_tokens_to_ids(self.q_marker)

    def tokenize(
        self,
        texts: List[str] | List[Tuple[str, str]],
        options: TokenizerOptions | None = None,
    ) -> TokenizedTexts:
        options = options or HFTokenizer.DEFAULT_OPTIONS
        max_length = options.max_length
        if max_length is None:
            max_length = self.tokenizer.model_max_length
        else:
            max_length = min(max_length, self.tokenizer.model_max_length)

        r = self.tokenizer(
            list(texts),
            # -1 because we need to pad the marker
            max_length=max_length - 1,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_length=options.return_length,
            return_attention_mask=options.return_mask,
        )

        ids = r["input_ids"]
        ids[ids == self.pad_token_id] = self.mask_token_id
        ids = _insert_prefix_token(ids, self.q_marker_id)
        if options.return_length:
            r["length"] = r["length"] + 1
        if options.return_mask:
            r["attention_mask"] = _insert_prefix_token(r["attention_mask"], 1)
        if r.get("token_type_ids", None) is not None:
            r["token_type_ids"] = _insert_prefix_token(r["token_type_ids"], 0)
        return TokenizedTexts(
            None,
            ids,
            r.get("length", None),
            r.get("attention_mask", None),
            r.get("token_type_ids", None),
        )


class HFTokenizerColBERTDocument(HFTokenizer):
    """The colbert's tokenzier, which pads the query with [MASK]
    to fix length"""

    add_d_tokens: Param[int] = 1
    """The number of additional [D] token prepend, normally initialize to [unused1]
    if the addition number of tokens are provided, the markers should be
    [unused2], [unused3], etc
    """

    def __initialize__(self, options: ModuleInitOptions):
        super().__initialize__(options)
        self.d_markers = [f"[unused{i}]" for i in range(1, self.add_d_tokens + 1)]
        self.d_marker_ids = self.tokenizer.convert_tokens_to_ids(self.d_markers)

    def tokenize(self, texts, options=None):
        options = options or HFTokenizer.DEFAULT_OPTIONS
        max_length = options.max_length
        if max_length is None:
            max_length = self.tokenizer.model_max_length
        else:
            max_length = min(max_length, self.maxtokens())

        r = self.tokenizer(
            list(texts),
            # -1 because we need to pad the marker
            max_length=max_length - self.add_d_tokens,
            truncation=True,
            padding=True,
            return_tensors="pt",
            return_length=options.return_length,
            return_attention_mask=options.return_mask,
        )

        ids = r["input_ids"]
        # try to append the d_markers to the mask and the ids
        ids = _insert_prefix_token(ids, self.d_marker_ids, self.add_d_tokens)
        if options.return_length:
            r["length"] = r["length"] + self.add_d_tokens
        if options.return_mask:
            r["attention_mask"] = _insert_prefix_token(
                r["attention_mask"], 1, self.add_d_tokens
            )
        if r.get("token_type_ids", None) is not None:
            r["token_type_ids"] = _insert_prefix_token(
                r["token_type_ids"], 0, self.add_d_tokens
            )

        return TokenizedTexts(
            None,
            ids,
            r.get("length", None),
            r.get("attention_mask", None),
            r.get("token_type_ids", None),
        )

class HFStringTokenizerColBERT(HFTokenizerAdapter):
    """A class which generate different tokenizer instance for query and
    documents"""

    @classmethod
    def from_pretrained_id(cls, hf_id: str, query=True, num_add_tokens=1, **kwargs):
        if query:
            return cls.C(tokenizer=HFTokenizerColBERTQuery.C(model_id=hf_id), **kwargs)
        else:
            return cls.C(
                tokenizer=HFTokenizerColBERTDocument.C(
                    model_id=hf_id, add_d_tokens=num_add_tokens
                ),
                **kwargs,
            )