from enum import Enum
from copy import deepcopy
from typing import List, Tuple, Optional, NamedTuple, Union
from datamaestro_ir.data.base import IDTextRecord
import torch
import torch.nn as nn
import logging
from pathlib import Path
from jinja2 import Template

from experimaestro import Param, LightweightTask, Constant, field
from xpm_torch.huggingface import TorchHFHub
from xpm_torch.utils import to_device
from xpm_torch.module import SimpleModuleLoader
from xpmir.letor.records import BaseItems
from xpmir.rankers import AbstractModuleScorer
from xpmir.text.encoders import (
    TokenizedTexts,
    EncoderOutput,
    TextEncoderBase,
    TokensEncoderOutput,
    TokensRepresentationOutput,
)
from xpmir.text.huggingface.tokenizers import HFTokenizer
from xpmir.text.tokenizers import TokenizerOptions

# Configuration and common types

# Transformers imports with safety checks
try:
    from transformers import (
        ModernBertConfig,
        AutoModel,
        AutoConfig,
        AutoModelForSequenceClassification,
    )
    from transformers.models.bert.modeling_bert import BertLayer
    from transformers.models.modernbert.modeling_modernbert import (
        ModernBertMLP,
        ModernBertAttention,
        ModernBertRotaryEmbedding,
    )
    from transformers.models.modernbert_decoder.modeling_modernbert_decoder import (
        eager_attention_forward,
    )
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

except Exception:
    logging.error("Install huggingface transformers to use these configurations")
    raise

logger = logging.getLogger(__name__)


class ExactMatchAttentionHead(nn.Module):
    def __init__(self, d_model, use_projections=True):
        super().__init__()
        self.use_projections = use_projections

        if self.use_projections:
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, attn_mask):
        B, Tv, C = x.shape
        _, _, Tq, _ = attn_mask.shape
        dtype = x.dtype
        device = x.device

        V = self.v_proj(x) if self.use_projections else x
        V = V.unsqueeze(1)

        dummy_Q = torch.zeros(B, 1, Tq, C, dtype=dtype, device=device)
        dummy_K = torch.zeros(B, 1, Tv, C, dtype=dtype, device=device)

        out = torch.nn.functional.scaled_dot_product_attention(
            dummy_Q, dummy_K, V, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )

        out = out.squeeze(1)

        if self.use_projections:
            out = self.out_proj(out)

        return out


def compute_mask(q_ids, doc_ids, is_causal: bool = False):
    B, Tq = q_ids.shape
    B, Td = doc_ids.shape

    tokens_q = q_ids.unsqueeze(2)
    tokens_k = doc_ids.unsqueeze(1)
    match_mask = (tokens_q == tokens_k).unsqueeze(1)

    if is_causal:
        causal_mask = torch.tril(
            torch.ones(Tq, Td, dtype=torch.bool, device=q_ids.device)
        )
        match_mask = match_mask & causal_mask.unsqueeze(0).unsqueeze(0)

    attn_mask = torch.full(
        match_mask.shape, float("-inf"), device=q_ids.device, dtype=torch.float32
    )
    attn_mask[match_mask] = 0.0

    no_matches = ~match_mask.any(dim=-1)

    if no_matches.any():
        attn_mask[:, 0, :, 0] = torch.where(
            no_matches[:, 0, :], 0.0, attn_mask[:, 0, :, 0]
        )

    return attn_mask


class MICETokenizedTexts(NamedTuple):
    """Container for MICE tokenized inputs (separate query and document streams)"""

    tokenized_q: TokenizedTexts
    """tokenized Queries"""

    tokenized_docs: TokenizedTexts
    """tokenized Documents"""


class QueryDocInput(Enum):
    """Enum to specify whether the input is a query or a document, for tokenization purposes."""

    QUERY = "query"
    DOCUMENT = "document"
    PAIRS = "pairs"  # Both query and document are present in the input records, and should be tokenized together (default)


class MICEQueryDocTokenizer(HFTokenizer):
    """Specific tokenizer for MICE that handles query and document independently"""

    max_query_length: Param[Optional[int]]
    """maximum number of tokens for the query side"""

    max_doc_length: Param[Optional[int]]
    """maximum number of tokens for the document side"""

    def __post_init__(self):
        super().__post_init__()
        if self.max_doc_length is None:
            self.max_doc_length = self.max_length
        if self.max_query_length is None:
            self.max_query_length = self.max_length

    def tokenize(
        self,
        input_records: BaseItems,
        options: Optional[TokenizerOptions] = None,
        input_nature: QueryDocInput = QueryDocInput.PAIRS,
    ) -> MICETokenizedTexts:
        # Determine per-side token limits
        q_max = self.max_query_length
        d_max = self.max_doc_length

        def _encode(texts: List[str], max_tokens: int):
            r = self.tokenizer(
                texts,
                add_special_tokens=True,
                truncation=True,
                max_length=max_tokens,
                padding=True,
                return_tensors="pt",
                return_length=True,
            )
            return TokenizedTexts(
                tokens=None,
                ids=r["input_ids"],
                lens=r["length"].tolist(),
                mask=r.get("attention_mask", None),
                token_type_ids=r.get("token_type_ids", None),
            )

        if input_nature == QueryDocInput.PAIRS:
            ix_qs, ix_ds = input_records.pairs()
            queries = [input_records.unique_topics[i]["text_item"].text for i in ix_qs]
            docs = [input_records.unique_documents[i]["text_item"].text for i in ix_ds]
            tokenized_texts = MICETokenizedTexts(
                tokenized_q=_encode(queries, q_max),
                tokenized_docs=_encode(docs, d_max),
            )
        elif input_nature == QueryDocInput.QUERY:
            if hasattr(input_records, "unique_topics"):
                queries = [
                    topic["text_item"].text
                    if hasattr(topic, "__getitem__") and "text_item" in topic
                    else getattr(
                        topic,
                        "text",
                        topic["text"] if hasattr(topic, "__getitem__") else str(topic),
                    )
                    for topic in input_records.unique_topics
                ]
            else:
                queries = [
                    query["text_item"].text
                    if hasattr(query, "__getitem__") and "text_item" in query
                    else getattr(
                        query,
                        "text",
                        query["text"] if hasattr(query, "__getitem__") else str(query),
                    )
                    for query in input_records
                ]
            tokenized_texts = (
                MICETokenizedTexts(
                    tokenized_q=_encode(queries, q_max),
                    tokenized_docs=None,  # No documents to encode
                ).tokenized_q
            )  # Returns only the queries in that case, to be compatible later on
        else:  # input_nature == QueryDocInput.DOCUMENT
            ### WARNING: Assumes the input is composed of a list of document records,
            # not an instance from BaseItems
            docs = [
                doc["text_item"].text
                if hasattr(doc, "__getitem__") and "text_item" in doc
                else getattr(
                    doc,
                    "text",
                    doc["text"] if hasattr(doc, "__getitem__") else str(doc),
                )
                for doc in input_records
            ]
            tokenized_texts = (
                MICETokenizedTexts(
                    tokenized_q=None,
                    tokenized_docs=_encode(docs, d_max),
                ).tokenized_docs
            )  # Returns only the docs in that case, to be compatible later on

        return tokenized_texts


class MiceCrossEncoder(AbstractModuleScorer):
    """
    Base class for Mice Cross Encoder base Architecture, with Cross-Attention in the top layers.
    The bottom layers encode query and document independently, while the top layers
    allow the query to attend to the document via cross-attention mechanisms.
    """

    ## Parameters for Config

    hf_id: Param[str]
    """Hugging Face checkpoint identifier that provides weights and config. Must be a BERT-like model."""

    tokenizer: Param[MICEQueryDocTokenizer]
    """The tokenizer for independent Q/D processing"""

    n_contextualization_layers: Param[int] = field(default=6, ignore_default=True)
    """Number of bottom encoder layers that process query and document independently"""

    n_docs_ctx_layers: Param[Optional[int]] = field(default=None, ignore_default=True)
    """Number of bottom encoder layers for the document. If None, use n_contextualization_layers."""

    n_interaction_layers: Param[Optional[int]] = field(
        default=None, ignore_default=True
    )
    """Number of top encoder layers with cross-attention. If None, use all remaining layers from the backbone."""

    bound_bottom_layers: Param[bool] = field(default=True, ignore_default=True)
    """Whether to use the same set of parameters for Query/Doc encoder"""

    cross_attn_first: Param[bool] = field(default=True, ignore_default=True)
    """Whether to perform cross-attention before self-attention in the top layers."""

    mask_cls_to_doc: Param[bool] = field(default=False, ignore_default=True)
    """Whether to mask the [CLS] token from attending to document tokens."""

    mask_query_to_cls: Param[bool] = field(default=True, ignore_default=True)
    """Whether to mask query tokens from attending to the [CLS] token (using it as a sink)"""

    freeze_base: Param[bool] = field(default=False, ignore_default=True)
    """Whether to freeze the bottom layers during finetuning"""

    random_top_layers: Param[bool] = field(default=False, ignore_default=True)
    """Whether to initialize top layers randomly instead of copying from backbone"""

    global_cls_token: Param[bool] = field(default=False, ignore_default=True)
    """Whether to add a fresh [CLS] token before the top layers."""

    extra_attn_bias: Param[bool] = field(default=False, ignore_default=True)
    """Whether to add an exact match cross-attention bias head."""

    lexical_bias: Param[bool] = field(default=False, ignore_default=True)
    """Whether to add a learnable lexical bias for top layers attention heads."""

    compress_dim: Param[float] = field(default=1.0, ignore_default=True)
    """Factor by which to divide the hidden dimensions of the top layers"""

    doc: Param[Optional[str]] = field(overrides=True)
    """Documentation for the model"""

    bibtex: Param[Optional[str]] = field(overrides=True)
    """BibTeX for the model"""

    _version: Constant[int] = 2
    """Model version"""

    ## Attributes (not Parameters)

    embeddings: nn.Module
    """Shared embeddings for query and document"""

    bottom_layers: nn.ModuleList
    """Bottom layers: independent encoding"""

    top_layers: nn.ModuleList
    """Top layers: cross-attention encoding"""

    _contextualization_config_path: str = "contextualization_layers"

    _interaction_config_path: str = "interaction_layers"

    def get_lexical_match_mask(
        self, query_ids: torch.Tensor, doc_ids: torch.Tensor
    ) -> torch.Tensor:
        """Computes a boolean lexical match mask (B, 1, Tq, Td) ignoring special tokens."""
        is_special_q = torch.isin(
            query_ids,
            torch.tensor(
                self.tokenizer.tokenizer.all_special_ids, device=query_ids.device
            ),
        )
        is_special_d = torch.isin(
            doc_ids,
            torch.tensor(
                self.tokenizer.tokenizer.all_special_ids, device=doc_ids.device
            ),
        )
        match_mask = (
            (query_ids.unsqueeze(2) == doc_ids.unsqueeze(1))
            & (~is_special_q.unsqueeze(2))
            & (~is_special_d.unsqueeze(1))
        )
        return match_mask.unsqueeze(1)  # shape (B, 1, Tq, Td)

    def _fetch_backbone_configs(self):
        """Helper to load the backbone config and head config from Hugging Face,
        used when initializing the model structure if local configs have not been loaded. (init from hf_id)
        """
        if (
            hasattr(self, "config")
            and self.config is not None
            and hasattr(self, "head_config")
            and self.head_config is not None
        ):
            return  # Configs already set (possibly loaded locally), no need to fetch from HF

        logger.warning(
            "Fetching backbone configs from Hugging Face for hf_id: %s", self.hf_id
        )

        base_config = AutoConfig.from_pretrained(self.hf_id)

        # 1. base config (used for the bottom layers, which follow the original backbone architecture more closely)
        self.config = deepcopy(base_config)
        # Ensure _attn_implementation is not None to avoid warnings
        # Configs should be set by InitTask or manually before calling initialize()
        if (
            not hasattr(self.config, "_attn_implementation")
            or self.config._attn_implementation is None
        ):
            self.config._attn_implementation = "eager"

        # 2. head config (used for the top layers, which may have different settings like cross-attention)
        self.head_config = deepcopy(base_config)
        self.head_config.is_decoder = True
        self.head_config.add_cross_attention = True

        if (
            not hasattr(self.head_config, "_attn_implementation")
            or self.head_config._attn_implementation is None
        ):
            self.head_config._attn_implementation = getattr(
                self.config, "_attn_implementation", "eager"
            )

    def __initialize__(self):
        """Instanciates the Config and skeleton of the model"""
        super().__initialize__()
        self.tokenizer.initialize()

        # Ensure configs are loaded before accessing them
        self._fetch_backbone_configs()

        # Check that we don't exceed the number of layers in the backbone
        num_backbone_layers = getattr(self.config, "num_hidden_layers", 0)

        if num_backbone_layers > 0:
            if self.n_contextualization_layers > num_backbone_layers:
                raise ValueError(
                    f"n_contextualization_layers ({self.n_contextualization_layers}) "
                    f"exceeds backbone layers ({num_backbone_layers}) for {self.hf_id}"
                )
            if (
                self.n_docs_ctx_layers is not None
                and self.n_docs_ctx_layers > num_backbone_layers
            ):
                raise ValueError(
                    f"n_docs_ctx_layers ({self.n_docs_ctx_layers}) "
                    f"exceeds backbone layers ({num_backbone_layers}) for {self.hf_id}"
                )
            if self.n_interaction_layers is not None:
                total_requested = (
                    self.n_contextualization_layers + self.n_interaction_layers
                )
                if total_requested > num_backbone_layers:
                    raise ValueError(
                        f"Total requested layers ({total_requested}) exceeds backbone "
                        f"layers ({num_backbone_layers}) for {self.hf_id}. "
                        f"(n_ctx={self.n_contextualization_layers}, n_inter={self.n_interaction_layers})"
                    )

    @property
    def dimension(self):
        return self.config.hidden_size

    @property
    def max_doc_len(self):
        return self.tokenizer.max_doc_length

    @property
    def max_query_len(self):
        return self.tokenizer.max_query_length

    def batch_tokenize(
        self,
        input_records: BaseItems,
        options=None,
        input_nature: QueryDocInput = QueryDocInput.PAIRS,
    ) -> MICETokenizedTexts:
        """Transform the text to tokens by using the tokenizer"""
        return self.tokenizer.tokenize(
            input_records, options=options, input_nature=input_nature
        )

    def get_tokenizer_fn(self):
        return self.batch_tokenize

    def get_extended_attention_mask(self, mask, dtype):
        """Helper to create the -inf mask for transformers"""
        inverted_mask = 1.0 - mask[:, None, None, :]
        return inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.finfo(dtype).min
        )

    def get_self_attention_mask(self, mask, dtype):
        """Helper to create the -inf mask for transformers"""
        mask_bool = mask.to(torch.bool)
        # Build a square attention map where both query and key positions must be valid tokens
        valid_pairs = mask_bool[:, None, :, None] & mask_bool[:, None, None, :]
        if self.mask_query_to_cls:
            # Additionally mask out all query tokens from attending to the [CLS] token
            valid_pairs[:, :, 1:, 0] = False
        attn_mask = torch.zeros(valid_pairs.shape, dtype=dtype, device=mask.device)
        attn_mask.masked_fill_(~valid_pairs, torch.finfo(dtype).min)

        return attn_mask

    def get_cross_attention_mask(self, query_mask, doc_mask, dtype):
        """Helper to create the -inf mask for cross-attention with rectangular support"""
        q_valid = query_mask.to(torch.bool)[:, :, None]
        d_valid = doc_mask.to(torch.bool)[:, None, :]
        valid_pairs = q_valid & d_valid
        if self.mask_cls_to_doc:
            # Additionally mask out the [CLS] token from attending to document tokens
            valid_pairs[:, 0, :] = False
        attn_mask = torch.zeros(
            (valid_pairs.size(0), 1, valid_pairs.size(1), valid_pairs.size(2)),
            dtype=dtype,
            device=query_mask.device,
        )
        attn_mask.masked_fill_(~valid_pairs[:, None, :, :], torch.finfo(dtype).min)
        return attn_mask

    def get_document_encoder(self) -> TextEncoderBase:
        """Returns a TokenizedTextEncoder initialized from the bottom layers of MICE"""
        raise NotImplementedError()

    def encode_documents(self, input_ids, attention_mask) -> torch.Tensor:
        """Encode inputs through bottom layers (independent encoding)"""
        raise NotImplementedError()

    def save_model(self, path: Path):
        """Save the model and tokenizer in standard pretrained format."""
        from safetensors.torch import save_file

        path.mkdir(parents=True, exist_ok=True)
        assert path.is_dir(), f"Save path '{path}' must be a directory"

        # Save model weights
        save_file(self.state_dict(), str(path / "model.safetensors"))
        # Save tokenizer
        if (
            hasattr(self.tokenizer, "tokenizer")
            and self.tokenizer.tokenizer is not None
        ):
            self.tokenizer.tokenizer.save_pretrained(path)

        # Save head config
        if hasattr(self, "head_config") and self.head_config:
            self.head_config.save_pretrained(path / self._interaction_config_path)
        else:
            logger.warning("No head config found to save for MICE model.")
        # Save configs
        if hasattr(self, "config") and self.config:
            self.config.save_pretrained(path / self._contextualization_config_path)
        else:
            logger.warning("No base config found to save for MICE model.")

    def save_pretrained(self, path: Union[str, Path]):
        """Allows direct saving from Module"""

        loader = self.loader_config(Path(path))
        TorchHFHub(loader).save_pretrained(path)

    def write_hub_extras(
        self,
        save_directory: Path,
        tb_path: Optional[Union[str, Path]] = None,
    ):
        """Called by TorchHFHub to save additional files."""
        import shutil

        if hasattr(self, "config") and self.config is not None:
            self.config.save_pretrained(save_directory)
        if hasattr(self, "head_config") and self.head_config is not None:
            (save_directory / "head").mkdir(exist_ok=True)
            self.head_config.save_pretrained(save_directory / "head")
        if (
            hasattr(self.tokenizer, "tokenizer")
            and self.tokenizer.tokenizer is not None
        ):
            self.tokenizer.tokenizer.save_pretrained(save_directory)

        # Copy TensorBoard log directory if available
        tb_dir = (
            Path(tb_path)
            if tb_path
            else getattr(self, "_tb_path", None)
            or (
                Path(self._export_metadata["tb_path"])
                if hasattr(self, "_export_metadata")
                and isinstance(self._export_metadata, dict)
                and self._export_metadata.get("tb_path")
                else None
            )
        )

        if tb_dir and Path(tb_dir).exists():
            target_tb_dir = Path(save_directory) / "runs"
            try:
                if target_tb_dir.exists():
                    shutil.rmtree(target_tb_dir)
                shutil.copytree(tb_dir, target_tb_dir, symlinks=False)
                logger.info(f"Copied TensorBoard logs from {tb_dir} to {target_tb_dir}")
            except Exception as e:
                logger.warning(
                    f"Failed to copy TensorBoard logs to {target_tb_dir}: {e}"
                )

        # Write model card (README.md)
        try:
            template_path = Path(__file__).parent / "MiceCard.md"
            if template_path.exists():
                template = Template(template_path.read_text())
                # Use getattr with defaults to be robust
                card = template.render(
                    base=getattr(self, "hf_id", "unknown"),
                    model_id=save_directory.name,
                    n_contextualization_layers=self.n_contextualization_layers,
                    n_interaction_layers=self.n_interaction_layers
                    or (len(self.top_layers) if hasattr(self, "top_layers") else "N/A"),
                    loss=getattr(self, "loss_name", "Unknown"),
                    k=getattr(self, "top_k", "100"),
                    retriever=getattr(self, "retriever_name", "BM25"),
                    results=getattr(self, "eval_results", "Results not available yet."),
                )
                with open(save_directory / "README.md", "w") as f:
                    f.write(card)
                logger.info(f"Model card written to {save_directory / 'README.md'}")
        except Exception as e:
            logger.warning(f"Could not write model card for MICE: {e}")

    def load_model(self, path: Path):
        """Load from given directory, prioritizing local configs if available."""
        from transformers import AutoConfig

        assert path.exists(), f"Model path '{path}' does not exist"

        if path.is_file() and path.suffix in [".safetensors", ".pt", ".pth", ".bin"]:
            path = path.parent

        # 1. Attempt to load head config locally
        head_config_path = path / self._interaction_config_path / "config.json"
        if head_config_path.exists():
            logger.info(f"Loading local head config from {head_config_path}")
            self.head_config = AutoConfig.from_pretrained(
                str(path / self._interaction_config_path)
            )
        else:
            logger.info(
                "Local head config not found, falling back to base/hf_id config."
            )
            if not hasattr(self, "head_config") or self.head_config is None:
                self.head_config = AutoConfig.from_pretrained(self.hf_id)
                self.head_config.is_decoder = True
                self.head_config.add_cross_attention = True

        # 2. Attempt to load base config locally
        config_path = path / self._contextualization_config_path / "config.json"
        if config_path.exists():
            logger.info(f"Loading local base config from {config_path}")
            self.config = AutoConfig.from_pretrained(
                str(path / self._contextualization_config_path)
            )
        else:
            logger.info(f"Local base config not found, relying on hf_id: {self.hf_id}")
            if not hasattr(self, "config") or self.config is None:
                self.config = AutoConfig.from_pretrained(self.hf_id)

        # Ensure _attn_implementation is set (similar to __initialize__)
        if (
            not hasattr(self.config, "_attn_implementation")
            or self.config._attn_implementation is None
        ):
            self.config._attn_implementation = "eager"

        if (
            not hasattr(self.head_config, "_attn_implementation")
            or self.head_config._attn_implementation is None
        ):
            self.head_config._attn_implementation = getattr(
                self.config, "_attn_implementation", "eager"
            )

        # 2. Initialize the model structure based on the config (weights will be loaded in the next step)
        self.initialize()

        # 3. Load weights (via state_dict)
        super().load_model(path)

    def loader_config(self, path: Path, *, settings=None) -> "SimpleModuleLoader":
        return SimpleModuleLoader.C(value=self, path=path, settings=settings)

    def export_action(self, loader, **kwargs):
        from xpmir.models import XPMIRExportAction

        if self.doc:
            kwargs.setdefault("doc", self.doc)
        if self.bibtex:
            kwargs.setdefault("bibtex", self.bibtex)
        return XPMIRExportAction.C(loader=loader, **kwargs)


class BertMiceCrossEncoder(MiceCrossEncoder):
    """Mice Cross Encoder based on BERT Architecture."""

    _version: Constant[int] = field(default=3, overrides=True)
    """Model version"""

    def __initialize__(self):
        super().__initialize__()
        self.head_config.is_decoder = True
        self.head_config.add_cross_attention = True

        if self.compress_dim > 1:
            self.head_config.hidden_size = int(
                self.head_config.hidden_size / self.compress_dim
            )
            self.head_config.intermediate_size = int(
                self.head_config.intermediate_size / self.compress_dim
            )
            self.head_config.num_attention_heads = int(
                self.head_config.num_attention_heads / self.compress_dim
            )
            self.adapter = nn.Linear(
                self.config.hidden_size, self.head_config.hidden_size
            )
        else:
            self.adapter = None

        # Build skeleton (initially random)
        # Note: We use the config to build the structure. Weights copied later by InitTask.
        temp_model = AutoModel.from_config(self.config)
        self.add_module("embeddings", temp_model.embeddings)

        # Build Bottom layers, either bound or distinct
        if self.bound_bottom_layers:
            if (
                self.n_docs_ctx_layers is not None
                and self.n_docs_ctx_layers != self.n_contextualization_layers
            ):
                raise ValueError(
                    "n_docs_ctx_layers must be equal to n_contextualization_layers if bound_bottom_layers is True"
                )
            self.add_module(
                "bottom_layers",
                nn.ModuleList(
                    [
                        BertLayer(self.config)
                        for _ in range(self.n_contextualization_layers)
                    ]
                ),
            )
        else:
            self.add_module(
                "query_bottom_layers",
                nn.ModuleList(
                    [
                        BertLayer(self.config)
                        for _ in range(self.n_contextualization_layers)
                    ]
                ),
            )
            n_doc_layers = (
                self.n_docs_ctx_layers
                if self.n_docs_ctx_layers is not None
                else self.n_contextualization_layers
            )
            self.add_module(
                "document_bottom_layers",
                nn.ModuleList(
                    [BertLayer(self.config) for _ in range(n_doc_layers)],
                ),
            )

        if self.n_interaction_layers is not None:
            num_top = self.n_interaction_layers
        else:
            num_top = len(temp_model.encoder.layer) - self.n_contextualization_layers

        self.add_module(
            "top_layers",
            nn.ModuleList([BertLayer(self.head_config) for _ in range(num_top)]),
        )

        if self.extra_attn_bias:
            self.add_module(
                "exact_match_heads",
                nn.ModuleList(
                    [
                        ExactMatchAttentionHead(self.head_config.hidden_size)
                        for _ in range(num_top)
                    ]
                ),
            )

        self.pooler = getattr(temp_model, "pooler", None)
        if self.pooler:
            if self.compress_dim > 1:
                self.pooler.dense = nn.Linear(
                    self.head_config.hidden_size,
                    self.head_config.hidden_size,
                    bias=True,
                )
            self.add_module("pooler", self.pooler)
        self.dropout_layer = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.head_config.hidden_size, 1)

        if self.global_cls_token:
            self.global_cls = nn.Parameter(
                torch.randn(1, 1, self.head_config.hidden_size) * 0.02
            )

    def get_document_encoder(self) -> TextEncoderBase:
        """Returns a TokenizedTextEncoder initialized from the bottom layers of MICE"""

        return MiceDocumentEncoder.C(model=self)

    def query_token_embeddings(self, records: List[IDTextRecord]) -> List[torch.Tensor]:
        """Encode a batch of queries and return the list of per-token
        embeddings, one tensor ``(num_tokens, dim)`` per query. Padding
        positions are filtered out.
        """
        options = TokenizerOptions(max_length=self.max_query_len)
        tokenized = self.batch_tokenize(
            records, options=options, input_nature=QueryDocInput.QUERY
        )
        if tokenized.ids.device != self.device:
            tokenized = tokenized.to(self.device)
        output = self.encode_queries(tokenized.ids, tokenized.mask)

        return output

    def document_token_embeddings(
        self, records: List[IDTextRecord]
    ) -> List[torch.Tensor]:
        """Encode a batch of documents and return the list of per-token
        embeddings, one tensor ``(num_tokens, dim)`` per document. Padding
        positions are filtered out.
        """
        options = TokenizerOptions(max_length=self.max_doc_len)
        tokenized = self.batch_tokenize(
            records, options=options, input_nature=QueryDocInput.DOCUMENT
        )
        if tokenized.ids.device != self.device:
            tokenized = tokenized.to(self.device)
        output = self.encode_documents(tokenized.ids, tokenized.mask)

        return output

    def encode_queries(self, input_ids, attention_mask):
        """Compute bottom layers (independent encoding)"""
        x = self.embeddings(input_ids)
        # Standard BERT extended mask logic
        ext_mask = self.get_extended_attention_mask(attention_mask, x.dtype)
        if self.bound_bottom_layers:
            query_bottom_layers = self.bottom_layers
        else:
            query_bottom_layers = self.query_bottom_layers

        for layer in query_bottom_layers:
            x = layer(x, ext_mask)
        return x

    def encode_documents(self, input_ids, attention_mask) -> torch.Tensor:
        """Encode inputs through bottom layers (independent encoding)"""
        x = self.embeddings(input_ids)
        # Standard BERT extended mask logic
        ext_mask = self.get_extended_attention_mask(attention_mask, x.dtype)

        if self.bound_bottom_layers:
            document_bottom_layers = self.bottom_layers
        else:
            document_bottom_layers = self.document_bottom_layers

        for layer in document_bottom_layers:
            x = layer(x, ext_mask)
        return x

    def forward_inverted_transformer(
        self, q_hidden, doc_hidden_states, q_ext_mask, d_ext_mask, exact_match_mask=None
    ):
        """Inverted transformer architecture: Cross-Attention followed by Self-Attention."""
        num_top = len(self.top_layers)
        for i, layer in enumerate(self.top_layers):
            if self.extra_attn_bias and exact_match_mask is not None:
                exact_match_out = self.exact_match_heads[i](
                    doc_hidden_states, exact_match_mask
                )
                q_hidden = q_hidden + exact_match_out

            # Cross attention First (full sequence) to allow better information flow from document tokens
            # BertAttention.forward(hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, ...)
            q_hidden = layer.crossattention(
                hidden_states=q_hidden,
                encoder_hidden_states=doc_hidden_states,
                encoder_attention_mask=d_ext_mask,
            )[0]

            # Then Self-Attention (Full)
            # BertLayer.attention returns (attention_output, ...)
            q_hidden = layer.attention(q_hidden, q_ext_mask)[0]

            if i == num_top - 1:
                # Optimized last layer: Slice to CLS token for MLP and output
                q_hidden = q_hidden[:, 0:1, :]

            # 4. MLP (CLS only)
            # BertLayer.intermediate returns hidden_states
            # BertLayer.output does residual + norm
            intermediate_output = layer.intermediate(q_hidden)
            q_hidden = layer.output(intermediate_output, q_hidden)
        return q_hidden

    def forward_vanilla_transformer(
        self, q_hidden, doc_hidden_states, q_ext_mask, d_ext_mask, exact_match_mask=None
    ):
        """Vanilla transformer architecture: Self-Attention followed by Cross-Attention."""
        num_top = len(self.top_layers)
        for i, layer in enumerate(self.top_layers):
            if self.extra_attn_bias and exact_match_mask is not None:
                exact_match_out = self.exact_match_heads[i](
                    doc_hidden_states, exact_match_mask
                )
                q_hidden = q_hidden + exact_match_out

            if i == num_top - 1:
                # Optimized last layer: Self-Attention on full sequence,
                # then slice to CLS for Cross-Attention and MLP.

                # 1. Self-Attention (Full)
                # BertLayer.attention returns (attention_output, ...)
                q_hidden = layer.attention(q_hidden, q_ext_mask)[0]

                # 2. Slice to CLS
                q_hidden = q_hidden[:, 0:1, :]

                # 3. Cross-Attention (CLS only)
                if doc_hidden_states is not None and not self.mask_cls_to_doc:
                    # Slice masks for CLS token
                    # BertAttention.forward(hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, ...)
                    q_hidden = layer.crossattention(
                        hidden_states=q_hidden,
                        encoder_hidden_states=doc_hidden_states,
                        encoder_attention_mask=d_ext_mask[:, :, 0:1, :],
                    )[0]

                # 4. MLP (CLS only)
                # BertLayer.intermediate returns hidden_states
                # BertLayer.output does residual + norm
                intermediate_output = layer.intermediate(q_hidden)
                q_hidden = layer.output(intermediate_output, q_hidden)
            else:
                # BertLayer with is_decoder=True accepts:
                # (hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask)
                layer_out = layer(
                    hidden_states=q_hidden,  # Query (Self-Attn)
                    attention_mask=q_ext_mask,
                    encoder_hidden_states=doc_hidden_states,  # Document (Cross-Attn Key/Value)
                    encoder_attention_mask=d_ext_mask,
                )
                q_hidden = layer_out
        return q_hidden

    def forward(
        self,
        inputs: Optional[BaseItems] = None,
        tokenized: Optional[MICETokenizedTexts] = None,
        doc_hidden_states: Optional[torch.Tensor] = None,
        doc_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the Mice Cross Encoder.
        inputs: BaseRecords containing 'topics' and 'documents' with TextItems.
        tokenized_queries: Optional pre-tokenized queries to skip tokenization step.
        tokenized_docs: Optional pre-tokenized documents to skip tokenization step.
        doc_hidden_states: Optional pre-computed document hidden states from bottom layers.
        doc_mask: Optional document attention mask when doc_hidden_states are provided.
        info: TrainerContext for additional context (not used here).
        """
        if inputs is None and tokenized is None:
            raise ValueError("Either raw inputs or tokenized inputs must be provided.")

        # Prepare inputs
        if tokenized is None:
            tokenized = self.batch_tokenize(inputs, input_nature=QueryDocInput.PAIRS)

        tokenized_q = to_device(tokenized.tokenized_q, self.device)
        tokenized_docs = to_device(tokenized.tokenized_docs, self.device)
        query_ids = tokenized_q.ids
        doc_ids = tokenized_docs.ids
        query_mask = tokenized_q.mask
        doc_mask = tokenized_docs.mask

        # 1. Process Query through Bottom Layers
        q_hidden = self.encode_queries(query_ids, query_mask)

        if self.global_cls_token:
            cls_token = self.global_cls.expand(q_hidden.shape[0], -1, -1)
            q_hidden = torch.cat([cls_token, q_hidden], dim=1)
            query_mask = torch.cat(
                [
                    torch.ones(
                        (query_mask.shape[0], 1),
                        dtype=query_mask.dtype,
                        device=query_mask.device,
                    ),
                    query_mask,
                ],
                dim=1,
            )

        if doc_hidden_states is None:
            # Process Doc through Bottom Layers

            doc_hidden_states = self.encode_documents(
                doc_ids, doc_mask
            )  # shape [batch, seq_len_doc, dim]
        else:
            # doc_hidden_states has been provided: we need to deduce the masks AND pad the documents for the
            # batch processing in the top layers.
            def repad_batch(
                doc_embeddings: List[
                    torch.Tensor
                ],  # each (seq_len_i, hidden) or (1, seq_len_i, hidden)
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                # Normalize to (seq_len, hidden)
                docs = [d.squeeze(0) if d.dim() == 3 else d for d in doc_embeddings]

                lengths = [doc.shape[0] for doc in docs]
                max_len = max(lengths)
                device = docs[0].device
                B = len(docs)

                padded = torch.zeros(B, max_len, *docs[0].shape[1:], device=device)
                mask = torch.zeros(B, max_len, device=device).long()

                for i, doc in enumerate(docs):
                    seq_len = doc.shape[0]
                    padded[i, :seq_len] = doc
                    mask[i, :seq_len] = 1

                return padded, mask

            doc_hidden_states, doc_mask = repad_batch(doc_hidden_states)

        # 2. Prepare Masks for Top Layers
        # Mask for Self-Attention (Query) shape [batch, 1, seq_len_query, seq_len_query]
        q_ext_mask = self.get_self_attention_mask(query_mask, q_hidden.dtype)
        # Mask for Cross-Attention (Query attending to Doc) shape [batch, 1, seq_len_query, seq_len_doc]
        d_ext_mask = self.get_cross_attention_mask(query_mask, doc_mask, q_hidden.dtype)

        if self.extra_attn_bias:
            exact_match_mask = compute_mask(query_ids, doc_ids)
        else:
            exact_match_mask = None

        # 3. Process Query through Top Layers (with Cross-Attention to Doc)
        if self.adapter is not None:
            q_hidden = self.adapter(q_hidden)
            doc_hidden_states = self.adapter(doc_hidden_states)

        if self.cross_attn_first:
            q_hidden = self.forward_inverted_transformer(
                q_hidden, doc_hidden_states, q_ext_mask, d_ext_mask, exact_match_mask
            )
        else:
            q_hidden = self.forward_vanilla_transformer(
                q_hidden, doc_hidden_states, q_ext_mask, d_ext_mask, exact_match_mask
            )

        # 4. Score (Use [CLS] of the Query)
        if self.pooler is not None:
            pooled = self.pooler(q_hidden)
        else:
            pooled = q_hidden[:, 0, :]

        pooled = self.dropout_layer(pooled)
        score = self.classifier(pooled)
        return score.squeeze(-1)


class ModernBertCrossAttention(nn.Module):
    """Cross-attention wrapper for ModernBERT that accepts separate query/key/value tensors."""

    def __init__(
        self,
        config: ModernBertConfig,
        layer_idx: Optional[int] = None,
        lexical_bias: bool = False,
    ):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.all_head_size = self.head_dim * self.num_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.attention_bias
        )
        self.Wo = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.attention_bias
        )
        self.out_drop = (
            nn.Dropout(config.attention_dropout)
            if config.attention_dropout > 0.0
            else nn.Identity()
        )
        self.lexical_bias = lexical_bias
        if self.lexical_bias:
            self.lexical_bias_weight = nn.Parameter(torch.zeros(self.num_heads))

    def forward(
        self, query, key, value, attention_mask, lexical_match_mask=None, **kwargs
    ):
        q_shape = query.shape[:-1]
        query_states = (
            self.q_proj(query).view(*q_shape, -1, self.head_dim).transpose(1, 2)
        )
        key_states = (
            self.k_proj(key).view(*key.shape[:-1], -1, self.head_dim).transpose(1, 2)
        )
        value_states = (
            self.v_proj(value)
            .view(*value.shape[:-1], -1, self.head_dim)
            .transpose(1, 2)
        )

        if self.lexical_bias and lexical_match_mask is not None:
            bias = self.lexical_bias_weight.view(1, -1, 1, 1) * lexical_match_mask.to(
                dtype=query_states.dtype
            )
            if attention_mask is not None:
                attention_mask = attention_mask + bias
            else:
                attention_mask = bias

        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]
        else:
            attention_interface = eager_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*q_shape, -1).contiguous()
        return self.out_drop(self.Wo(attn_output)), attn_weights


class ModernBertCrossAttentionLayer(nn.Module):
    """
    A ModernBERT encoder layer with added cross-attention for mid-fusion ranking.
    """

    def __init__(
        self,
        config: ModernBertConfig,
        layer_id: Optional[int] = None,
        lexical_bias: bool = False,
    ):
        super().__init__()
        self.config = config
        self.attn_norm = (
            nn.Identity()
            if layer_id == 0
            else nn.LayerNorm(
                config.hidden_size,
                eps=config.norm_eps,
                bias=getattr(config, "norm_bias", True),
            )
        )
        self.attn = ModernBertAttention(config=config, layer_idx=layer_id)
        self.crossattention = ModernBertCrossAttention(
            config=config, layer_idx=layer_id, lexical_bias=lexical_bias
        )
        self.mlp_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.norm_eps,
            bias=getattr(config, "norm_bias", True),
        )
        self.mlp = ModernBertMLP(config)
        self.attention_type = (
            config.layer_types[layer_id] if layer_id is not None else "full_attention"
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings=None,
        lexical_match_mask=None,
    ):
        # Self-attn residual
        attn_out = self.attn(
            self.attn_norm(hidden_states),
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )[0]
        hidden_states = hidden_states + attn_out

        # Cross-attn residual
        if encoder_hidden_states is not None:
            cross_out = self.crossattention(
                query=self.attn_norm(hidden_states),
                key=self.attn_norm(encoder_hidden_states),
                value=self.attn_norm(encoder_hidden_states),
                attention_mask=encoder_attention_mask,
                lexical_match_mask=lexical_match_mask,
            )[0]
            hidden_states = hidden_states + cross_out

        # MLP residual
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states


class ModernBertMiceCrossEncoder(MiceCrossEncoder):
    """Mice Cross Encoder based on ModernBERT Architecture."""

    pooling_method: Param[Optional[str]] = field(default=None, ignore_default=True)
    """Pooling method to use for the ModernBert based scorer: cls or mean.
    Leave it to None for models coming from the Hub, as it will be inferred from the model config."""

    def __initialize__(self):
        super().__initialize__()
        pm = self.pooling_method or getattr(self.config, "classifier_pooling", "cls")
        if pm == "cls":
            self.pooling_function = lambda x: x[:, 0]
        else:
            self.pooling_function = lambda x: x.mean(dim=1)

        # Structure setup
        # Weights copied later by InitTask
        temp_model = AutoModelForSequenceClassification.from_config(self.config)
        self.rotary_emb = ModernBertRotaryEmbedding(self.config)
        self.add_module("embeddings", temp_model.model.embeddings)

        # Build Bottom layers, either bound or distinct
        if self.bound_bottom_layers:
            if (
                self.n_docs_ctx_layers is not None
                and self.n_docs_ctx_layers != self.n_contextualization_layers
            ):
                raise ValueError(
                    "n_docs_ctx_layers must be equal to n_contextualization_layers if bound_bottom_layers is True"
                )
            self.add_module(
                "bottom_layers",
                nn.ModuleList(
                    [
                        type(temp_model.model.layers[0])(self.config, i)
                        for i in range(self.n_contextualization_layers)
                    ]
                ),
            )
        else:
            self.add_module(
                "query_bottom_layers",
                nn.ModuleList(
                    [
                        type(temp_model.model.layers[0])(self.config, i)
                        for i in range(self.n_contextualization_layers)
                    ]
                ),
            )
            n_doc_layers = (
                self.n_docs_ctx_layers
                if self.n_docs_ctx_layers is not None
                else self.n_contextualization_layers
            )
            self.add_module(
                "document_bottom_layers",
                nn.ModuleList(
                    [
                        type(temp_model.model.layers[0])(self.config, i)
                        for i in range(n_doc_layers)
                    ]
                ),
            )

        if self.n_interaction_layers is not None:
            num_top = self.n_interaction_layers
        else:
            num_top = len(temp_model.model.layers) - self.n_contextualization_layers

        self.add_module(
            "top_layers",
            nn.ModuleList(
                [
                    ModernBertCrossAttentionLayer(
                        self.config,
                        self.n_contextualization_layers + i,
                        lexical_bias=self.lexical_bias,
                    )
                    for i in range(num_top)
                ]
            ),
        )

        if self.extra_attn_bias:
            self.add_module(
                "exact_match_heads",
                nn.ModuleList(
                    [
                        ExactMatchAttentionHead(self.config.hidden_size)
                        for _ in range(num_top)
                    ]
                ),
            )

        self.add_module("final_norm", temp_model.model.final_norm)
        self.add_module("head", temp_model.head)
        self.dropout_layer = nn.Dropout(self.config.classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, 1)

        if self.global_cls_token:
            self.global_cls = nn.Parameter(
                torch.randn(1, 1, self.config.hidden_size) * 0.02
            )

    def get_document_encoder(self) -> TextEncoderBase:
        """Returns a TokenizedTextEncoder initialized from the bottom layers of MICE"""
        return MiceDocumentEncoder.C(model=self)

    def query_token_embeddings(self, records: List[IDTextRecord]) -> List[torch.Tensor]:
        """Encode a batch of queries and return the list of per-token
        embeddings, one tensor ``(num_tokens, dim)`` per query. Padding
        positions are filtered out.
        """
        options = TokenizerOptions(max_length=self.max_query_len)
        tokenized = self.batch_tokenize(
            records, options=options, input_nature=QueryDocInput.QUERY
        )
        if tokenized.ids.device != self.device:
            tokenized = tokenized.to(self.device)
        output = self.encode_queries(tokenized.ids, tokenized.mask)

        return output

    def document_token_embeddings(
        self, records: List[IDTextRecord]
    ) -> List[torch.Tensor]:
        """Encode a batch of documents and return the list of per-token
        embeddings, one tensor ``(num_tokens, dim)`` per document. Padding
        positions are filtered out.
        """
        options = TokenizerOptions(max_length=self.max_doc_len)
        tokenized = self.batch_tokenize(
            records, options=options, input_nature=QueryDocInput.DOCUMENT
        )
        if tokenized.ids.device != self.device:
            tokenized = tokenized.to(self.device)
        output = self.encode_documents(tokenized.ids, tokenized.mask)

        return output

    def encode_queries(self, input_ids, attention_mask) -> torch.Tensor:
        """Compute bottom layers (independent encoding)"""
        x_q = self.embeddings(input_ids)
        q_ext = self.get_extended_attention_mask(attention_mask, torch.float32)

        # Pos IDs for RoPE
        b, s = input_ids.size()
        q_pos = torch.arange(s, device=input_ids.device).unsqueeze(0).expand(b, s)

        # Precompute RoPE for all layer types in config
        unique_layer_types = set(self.config.layer_types)
        q_pos_embeds = {
            lt: self.rotary_emb(x_q, q_pos, layer_type=lt) for lt in unique_layer_types
        }

        if self.bound_bottom_layers:
            query_bottom_layers = self.bottom_layers
        else:
            query_bottom_layers = self.query_bottom_layers

        for layer in query_bottom_layers:
            x_q = layer(
                x_q, q_ext, position_embeddings=q_pos_embeds[layer.attention_type]
            )
        return x_q

    def encode_documents(self, input_ids, attention_mask) -> torch.Tensor:
        """Encode inputs through bottom layers (independent encoding)"""
        x_d = self.embeddings(input_ids)
        d_ext = self.get_extended_attention_mask(attention_mask, torch.float32)

        # Pos IDs for RoPE
        b, s = input_ids.size()
        d_pos = torch.arange(s, device=input_ids.device).unsqueeze(0).expand(b, s)

        # Precompute RoPE for all layer types in config
        unique_layer_types = set(self.config.layer_types)
        d_pos_embeds = {
            lt: self.rotary_emb(x_d, d_pos, layer_type=lt) for lt in unique_layer_types
        }

        if self.bound_bottom_layers:
            document_bottom_layers = self.bottom_layers
        else:
            document_bottom_layers = self.document_bottom_layers

        for layer in document_bottom_layers:
            x_d = layer(
                x_d, d_ext, position_embeddings=d_pos_embeds[layer.attention_type]
            )
        return x_d

    def forward_vanilla_transformer(
        self,
        x_q,
        x_d,
        q_self,
        cross_mask,
        q_pos_embeds,
        exact_match_mask=None,
        lexical_match_mask=None,
    ):
        """Vanilla transformer architecture: Self-Attention followed by Cross-Attention."""
        pm = self.pooling_method or getattr(self.config, "classifier_pooling", "cls")
        num_top = len(self.top_layers)
        for i, layer in enumerate(self.top_layers):
            if self.extra_attn_bias and exact_match_mask is not None:
                exact_match_out = self.exact_match_heads[i](x_d, exact_match_mask)
                x_q = x_q + exact_match_out

            if i == num_top - 1 and pm == "cls":
                # Optimized last layer: Self-Attention on full sequence,
                # then slice to CLS for Cross-Attention and MLP.

                # 1. Self-attn residual (Full)
                attn_out = layer.attn(
                    layer.attn_norm(x_q),
                    attention_mask=q_self,
                    position_embeddings=q_pos_embeds[layer.attention_type],
                )[0]
                x_q = x_q + attn_out

                # 2. Slice to CLS
                x_q = x_q[:, 0:1, :]

                # 3. Cross-attn residual (CLS only)
                if not self.mask_cls_to_doc:
                    cross_out = layer.crossattention(
                        query=layer.attn_norm(x_q),
                        key=layer.attn_norm(x_d),
                        value=layer.attn_norm(x_d),
                        attention_mask=cross_mask[:, :, 0:1, :],
                        lexical_match_mask=lexical_match_mask[:, :, 0:1, :]
                        if lexical_match_mask is not None
                        else None,
                    )[0]
                    x_q = x_q + cross_out

                # 4. MLP residual (CLS only)
                x_q = x_q + layer.mlp(layer.mlp_norm(x_q))
            else:
                x_q = layer(
                    x_q,
                    attention_mask=q_self,
                    encoder_hidden_states=x_d,
                    encoder_attention_mask=cross_mask,
                    position_embeddings=q_pos_embeds[layer.attention_type],
                    lexical_match_mask=lexical_match_mask,
                )
        return x_q

    def forward_inverted_transformer(
        self,
        x_q,
        x_d,
        q_self,
        cross_mask,
        q_pos_embeds,
        exact_match_mask=None,
        lexical_match_mask=None,
    ):
        """Inverted transformer architecture: Cross-Attention followed by Self-Attention."""

        pm = self.pooling_method or getattr(self.config, "classifier_pooling", "cls")
        num_top = len(self.top_layers)

        for i, layer in enumerate(self.top_layers):
            if self.extra_attn_bias and exact_match_mask is not None:
                exact_match_out = self.exact_match_heads[i](x_d, exact_match_mask)
                x_q = x_q + exact_match_out

            # 1. Cross-attn residual (Full)
            cross_out = layer.crossattention(
                query=layer.attn_norm(x_q),
                key=layer.attn_norm(x_d),
                value=layer.attn_norm(x_d),
                attention_mask=cross_mask,
                lexical_match_mask=lexical_match_mask,
            )[0]
            x_q = x_q + cross_out

            # 2. Self-attn residual (Full)
            attn_out = layer.attn(
                layer.attn_norm(x_q),
                attention_mask=q_self,
                position_embeddings=q_pos_embeds[layer.attention_type],
            )[0]
            x_q = x_q + attn_out

            if i == num_top - 1 and pm == "cls":
                # 3. Slice to CLS only to save compute in MLP and output
                x_q = x_q[:, 0:1, :]

            # 4. MLP residual (CLS only)
            x_q = x_q + layer.mlp(layer.mlp_norm(x_q))

        return x_q

    def forward(
        self,
        inputs: Optional[BaseItems] = None,
        tokenized: Optional[MICETokenizedTexts] = None,
        doc_hidden_states: Optional[torch.Tensor] = None,
        doc_mask: Optional[torch.Tensor] = None,
    ):
        if inputs is None and tokenized is None:
            raise ValueError("Either raw inputs or tokenized inputs must be provided.")

        # Prepare inputs
        if tokenized is None:
            tokenized = self.batch_tokenize(inputs, input_nature=QueryDocInput.PAIRS)

        tokenized_q = to_device(tokenized.tokenized_q, self.device)
        query_ids = tokenized_q.ids
        query_mask = tokenized_q.mask

        tokenized_docs = to_device(tokenized.tokenized_docs, self.device)
        doc_ids = tokenized_docs.ids
        doc_mask = tokenized_docs.mask

        # 1. Process Query through Bottom Layers
        x_q = self.encode_queries(query_ids, query_mask)

        if self.global_cls_token:
            cls_token = self.global_cls.expand(x_q.shape[0], -1, -1)
            x_q = torch.cat([cls_token, x_q], dim=1)
            query_mask = torch.cat(
                [
                    torch.ones(
                        (query_mask.shape[0], 1),
                        dtype=query_mask.dtype,
                        device=query_mask.device,
                    ),
                    query_mask,
                ],
                dim=1,
            )

        # Recompute pos embeds for top layers (only query side)
        b, s = query_mask.size()
        q_pos = torch.arange(s, device=query_mask.device).unsqueeze(0).expand(b, s)
        unique_layer_types = set(self.config.layer_types)
        q_pos_embeds = {
            lt: self.rotary_emb(x_q, q_pos, layer_type=lt) for lt in unique_layer_types
        }

        if doc_hidden_states is None:
            # Process Doc through Bottom Layers
            x_d = self.encode_documents(doc_ids, doc_mask)
        else:
            x_d = doc_hidden_states
            if doc_mask is None:
                doc_mask = tokenized_docs.mask

        # 2. Prepare Masks for Top Layers
        q_self = self.get_self_attention_mask(query_mask, x_q.dtype)
        cross_mask = self.get_cross_attention_mask(query_mask, doc_mask, x_q.dtype)

        if self.extra_attn_bias:
            exact_match_mask = compute_mask(query_ids, doc_ids)
        else:
            exact_match_mask = None

        if self.lexical_bias:
            lexical_match_mask = self.get_lexical_match_mask(query_ids, doc_ids)
        else:
            lexical_match_mask = None

        # 3. Process Query through Top Layers (with Cross-Attention to Doc)
        if self.cross_attn_first:
            x_q = self.forward_inverted_transformer(
                x_q,
                x_d,
                q_self,
                cross_mask,
                q_pos_embeds,
                exact_match_mask,
                lexical_match_mask,
            )
        else:
            x_q = self.forward_vanilla_transformer(
                x_q,
                x_d,
                q_self,
                cross_mask,
                q_pos_embeds,
                exact_match_mask,
                lexical_match_mask,
            )

        # 4. Score (Use [CLS] of the Query)
        x_q = self.final_norm(x_q)
        pooled = self.pooling_function(x_q)
        return self.classifier(self.dropout_layer(self.head(pooled))).squeeze(-1)


class MiceDocumentEncoder(TextEncoderBase):
    """Overrides TokenizedTextEncoder to load the bottom layers from MICE and
    instantiate a document encoder from them"""

    model: Param[MiceCrossEncoder]

    def __initialize__(self) -> None:
        super().__initialize__()
        self.model.initialize()

    @property
    def dimension(self):
        return self.model.dimension

    @staticmethod
    def _token_mask(output: TokensRepresentationOutput) -> Optional[torch.Tensor]:
        mask = output.tokenized.mask
        if mask is None:
            return None
        return mask.to(output.value.device).bool()

    def document_token_embeddings(
        self, records: List[IDTextRecord]
    ) -> List[torch.Tensor]:
        """Encode a batch of documents and return the list of per-token
        embeddings, one tensor ``(num_tokens, dim)`` per document. Padding
        positions are filtered out.
        """
        output = self.encode_documents(records)
        mask = self._token_mask(output)
        value = output.value
        if mask is None:
            return [value[i] for i in range(value.shape[0])]
        return [value[i][mask[i]] for i in range(value.shape[0])]

    def encode_documents(
        self, records: List[IDTextRecord]
    ) -> TokensRepresentationOutput:
        options = TokenizerOptions(max_length=self.model.max_doc_len)
        output = self(records, options=options)
        return output

    def forward(
        self,
        inputs: List[IDTextRecord],
        *args,
        options: Optional[TokenizerOptions] = None,
    ) -> EncoderOutput:
        assert len(args) == 0, "Unhandled extra arguments"
        tokenized = self.model.batch_tokenize(
            inputs, options=options, input_nature=QueryDocInput.DOCUMENT
        )
        if tokenized.ids.device != self.model.device:
            tokenized = tokenized.to(self.model.device)
        return TokensEncoderOutput(
            tokenized, self.model.encode_documents(tokenized.ids, tokenized.mask)
        )


class InitMICEBERTFromHFID(LightweightTask):
    """Worker-node task to load weights into MICE BERT model"""

    model: Param[BertMiceCrossEncoder]

    def execute(self):
        # Ensure model is instantiated and initialized
        model = self.model
        hf_id = model.hf_id

        # Ensure configs are available
        if not hasattr(model, "config") or model.config is None:
            model.config = AutoConfig.from_pretrained(hf_id)

        if not hasattr(model, "head_config") or model.head_config is None:
            model.head_config = AutoConfig.from_pretrained(hf_id)
            model.head_config.is_decoder = True
            model.head_config.add_cross_attention = True

        # Build the model structure first
        model.initialize()

        logger.info(f"Building MICE from pretrained Bert model: {hf_id}")

        full_bert = AutoModel.from_pretrained(hf_id)

        if hasattr(full_bert, "embeddings"):
            logger.info("Seeding embeddings from backbone")
            model.embeddings = full_bert.embeddings
        else:
            logger.warning(
                f"Backbone {hf_id} has no 'embeddings' attribute; skipping seeding"
            )

        # Copy bottom layers
        if hasattr(full_bert, "encoder") and hasattr(full_bert.encoder, "layer"):
            logger.info(
                f"Seeding {model.n_contextualization_layers} bottom layers from backbone"
            )
            if model.bound_bottom_layers:
                contextualization_module_names = ["bottom_layers"]
            else:
                contextualization_module_names = [
                    "document_bottom_layers",
                    "query_bottom_layers",
                ]

            for cntx_module_name in contextualization_module_names:
                contextualization_module = getattr(model, cntx_module_name)
                for i in range(len(contextualization_module)):
                    if i < len(full_bert.encoder.layer):
                        contextualization_module[i].load_state_dict(
                            full_bert.encoder.layer[i].state_dict()
                        )
                    else:
                        logger.warning(
                            f"Backbone has only {len(full_bert.encoder.layer)} layers; cannot seed bottom layer {i}"
                        )
        else:
            logger.warning(
                f"Backbone {hf_id} has no encoder layers; skipping bottom layer seeding"
            )

        model.top_layers = nn.ModuleList()

        # Load original top layers to copy weights from
        start_idx = model.n_contextualization_layers
        if model.n_interaction_layers is not None:
            end_idx = start_idx + model.n_interaction_layers
            assert end_idx <= len(full_bert.encoder.layer), (
                f"Total layers ({end_idx}) exceeds backbone layers: {len(full_bert.encoder.layer)}"
            )
            original_top_layers = full_bert.encoder.layer[start_idx:end_idx]
            logging.info(
                f"Using {model.n_interaction_layers} layers for interaction, dropping remaining backbone layers if any"
            )
        else:
            original_top_layers = full_bert.encoder.layer[start_idx:]
            logging.info(
                f"Using all remaining {len(original_top_layers)} backbone layers for interaction"
            )

        for i in range(len(original_top_layers)):
            # Instantiate a fresh layer with Cross-Attention enabled
            new_layer = BertLayer(model.head_config)

            # COPY trained weights (Self-Attention + FFN) from original BERT to new layer
            # Also seed Cross-Attention from the same weights
            if not model.random_top_layers:
                logger.info(
                    f"Copying weights from original BERT to Interaction top layer {i}"
                )
                self._copy_bert_weights(original_top_layers[i], new_layer)
            else:
                logger.info(f"Initializing Interaction top layer {i} randomly")

            model.top_layers.append(new_layer)

        # pooler
        # Preserve the pretrained pooler to keep the original [CLS] projection
        model.pooler = getattr(full_bert, "pooler", None)

        if model.pooler is None:
            logger.warning(
                "No pooler found in the base model; using [CLS] token directly."
            )

        if model.global_cls_token:
            cls_token_id = model.tokenizer.tokenizer.cls_token_id
            if cls_token_id is not None:
                logger.info(
                    f"Seeding global_cls with [CLS] embedding (ID {cls_token_id})"
                )
                with torch.no_grad():
                    cls_embedding = full_bert.embeddings.word_embeddings.weight[
                        cls_token_id
                    ]
                    model.global_cls.data.copy_(cls_embedding.view(1, 1, -1))
            else:
                logger.warning(
                    "global_cls is True but no [CLS] token found in tokenizer; skipping seeding"
                )

    def _copy_bert_weights(self, src, target):
        """
        Copies Self-Attention and FFN weights from src to target.
        Also seeds Cross-Attention weights from Self-Attention weights.
        """
        logger.debug("Copying BERT self-attention and MLP weights")
        target.attention.self.load_state_dict(src.attention.self.state_dict())
        target.attention.output.load_state_dict(src.attention.output.state_dict())
        target.intermediate.load_state_dict(src.intermediate.state_dict())
        target.output.load_state_dict(src.output.state_dict())
        if hasattr(target, "crossattention") and target.crossattention:
            logger.debug("Seeding BERT cross-attention from self-attention weights")
            target.crossattention.self.load_state_dict(src.attention.self.state_dict())
            target.crossattention.output.load_state_dict(
                src.attention.output.state_dict()
            )


class InitMICEModernBERTFromHFID(LightweightTask):
    """Worker-node task to load weights into MICE ModernBERT model"""

    model: Param[ModernBertMiceCrossEncoder]

    def execute(self):
        # Ensure model is instantiated and initialized
        model = self.model
        hf_id = model.hf_id

        # Ensure configs are available
        if not hasattr(model, "config") or model.config is None:
            model.config = AutoConfig.from_pretrained(hf_id)

        if not hasattr(model, "head_config") or model.head_config is None:
            model.head_config = AutoConfig.from_pretrained(hf_id)
            model.head_config.is_decoder = True
            model.head_config.add_cross_attention = True

        # Build the model structure first
        model.initialize()

        logger.info(f"Loading MICE ModernBERT weights from {hf_id}")

        full_backbone = AutoModelForSequenceClassification.from_pretrained(hf_id)

        if hasattr(full_backbone.model, "embeddings"):
            logger.info("Seeding embeddings from backbone")
            model.embeddings.load_state_dict(
                full_backbone.model.embeddings.state_dict()
            )
        else:
            logger.warning(
                f"Backbone {hf_id} has no 'embeddings' attribute; skipping seeding"
            )

        if hasattr(full_backbone.model, "layers"):
            logger.info("Seeding bottom layers from backbone")
            if model.bound_bottom_layers:
                contextualization_module_names = ["bottom_layers"]
            else:
                contextualization_module_names = [
                    "document_bottom_layers",
                    "query_bottom_layers",
                ]

            for cntx_module_name in contextualization_module_names:
                contextualization_module = getattr(model, cntx_module_name)
                for i in range(len(contextualization_module)):
                    if i < len(full_backbone.model.layers):
                        contextualization_module[i].load_state_dict(
                            full_backbone.model.layers[i].state_dict()
                        )
                    else:
                        logger.warning(
                            f"Backbone has only {len(full_backbone.model.layers)} layers; cannot seed bottom layer {i}"
                        )
        else:
            logger.warning(
                f"Backbone {hf_id} has no layers; skipping bottom layer seeding"
            )

        if hasattr(full_backbone.model, "layers"):
            start_idx = model.n_contextualization_layers
            if model.n_interaction_layers is not None:
                end_idx = start_idx + model.n_interaction_layers
                src_layers = full_backbone.model.layers[start_idx:end_idx]
            else:
                src_layers = full_backbone.model.layers[start_idx:]

            logger.info(f"Seeding {len(model.top_layers)} top layers from backbone")
            for i, target_layer in enumerate(model.top_layers):
                if i < len(src_layers):
                    if not model.random_top_layers:
                        self._copy_modernbert_weights(src_layers[i], target_layer)
                else:
                    logger.warning(
                        f"Backbone has only {len(src_layers)} remaining layers; cannot seed top layer {i}"
                    )
        else:
            logger.warning(
                f"Backbone {hf_id} has no layers; skipping top layer seeding"
            )

        if hasattr(full_backbone.model, "rotary_emb"):
            logger.info("Seeding rotary_emb from backbone")
            model.rotary_emb.load_state_dict(
                full_backbone.model.rotary_emb.state_dict()
            )
        else:
            logger.warning(
                f"Backbone {hf_id} has no 'rotary_emb' attribute; skipping seeding"
            )

        if hasattr(full_backbone.model, "final_norm"):
            logger.info("Seeding final_norm from backbone")
            model.final_norm.load_state_dict(
                full_backbone.model.final_norm.state_dict()
            )

        if hasattr(full_backbone, "head"):
            logger.info("Seeding head from backbone")
            model.head.load_state_dict(full_backbone.head.state_dict())

        if model.global_cls_token:
            cls_token_id = model.tokenizer.tokenizer.cls_token_id
            if cls_token_id is not None:
                logger.info(
                    f"Seeding global_cls with [CLS] embedding (ID {cls_token_id})"
                )
                with torch.no_grad():
                    # ModernBert uses tok_embeddings
                    emb_layer = getattr(
                        full_backbone.model.embeddings,
                        "tok_embeddings",
                        getattr(
                            full_backbone.model.embeddings, "word_embeddings", None
                        ),
                    )
                    if emb_layer is not None:
                        cls_embedding = emb_layer.weight[cls_token_id]
                        model.global_cls.data.copy_(cls_embedding.view(1, 1, -1))
                    else:
                        logger.warning(
                            "Could not find embedding layer in ModernBERT; skipping seeding"
                        )
            else:
                logger.warning(
                    "global_cls is True but no [CLS] token found in tokenizer; skipping seeding"
                )

    def _copy_modernbert_weights(self, src, target):
        """
        Copies Attention and MLP weights from src to target.
        Also seeds Cross-Attention weights from Self-Attention weights.
        """
        logger.debug("Copying ModernBERT self-attention and MLP weights")
        target.attn_norm.load_state_dict(src.attn_norm.state_dict())
        target.attn.load_state_dict(src.attn.state_dict())
        target.mlp_norm.load_state_dict(src.mlp_norm.state_dict())
        target.mlp.load_state_dict(src.mlp.state_dict())

        # Seed cross-attention
        logger.debug("Seeding ModernBERT cross-attention from Wqkv weights")
        all_head = src.attn.head_dim * src.attn.config.num_attention_heads
        with torch.no_grad():
            target.crossattention.q_proj.weight.copy_(
                src.attn.Wqkv.weight[0:all_head, :]
            )
            target.crossattention.k_proj.weight.copy_(
                src.attn.Wqkv.weight[all_head : 2 * all_head, :]
            )
            target.crossattention.v_proj.weight.copy_(
                src.attn.Wqkv.weight[2 * all_head : 3 * all_head, :]
            )
            if src.attn.Wqkv.bias is not None:
                target.crossattention.q_proj.bias.copy_(src.attn.Wqkv.bias[0:all_head])
                target.crossattention.k_proj.bias.copy_(
                    src.attn.Wqkv.bias[all_head : 2 * all_head]
                )
                target.crossattention.v_proj.bias.copy_(
                    src.attn.Wqkv.bias[2 * all_head : 3 * all_head]
                )
            target.crossattention.Wo.load_state_dict(src.attn.Wo.state_dict())
            target.crossattention.out_drop.load_state_dict(
                src.attn.out_drop.state_dict()
            )


def mice_scorer(
    hf_id: str,
    n_contextualization_layers: int,
    n_docs_ctx_layers: Optional[int] = None,
    n_interaction_layers: Optional[int] = None,
    bound_bottom_layers: bool = True,
    mask_cls_to_doc: bool = True,
    mask_query_to_cls: bool = True,
    cross_attn_first: bool = True,
    freeze_base: bool = False,
    random_top_layers: bool = False,
    compress_dim: float = 1.0,
    global_cls_token: bool = False,
    extra_attn_bias: bool = False,
    lexical_bias: bool = False,
    pooling_method: Optional[str] = None,
    max_query_length: Optional[int] = None,
    max_doc_length: Optional[int] = None,
    max_length: int = 512,
) -> Tuple[MiceCrossEncoder, List[LightweightTask]]:
    """
    Unified entry point for creating a MICE scorer.
    Automatically selects between BERT and ModernBERT architectures based on the hf_id.

    Args:
        hf_id: Hugging Face checkpoint identifier.
        n_contextualization_layers: Number of bottom encoder layers that process query and document independently.
        n_docs_ctx_layers: Number of bottom encoder layers for the document. If None, use n_contextualization_layers.
        n_interaction_layers: Number of top encoder layers with cross-attention. If None, use all remaining layers from the backbone.
        bound_bottom_layers: whether to bound bottom query and document encoding layers
        mask_cls_to_doc: If True, prevents [CLS] from attending to document tokens.
        mask_query_to_cls: If True, prevents query tokens from attending to [CLS].
        cross_attn_first: Whether to perform cross-attention before self-attention in the top layers.
        freeze_base: If True, freezes the bottom layers.
        random_top_layers: If True, initializes top layers randomly.
        compress_dim: Dimensionality compression factor for top layers.
        pooling_method: (ModernBERT only) "cls" or "mean" pooling.
        max_query_length: Maximum number of tokens for the query.
        max_doc_length: Maximum number of tokens for the document.
        max_length: Maximum total number of tokens (used as default for query/doc if not specified).
    """
    # Automatically unbind bottom layers if n_docs_ctx_layers is specified
    if n_docs_ctx_layers is not None and bound_bottom_layers:
        logger.info(
            "n_docs_ctx_layers provided: automatically setting bound_bottom_layers to False"
        )
        bound_bottom_layers = False

    tokenizer = MICEQueryDocTokenizer.C(
        model_id=hf_id,
        max_query_length=max_query_length,
        max_doc_length=max_doc_length,
        max_length=max_length,
    )

    common_kwargs = dict(
        hf_id=hf_id,
        tokenizer=tokenizer,
        n_contextualization_layers=n_contextualization_layers,
        n_docs_ctx_layers=n_docs_ctx_layers,
        n_interaction_layers=n_interaction_layers,
        bound_bottom_layers=bound_bottom_layers,
        mask_cls_to_doc=mask_cls_to_doc,
        mask_query_to_cls=mask_query_to_cls,
        cross_attn_first=cross_attn_first,
        freeze_base=freeze_base,
        random_top_layers=random_top_layers,
        compress_dim=compress_dim,
        global_cls_token=global_cls_token,
        extra_attn_bias=extra_attn_bias,
        lexical_bias=lexical_bias,
    )

    if "modernbert" in hf_id.lower() or "ettin" in hf_id.lower():
        model = ModernBertMiceCrossEncoder.C(
            **common_kwargs,
            pooling_method=pooling_method,
        )
        return model, [InitMICEModernBERTFromHFID.C(model=model)]

    elif "qwen" in hf_id.lower():
        from .qwen_mice import QwenMiceCrossEncoder, InitMICEQwenFromHFID

        model = QwenMiceCrossEncoder.C(
            **common_kwargs,
            pooling_method=pooling_method or "cls",
        )
        return model, [InitMICEQwenFromHFID.C(model=model)]

    elif "electra" in hf_id.lower():
        from .electra_mice import ElectraMiceCrossEncoder, InitMICEElectraFromHFID

        model = ElectraMiceCrossEncoder.C(**common_kwargs)
        return model, [InitMICEElectraFromHFID.C(model=model)]

    else:
        if not any(
            key in hf_id.lower() for key in ["bert", "minilm", "roberta", "deberta"]
        ):
            logger.warning(
                f"No Backbone recognized for {hf_id}, using default 'BertMiceCrossEncoder' architecture"
            )

        model = BertMiceCrossEncoder.C(**common_kwargs)
        return model, [InitMICEBERTFromHFID.C(model=model)]
