from enum import Enum
from attrs import Factory, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)
from experimaestro.experiments.grid import GridSearch
from xpm_torch.experiments.configuration import TransformerOptimization, Fabric
from xpmir.papers import configuration
from xpmir.experiments.helpers import LauncherSpecification
from xpmir.datasets.msmarco import RerankerMSMarcoV1Configuration
import logging

logger = logging.getLogger(__name__)


class Losses(str, Enum):
    """Possible losses"""

    BCE = "bce"
    """ Binary Cross Entropy loss """

    hingeLoss = "hingeLoss"
    """Hinge loss"""

    infoNCE = "infoNCE"
    """InfoNCE loss, with in-batch negatives"""

    infoNCE_Colbertv2Neg = "infoNCE_Colbertv2Neg"
    """InfoNCE using the negatives sampled by Schlatt et al. 2025 with ColBERTv2"""

    infoNCE_RankDistiLLM = "infoNCE_RankDistiLLM"
    """InfoNCE using the negatives sampled from the RankDistiLLM top-50 pool using MS MARCO qrels"""

    BCE_Colbertv2Neg = "BCE_Colbertv2Neg"
    """Binary Cross Entropy loss with ColBERTv2 negatives"""

    hingeLoss_Colbertv2Neg = "hingeLoss_Colbertv2Neg"
    """Hinge loss with ColBERTv2 negatives"""

    marginMSE = "marginMSE"
    """Margin Mean Squared Error loss from hofstatter et al. 2020"""

    distillRankNET = "distillRankNET"
    """Distillation version of RankNET loss from Schlatt et al. 2025"""

    ADR_MSE = "ADR_MSE"
    """Listwise distillation loss proposed by Schlatt et al. 2025"""

    MSE_mixedbread_large = "MSE_mixedbread_large"
    """Plain pointwise MSE loss on cross-encoder/ettin-reranker-v1-data teacher scores"""

    pearson_mixedbread_large = "pearson_mixedbread_large"
    """Pearson correlation loss (1 - r) across batch pairs on cross-encoder/ettin-reranker-v1-data teacher scores"""


class PoolingMethod(str, Enum):
    """Possible pooling methods"""

    CLS = "cls"
    """CLS token pooling"""

    MEAN = "mean"
    """Mean pooling"""


class ScorerType(str, Enum):
    """Supported neural scorer architectures."""

    stCrossEncoder = "stCrossEncoder"
    stMultiVectorEncoder = "stMultiVectorEncoder"
    hfCrossScorer = "hfCrossScorer"
    miceV2 = "miceV2"
    from_experiment = "from_experiment"
    auto = "auto"

    @classmethod
    def from_string(cls, val: Union[str, "ScorerType"]) -> "ScorerType":
        """Resolve a string or ScorerType instance to a ScorerType enum member."""
        if isinstance(val, cls):
            return val
        if not isinstance(val, str):
            raise TypeError(f"Expected str or ScorerType, got {type(val).__name__}")

        normalized = val.strip()
        cleaned = normalized.lower().replace("-", "").replace("_", "")
        for member in cls:
            if cleaned == member.value.lower().replace("-", "").replace(
                "_", ""
            ) or cleaned == member.name.lower().replace("-", "").replace("_", ""):
                return member

        aliases = {
            "hf": cls.hfCrossScorer,
            "hfcrossscorer": cls.hfCrossScorer,
            "st": cls.stCrossEncoder,
            "stcrossencoder": cls.stCrossEncoder,
            "sentencetransformers": cls.stCrossEncoder,
            "multivector": cls.stMultiVectorEncoder,
            "stmultivector": cls.stMultiVectorEncoder,
            "stmultivectorencoder": cls.stMultiVectorEncoder,
            "colbert": cls.stMultiVectorEncoder,
            "micev2": cls.miceV2,
            "mice": cls.miceV2,
            "fromexperiment": cls.from_experiment,
            "experiment": cls.from_experiment,
            "auto": cls.auto,
        }
        if cleaned in aliases:
            return aliases[cleaned]

        raise ValueError(
            f"Unknown scorer_type: '{val}'. "
            f"Expected one of {[e.value for e in cls]} or aliases {list(aliases.keys())}"
        )


class Validation(str, Enum):
    """Possible validation subsets"""

    MSMARCO = "msmarco"
    """Nano MSMARCO validation subset"""

    NanoBEIR = "nanobeir"
    """A small subset of BEIR datasets designed specifically for validation"""

    NanoBEIR11 = "nanobeir11"
    """NanoBEIR excluding argana and touche-2020 as done in [Sentence-Transformers](https://www.sbert.net/docs/package_reference/cross_encoder/evaluation.html#crossencodernanobeirevaluator)"""

    ALL = "all"
    """ Both Nano MSMARCO and NanoBEIR validations"""


@configuration()
class Indexation(LauncherSpecification):
    batch_size: int = 512
    max_indexed: int = 0

    requirements: str = "duration=2 days & cpu(cores=8)"
    sparse2bmp_requirements: str = "duration=1d & cuda(mem=24G)"
    sparse2seismic_requirements: str = "duration=1d & cpu(cores=8)"

    long_evals: Optional[List[str]] = None
    """List of substrings. If a dataset name matches one, indexing will route to long_launcher_index. Defaults to None, inheriting evaluation's long_evals."""

    long_requirements: Optional[str] = None
    """Requirements string used to instantiate the specialized long_launcher_index for indexing."""


@configuration()
class Retrieval:
    k: GridSearch[int] = 1000
    batch_size: GridSearch[int] = 128
    chunk_size: GridSearch[Optional[int]] = None
    """Micro-batch chunk size for Stage 1 candidate retrieval and Stage 2 reranking.
    If None, retrieves all candidates in one go before Stage 2 reranking."""

    in_memory: bool = False
    """Whether to load the sparse retriever index in memory instead of memory-mapping."""

    requirements: str = "duration=2 days & cuda(mem=24G)"

    long_evals: List[str] = Factory(list)
    """List of substrings. If a dataset name matches one, evaluation will route to the long_launcher."""

    long_requirements: Optional[str] = None
    """Requirements string used to instantiate the specialized long_launcher (e.g., requesting multi-GPU)."""


@configuration()
class Preprocessing:
    requirements: str = "duration=12h & cpu(cores=4)"
    blocking_download: bool = False
    """Whether to download and build dataset document stores sequentially in the main process before experiment task submission"""


class CheckpointSelection(str, Enum):
    """Possible checkpoints to evaluate"""

    VAL = "val"
    """Evaluate only the best validation checkpoint(s)"""

    BEST = "best"
    """Alias for VAL: evaluate best validation checkpoint(s)"""

    LAST = "last"
    """Evaluate only the last checkpoint"""


@configuration()
class Evaluation:
    """What datasets to evaluate on, eventually limit the number of queries for debug"""

    test_max_topics: int = 0
    """Development test size (0 to leave it like this)"""

    all_datasets: bool = False
    """Whether to evaluate on all BEIR datasets (minus the 5 not publicly available)"""

    in_domain: bool = False
    """Whether to evaluate on in-domain datasets (MSMarco, TREC DL 19 and 20)"""

    beir13: bool = False
    """Whether to evaluate on all BEIR13 datasets"""

    beir13_decontaminated: bool = False
    """Whether to evaluate on all LightOn BEIR13 decontaminated datasets"""

    lotte_search: bool = False
    """Whether to evaluate on all LOTTE Search datasets"""

    robust04: bool = False
    """Whether to evaluate on Robust04"""

    nanobeir: bool = False
    """Whether to evaluate on NanoBEIR datasets"""

    limit: bool = False
    """Whether to evaluate on LIMIT theoretical stress-test benchmarks"""

    eval_checkpoint: Union[
        List[Union[CheckpointSelection, int, str]], CheckpointSelection, int, str
    ] = CheckpointSelection.LAST
    """Checkpoint(s) to evaluate: 'last', 'val'/'best', a specific epoch int, or a list like ['last', 500]"""

    datasets: List[str] = Factory(list)
    """List of specific datasets to evaluate on"""

    include_recall_100: bool = False
    """Whether to additionally report R@100 alongside CE_MEASURES (Success@5, nDCG@10, nDCG@20, RR@10)"""

    fabric: Fabric = Factory(Fabric)
    """Configuration for Fabric device management during evaluation"""

    def get_checkpoint_modes(self) -> List[Union[CheckpointSelection, int]]:
        raw = self.eval_checkpoint
        items = raw if isinstance(raw, list) else [raw]
        modes: List[Union[CheckpointSelection, int]] = []
        for item in items:
            if isinstance(item, int):
                modes.append(CheckpointSelection.LAST if item == -1 else item)
            elif isinstance(item, str):
                item_lower = item.strip().lower()
                if item_lower == "both":
                    raise ValueError(
                        "Option 'both' has been removed. Please specify a list of checkpoints instead, e.g. ['val', 'last'] or [last, '500']."
                    )
                elif item_lower in ("val", "best", "best_val", "best val"):
                    modes.append(CheckpointSelection.VAL)
                elif item_lower in ("last", "-1"):
                    modes.append(CheckpointSelection.LAST)
                else:
                    try:
                        modes.append(int(item_lower))
                    except ValueError:
                        modes.append(CheckpointSelection(item))
            elif isinstance(item, CheckpointSelection):
                modes.append(item)
            else:
                modes.append(CheckpointSelection(item))

        # Deduplicate while preserving order
        seen = set()
        unique_modes = []
        for m in modes:
            if m not in seen:
                seen.add(m)
                unique_modes.append(m)
        return unique_modes

    def get_checkpoint_mode(self) -> List[Union[CheckpointSelection, int]]:
        return self.get_checkpoint_modes()


@configuration()
class xpm_torch_Learner:
    validation_interval: GridSearch[int] = field(default=32)

    validation_top_k: GridSearch[int] = 1000

    checkpoint_interval: GridSearch[int] = field(default=32)

    optimization: TransformerOptimization = Factory(TransformerOptimization)

    requirements: str = "duration=4 days & cuda(mem=24G) * 2"

    sample_rate: GridSearch[float] = 1.0
    """Sample rate for triplets"""

    sample_max: GridSearch[int] = 0
    """Maximum number of samples considered (before shuffling). 0 for no limit."""

    max_grad_norm: GridSearch[float] = 0.0
    """Maximum gradient norm (0 for no clipping)"""

    loss: GridSearch[str] = Losses.marginMSE.value
    """Loss function to use"""

    validation: str = Validation.NanoBEIR.value
    """ The validation subset to use """

    early_stop_epochs: GridSearch[int] = 0
    """ number of **epochs** without improvements before early stopping based on validation"""

    fabric: Fabric = Factory(Fabric)
    """Configuration for Fabric device management"""


@configuration()
class PlaidConfiguration:
    """
    Configuration for PLAID integration in MICE retrieval pipeline.
    """

    use_colbert_first_stage: bool = False
    """Whether to additionally evaluate a genuine ColBERT-style first-stage
    retriever using MICE's own bottom-layer contextualized embeddings
    (standalone, and reranked by MICE's top interaction layers), alongside
    the existing SPLADE/BM25-based baselines. Requires use_plaid=True."""

    ### Indexation params ###
    # TODO: Remove `dim` once previous runs with hardcoded dim=128 are finished/deprecated
    dim: GridSearch[Optional[int]] = None
    """Projection dimension of the per-token vectors (if None, inferred from model weights)."""

    max_doc_length: GridSearch[Optional[int]] = None
    """Maximum number of tokens per document stored in the PLAID index.
    When set, overrides the encoder's default to force storing only up to this many vectors."""

    max_query_length: GridSearch[Optional[int]] = None
    """Maximum number of tokens for queries in PLAID search and reranking.
    When set, overrides the encoder's default max_query_length."""

    buffer_size: GridSearch[int] = 1000
    """Number of documents to use for creating/updating the PLAID index"""

    fast_plaid_batch_size: GridSearch[int] = 25000
    """Fast-PLAID internal batch size (in tokens) used by the Rust engine for quantization
    and shard writing. Defaults to 25000 (native Fast-PLAID default)."""

    n_bits: GridSearch[int] = 2
    """Number of bits for residual quantization in PLAID"""

    kmeans_niters: int = 4
    """Number of K-means iterations for PLAID clustering"""

    n_samples_kmeans: int = 0
    """Number of token samples used to train the centroids (0 = fast-plaid
    default)."""

    max_points_per_centroid: int = 256
    """Maximum number of points (documents) per centroid. Controls the creation of new centroids."""

    compress_only: bool = False
    """Whether to build a compress-only index (no IVF search). For MICE models,
    compress_only is automatically enforced as True in MicePlaidRetrieverFactory
    unless use_colbert_first_stage is enabled."""

    force_cpu_indexing: bool = False
    """When True, forces the use of CPU for indexing even if a GPU is available.
    This can be useful to avoid GPU OOM errors during indexing, especially for large corpora."""

    ### Retrieval params ###
    n_ivf_probe: int = 8
    """Number of IVF clusters to probe in PLAID (lower = faster, less accurate)"""

    n_full_scores: int = 0
    """Number of candidates for which fast-plaid computes full scores
    (0 = fast-plaid default)."""


@configuration()
class SeismicConfiguration:
    """Configuration for Seismic approximate search on learned sparse indices (SPLADE)."""

    enabled: GridSearch[bool] = False
    """Whether to use Seismic approximate search instead of exact MaxScore."""

    # Build / conversion parameters
    n_postings: GridSearch[int] = 6000
    """Average number of postings kept per term (global budget pruning)."""

    centroid_fraction: float = 0.1
    """Number of blocks (k-means centroids) per posting list, as a fraction of its length."""

    summary_energy: float = 0.4
    """Fraction of the L1 mass kept in each block summary."""

    max_fraction: float = 1.5
    """Maximum posting list length, as a multiple of n_postings."""

    knn: int = 0
    """Number of nearest neighbours stored per document (0 = no kNN graph)."""

    # Query / retrieval parameters
    query_cut: GridSearch[int] = 10
    """Only the query_cut highest-weighted query terms are traversed."""

    heap_factor: GridSearch[float] = 0.7
    """Blocks scoring below heap_factor * current k-th score are skipped (1.0 = no summary pruning)."""

    n_knn: int = 0
    """Number of kNN neighbours used to refine the results (requires knn > 0 at build time)."""


@configuration()
class CE_FineTuning(RerankerMSMarcoV1Configuration):
    use_plaid: GridSearch[bool] = False
    """Whether to evaluate with PLAID index and retriever."""

    seismic: SeismicConfiguration = Factory(SeismicConfiguration)
    """Configuration for Seismic approximate search on SPLADE index."""

    nb_repetitions: int = field(default=1)
    """Number of repetitions of the training process"""

    pref_attn_implementation: Optional[str] = field(default=None)
    """Attention implementation for HuggingFace models (e.g. 'flash_attention_2', 'sdpa', 'eager')"""

    scorer_type: GridSearch[str] = ScorerType.stCrossEncoder.value
    """Type of scorer model: stCrossEncoder, stMultiVectorEncoder, or hfCrossScorer"""

    similarity_fn_name: Optional[str] = field(default=None)
    """Similarity function for multi-vector scorer ('maxsim' or 'meanmaxsim'). When None, defaults to 'meanmaxsim' for MSE_mixedbread_large / pearson_mixedbread_large and 'maxsim' otherwise."""

    pearson_eps: float = field(default=1e-6)
    """Numerical stability epsilon for Pearson correlation loss to prevent zero-variance collapse."""

    freeze_LN: GridSearch[bool] = False
    """Whether to freeze LayerNorm / normalization parameters during training."""

    freeze_LN_pattern: str = r"(?i)norm"
    """Regex pattern used to match parameters to freeze when freeze_LN is True (defaults to '(?i)norm')."""

    ettin_subset_exclude: List[str] = field(factory=list)
    """The subset config names of cross_encoder.ettin_reranker_v1_data to exclude from training"""

    indexation: Indexation = Factory(Indexation)
    retrieval: Retrieval = Factory(Retrieval)

    learner: xpm_torch_Learner = Factory(xpm_torch_Learner)

    preprocessing: Preprocessing = Factory(Preprocessing)

    evaluation: Evaluation = Factory(Evaluation)

    plaid: PlaidConfiguration = Factory(PlaidConfiguration)

    ## Retriever Model
    retriever: str = ""
    """Identifier for the retriever model. If empty, uses BM25."""

    precompute_first_stage: bool = True
    """If true, will save the run for the first stage - just reload it for evaluation rather than recomputing"""

    from_experiment: GridSearch[Optional[str]] = None
    """Optional experiment ID (e.g. 'exp_id' or 'exp_id/run_id' to pin a specific run), local model directory path, or HF Hub model ID to instantiate a trained model from results/models"""

    from_run_id: Optional[str] = None
    """Optional run ID within from_experiment (e.g. '20260924_180227')"""

    ## Cross Encoder Model
    base: GridSearch[str] = ""
    """Identifier for the base model"""

    max_length: Optional[int] = None
    """max len for scorer, default to 0 = max len of the model"""

    max_query_length: Optional[int] = None
    """Maximum query length for cross-encoders"""

    max_doc_length: Optional[int] = None
    """Maximum document length for cross-encoders"""

    pooling_method: str = PoolingMethod.CLS.value
    """Pooling method to use for the ModernBert based scorer: cls or mean"""

    compare_with_baseline: bool = False
    """After evaluations are done, whether to test statistical significance against a baseline.
    By default, the baseline is BM25 + the CE simply fine-tuned on the same setup."""

    save_runs: bool = False
    """Whether to save the evaluation runs in the best model folders"""

    eval_sparsity: bool = False
    """Whether to compute token sparsity, mean utility, and token retention ratio during evaluation benchmarks"""

    get_latency: bool = False
    """Whether to extract evaluation query latency from task logs and sync into aggregated.txt"""

    export_trained_models: bool = True
    """Whether to export the best models to the models/ folder"""

    export_only_best: bool = True
    """Whether to export only the single best performing model across all configurations (default: True).
    If False, exports the best model for each grid search configuration."""

    normalize_docs_per_batch: bool = True
    """whether to normalize documents per batch for listwise losses"""

    grid_search: Dict[str, Any] = field(factory=dict)
    """
    Grid search parameters. Maps a dot-separated parameter path to values,
    or contains 'config_dicts' for coupled parameter combinations.
    Example in YAML:
    grid_search:
      learner.optimization.lr:
        values_list: [1e-5, 2e-5]
      pooling_method:
        value: "cls"
      config_dicts:
        - reg_budget: null
          l2_reg: false
    """
