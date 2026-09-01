from enum import Enum
from attrs import Factory, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
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


class PoolingMethod(str, Enum):
    """Possible pooling methods"""

    CLS = "cls"
    """CLS token pooling"""

    MEAN = "mean"
    """Mean pooling"""


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


@configuration()
class Retrieval:
    k: GridSearch[int] = 1000
    batch_size: GridSearch[int] = 128
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

    LAST = "last"
    """Evaluate only the last checkpoint"""

    BOTH = "both"
    """Evaluate both best validation and last checkpoints"""


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

    eval_checkpoint: CheckpointSelection = CheckpointSelection.LAST
    """Checkpoint(s) to evaluate: 'val' (best validation), 'last' (final epoch), or 'both'"""

    datasets: List[str] = Factory(list)
    """List of specific datasets to evaluate on"""

    fabric: Fabric = Factory(Fabric)
    """Configuration for Fabric device management during evaluation"""

    def get_checkpoint_mode(self) -> CheckpointSelection:
        return CheckpointSelection(self.eval_checkpoint)


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

    validation: GridSearch[str] = Validation.NanoBEIR.value
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

    use_plaid: bool = False
    """Whether to use PLAID for retrieval"""

    ### Indexation params ###
    buffer_size: int = 1000
    """Number of documents to use for creating/updating the PLAID index"""

    batch_size: int = 25_000
    """Batch size (in tokens) when encoding documents for PLAID"""

    dim: int = 128
    """Per-token embedding dimension for PLAID index"""

    n_bits: int = 2
    """Number of bits for residual quantization in PLAID"""

    kmeans_niters: int = 4
    """Number of K-means iterations for PLAID clustering"""

    n_samples_kmeans: int = 0
    """Number of token samples used to train the centroids (0 = fast-plaid
    default)."""

    max_points_per_centroid: int = 256
    """Maximum number of points (documents) per centroid. Controls the creation of new centroids."""

    compress_only: bool = False
    """Whether to build a compress-only index (no IVF search)"""

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
class CE_FineTuning(RerankerMSMarcoV1Configuration):
    nb_repetitions: int = field(default=1)
    """Number of repetitions of the training process"""

    pref_attn_implementation: Optional[str] = field(default=None)
    """Attention implementation for HuggingFace models (e.g. 'flash_attention_2', 'sdpa', 'eager')"""

    use_st_scorer: bool = field(default=True)
    """Use sentence-transformers STCrossEncoder instead of HFCrossScorer"""

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

    export_trained_models: bool = True
    """Whether to export the best models to the models/ folder"""

    normalize_docs_per_batch: bool = True
    """whether to normalize documents per batch for listwise losses"""

    grid_search: Dict[str, GridSearch[Any]] = field(factory=dict)
    """
    Grid search parameters. Maps a dot-separated parameter path to a GenericParams object.
    Example in YAML:
    grid_search:
      learner.optimization.lr:
        values_list: [1e-5, 2e-5]
      pooling_method:
        value: "cls"
    """
