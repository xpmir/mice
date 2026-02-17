from attrs import Factory, field
from xpmir.papers import configuration
from xpmir.papers.helpers.optim import TransformerOptimization
from xpmir.papers.helpers.msmarco import RerankerMSMarcoV1Configuration

from configuration import Indexation, Retrieval


@configuration
class TransformerOptimization:
    """Configuration for a transformer optimization"""
    warmup_ratio: float = 0
    batch_size: int = 64
    num_epochs: int = 1
    lr: float = 3.0e-6
    weight_decay: float = 1e-2
    eps: float = 1e-8


@configuration()
class Learner:
    validation_steps: int = field(default=5000)
    validation_top_k: int = 1000

    checkpoint_steps: int = field(default=5000)

    logging_steps: int = field(default=4000)

    optimization: TransformerOptimization = Factory(TransformerOptimization)

    requirements: str = "duration=4 days & cuda(mem=24G) * 2"

    sample_max: int = field(default=2_000_000)
    """Maximum number of samples considered (before shuffling). 0 for no limit."""

    max_grad_norm: float = field(default=1.0)
    """Maximum gradient norm"""

    use_fp16: bool = False
    """Use float16 when training"""

    use_bf16: bool = False
    """Use bfloat16 when training"""

    dataloader_num_workers: int = 4

@configuration()
class Evaluation:
    retrieval: Retrieval = Factory(Retrieval)
    indexation: Indexation = Factory(Indexation)

    test_max_topics: int = 0
    """Development test size (0 to leave it like this)"""

    all_datasets: bool = False
    """Whether to use all datasets defined in the paper_tests function"""

    save_model_requirements: str = "duration=1h & cpu(cores=5)"


@configuration()
class HFFinetuning(RerankerMSMarcoV1Configuration):
    nb_repetitions: int = field(default=1)
    """Number of repetitions of the training process"""
    
    learner: Learner = Factory(Learner)
    test_size: int = field(default=10_000)

    model_id: str = "microsoft/MiniLM-L12-H384-uncased"
    """Identifier for the base model"""

    evaluation: Evaluation = Factory(Evaluation)
    loader_requirements: str = "duration=12h & cpu(cores=4)"

    deterministic_gpu: bool = False
    """Make GPU operations deterministic (may be slower)"""