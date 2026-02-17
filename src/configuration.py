from enum import Enum
from attrs import Factory, field
from typing import Any, List, Optional, Tuple
from xpmir.papers import configuration
from xpmir.papers.helpers import LauncherSpecification
from xpmir.papers.helpers.optim import TransformerOptimization
from xpmir.papers.helpers.msmarco import RerankerMSMarcoV1Configuration
from functools import cached_property as attrs_cached_property
from xpmir.learning.devices import CudaDevice, BestDevice, Device

import logging

logging.basicConfig(level=logging.INFO)

class Losses(str, Enum):
    """Possible losses"""

    marginMSE = "marginMSE"
    """Margin Mean Squared Error loss from hofstatter et al. 2020"""

    PointWiseMSE = "PointWiseMSE"
    """ Point Wise Mean Squared Error loss """

class PoolingMethod(str, Enum):
    """Possible pooling methods"""

    CLS = "cls"
    """CLS token pooling"""

    MEAN = "mean"
    """Mean pooling"""

@configuration()
class Indexation(LauncherSpecification):
    batch_size: int = 512
    max_indexed: int = 0

    requirements: str = "duration=2 days & cpu(cores=8)"
    sparse2bmp_requirements: str = "duration=1d & cuda(mem=24G)"

@configuration()
class xpm_torch_Learner:
    validation_interval: int = field(default=32)
    validation_top_k: int = 1000

    checkpoint_interval: int = field(default=32)

    optimization: TransformerOptimization = Factory(TransformerOptimization)
    requirements: str = "duration=4 days & cuda(mem=24G) * 2"
    sample_rate: float = 1.0
    """Sample rate for triplets"""

    sample_max: int = 0
    """Maximum number of samples considered (before shuffling). 0 for no limit."""

    max_grad_norm: float = 0.0
    """Maximum gradient norm (0 for no clipping)"""

    loss: str = Losses.marginMSE.value
    """Loss function to use"""

    ## Lighnting Fabric parameters see https://lightning.ai/docs/fabric/stable/api/generated/lightning.fabric.fabric.Fabric.html#lightning.fabric.fabric.Fabric 

    strategy: str = "auto"
    """Distributed training strategy"""

    precision: Optional[str] = None
    """Precision to use - e.g., '16-mixed', 'bf16-mixed', etc."""

    accelerator: str = "auto"
    """ Accelerator to use """


@configuration()
class Retrieval:
    k: int = 1000
    batch_size: int = 128
    requirements: str = "duration=2 days & cuda(mem=24G)"


@configuration()
class Preprocessing:
    requirements: str = "duration=12h & cpu(cores=4)"

@configuration()
class Layer_params:
    """Either a single layer (value) or an explicit range (values_range)."""
    value: Optional[int] = 0
    values_range: Optional[Tuple[int, int]] = None

    def get_content(self) -> Any:
        if self.value is not None:
            return self.value
        if self.values_range is not None:
            return self.values_range
        raise ValueError("Either value or values_range must be set.")

    def get_content_as_list(self) -> str:
        if self.value is not None:
            return [self.value]
        if self.values_range is not None:
            return list(range(self.values_range[0], self.values_range[1]))
        raise ValueError("Either value or values_range must be set.")
    
    def _validate(self):
        if self.value is not None and not isinstance(self.value, int):
            raise TypeError(f"value must be an int or None, got {self.value!r}")
        if self.values_range is not None:
            if not (
                isinstance(self.values_range, (tuple, list))
                and len(self.values_range) == 2
                and isinstance(self.values_range[0], int)
                and isinstance(self.values_range[1], int)
                and self.values_range[1] >= self.values_range[0]
            ):
                raise TypeError(
                    f"values_range must be a pair of ints (start <= end), got {self.values_range!r}"
                )
        if self.value is not None and self.values_range is not None:
            logging.warning("Both value and values_range are set. Defaulting to value.")
        if self.value is None and self.values_range is None:
            raise ValueError("Either value or values_range must be set.")
        
    @staticmethod
    def from_any(obj: Any, default: int = 0) -> "Layer_params":
        """Normalize int, dict/DictConfig or Layer_params into a Layer_params instance."""
        from omegaconf import DictConfig  # local import to avoid top-level dependency issues
        if isinstance(obj, Layer_params):
            return obj
        if isinstance(obj, int):
            return Layer_params(value=obj)
        if obj is None:
            return Layer_params(value=default)
        if isinstance(obj, DictConfig) or isinstance(obj, dict):
            od = dict(obj)
            if "value" in od:
                return Layer_params(value=int(od["value"]))
            if "values_range" in od:
                vr = od["values_range"]
                if isinstance(vr, (list, tuple)) and len(vr) == 2:
                    return Layer_params(value=None, values_range=(int(vr[0]), int(vr[1] + 1))) # Add +1 to include the upper_bound specified in the config
        raise TypeError(f"Cannot convert {obj!r} to Layer_params")


@configuration()
class Attn_patch:
    mask_attention_from : List[str]
    """tokens from which attention will be masked
    can be  in 'cls', 'query', 'document' or 'sep_1', 'sep_2'
    """

    mask_attention_to : List[str]
    """tokens to which attention will be masked
    can be in 'cls', 'query', 'document' or 'sep_1', 'sep_2'
    """

    # use Layer_params for both start and end; defaults use single int values
    start_layer: Any = 0
    """Start layer for attention masking (use .value for single layer or .values_range for a range)"""

    end_layer: Any = -1
    """End layer for attention masking (use .value for single layer or .values_range for a range)"""

@configuration()
class Evaluation:
    test_max_topics: int = 0
    """Development test size (0 to leave it like this)"""

    in_domain_only: bool = False
    """Whether to evaluate only on in-domain datasets (MSMarco, TREC DL 19 and 20)"""
    
    all_datasets: bool = False
    """Whether to evaluate on all BEIR datasets (minus the 5 not publicly available)"""

@configuration()
class FrankenCE_Finetuning(RerankerMSMarcoV1Configuration):
    
    nb_repetitions: int = field(default=1)
    """Number of repetitions of the training process"""
    
    indexation: Indexation = Factory(Indexation)
    retrieval: Retrieval = Factory(Retrieval)

    learner: xpm_torch_Learner = Factory(xpm_torch_Learner)
    
    preprocessing: Preprocessing = Factory(Preprocessing)

    evaluation: Evaluation = Factory(Evaluation)

    ## Retriever Model
    retriever: str = ""
    """Identifier for the retriever model. If empty, uses BM25."""
    
    ## Cross Encoder Model
    base: str = "bert-base-uncased"
    """Identifier for the base model"""

    pooling_method: str = PoolingMethod.CLS.value
    """Pooling method to use for the Ettin based scorer: cls or mean"""

    attn_patches: List[Attn_patch] = []    

    compare_with_baseline: bool = False
    """After evaluations are done, whether to test statistical significance against a baseline.
    By default, the baseline is BM25 + the CE simply fine-tuned on the same setup."""

    @attrs_cached_property
    def deterministic_device(self) -> Device:
        if self.use_best_device:
            return BestDevice.C()
        return CudaDevice.C(gpu_determ=True) if self.gpu else Device.C()

@configuration()
class   MidFusionCE_Finetuning(RerankerMSMarcoV1Configuration):
    
    nb_repetitions: int = field(default=1)
    """Number of repetitions of the training process"""
    
    indexation: Indexation = Factory(Indexation)
    retrieval: Retrieval = Factory(Retrieval)

    learner: xpm_torch_Learner = Factory(xpm_torch_Learner)
    
    preprocessing: Preprocessing = Factory(Preprocessing)

    evaluation: Evaluation = Factory(Evaluation)

    ## Retriever Model
    retriever: str = ""
    """Identifier for the retriever model. If empty, uses BM25."""
    
    ## Cross Encoder Model
    base: str = "bert-base-uncased"
    """Identifier for the base model"""

    merge_layer: Any = 6
    """Layer at which to split the model for mid-fusion : value or values_range"""

    drop_layer: Any = 0
    """Layer at which to drop backbone layers, Layer_params : value or values_range"""

    compress_dim: Any = 1
    """Factor to use for dimension compression in the model."""

    use_self_attention: bool = True
    """Whether to use self-attention in the fusion layers"""

    random_top_layers: bool = False
    """Whether to initialize the top layers randomly instead of copying from the base model"""

    freeze_base: bool = False
    """Whether to freeze the base model during finetuning"""
    
    pooling_method: str = PoolingMethod.CLS.value
    """Pooling method to use for the Ettin based scorer: cls or mean"""

    compare_with_baseline: bool = False
    """After evaluations are done, whether to test statistical significance against a baseline.
    By default, the baseline is BM25 + the CE simply fine-tuned on the same setup."""

    @attrs_cached_property
    def deterministic_device(self) -> Device:
        if self.use_best_device:
            return BestDevice.C()
        return CudaDevice.C(gpu_determ=True) if self.gpu else Device.C()
