import sys
from experimaestro.experiments import mock_modules

# Mock PyTorch and related modules
mock_modules(
    # Modules to mock (submodules are automatically included)
        [
            "torch",
            "torchmetrics",
            "pytorch_lightning",
            "lightning",
            "transformers",
            "sentence_transformers",
            "pylate",
            "huggingface_hub",
            "xpmir.learning.losses",
        ],
    # Decorators to make no-ops
    decorators=[
        "torch.compile",
        "torch.jit.script",
        "torch.jit.unused",
        "torch.jit.export",
        "torch.jit.ignore",
        "torch.no_grad",
        "torch.inference_mode",
    ],
)