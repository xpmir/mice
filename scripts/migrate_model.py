import torch
from xpmir.letor.records import PointwiseItems
from xpm_torch.huggingface import TorchHFHub
import logging, shutil
from pathlib import Path
from experimaestro import Config

from logging_utils import setup_logging
setup_logging(level=logging.INFO)

xps_root = Path.home() / "code" / "experiments" / "JZ"
model_path = "train_mice_ettin32m_l4+4/20260521_194044/results/models/Mice-l4+4-ettin-32m"

def get_scores(model):
    """Run a simple inference pass"""
    model.eval()
    queries = ["What is the capital of France?"]
    documents = ["Paris is the capital and most populous city of France."]
    input_records = PointwiseItems.from_texts(topics=queries, documents=documents)
    with torch.no_grad():
        output = model(input_records)
    return output

# 1. Load the original model
print("\n--- Phase 1: Loading original model ---")
# mice_model = TorchHFHub.from_pretrained(xps_root / model_path, as_instance=False)
loader_cfg = TorchHFHub.pretrained_loader(xps_root / model_path, as_instance=False)
mice_config = loader_cfg.model
loader = loader_cfg.instance(keep=True)
loader.execute()  # This will load the model and its configuration

mice_model = loader.model
original_scores = get_scores(mice_model)


print("Original model loaded.")
print(mice_config)
print(loader_cfg)
print("Model architecture:")
print(mice_model)
print(f"Original scores: {original_scores}")


# 2. Save it locally
save_path = Path("./local/test_mice_saving")
if save_path.exists():
    print(f"Warning: {save_path} already exists. It will be overwritten.")
    shutil.rmtree(save_path)


print(f"\n--- Phase 2: Saving model to {save_path} ---")

mice_model.save_model(save_path)  # This saves the model and its config to the local path
# then save the config with it.

new_loader = mice_config.loader_config(loader_cfg.path)

TorchHFHub(new_loader).save_pretrained(save_path)

# List files in the local directory to verify
print("Files in local directory:")
for f in save_path.iterdir():
    print(f" - {f.name}")

# 3. Reload from local to verify self-contained loading
print("\n--- Phase 3: Reloading model from local path ---")
# Instantiate a new model using the original configuration
# and call load_model(save_path) which we updated to prefer local configs.

reloaded_model = TorchHFHub.from_pretrained(save_path)
reloaded_model.initialize()

reloaded_scores = get_scores(reloaded_model)

print("\nFull migration cycle successful!")
print(f"Reloaded model config name/path: {reloaded_model.config._name_or_path}")
print(f"Reloaded scores: {reloaded_scores}")

# Comparison
if torch.allclose(original_scores, reloaded_scores, atol=1e-5):
    print("SUCCESS: Inference results are identical!")
else:
    print("ERROR: Inference results mismatch!")
    print(f"Difference: {original_scores - reloaded_scores}")

# Check if it loaded from local
if str(save_path) in reloaded_model.config._name_or_path:
    print("SUCCESS: Model configuration loaded from local path.")
else:
    print("WARNING: Model configuration might have been fetched from the Hub.")
