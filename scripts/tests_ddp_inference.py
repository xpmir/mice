import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from lightning.fabric import Fabric
import os

# --- 1. Define a Mock Dataset & Model ---
class RandomDocDataset(Dataset):
    def __init__(self, num_samples=10000, seq_len=128):
        self.num_samples = num_samples
        self.seq_len = seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return dummy input_ids and a document ID
        return {
            "input_ids": torch.randint(0, 1000, (self.seq_len,)), 
            "doc_id": idx
        }

class SimpleBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(128, 768)
    
    def forward(self, x):
        # Simulate an embedding generation
        return self.layer(x.float()).mean(dim=1)

# --- 2. The Inference Loop ---
def run_inference(fabric: Fabric):
    # A. Setup Data
    # Fabric will automatically inject a DistributedSampler here for DDP
    dataset = RandomDocDataset(num_samples=10000)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4)
    dataloader = fabric.setup_dataloaders(dataloader)

    # B. Setup Model
    model = SimpleBackbone()
    model = fabric.setup(model)
    model.eval()

    # C. Prepare Output Directory
    output_dir = "./embeddings_output"
    if fabric.is_global_zero:
        os.makedirs(output_dir, exist_ok=True)
    
    # Ensure dir exists before workers start writing
    fabric.barrier()

    # D. Run Inference
    fabric.print(f"Rank {fabric.global_rank} starting inference...")
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids = batch["input_ids"]
            doc_ids = batch["doc_id"]

            # Fabric handles device placement automatically!
            # No need for input_ids.to(device)
            embeddings = model(input_ids)

            # E. Save Results Immediately (No OOM)
            # We save a separate file per rank and per batch to avoid locking
            save_path = f"{output_dir}/part_rank{fabric.global_rank}_batch{i}.pt"
            
            torch.save({
                "doc_ids": doc_ids.cpu(), 
                "embeddings": embeddings.cpu()
            }, save_path)

    fabric.print("Inference completed successfully.")

if __name__ == "__main__":
    # Initialize Fabric
    # 'devices=4' and 'strategy="ddp"' enables the parallel batch processing
    fabric = Fabric(accelerator="gpu", devices=4, strategy="ddp")
    
    # Launch the multi-GPU process
    fabric.launch(run_inference)