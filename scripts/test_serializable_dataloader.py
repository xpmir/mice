import os
import pickle
import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from lightning.fabric import Fabric
from typing import Iterator, Sized, Optional, List, Tuple, NamedTuple
import itertools

# --- Mock objects from the user's project for a self-contained test ---
class TopicRecord(NamedTuple):
    id: int
class DocumentRecord(NamedTuple):
    id: int
    score: float
class PairwiseDistillationSample(NamedTuple):
    query: TopicRecord
    documents: Tuple[DocumentRecord, DocumentRecord]

# --- 1. A Custom Serializable Sampler ---

class SerializableDistributedSampler(Sampler[int]):
    """
    A Sampler that can be serialized and resumed, behaving like DistributedSampler.
    """
    def __init__(self, dataset: Sized, num_replicas: Optional[int] = None, rank: Optional[int] = None, shuffle: bool = True, seed: int = 0):
        if num_replicas is None or rank is None:
            raise ValueError("num_replicas and rank must be provided.")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed

        self.num_samples = -(-len(self.dataset) // self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

        self._indices: List[int] = []
        self._current_index = 0

    def _generate_indices(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Pad indices to be divisible by num_replicas
        padding_size = self.total_size - len(indices)
        if padding_size > 0:
            indices += indices[:padding_size]

        self._indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(self._indices) == self.num_samples, f"Expected {self.num_samples} indices for rank {self.rank}, but got {len(self._indices)}"

    def __iter__(self) -> Iterator[int]:
        if not self._indices:
            self._generate_indices()

        indices_to_yield = self._indices[self._current_index:]
        for i in indices_to_yield:
            yield i

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._current_index = 0
        self._indices = [] # Clear indices to force regeneration

    def __getstate__(self):
        return {
            "num_replicas": self.num_replicas,
            "rank": self.rank,
            "epoch": self.epoch,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "_current_index": self._current_index,
        }

    def __setstate__(self, state):
        # Just load the values, don't trigger side effects
        self.num_replicas = state["num_replicas"]
        self.rank = state["rank"]
        self.epoch = state["epoch"]
        self.shuffle = state["shuffle"]
        self.seed = state["seed"]
        self._current_index = state["_current_index"]

        # Clear computed properties; they will be rebuilt
        self.dataset = None
        self._indices = []
        self.num_samples = 0
        self.total_size = 0

# --- 2. A Stateful DataLoader Wrapper ---
class StatefulDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, **kwargs):
        if 'sampler' not in kwargs or not isinstance(kwargs['sampler'], SerializableDistributedSampler):
             raise ValueError("StatefulDataLoader requires an instance of SerializableDistributedSampler.")
        super().__init__(dataset, **kwargs)

    @classmethod
    def from_state(cls, path: str, dataset: Dataset, fabric: Fabric, **kwargs) -> 'StatefulDataLoader':
        with open(path, "rb") as f:
            loaded_sampler_state = pickle.load(f)

        sampler = SerializableDistributedSampler(
            dataset,
            num_replicas=fabric.world_size,
            rank=fabric.global_rank
        )
        sampler.__setstate__(loaded_sampler_state)
        # Re-assign the actual dataset and let __iter__ handle lazy index generation
        sampler.dataset = dataset
        sampler.num_samples = -(-len(dataset) // sampler.num_replicas)
        sampler.total_size = sampler.num_samples * sampler.num_replicas


        kwargs['sampler'] = sampler
        return cls(dataset, **kwargs)

# --- 3. Test Components ---

class MockDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        return (idx, torch.randn(4))

class MockDistillationDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
        # Return dicts so default_collate can batch them into tensors
        self.samples = [{"query_id": i} for i in range(size)]
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

def run_test_logic(fabric: Fabric, use_stateful_loader: bool, checkpoint_dir: str, dataset_type: str):
    if dataset_type == "simple":
        dataset = MockDataset(size=101)
    else:
        dataset = MockDistillationDataset(size=97)

    state_path = os.path.join(checkpoint_dir, f"dataloader_rank_{fabric.global_rank}_{dataset_type}.pkl")

    sampler = SerializableDistributedSampler(dataset, num_replicas=fabric.world_size, rank=fabric.global_rank, shuffle=False)
    dataloader = StatefulDataLoader(dataset, batch_size=10, sampler=sampler, num_workers=0)

    part1_items = []
    fabric.print(f"[{dataset_type.capitalize()}] Starting first run...")
    for i, batch in enumerate(dataloader):
        items = batch[0] if dataset_type == "simple" else batch["query_id"]
        part1_items.extend(items.tolist())
        if i >= 3:
            break

    if use_stateful_loader:
        fabric.print(f"Rank {fabric.global_rank} saving state...")
        # Save the number of items we processed
        dataloader.sampler._current_index = len(part1_items)
        with open(state_path, "wb") as f:
            pickle.dump(dataloader.sampler.__getstate__(), f)

    fabric.barrier()

    part2_items = []
    fabric.print(f"[{dataset_type.capitalize()}] Starting second run (resuming)...")

    if use_stateful_loader:
        dataloader_resume = StatefulDataLoader.from_state(state_path, dataset, fabric=fabric, batch_size=10, num_workers=0)
    else:
        # Standard run restarts from the beginning
        sampler_resume = SerializableDistributedSampler(dataset, num_replicas=fabric.world_size, rank=fabric.global_rank, shuffle=False)
        dataloader_resume = StatefulDataLoader(dataset, batch_size=10, sampler=sampler_resume, num_workers=0)

    for batch in dataloader_resume:
        items = batch[0] if dataset_type == "simple" else batch["query_id"]
        part2_items.extend(items.tolist())

    # --- Verification ---
    part1_all_ranks = [None] * fabric.world_size
    torch.distributed.all_gather_object(part1_all_ranks, part1_items)
    part2_all_ranks = [None] * fabric.world_size
    torch.distributed.all_gather_object(part2_all_ranks, part2_items)

    if fabric.is_global_zero:
        print(f"\n--- VERIFICATION for {'Stateful' if use_stateful_loader else 'Standard'} {dataset_type.capitalize()} Run ---")

        if use_stateful_loader:
            # In the stateful run, the sets of items from part 1 and part 2 should be disjoint for EACH rank
            for r in range(fabric.world_size):
                if not set(part1_all_ranks[r]).isdisjoint(set(part2_all_ranks[r])):
                    intersection = set(part1_all_ranks[r]).intersection(set(part2_all_ranks[r]))
                    raise AssertionError(f"STATEFUL FAILED: Rank {r} has overlapping data after resuming! Overlap: {intersection}")
            print("OK: Stateful run passed. No overlap found after resuming.")
        else: # Standard run
            # In the standard run, part 2 should restart, so it should NOT be disjoint with part 1
            if set(part1_all_ranks[0]).isdisjoint(set(part2_all_ranks[0])):
                 raise AssertionError("STANDARD FAILED: Expected overlap from reprocessing, but none was found.")
            print("OK: Standard run correctly showed overlap from reprocessing.")

        # Final check for completeness across all parts of the stateful run
        if use_stateful_loader:
            full_run_items = list(itertools.chain.from_iterable(part1_all_ranks + part2_all_ranks))
            original_ids = set(range(len(dataset)))

            # We only care about original IDs, not padded ones, for the completeness check
            processed_ids = set(i for i in full_run_items if i < len(dataset))

            if processed_ids != original_ids:
                missed = original_ids - processed_ids
                raise AssertionError(f"COMPLETENESS FAILED: Did not process all original items. Missed: {missed}")
            print("OK: Stateful run processed all original items exactly once.")

        print("--- VERIFICATION PASSED ---\\n")

def main():
    checkpoint_dir = "checkpoints_test_serializable_loader"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    fabric = Fabric(accelerator="cpu", devices=2, strategy="ddp")

    # --- Test Suite ---
    fabric.launch(run_test_logic, use_stateful_loader=False, checkpoint_dir=checkpoint_dir, dataset_type="simple")
    fabric.launch(run_test_logic, use_stateful_loader=True, checkpoint_dir=checkpoint_dir, dataset_type="simple")

    print("\n" + "="*50 + "\n")

    fabric.launch(run_test_logic, use_stateful_loader=False, checkpoint_dir=checkpoint_dir, dataset_type="distillation")
    fabric.launch(run_test_logic, use_stateful_loader=True, checkpoint_dir=checkpoint_dir, dataset_type="distillation")

if __name__ == "__main__":
    main()
