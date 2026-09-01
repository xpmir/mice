
import torch
from torch.utils.data import DataLoader, IterableDataset
import os

class MockShardedDataset(IterableDataset):
    def __init__(self, data, world_size, rank):
        self.data = data
        self.world_size = world_size
        self.rank = rank

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0

        num_shards = self.world_size * num_workers
        shard_id = self.rank * num_workers + worker_id

        for i in range(shard_id, len(self.data), num_shards):
            yield self.data[i]

def test_sharding(N, world_size, num_workers):
    data = list(range(N))
    counts = []
    for rank in range(world_size):
        ds = MockShardedDataset(data, world_size, rank)
        dl = DataLoader(ds, num_workers=num_workers, batch_size=1)
        count = sum(1 for _ in dl)

        # Formula to verify
        effective_nw = max(1, num_workers)
        expected = sum(1 for i in range(N) if (i // effective_nw) % world_size == rank)

        print(f"Rank {rank}: actual={count}, expected={expected}")
        assert count == expected
        counts.append(count)

    print(f"Total: {sum(counts)} / {N}")
    assert sum(counts) == N

if __name__ == "__main__":
    print("Testing N=10, WS=2, NW=2")
    test_sharding(10, 2, 2)
    print("\nTesting N=100, WS=3, NW=4")
    test_sharding(100, 3, 4)
    print("\nTesting N=13, WS=4, NW=0 (no workers)")
    test_sharding(13, 4, 0)
