import torch
from torch.utils.data import Dataset
from pathlib import Path
from .random_clifford_circuit import load_circuit_from_file

class CachedSyndromeDataset(Dataset):
    def __init__(self, root_dir, code_distance, circuit_index, depth, batch_size):
        self.root_dir = Path(root_dir)
        self.code_distance = code_distance
        self.circuit_index = circuit_index
        self.depth = depth
        self.batch_size = batch_size

        prefix = f"d{code_distance}_c{circuit_index}_bs{batch_size}_D{depth}"
        self.syndromes = torch.load(self.root_dir / f"syndromes_{prefix}.pt")
        self.labels = torch.load(self.root_dir / f"labels_{prefix}.pt")
        self.final_round_syndromes = torch.load(self.root_dir / f"final_round_syndromes_{prefix}.pt")
        self.circuits_file = self.root_dir / f"circuits_{prefix}.txt"

        assert self.syndromes.shape[0] % batch_size == 0, \
            "Total number of samples must be divisible by batch_size"

        self.num_batches = self.syndromes.shape[0] // batch_size

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        i0 = idx * self.batch_size
        i1 = (idx + 1) * self.batch_size

        circuit = load_circuit_from_file(self.circuits_file, idx)

        return {
            'syndromes': self.syndromes[i0:i1],
            'label': self.labels[i0:i1],
            'final_round_syndromes': self.final_round_syndromes[i0:i1],
            'circuit': circuit
        }

import itertools
from torch.utils.data import IterableDataset

class MultiDepthCachedSyndromeDataset(IterableDataset):
    def __init__(self, root_dir, code_distance, circuit_index, batch_size, depth_list):
        self.datasets = [
            CachedSyndromeDataset(
                root_dir=root_dir,
                code_distance=code_distance,
                circuit_index=circuit_index,
                depth=depth,
                batch_size=batch_size
            )
            for depth in depth_list
        ]

    def __iter__(self):
        datasets = self.datasets
        iterators = [iter(ds) for ds in datasets]
        exhausted = [False] * len(datasets)
        depth_cycle = itertools.cycle(range(len(datasets)))

        while not all(exhausted):
            idx = next(depth_cycle)

            if exhausted[idx]:
                continue

            try:
                yield next(iterators[idx])
            except StopIteration:
                exhausted[idx] = True
            
            
