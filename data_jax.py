"""
JAX Native Data Loading with Multi-Mode Support

Mode:
- 'variable': segment_ids 사용 (가변 atom 수)
- 'fixed': reshape 사용 (고정 atom 수, alloy 등)

Usage:
    # Variable mode (다양한 crystal)
    dataset = load_dataset_variable(data_dir)
    
    # Fixed mode (alloy 160 atoms)
    dataset = load_dataset_fixed(data_dir, n_atoms_per_crystal=160)
"""

import os
import csv
import json
from typing import Dict, Any, List, Tuple, Optional, Literal

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

from pymatgen.core.structure import Structure
from pymatgen.core import DummySpecies


# =============================================================================
# Preprocessing (공통)
# =============================================================================

class GaussianDistance:
    """Gaussian distance filter."""
    def __init__(self, dmin: float, dmax: float, step: float, var: float = None):
        self.filter = np.arange(dmin, dmax + step, step)
        self.var = var if var else step

    def expand(self, distances: np.ndarray) -> np.ndarray:
        return np.exp(-(distances[..., np.newaxis] - self.filter) ** 2 / self.var ** 2)


def load_atom_embeddings(path: str) -> Dict[int, np.ndarray]:
    """Load atom embeddings from JSON."""
    with open(path) as f:
        data = json.load(f)
    return {int(k): np.array(v, dtype=np.float32) for k, v in data.items()}


def get_specie_number(specie) -> int:
    """Get atomic number, handling DummySpecies."""
    if isinstance(specie, DummySpecies):
        return 0
    return specie.number


def process_crystal(
    cif_path: str,
    atom_embeddings: Dict[int, np.ndarray],
    gdf: GaussianDistance,
    max_num_nbr: int = 12,
    radius: float = 8.0,
) -> Dict[str, np.ndarray]:
    """Process single CIF file to graph features."""
    crystal = Structure.from_file(cif_path)

    # Atom features
    atom_fea = np.vstack([
        atom_embeddings[get_specie_number(crystal[i].specie)]
        for i in range(len(crystal))
    ]).astype(np.float32)

    # Neighbor information
    all_nbrs = crystal.get_all_neighbors(radius, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

    nbr_fea_idx, nbr_fea = [], []
    for nbr in all_nbrs:
        if len(nbr) < max_num_nbr:
            nbr_fea_idx.append(
                [x[2] for x in nbr] + [0] * (max_num_nbr - len(nbr))
            )
            nbr_fea.append(
                [x[1] for x in nbr] + [radius + 1.] * (max_num_nbr - len(nbr))
            )
        else:
            nbr_fea_idx.append([x[2] for x in nbr[:max_num_nbr]])
            nbr_fea.append([x[1] for x in nbr[:max_num_nbr]])

    nbr_fea_idx = np.array(nbr_fea_idx, dtype=np.int32)
    nbr_fea = gdf.expand(np.array(nbr_fea, dtype=np.float32)).astype(np.float32)

    return {
        'atom_fea': atom_fea,
        'nbr_fea': nbr_fea,
        'nbr_fea_idx': nbr_fea_idx,
        'n_atoms': len(crystal),
    }


# =============================================================================
# Multi-Mode Dataset
# =============================================================================

class FullMemoryDataset:
    """
    Load entire dataset into memory with multi-mode support.
    
    Modes:
    - 'variable': segment_ids used (variable atom count, for training)
    - 'fixed': reshape used (fixed atom count, for alloy inference)
    """

    def __init__(
        self,
        data_dir: str,
        target_file: str = "detailed_info.csv",
        prop_name: str = "total_energy",
        id_col: str = "id",
        max_num_nbr: int = 12,
        radius: float = 8.0,
        seed: int = 42,
        mode: Literal['variable', 'fixed'] = 'variable',
        n_atoms_per_crystal: Optional[int] = None,
    ):
        self.data_dir = data_dir
        self.max_num_nbr = max_num_nbr
        self.radius = radius
        self.mode = mode
        self.n_atoms_per_crystal = n_atoms_per_crystal
        self.prop_name = prop_name
        
        if mode == 'fixed' and n_atoms_per_crystal is None:
            raise ValueError("n_atoms_per_crystal required for mode='fixed'")

        # Load metadata
        id_prop_file = os.path.join(data_dir, target_file)
        print(f"Loading metadata from: {id_prop_file}")

        with open(id_prop_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.raw_data = list(reader)  # Defined here

        np.random.seed(seed)
        np.random.shuffle(self.raw_data) # Shuffled here

        # Preprocessing tools
        self.atom_embeddings = load_atom_embeddings(
            os.path.join(data_dir, 'atom_init.json')
        )
        self.gdf = GaussianDistance(dmin=0.0, dmax=radius, step=0.2)
        
        self.orig_atom_fea_len = len(list(self.atom_embeddings.values())[0])
        self.nbr_fea_len = len(self.gdf.filter)

        # Load all data
        print(f"Loading {len(self.raw_data)} structures (mode={mode})...")
        self.data = []
        
        # [Fix] Iterate over self.raw_data, NOT self.id_prop_data
        for i, row in enumerate(self.raw_data):
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(self.raw_data)}")

            try:
                id = row[id_col]
                target_val = float(row[prop_name])
                
                cif_path = os.path.join(data_dir, f"{id}.cif")
                
                if not os.path.exists(cif_path):
                    continue

                processed = process_crystal(
                    cif_path, self.atom_embeddings, self.gdf,
                    max_num_nbr, radius
                )
                
                if mode == 'fixed' and processed['n_atoms'] != n_atoms_per_crystal:
                    raise ValueError(
                        f"Crystal {id} has {processed['n_atoms']} atoms, "
                        f"expected {n_atoms_per_crystal}"
                    )
                
                processed['target'] = np.array([target_val], dtype=np.float32)
                processed['id'] = id
                self.data.append(processed)
                
            except KeyError as e:
                raise KeyError(f"Column not found in CSV: {e}. Available: {list(row.keys())}")
            except ValueError as e:
                print(f"Skipping {id}: {e}")
                continue

        print(f"Successfully loaded {len(self.data)} structures")

        # Precompute all targets for normalizer
        self.all_targets = np.array([d['target'][0] for d in self.data])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_batch(self, indices: List[int]) -> Dict[str, jnp.ndarray]:
        """Get a collated batch by indices (mode-aware)."""
        batch_atom_fea = []
        batch_nbr_fea = []
        batch_nbr_fea_idx = []
        batch_target = []
        base_idx = 0

        for idx in indices:
            d = self.data[idx]
            n = d['n_atoms']

            batch_atom_fea.append(d['atom_fea'])
            batch_nbr_fea.append(d['nbr_fea'])
            batch_nbr_fea_idx.append(d['nbr_fea_idx'] + base_idx)
            batch_target.append(d['target'])

            base_idx += n

        batch = {
            'atom_fea': jnp.concatenate(batch_atom_fea, axis=0),
            'nbr_fea': jnp.concatenate(batch_nbr_fea, axis=0),
            'nbr_fea_idx': jnp.concatenate(batch_nbr_fea_idx, axis=0),
            'target': jnp.stack(batch_target, axis=0), 
        }
        
        if self.mode == 'fixed':
            batch['batch_size'] = len(indices)
            batch['n_atoms_per_crystal'] = self.n_atoms_per_crystal
        else:
            segment_ids = []
            for i, idx in enumerate(indices):
                n = self.data[idx]['n_atoms']
                segment_ids.extend([i] * n)
            batch['segment_ids'] = jnp.array(segment_ids, dtype=jnp.int32)
            batch['num_crystals'] = len(indices)

        return batch

    def get_split_indices(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(self.data)
        indices = np.arange(n)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        return (indices[:train_end], indices[train_end:val_end], indices[val_end:])

    def iterate_batches(
        self,
        indices: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        rng: Optional[random.PRNGKey] = None,
        drop_last: bool = False, # <--- Required argument
    ):
        """Iterate over batches (generator)."""
        indices = np.array(indices, copy=True)

        if shuffle:
            if rng is not None:
                perm = random.permutation(rng, len(indices))
                indices = indices[np.array(perm)]
            else:
                np.random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            
            # [CRITICAL] Prevents Fixed Mode crashes & JAX recompilation
            if (self.mode == 'fixed' or drop_last) and len(batch_indices) < batch_size:
                break
                
            yield self.get_batch(batch_indices.tolist())
    
    def set_mode(self, mode: Literal['variable', 'fixed'], n_atoms_per_crystal: Optional[int] = None):
        if mode == 'fixed':
            if n_atoms_per_crystal is None:
                raise ValueError("n_atoms_per_crystal required for mode='fixed'")
            for d in self.data:
                if d['n_atoms'] != n_atoms_per_crystal:
                    raise ValueError(f"Atom count mismatch in {d['id']}")
            self.n_atoms_per_crystal = n_atoms_per_crystal
        self.mode = mode
        print(f"Mode changed to '{mode}'")


# =============================================================================
# Convenience functions
# =============================================================================

def load_dataset_variable(
    data_dir: str,
    target_file: str = "detailed_info.csv",  # Added
    prop_name: str = "total_energy",         # Added
    id_col: str = "id",                  # Added
    max_num_nbr: int = 12,
    radius: float = 8.0,
    seed: int = 42,
) -> FullMemoryDataset:
    return FullMemoryDataset(
        data_dir, 
        target_file=target_file,
        prop_name=prop_name,
        id_col=id_col,
        max_num_nbr=max_num_nbr, 
        radius=radius, 
        seed=seed,
        mode='variable'
    )


def load_dataset_fixed(
    data_dir: str,
    n_atoms_per_crystal: int,
    target_file: str = "detailed_info.csv",  # Added
    prop_name: str = "total_energy",         # Added
    id_col: str = "id",                  # Added
    max_num_nbr: int = 12,
    radius: float = 8.0,
    seed: int = 42,
) -> FullMemoryDataset:
    return FullMemoryDataset(
        data_dir, 
        target_file=target_file,
        prop_name=prop_name,
        id_col=id_col,
        max_num_nbr=max_num_nbr, 
        radius=radius, 
        seed=seed,
        mode='fixed',
        n_atoms_per_crystal=n_atoms_per_crystal
    )

# =============================================================================
# Usage Example
# =============================================================================

if __name__ == "__main__":
    """
    # Case 1: 처음부터 fixed mode (alloy만 다룰 때)
    dataset = load_dataset_fixed("./data/alloy", n_atoms_per_crystal=160)
    train_idx, val_idx, test_idx = dataset.get_split_indices()
    
    for batch in dataset.iterate_batches(train_idx, batch_size=32):
        # batch['batch_size'], batch['n_atoms_per_crystal'] 포함
        out = model.apply(params, **batch, mode='fixed')
    
    
    # Case 2: 학습은 variable, 추론은 fixed (다양한 데이터로 학습 후 alloy 추론)
    dataset = load_dataset_variable("./data/mixed_crystals")
    train_idx, val_idx, test_idx = dataset.get_split_indices()
    
    # 학습
    for batch in dataset.iterate_batches(train_idx, batch_size=32):
        # batch['segment_ids'], batch['num_crystals'] 포함
        out = model.apply(params, **batch, mode='variable')
    
    # 추론 시 mode 변경 (모든 crystal이 160 atoms라면)
    dataset.set_mode('fixed', n_atoms_per_crystal=160)
    
    for batch in dataset.iterate_batches(test_idx, batch_size=64):
        # batch['batch_size'], batch['n_atoms_per_crystal'] 포함
        out = model.apply(params, **batch, mode='fixed')
    """
    pass