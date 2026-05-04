"""
swap_test.py — chemical-specificity test by paired energy comparison.

Given two cif files for the same composition (e.g. an "Fe-near-Vo" config
and its "Ti-near-Vo swap"), compute ΔE = E(swap) − E(orig) under one or
more cluster-expansion models.

Typical use:

    >>> rows = run_swap_test(
    ...     ce, primitive_for_supercell,
    ...     pairs=[('rank01', orig1, swap1), ('rank02', orig2, swap2), ...])
    >>> print(rows)        # rank, E_orig, E_swap, dE, dE_per_swap_site

The function performs vacancy-marker normalisation (X → Xe) and verifies
both members of each pair share the same composition.
"""

from __future__ import annotations
from collections import Counter
from typing import Iterable, Tuple, List, Optional, Dict
import numpy as np
import pandas as pd
from ase.atoms import Atoms
from icet import ClusterExpansion
from icet.tools import map_structure_to_reference

from neuralce.utils.vacancy import load_atoms_xe


def boltzmann_ratio(delta_E: float, T_K: float = 1000.0) -> float:
    kB = 8.617333e-5
    return float(np.exp(-delta_E / (kB * T_K)))


def predict_one(ce: ClusterExpansion, atoms: Atoms,
                primitive_for_supercell,
                inert_species=('Sr',)) -> float:
    mapped, _ = map_structure_to_reference(
        atoms, primitive_for_supercell, inert_species=list(inert_species),
        assume_no_cell_relaxation=False)
    return float(ce.predict(mapped))


def run_swap_test(ce: ClusterExpansion,
                   primitive_for_supercell,
                   pairs: Iterable[Tuple[str, str, str]],
                   inert_species=('Sr',),
                   n_swap_sites: Optional[int] = None,
                   composition_check: Optional[Dict[str, int]] = None,
                   ) -> pd.DataFrame:
    """Predict the energy of every (orig, swap) pair and return a table.

    Parameters
    ----------
    ce : ClusterExpansion
    primitive_for_supercell : Atoms
        Primitive used by ``map_structure_to_reference`` to build the
        supercell (typically the no-idealize primitive matching ``ce``).
    pairs : iterable of (label, orig_cif_path, swap_cif_path)
    inert_species : sublattices held fixed by ``map_structure_to_reference``
    n_swap_sites : optional int
        If given, ``dE_per_swap_site = dE / n_swap_sites`` is included.
    composition_check : optional dict
        e.g. ``{'Sr':32,'Fe':16,'Ti':16,'O':88,'Xe':8}`` — assert both
        members of each pair match.

    Returns
    -------
    DataFrame with columns:
        rank, E_orig, E_swap, dE, dE_per_swap_site (optional).
    """
    rows = []
    for rank_label, orig_path, swap_path in pairs:
        orig = load_atoms_xe(orig_path)
        swap = load_atoms_xe(swap_path)
        co, cs_ = Counter(orig.get_chemical_symbols()), \
                   Counter(swap.get_chemical_symbols())
        assert co == cs_, f"composition mismatch: {co} vs {cs_}"
        if composition_check is not None:
            assert co == composition_check, (
                f"unexpected composition: {co} != {composition_check}")
        E_o = predict_one(ce, orig, primitive_for_supercell, inert_species)
        E_s = predict_one(ce, swap, primitive_for_supercell, inert_species)
        dE = E_s - E_o
        row = dict(rank=rank_label, E_orig=E_o, E_swap=E_s, dE=dE)
        if n_swap_sites is not None and n_swap_sites > 0:
            row['dE_per_swap_site'] = dE / n_swap_sites
        rows.append(row)
    return pd.DataFrame(rows)


def summarise_swap_test(df: pd.DataFrame, T_K: float = 1000.0) -> dict:
    """One-row summary suitable for cross-model comparison."""
    arr = df['dE'].values
    out = dict(
        n=len(arr),
        mean_dE=float(arr.mean()),
        std_dE=float(arr.std()),
        min_dE=float(arr.min()),
        max_dE=float(arr.max()),
        boltzmann_ratio_T_K=T_K,
        boltzmann_ratio=boltzmann_ratio(float(arr.mean()), T_K),
    )
    if 'dE_per_swap_site' in df.columns:
        out['mean_dE_per_swap_site'] = float(df['dE_per_swap_site'].mean())
    return out


__all__ = [
    'predict_one',
    'run_swap_test',
    'summarise_swap_test',
    'boltzmann_ratio',
]
