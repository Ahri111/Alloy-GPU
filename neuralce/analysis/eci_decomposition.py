"""
eci_decomposition.py — decompose ΔE between two configurations into
per-orbit / per-order / per-sublattice contributions.

The energy is the dot product of the cluster-vector with the ECI:
    E = Σ_i  ECI_i · cv_i

so for any pair of configurations the difference decomposes exactly:
    ΔE = Σ_i  ECI_i · (cv_swap_i − cv_orig_i)

This module computes the full decomposition table for arbitrary cif pairs
and groups the contributions by orbit metadata (order, radius, sublattice).
"""

from __future__ import annotations
from itertools import combinations
import numpy as np
import pandas as pd
from ase.atoms import Atoms
from icet import ClusterExpansion
from icet.tools import map_structure_to_reference


def _site_label(site_idx: int, cs) -> str:
    syms = list(cs.chemical_symbols[site_idx])
    if 'Fe' in syms and 'Ti' in syms: return 'A'
    if 'Xe' in syms and 'O' in syms:  return 'B'
    if syms == ['Sr']:                return 'C'
    return '?'


def _chemical_label(site_indices, cs) -> str:
    """Convert sublattice labels to a canonical chemical label using
    A→Fe, B→Vo, C→Sr (interpretation tag for Fe-Vo physics)."""
    parts = []
    for i in site_indices:
        sl = _site_label(i, cs)
        parts.append({'A': 'Fe', 'B': 'Vo', 'C': 'Sr'}.get(sl, '?'))
    return '-'.join(sorted(parts))


def _lattice_site_position(ls, primitive):
    return (primitive.positions[ls.index]
            + np.dot(np.asarray(ls.unitcell_offset, dtype=float),
                      primitive.cell.array))


def build_orbit_metadata(ce: ClusterExpansion) -> pd.DataFrame:
    """Per-orbit metadata table indexed by cv-element index.

    Columns: cv_idx, orbit_idx, order, multiplicity, radius, d_max, d_min,
             sublattice, chemical, eci.
    """
    cs = ce._cluster_space
    primitive = cs.primitive_structure
    params = np.asarray(ce.parameters)
    rows = []
    cv_idx = 1   # ECI[0] is the zerolet
    for o_idx in range(len(cs.orbit_list)):
        orbit = cs.orbit_list.get_orbit(o_idx)
        rep = orbit.representative_cluster
        order = int(rep.order)
        radius = float(rep.radius)
        site_indices = [int(ls.index) for ls in rep.lattice_sites]
        sublabel = '-'.join(sorted(_site_label(i, cs) for i in site_indices))
        chem = _chemical_label(site_indices, cs)
        positions = np.array([_lattice_site_position(ls, primitive)
                                for ls in rep.lattice_sites])
        if order >= 2:
            d_pair = np.array([np.linalg.norm(positions[i] - positions[j])
                                for i, j in combinations(range(order), 2)])
            d_max = float(d_pair.max()); d_min = float(d_pair.min())
        else:
            d_max = d_min = 0.0
        for cv_elem in orbit.cluster_vector_elements:
            mult = int(cv_elem.get('multiplicity', 1))
            rows.append(dict(
                cv_idx=cv_idx, orbit_idx=o_idx, order=order,
                multiplicity=mult, radius=radius,
                d_max=d_max, d_min=d_min,
                sublattice=sublabel, chemical=chem,
                eci=float(params[cv_idx]),
            ))
            cv_idx += 1
    return pd.DataFrame(rows)


def decompose_pair(ce: ClusterExpansion,
                   atoms_orig: Atoms,
                   atoms_swap: Atoms,
                   primitive_for_supercell,
                   inert_species: list = ['Sr']) -> pd.DataFrame:
    """Return per-orbit decomposition of ΔE = E(swap) − E(orig).

    Both configurations are mapped through ``primitive_for_supercell``
    using ``map_structure_to_reference`` (consistent with how the model
    was trained / evaluated).

    Returns a DataFrame with one row per cv element, columns:
        cv_idx, orbit_idx, order, multiplicity, radius, d_max, d_min,
        sublattice, chemical, eci, delta_cv, contribution.
    """
    cs = ce._cluster_space
    mapped_o, _ = map_structure_to_reference(
        atoms_orig, primitive_for_supercell, inert_species=inert_species,
        assume_no_cell_relaxation=False)
    mapped_s, _ = map_structure_to_reference(
        atoms_swap, primitive_for_supercell, inert_species=inert_species,
        assume_no_cell_relaxation=False)
    cv_o = np.asarray(cs.get_cluster_vector(mapped_o))
    cv_s = np.asarray(cs.get_cluster_vector(mapped_s))
    dcv = cv_s - cv_o
    params = np.asarray(ce.parameters)
    contribution = params * dcv

    meta = build_orbit_metadata(ce)
    meta = meta.copy()
    meta['delta_cv'] = dcv[meta['cv_idx'].values]
    meta['contribution'] = contribution[meta['cv_idx'].values]
    return meta


def aggregate_by_order(decomp: pd.DataFrame) -> pd.DataFrame:
    """Sum contributions by cluster order."""
    return decomp.groupby('order')['contribution'].agg(['sum', 'count']).reset_index()


def aggregate_by_sublattice(decomp: pd.DataFrame) -> pd.DataFrame:
    """Sum contributions by sublattice composition."""
    return decomp.groupby('sublattice')['contribution'] \
                  .agg(['sum', 'count']).reset_index() \
                  .sort_values('sum', key=lambda s: -np.abs(s)) \
                  .reset_index(drop=True)


def aggregate_per_rank(per_rank_decomp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a stacked per-rank decomposition (long format with a
    ``rank`` column) into per-orbit mean/std contributions across ranks.
    """
    grouped = (per_rank_decomp
               .groupby(['cv_idx', 'orbit_idx', 'order',
                          'multiplicity', 'radius', 'sublattice',
                          'chemical', 'eci'])
               ['contribution']
               .agg(['mean', 'std', 'min', 'max', 'count'])
               .reset_index())
    grouped.rename(columns={'mean': 'mean_contr', 'std': 'std_contr',
                             'min': 'min_contr', 'max': 'max_contr',
                             'count': 'n_ranks'}, inplace=True)
    grouped['abs_mean_contr'] = grouped['mean_contr'].abs()
    return grouped.sort_values('abs_mean_contr', ascending=False).reset_index(drop=True)


__all__ = [
    'build_orbit_metadata',
    'decompose_pair',
    'aggregate_by_order',
    'aggregate_by_sublattice',
    'aggregate_per_rank',
]
