"""
cluster_extraction.py — extract orbit metadata and write per-orbit cif files.

Two output styles supported:

  • ``cluster_only_atoms``  — Atoms object containing ONLY the cluster
    sites in a tight box around their centroid.  Default species at
    each site (cation→Ti, anion→O, inert→Sr).  Use for paper-figure
    cluster-shape illustrations.

  • ``cluster_in_supercell`` — full 160-atom supercell where the cluster
    sites are recoloured (Au=cation, Ag=anion, Pt=inert).  Use for
    showing a cluster in context of the perovskite framework.
"""

from __future__ import annotations
from itertools import combinations
import numpy as np
import pandas as pd
from ase import Atoms
from icet import ClusterExpansion


_CLUSTER_MARKER = {'A': 'Au', 'B': 'Ag', 'C': 'Pt'}
_DEFAULT_SPECIES = {'A': 'Ti', 'B': 'O', 'C': 'Sr'}


def _site_label(site_idx: int, cs) -> str:
    syms = list(cs.chemical_symbols[site_idx])
    if 'Fe' in syms and 'Ti' in syms: return 'A'
    if 'Xe' in syms and 'O' in syms:  return 'B'
    if syms == ['Sr']:                return 'C'
    return '?'


def _lattice_site_position(ls, primitive):
    return (primitive.positions[ls.index]
            + np.dot(np.asarray(ls.unitcell_offset, dtype=float),
                      primitive.cell.array))


def _find_supercell_index(supercell: Atoms, target_pos, tol: float = 0.05) -> int:
    """Return supercell atom index whose position matches ``target_pos``
    under PBC, or -1 if no match within ``tol``."""
    cell = supercell.cell.array
    inv = np.linalg.inv(cell)
    df = (supercell.positions - target_pos) @ inv
    df -= np.round(df)
    d = np.linalg.norm(df @ cell, axis=1)
    j = int(np.argmin(d))
    if d[j] >= tol:
        # try the wrapped position
        pos_frac = (target_pos @ inv) % 1.0
        df2 = (supercell.positions - pos_frac @ cell) @ inv
        df2 -= np.round(df2)
        d2 = np.linalg.norm(df2 @ cell, axis=1)
        j2 = int(np.argmin(d2))
        if d2[j2] < tol:
            return j2
        return -1
    return j


def orbit_metadata_table(ce: ClusterExpansion) -> pd.DataFrame:
    """One row per primitive orbit (binary CE: 1 cv element per orbit)."""
    cs = ce._cluster_space
    primitive = cs.primitive_structure
    params = np.asarray(ce.parameters)
    rows = []
    cv_idx = 1
    for o_idx in range(len(cs.orbit_list)):
        orbit = cs.orbit_list.get_orbit(o_idx)
        rep = orbit.representative_cluster
        order = int(rep.order)
        radius = float(rep.radius)
        site_indices = [int(ls.index) for ls in rep.lattice_sites]
        sublabels = [_site_label(i, cs) for i in site_indices]
        positions = np.array([_lattice_site_position(ls, primitive)
                                for ls in rep.lattice_sites])
        if order >= 2:
            d_pair = np.array([np.linalg.norm(positions[i] - positions[j])
                                for i, j in combinations(range(order), 2)])
            d_max = float(d_pair.max()); d_min = float(d_pair.min())
        else:
            d_max = d_min = 0.0
        sub_str = '-'.join(sorted(sublabels))
        for cv_elem in orbit.cluster_vector_elements:
            mult = int(cv_elem.get('multiplicity', 1))
            rows.append(dict(
                cv_idx=cv_idx, orbit_idx=o_idx, order=order,
                multiplicity=mult, radius=radius,
                d_max=d_max, d_min=d_min,
                sublattice=sub_str, site_indices=site_indices,
                site_offsets=[list(map(int, ls.unitcell_offset))
                              for ls in rep.lattice_sites],
                site_positions=[p.tolist() for p in positions],
                eci=float(params[cv_idx]),
            ))
            cv_idx += 1
    return pd.DataFrame(rows)


def cluster_only_atoms(ce: ClusterExpansion, orbit_idx: int,
                        pad: float = 2.5) -> Atoms:
    """Return an Atoms with only the orbit's representative cluster
    sites (default species per sublattice) inside a tight box."""
    cs = ce._cluster_space
    primitive = cs.primitive_structure
    orbit = cs.orbit_list.get_orbit(orbit_idx)
    rep = orbit.representative_cluster
    site_indices = [int(ls.index) for ls in rep.lattice_sites]
    pos = np.array([_lattice_site_position(ls, primitive)
                     for ls in rep.lattice_sites])
    species = [_DEFAULT_SPECIES.get(_site_label(i, cs), 'X')
                for i in site_indices]
    if len(pos) == 1:
        side = 2 * pad
        cell = np.eye(3) * side
        rel = pos - pos[0] + pad
    else:
        mn = pos.min(axis=0); mx = pos.max(axis=0)
        side = np.maximum((mx - mn) + 2 * pad, 2 * pad)
        cell = np.diag(side)
        rel = pos - mn + pad
    return Atoms(symbols=species, positions=rel, cell=cell, pbc=False)


def cluster_in_supercell(ce: ClusterExpansion, orbit_idx: int,
                          supercell: Atoms,
                          marker: dict = None) -> Atoms:
    """Return a copy of ``supercell`` (with default Ti/O/Sr) where the
    orbit's representative cluster sites are recoloured with markers
    (default: Au cation / Ag anion / Pt inert)."""
    cs = ce._cluster_space
    primitive = cs.primitive_structure
    orbit = cs.orbit_list.get_orbit(orbit_idx)
    rep = orbit.representative_cluster
    site_indices = [int(ls.index) for ls in rep.lattice_sites]
    positions = np.array([_lattice_site_position(ls, primitive)
                            for ls in rep.lattice_sites])

    # Default-decorated supercell
    sc = supercell.copy()
    for at in sc:
        if at.symbol in ('Fe', 'Ti'): at.symbol = 'Ti'
        elif at.symbol in ('O', 'Xe'): at.symbol = 'O'

    mapping = dict(_CLUSTER_MARKER)
    if marker:
        mapping.update(marker)

    for ls_pos, ls_index in zip(positions, site_indices):
        j = _find_supercell_index(supercell, ls_pos)
        if j == -1:
            continue
        sl = _site_label(ls_index, cs)
        sc[j].symbol = mapping.get(sl, sc[j].symbol)
    return sc


__all__ = [
    'orbit_metadata_table',
    'cluster_only_atoms',
    'cluster_in_supercell',
]
