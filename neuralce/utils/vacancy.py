"""
vacancy.py — vacancy-marker conversions.

Two conventions show up in the project:

  * 'Xe' (Z = 54) — used by the icet ClusterSpace and all training cif
    files in ``data/processed/``.
  * 'X'  (Z = 0,  ASE / pymatgen DummySpecies) — used by NeuralCE
    PT-MCMC outputs (``data/check/pt_result_fe*_2_cifs/``).

This module provides the two converters and a fill-vacancy helper so any
downstream evaluator can normalise to the icet convention before calling
``map_structure_to_reference`` or ``ce.predict``.
"""

from __future__ import annotations
import numpy as np
from ase import Atom
from ase.io import read
from ase.atoms import Atoms


def load_atoms_xe(path: str) -> Atoms:
    """Read a cif and rename every 'X' (atomic number 0) site to 'Xe'.
    Cifs that already use the 'Xe' convention pass through untouched.
    """
    a = read(path)
    syms = a.get_chemical_symbols()
    a.set_chemical_symbols(['Xe' if s == 'X' else s for s in syms])
    return a


def to_x_convention(atoms: Atoms) -> Atoms:
    """Rename every 'Xe' site to 'X' (NeuralCE / DummySpecies style)."""
    out = atoms.copy()
    syms = out.get_chemical_symbols()
    out.set_chemical_symbols(['X' if s == 'Xe' else s for s in syms])
    return out


def fill_xe_at_vacancies(atoms: Atoms,
                          ref_atoms: Atoms,
                          anion_tol: float = 1.0) -> Atoms:
    """Insert 'Xe' atoms at the lattice positions of ``ref_atoms`` that
    have no anion within ``anion_tol`` Å in ``atoms`` (PBC-aware).

    Useful for cifs that encode vacancies as missing atoms.  ``ref_atoms``
    is expected to be a fully-decorated reference supercell (typically a
    training cif) whose anion lattice positions are taken as ground truth.

    Parameters
    ----------
    atoms : Atoms
        Input structure (may have fewer atoms than ``ref_atoms`` because
        of missing-atom vacancies).
    ref_atoms : Atoms
        Reference structure whose anion lattice (O ∪ Xe) defines the
        full anion sublattice positions.
    anion_tol : float
        Distance threshold under PBC for a position to count as
        “occupied” by an anion in ``atoms``.

    Returns
    -------
    Atoms
        Copy of ``atoms`` with 'Xe' atoms appended at all anion lattice
        positions of ``ref_atoms`` that were unoccupied.
    """
    cell = ref_atoms.cell.array
    inv = np.linalg.inv(cell)
    ref_syms = np.array(ref_atoms.get_chemical_symbols())
    ref_anion_pos = ref_atoms.positions[np.isin(ref_syms, ['O', 'Xe'])]

    in_syms = np.array(atoms.get_chemical_symbols())
    in_anion_pos = atoms.positions[np.isin(in_syms, ['O', 'Xe'])]

    df = (ref_anion_pos @ inv)[:, None, :] - (in_anion_pos @ inv)[None, :, :]
    df -= np.round(df)
    closest = np.linalg.norm(df @ cell, axis=-1).min(axis=1)
    vac_pos = ref_anion_pos[closest > anion_tol]

    out = atoms.copy()
    for p in vac_pos:
        out.append(Atom('Xe', position=p))
    return out


__all__ = ['load_atoms_xe', 'to_x_convention', 'fill_xe_at_vacancies']
