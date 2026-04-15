"""
cif_utils.py — Safe CIF loading with vacancy restoration.
"""

from pymatgen.core.structure import Structure
from pymatgen.core import DummySpecies


def load_cif_safe(path):
    """Load CIF and restore Xe sites to DummySpecies('X0+').

    pymatgen's CIF writer encodes DummySpecies('X0+') as 'X',
    which the reader misinterprets as Xe (xenon). This function
    reverses that corruption.
    """
    struct = Structure.from_file(path)
    for i in range(len(struct)):
        if str(struct[i].specie) == 'Xe':
            struct[i] = DummySpecies('X0+'), struct[i].frac_coords
    return struct


def get_specie_number(specie):
    """Get atomic number. DummySpecies → 0."""
    if isinstance(specie, DummySpecies):
        return 0
    return specie.number
