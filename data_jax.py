"""Backwards-compat shim. Real module lives at neuralce.data.data_jax."""
from neuralce.data.data_jax import *  # noqa: F401,F403
from neuralce.data.data_jax import (  # noqa: F401
    GaussianDistance, load_atom_embeddings, process_crystal,
    load_dataset_variable, load_dataset_fixed, get_specie_number,
)
