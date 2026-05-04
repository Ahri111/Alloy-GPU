"""
neuralce.analysis — analysis utilities for trained CE models.

Modules added in the Fe-Vo specificity / decomposition workflow:
  * eci_decomposition  — split ΔE between two configurations into per-orbit
                         / per-order / per-sublattice contributions.
  * swap_test          — paired energy comparison for chemical-specificity
                         testing (e.g. Fe-near-Vo vs Ti-near-Vo).
  * cluster_extraction — orbit metadata table + cif export of representative
                         clusters (cluster-only or highlighted-in-supercell).

The companion vacancy-marker conversion lives in
``neuralce.utils.vacancy``.
"""

from neuralce.analysis.eci_decomposition import (
    build_orbit_metadata,
    decompose_pair,
    aggregate_by_order,
    aggregate_by_sublattice,
    aggregate_per_rank,
)
from neuralce.analysis.swap_test import (
    predict_one,
    run_swap_test,
    summarise_swap_test,
    boltzmann_ratio,
)
from neuralce.analysis.cluster_extraction import (
    orbit_metadata_table,
    cluster_only_atoms,
    cluster_in_supercell,
)

__all__ = [
    # eci_decomposition
    'build_orbit_metadata',
    'decompose_pair',
    'aggregate_by_order',
    'aggregate_by_sublattice',
    'aggregate_per_rank',
    # swap_test
    'predict_one',
    'run_swap_test',
    'summarise_swap_test',
    'boltzmann_ratio',
    # cluster_extraction
    'orbit_metadata_table',
    'cluster_only_atoms',
    'cluster_in_supercell',
]
