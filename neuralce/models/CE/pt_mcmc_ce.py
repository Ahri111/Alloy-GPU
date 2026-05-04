"""
pt_mcmc_ce.py
═══════════════════════════════════════════════════════════════════════════════

GPU-accelerated Parallel Tempering MCMC for classical Cluster Expansion.
Supports arbitrary species count per sublattice and optional discrete spin.

Two modes (auto-detected from config):
    Compositional only:  swap moves on σ
    Compositional + spin: swap moves on σ + spin flip moves on s

All inner-loop operations are JIT-compiled. Replicas are vmapped.
No Python in the hot path.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, random as jrandom

from neuralce.models.CE.classical_ce_gpu import GPUClusterExpansion


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class MCMCConfig:
    ce_path: str
    supercell_matrix: list

    # Temperature
    T_min: float = 100.0
    T_max: float = 2000.0
    n_replicas: int = 64

    # MCMC
    n_steps: int = 100000
    n_equilib: int = 10000
    sample_interval: int = 100
    exchange_interval: int = 10

    # Spin (optional)
    spin_orbits_pkl: str = ''
    spin_states: dict = field(default_factory=dict)

    # Initial composition — fraction per sublattice
    # e.g. {'A': [0.5, 0.5]} for 50/50 binary
    # e.g. {'A': [0.6, 0.2, 0.2]} for ternary
    # If empty, defaults to uniform distribution
    initial_composition: dict = field(default_factory=dict)

    seed: int = 42
    output_dir: str = 'mcmc_results'


# ═══════════════════════════════════════════════════════════════════════════════
# MCMC tables
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class MCMCTables:
    # Sublattice info for swap moves
    sub_indices: jnp.ndarray     # (n_active_subs, max_sub_size) int32 padded with -1
    sub_mask: jnp.ndarray        # (n_active_subs, max_sub_size) bool
    sub_n_species: jnp.ndarray   # (n_active_subs,) int32
    sub_log_probs: jnp.ndarray   # (n_active_subs,) float32 — log selection probability

    # Site info
    site_to_sl: jnp.ndarray      # (n_atoms,) int32 — sublattice index per site
    n_atoms: int
    n_active_subs: int

    # Spin tables (None if no spin)
    has_spin: bool
    spin_allowed: Optional[jnp.ndarray] = None   # (n_subs, max_sp, max_spin_states) float32
    n_spin_states: Optional[jnp.ndarray] = None   # (n_subs, max_sp) int32
    max_spin_states: int = 1


def build_mcmc_tables(gce: GPUClusterExpansion, spin_states_config: dict = None) -> MCMCTables:
    cs = gce.ce._cluster_space
    supercell = gce.supercell
    n_atoms = gce.n_atoms

    # Sublattice info
    active_subs = gce.sublattices
    n_active = len(active_subs)
    max_sub_size = max(len(sl['indices']) for sl in active_subs)

    sub_indices_np = np.full((n_active, max_sub_size), -1, dtype=np.int32)
    sub_mask_np = np.zeros((n_active, max_sub_size), dtype=bool)
    sub_n_species_np = np.zeros(n_active, dtype=np.int32)
    sub_sizes = np.zeros(n_active, dtype=np.float32)

    for i, sl in enumerate(active_subs):
        idx = np.array(sl['indices'])
        n = len(idx)
        sub_indices_np[i, :n] = idx
        sub_mask_np[i, :n] = True
        sub_n_species_np[i] = sl['n_species']
        sub_sizes[i] = n

    sub_probs = sub_sizes / sub_sizes.sum()
    sub_log_probs = np.log(sub_probs).astype(np.float32)

    # site_to_sl — full supercell
    sup_subs = cs.get_sublattices(supercell)
    n_total_subs = max(sup_subs.get_sublattice_index_from_site_index(i)
                       for i in range(n_atoms)) + 1
    site_to_sl_np = np.zeros(n_atoms, dtype=np.int32)
    for i in range(n_atoms):
        site_to_sl_np[i] = sup_subs.get_sublattice_index_from_site_index(i)

    # Spin tables
    has_spin = bool(spin_states_config) and gce.has_spin
    spin_allowed = None
    n_spin_states_arr = None
    max_spin_states = 1

    if spin_states_config:
        species_to_sl_sp = {}
        for sl_idx, sl in enumerate(active_subs):
            for sp_idx, sym in enumerate(sl['chemical_symbols']):
                species_to_sl_sp[sym] = (sl_idx, sp_idx)

        all_spin_lists = {}
        for sym, states in spin_states_config.items():
            if sym in species_to_sl_sp:
                sl_i, sp_i = species_to_sl_sp[sym]
                all_spin_lists[(sl_i, sp_i)] = list(states)

        max_sp = max(sl['n_species'] for sl in active_subs)
        max_spin_states = max(len(v) for v in all_spin_lists.values()) if all_spin_lists else 1
        max_spin_states = max(max_spin_states, 1)

        spin_allowed_np = np.zeros((n_active, max_sp, max_spin_states), dtype=np.float32)
        n_spin_states_np = np.ones((n_active, max_sp), dtype=np.int32)

        for sl_i in range(n_active):
            for sp_i in range(active_subs[sl_i]['n_species']):
                if (sl_i, sp_i) in all_spin_lists:
                    states = all_spin_lists[(sl_i, sp_i)]
                    n_s = len(states)
                    spin_allowed_np[sl_i, sp_i, :n_s] = states
                    n_spin_states_np[sl_i, sp_i] = n_s

        spin_allowed = jnp.array(spin_allowed_np)
        n_spin_states_arr = jnp.array(n_spin_states_np)
        has_spin = True

    return MCMCTables(
        sub_indices=jnp.array(sub_indices_np),
        sub_mask=jnp.array(sub_mask_np),
        sub_n_species=jnp.array(sub_n_species_np),
        sub_log_probs=jnp.array(sub_log_probs),
        site_to_sl=jnp.array(site_to_sl_np),
        n_atoms=n_atoms,
        n_active_subs=n_active,
        has_spin=has_spin,
        spin_allowed=spin_allowed,
        n_spin_states=n_spin_states_arr,
        max_spin_states=max_spin_states,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Temperature ladder
# ═══════════════════════════════════════════════════════════════════════════════
def make_temperature_ladder(T_min, T_max, n_replicas):
    if n_replicas == 1:
        temps = np.array([T_min])
    else:
        temps = np.geomspace(T_min, T_max, n_replicas)
    betas = 1.0 / (8.617333e-5 * temps)  # 1/(kB*T), kB in eV/K
    return jnp.array(temps, dtype=jnp.float32), jnp.array(betas, dtype=jnp.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Initial state
# ═══════════════════════════════════════════════════════════════════════════════
def make_initial_state(key, gce, mcmc_tables, initial_composition, n_replicas):
    n_atoms = gce.n_atoms
    active_subs = gce.sublattices

    # Build one template sigma
    sigma_template = np.zeros(n_atoms, dtype=np.int32)

    for sl_idx, sl in enumerate(active_subs):
        indices = np.array(sl['indices'])
        n_sites = len(indices)
        n_species = sl['n_species']
        sl_name = sl['name']

        if sl_name in initial_composition:
            fracs = initial_composition[sl_name]
            if not isinstance(fracs, list):
                fracs = [1.0 - fracs, fracs]
        else:
            fracs = [1.0 / n_species] * n_species

        counts = np.round(np.array(fracs) * n_sites).astype(int)
        diff = n_sites - counts.sum()
        counts[0] += diff

        species_arr = np.concatenate([
            np.full(c, sp_i, dtype=np.int32)
            for sp_i, c in enumerate(counts)
        ])
        np.random.seed(42)
        np.random.shuffle(species_arr)
        sigma_template[indices] = species_arr

    # Replicate and shuffle independently per replica
    sigma_batch = np.tile(sigma_template, (n_replicas, 1))
    for r in range(n_replicas):
        for sl in active_subs:
            indices = np.array(sl['indices'])
            sub_sigma = sigma_batch[r, indices].copy()
            np.random.shuffle(sub_sigma)
            sigma_batch[r, indices] = sub_sigma

    sigma_batch = jnp.array(sigma_batch)

    # Initial spins
    if mcmc_tables.has_spin:
        spins_batch = jnp.zeros((n_replicas, n_atoms), dtype=jnp.float32)
        spin_allowed_np = np.array(mcmc_tables.spin_allowed)
        n_spin_states_np = np.array(mcmc_tables.n_spin_states)
        site_to_sl_np = np.array(mcmc_tables.site_to_sl)
        sigma_np = np.array(sigma_batch)

        spins_np = np.zeros((n_replicas, n_atoms), dtype=np.float32)
        for r in range(n_replicas):
            for i in range(n_atoms):
                sl_i = site_to_sl_np[i]
                found = False
                for a_sl_idx, sl in enumerate(active_subs):
                    if any(i == idx for idx in np.array(sl['indices'])):
                        sp_i = sigma_np[r, i]
                        n_s = n_spin_states_np[a_sl_idx, sp_i]
                        if n_s > 1:
                            spins_np[r, i] = spin_allowed_np[
                                a_sl_idx, sp_i, np.random.randint(0, n_s)]
                        found = True
                        break

        spins_batch = jnp.array(spins_np)
    else:
        spins_batch = None

    return sigma_batch, spins_batch


# ═══════════════════════════════════════════════════════════════════════════════
# Move proposals
# ═══════════════════════════════════════════════════════════════════════════════
def make_swap_fn(mcmc_tables):
    sub_idx = mcmc_tables.sub_indices
    sub_mask = mcmc_tables.sub_mask
    sub_log_probs = mcmc_tables.sub_log_probs
    n_active = mcmc_tables.n_active_subs

    def propose_swap(key, sigma):
        k_sub, k_s1, k_s2 = jrandom.split(key, 3)

        # Pick sublattice (Gumbel-max categorical)
        sl_choice = jnp.argmax(sub_log_probs + jrandom.gumbel(k_sub, (n_active,)))

        # Get sites on chosen sublattice
        sites = sub_idx[sl_choice]
        mask = sub_mask[sl_choice]
        sub_sigma = sigma[sites]

        # Pick site1 (uniform among valid sites)
        noise1 = jnp.where(mask, jrandom.gumbel(k_s1, sites.shape), -jnp.inf)
        i1_local = jnp.argmax(noise1)
        site1 = sites[i1_local]
        species1 = sigma[site1]

        # Pick site2 (different species, Gumbel-max)
        diff_mask = mask & (sub_sigma != species1)
        noise2 = jnp.where(diff_mask, jrandom.gumbel(k_s2, sites.shape), -jnp.inf)
        i2_local = jnp.argmax(noise2)
        site2 = sites[i2_local]

        # Swap
        sigma_new = sigma.at[site1].set(sigma[site2])
        sigma_new = sigma_new.at[site2].set(species1)

        # valid only if a different-species site exists
        valid = jnp.any(diff_mask)
        return sigma_new, valid

    return propose_swap


def make_spin_flip_fn(mcmc_tables):
    if not mcmc_tables.has_spin:
        return None

    site_to_sl = mcmc_tables.site_to_sl
    spin_allowed = mcmc_tables.spin_allowed
    n_spin_st = mcmc_tables.n_spin_states
    max_ss = mcmc_tables.max_spin_states
    sub_indices = mcmc_tables.sub_indices
    sub_mask = mcmc_tables.sub_mask
    n_active = mcmc_tables.n_active_subs
    n_atoms = mcmc_tables.n_atoms

    # Build active_sl_index: site → index into active sublattice list, or -1
    # We need this because site_to_sl gives the global sublattice index,
    # but spin_allowed is indexed by active sublattice index
    # For now, assume active sublattice indices are contiguous and match
    # the ordering in gce.sublattices. We build a lookup.

    def propose_spin_flip(key, sigma, spins, active_sl_lookup):
        k_site, k_spin = jrandom.split(key)

        asl = active_sl_lookup
        asl_safe = jnp.maximum(asl, 0)  # clamp -1 → 0 for safe indexing

        is_on_active = asl >= 0
        n_ss = jnp.where(is_on_active, n_spin_st[asl_safe, sigma], 1)
        is_magnetic = is_on_active & (n_ss > 1)

        # Pick a magnetic site (Gumbel-max)
        noise = jnp.where(is_magnetic, jrandom.gumbel(k_site, (n_atoms,)), -jnp.inf)
        site = jnp.argmax(noise)
        valid = jnp.any(is_magnetic)

        # Get allowed spins for this site (safe indexing)
        sl_i = jnp.maximum(asl[site], 0)
        sp_i = sigma[site]
        allowed = spin_allowed[sl_i, sp_i]
        n_allowed = n_spin_st[sl_i, sp_i]
        current_spin = spins[site]

        # Pick new spin != current (Gumbel-max with mask)
        spin_mask = (jnp.arange(max_ss) < n_allowed) & (allowed != current_spin)
        noise_spin = jnp.where(spin_mask, jrandom.gumbel(k_spin, (max_ss,)), -jnp.inf)
        new_spin_idx = jnp.argmax(noise_spin)
        new_spin = allowed[new_spin_idx]

        spins_new = spins.at[site].set(new_spin)
        return spins_new, valid

    return propose_spin_flip


# ═══════════════════════════════════════════════════════════════════════════════
# Replica exchange
# ═══════════════════════════════════════════════════════════════════════════════
def make_exchange_fn(betas):
    """Permutation-based replica exchange — no Python for-loop."""
    n_replicas = len(betas)

    @jit
    def replica_exchange(key, sigma_batch, spins_batch, E_batch):
        k_parity, k_accept = jrandom.split(key)
        parity = jrandom.randint(k_parity, (), 0, 2)

        i_indices = jnp.arange(0, n_replicas - 1, 2) + parity
        j_indices = i_indices + 1
        valid_mask = j_indices < n_replicas

        i_safe = jnp.where(valid_mask, i_indices, 0)
        j_safe = jnp.where(valid_mask, j_indices, 0)

        dBeta = jnp.where(valid_mask, betas[j_safe] - betas[i_safe], 0.0)
        dE = jnp.where(valid_mask, E_batch[j_safe] - E_batch[i_safe], 0.0)

        log_accept = dBeta * dE
        u = jrandom.uniform(k_accept, shape=i_indices.shape)
        do_swap = valid_mask & (jnp.log(u) < log_accept)

        new_perm = jnp.arange(n_replicas)
        new_perm = new_perm.at[i_safe].set(jnp.where(do_swap, j_safe, i_safe))
        new_perm = new_perm.at[j_safe].set(jnp.where(do_swap, i_safe, j_safe))

        sigma_batch = sigma_batch[new_perm]
        spins_batch = spins_batch[new_perm]
        E_batch = E_batch[new_perm]

        return sigma_batch, spins_batch, E_batch, jnp.sum(do_swap)

    return replica_exchange


# ═══════════════════════════════════════════════════════════════════════════════
# MC step (single replica)
# ═══════════════════════════════════════════════════════════════════════════════
def make_mc_step_fn(energy_fn, swap_fn, spin_flip_fn, has_spin, active_sl_lookup):
    """Build a single-replica MC step function.

    Returns a function with signature:
        (key, sigma, spins, E, beta) → (sigma, spins, E, accepted_swap, accepted_flip)
    """
    if has_spin:
        asl = active_sl_lookup

        def mc_step(key, sigma, spins, E, beta):
            k_swap, k_flip, k_a1, k_a2 = jrandom.split(key, 4)

            # Compositional swap
            sigma_new, swap_valid = swap_fn(k_swap, sigma)
            E_new_swap = energy_fn(sigma_new, spins)
            dE_swap = E_new_swap - E
            accept_swap = swap_valid & (jnp.log(jrandom.uniform(k_a1)) < -beta * dE_swap)
            sigma = jnp.where(accept_swap, sigma_new, sigma)
            E = jnp.where(accept_swap, E_new_swap, E)

            # Spin flip
            spins_new, flip_valid = spin_flip_fn(k_flip, sigma, spins, asl)
            E_new_flip = energy_fn(sigma, spins_new)
            dE_flip = E_new_flip - E
            accept_flip = flip_valid & (jnp.log(jrandom.uniform(k_a2)) < -beta * dE_flip)
            spins = jnp.where(accept_flip, spins_new, spins)
            E = jnp.where(accept_flip, E_new_flip, E)

            return sigma, spins, E, accept_swap, accept_flip

    else:
        def mc_step(key, sigma, spins, E, beta):
            k_swap, k_a1 = jrandom.split(key)

            sigma_new, swap_valid = swap_fn(k_swap, sigma)
            E_new = energy_fn(sigma_new)
            dE = E_new - E
            accept_swap = swap_valid & (jnp.log(jrandom.uniform(k_a1)) < -beta * dE)
            sigma = jnp.where(accept_swap, sigma_new, sigma)
            E = jnp.where(accept_swap, E_new, E)

            return sigma, spins, E, accept_swap, jnp.bool_(False)

    return mc_step


# ═══════════════════════════════════════════════════════════════════════════════
# Main PT-MCMC loop
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class MCMCResult:
    energies: np.ndarray           # (n_samples, n_replicas)
    temperatures: np.ndarray       # (n_replicas,)
    swap_accept_rate: float
    flip_accept_rate: float
    exchange_accept_rate: float
    sigma_final: np.ndarray        # (n_replicas, n_atoms)
    spins_final: Optional[np.ndarray]
    wall_time: float


def run_pt_mcmc(gce, cfg):
    """Run PT-MCMC simulation."""
    os.makedirs(cfg.output_dir, exist_ok=True)
    has_spin = bool(cfg.spin_orbits_pkl) and bool(cfg.spin_states)

    if has_spin and not gce.has_spin:
        gce.load_spin(cfg.spin_orbits_pkl)

    mcmc_tables = build_mcmc_tables(gce, cfg.spin_states if has_spin else None)
    temps, betas = make_temperature_ladder(cfg.T_min, cfg.T_max, cfg.n_replicas)

    print(f"PT-MCMC: {cfg.n_replicas} replicas, T=[{cfg.T_min:.0f}, {cfg.T_max:.0f}] K")
    print(f"  n_steps={cfg.n_steps}, n_equilib={cfg.n_equilib}")
    print(f"  spin: {'yes' if has_spin else 'no'}")
    print(f"  exchange_interval={cfg.exchange_interval}")

    # Build active_sl_lookup: site → active sublattice index (or 0 for inactive)
    active_subs = gce.sublattices
    active_sl_lookup_np = np.zeros(gce.n_atoms, dtype=np.int32)
    active_mask_np = np.zeros(gce.n_atoms, dtype=bool)
    for a_idx, sl in enumerate(active_subs):
        for site in np.array(sl['indices']):
            active_sl_lookup_np[site] = a_idx
            active_mask_np[site] = True
    # Sites not on active sublattices get -1 (won't be selected for spin flip)
    active_sl_lookup_np[~active_mask_np] = -1
    active_sl_lookup = jnp.array(active_sl_lookup_np)

    # Build functions
    swap_fn = make_swap_fn(mcmc_tables)
    spin_flip_fn = make_spin_flip_fn(mcmc_tables) if has_spin else None
    energy_fn = gce.energy
    mc_step = make_mc_step_fn(energy_fn, swap_fn, spin_flip_fn, has_spin, active_sl_lookup)
    mc_step_batched = jit(vmap(mc_step, in_axes=(0, 0, 0, 0, 0)))
    exchange_fn = make_exchange_fn(betas)

    # Initial state
    key = jrandom.PRNGKey(cfg.seed)
    key, k_init = jrandom.split(key)
    sigma_batch, spins_batch = make_initial_state(
        k_init, gce, mcmc_tables, cfg.initial_composition, cfg.n_replicas)

    if spins_batch is None:
        spins_batch = jnp.zeros((cfg.n_replicas, gce.n_atoms), dtype=jnp.float32)

    # Initial energies
    if has_spin:
        E_batch = gce.batched_energy(sigma_batch, spins_batch)
    else:
        E_batch = gce.batched_energy(sigma_batch)

    # Sampling buffers
    n_production = cfg.n_steps - cfg.n_equilib
    n_samples = max(1, n_production // cfg.sample_interval)
    energy_samples = np.zeros((n_samples, cfg.n_replicas), dtype=np.float32)
    sample_idx = 0

    total_swap_accept = 0
    total_flip_accept = 0
    total_exchange_accept = 0
    total_swap_attempts = 0
    total_exchange_attempts = 0

    print(f"\nRunning {cfg.n_steps} steps...")
    t0 = time.time()

    for step in range(cfg.n_steps):
        key, k_step, k_ex = jrandom.split(key, 3)
        keys = jrandom.split(k_step, cfg.n_replicas)

        sigma_batch, spins_batch, E_batch, acc_swap, acc_flip = mc_step_batched(
            keys, sigma_batch, spins_batch, E_batch, betas)

        total_swap_accept += int(jnp.sum(acc_swap))
        total_flip_accept += int(jnp.sum(acc_flip))
        total_swap_attempts += cfg.n_replicas

        # Replica exchange
        if (step + 1) % cfg.exchange_interval == 0:
            sigma_batch, spins_batch, E_batch, n_ex = exchange_fn(
                k_ex, sigma_batch, spins_batch, E_batch)
            total_exchange_accept += int(n_ex)
            total_exchange_attempts += cfg.n_replicas // 2

        # Sample
        if step >= cfg.n_equilib and (step - cfg.n_equilib) % cfg.sample_interval == 0:
            if sample_idx < n_samples:
                energy_samples[sample_idx] = np.array(E_batch)
                sample_idx += 1

        # Progress
        if (step + 1) % (cfg.n_steps // 10) == 0:
            elapsed = time.time() - t0
            rate = (step + 1) / elapsed
            swap_rate = total_swap_accept / max(1, total_swap_attempts)
            print(f"  step {step+1:>8d}/{cfg.n_steps}  "
                  f"E_min={float(E_batch.min()):.2f}  "
                  f"swap_acc={swap_rate:.3f}  "
                  f"{rate:.0f} steps/s")

    wall_time = time.time() - t0
    swap_accept_rate = total_swap_accept / max(1, total_swap_attempts)
    flip_accept_rate = total_flip_accept / max(1, total_swap_attempts) if has_spin else 0.0
    exchange_accept_rate = total_exchange_accept / max(1, total_exchange_attempts)

    print(f"\nDone in {wall_time:.1f}s")
    print(f"  swap acceptance:     {swap_accept_rate:.4f}")
    if has_spin:
        print(f"  flip acceptance:     {flip_accept_rate:.4f}")
    print(f"  exchange acceptance: {exchange_accept_rate:.4f}")

    result = MCMCResult(
        energies=energy_samples[:sample_idx],
        temperatures=np.array(temps),
        swap_accept_rate=swap_accept_rate,
        flip_accept_rate=flip_accept_rate,
        exchange_accept_rate=exchange_accept_rate,
        sigma_final=np.array(sigma_batch),
        spins_final=np.array(spins_batch) if has_spin else None,
        wall_time=wall_time,
    )

    # Save
    np.savez(os.path.join(cfg.output_dir, 'mcmc_result.npz'),
             energies=result.energies,
             temperatures=result.temperatures,
             sigma_final=result.sigma_final,
             spins_final=result.spins_final if result.spins_final is not None else np.array([]),
             swap_accept_rate=result.swap_accept_rate,
             flip_accept_rate=result.flip_accept_rate,
             exchange_accept_rate=result.exchange_accept_rate)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    from icet import ClusterExpansion
    from ase.build import make_supercell

    CE_PATH = os.environ.get('CE_PATH', 'models/stfo_comp_test.ce')
    SUPERCELL_MATRIX = [[0, 2, 2], [2, 0, 2], [2, 2, 0]]

    cfg = MCMCConfig(
        ce_path=CE_PATH,
        supercell_matrix=SUPERCELL_MATRIX,
        T_min=300.0, T_max=2000.0, n_replicas=32,
        n_steps=10000, n_equilib=2000,
        sample_interval=10, exchange_interval=5,
        initial_composition={'A': [0.5, 0.5], 'B': [0.75, 0.25]},
    )

    ce = ClusterExpansion.read(cfg.ce_path)
    prim = ce._cluster_space.primitive_structure
    supercell = make_supercell(prim, np.array(cfg.supercell_matrix))
    gce = GPUClusterExpansion(ce, supercell)
    print(gce)

    result = run_pt_mcmc(gce, cfg)
    print(f"\nEnergy samples shape: {result.energies.shape}")
