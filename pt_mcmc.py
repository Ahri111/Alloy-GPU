"""
PT-MCMC v4 — Parallel Tempering Monte Carlo with NeuralCE

Changes from v3:
  [1] Vectorized replica exchange (JAX array ops, no Python for-loop)
      → moved inside lax.scan, runs every SWAP_INTERVAL *steps* (not chunks)
  [2] Architecture: lax.scan(vmap(step) + exchange) instead of vmap(lax.scan(step))
      → exchange decoupled from chunk boundary
  [3] CHUNK_SIZE = save interval (no separate SAVE_EVERY)
      → SWAP_INTERVAL removed from chunk-level; now step-level inside scan
  [4] Vectorized spin initialization (no Python for-loop)
  [5] Single checkpoint load (no duplicate pickle.load)
  [6] Vectorized shell one-hot (advanced indexing)
  [7] Deduplicated energy computation (single_energy reused in local_step)

General-purpose: works for any fixed-lattice alloy system.
  - STFO (ABO3 perovskite): sublattice-aware B-site/O-site swaps + Fe spin flip
  - Fe-Ni-Cr (FCC alloy): global site swaps + spin flip
  - Any system: configure sublattices, swap types, spin in YAML

Key features:
  - YAML config driven (no argparse, Colab compatible)
  - Lite models only (shell one-hot edges) — graph topology fixed during MC
  - vmap single-crystal forward (no jnp.tile)
  - Gumbel-trick random site selection (branch-free, vmap compatible)
  - Configurable move types: sublattice_swap, spin_flip
  - Vectorized parallel tempering with replica exchange inside lax.scan

Usage (Colab):
  CONFIG_PATH = './configs/stfo_ptmcmc.yaml'
  %run pt_mcmc.py

Usage (CLI):
  CONFIG_PATH=./configs/stfo_ptmcmc.yaml python pt_mcmc.py
"""

import os, time, pickle, json
import numpy as np
import yaml

import jax
import jax.numpy as jnp
from jax import random, lax, vmap

from pymatgen.core.structure import Structure

from NeuralCE_jax import (create_neuralce, is_spin_model, is_sisj_model,
                           LITE_MODELS)


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════
CONFIG_PATH = os.environ.get('CONFIG_PATH', './configs/stfo_nospin_ptmcmc.yaml')

with open(CONFIG_PATH) as f:
    _cfg = yaml.safe_load(f)

assert _cfg.get('mode') == 'pt_mcmc', \
    f"This script requires mode: pt_mcmc, got '{_cfg.get('mode')}'"

# Dataset
DATASET_NAME = _cfg['dataset_name']
SPECIES_MAP  = {int(k): v for k, v in _cfg['species_map'].items()}
N_SPECIES    = len(SPECIES_MAP)
EXCLUDE_Z    = set(_cfg.get('exclude_species', []))

# Graph (from YAML — will be overridden by checkpoint hp if available)
_graph       = _cfg.get('graph', {})
_CUTOFF_YAML = _graph.get('cutoff')
_N_SHELLS_YAML = _graph.get('n_shells')
_SHELL_EDGES_YAML = _graph.get('shell_edges')
_MAX_NUM_NBR_YAML = _graph.get('max_num_nbr', 12)

# PT-MCMC
_mc          = _cfg['pt_mcmc']
MODEL_CKPT   = _mc['model_ckpt']
MODEL_TYPE   = _mc['model_type']
CIF_TEMPLATE = _mc['cif_template']
N_REPLICAS   = _mc.get('n_replicas', 3000)
T_MIN        = _mc.get('t_min', 200.0)
T_MAX        = _mc.get('t_max', 1500.0)
N_STEPS      = _mc.get('n_steps', 100000)
CHUNK_SIZE   = _mc.get('chunk_size', 200)         # v4 hybrid: MC steps per scan, = save interval
SWAP_INTERVAL = _mc.get('swap_interval', 1)       # v4 hybrid: exchange every N chunks
BURNIN_FRAC  = _mc.get('burnin_frac', 0.2)        # v4: fraction of total steps
SEED         = _mc.get('seed', 42)
OUTPUT       = _mc.get('output', 'pt_result.npz')

# Derived
BURNIN_STEPS = int(N_STEPS * BURNIN_FRAC)
BURNIN_CHUNKS = BURNIN_STEPS // CHUNK_SIZE

# ── Single checkpoint load ────────────────────────────────────────
# v4: load once, reuse for both graph config and model params
with open(MODEL_CKPT, 'rb') as _f:
    _CKPT = pickle.load(_f)
_ckpt_hp = _CKPT['hp']

CUTOFF      = _ckpt_hp.get('cutoff', _CUTOFF_YAML)
N_SHELLS    = _ckpt_hp.get('n_shells', _N_SHELLS_YAML)
MAX_NUM_NBR = _ckpt_hp.get('max_num_nbr', _MAX_NUM_NBR_YAML)

# Shell edges: try YAML candidates matching checkpoint cutoff, then YAML global
_candidates = _graph.get('candidates', {})
SHELL_EDGES = None
if _candidates:
    for _ck, _cv in _candidates.items():
        if abs(float(_ck) - CUTOFF) < 1e-6:
            SHELL_EDGES = _cv.get('shell_edges')
            if 'max_num_nbr' in _cv:
                MAX_NUM_NBR = _cv['max_num_nbr']
            break
if SHELL_EDGES is None:
    SHELL_EDGES = _SHELL_EDGES_YAML

# Validate
if CUTOFF is None or N_SHELLS is None:
    raise ValueError("Graph cutoff/n_shells not found in checkpoint hp or YAML config")

_yaml_mismatch = []
if _CUTOFF_YAML and abs(_CUTOFF_YAML - CUTOFF) > 1e-6:
    _yaml_mismatch.append(f"cutoff: YAML={_CUTOFF_YAML} → ckpt={CUTOFF}")
if _N_SHELLS_YAML and _N_SHELLS_YAML != N_SHELLS:
    _yaml_mismatch.append(f"n_shells: YAML={_N_SHELLS_YAML} → ckpt={N_SHELLS}")
if _MAX_NUM_NBR_YAML != MAX_NUM_NBR:
    _yaml_mismatch.append(f"max_num_nbr: YAML={_MAX_NUM_NBR_YAML} → ckpt={MAX_NUM_NBR}")
if _yaml_mismatch:
    print(f"  ⚠ Graph config overridden by checkpoint: {', '.join(_yaml_mismatch)}")
print(f"  Graph: cutoff={CUTOFF}, n_shells={N_SHELLS}, max_num_nbr={MAX_NUM_NBR}")

# Composition target (optional — for random init)
COMPOSITION  = _mc.get('composition', None)

# Move types
SUBLATTICES  = _mc.get('sublattices', [])
HAS_SPIN_FLIP = _mc.get('spin_flip', False)
SPIN_SPECIES  = _mc.get('spin_species', [])

# Precision
MATMUL_PREC  = _mc.get('matmul_precision', 'float32')
ENABLE_X64   = _mc.get('enable_x64', False)

if ENABLE_X64:
    jax.config.update('jax_enable_x64', True)
jax.config.update('jax_default_matmul_precision', MATMUL_PREC)

print(f"{'═' * 70}")
print(f"  PT-MCMC v4 — {DATASET_NAME}")
print(f"  Model: {MODEL_TYPE} from {MODEL_CKPT}")
print(f"  Template: {CIF_TEMPLATE}")
print(f"  Replicas: {N_REPLICAS}, Steps: {N_STEPS}")
print(f"  T: {T_MIN} → {T_MAX} K")
print(f"  Chunk: {CHUNK_SIZE} steps (= save interval)")
print(f"  Swap interval: every {SWAP_INTERVAL} step(s) (inside lax.scan)")
print(f"  Burn-in: {BURNIN_FRAC*100:.0f}% = {BURNIN_STEPS} steps ({BURNIN_CHUNKS} chunks)")
print(f"  Sublattices: {[s['name'] for s in SUBLATTICES]}")
print(f"  Spin flip: {HAS_SPIN_FLIP} (species: {SPIN_SPECIES})")
print(f"{'═' * 70}")


# ═══════════════════════════════════════════════════════════════════════
# 1. GRAPH BUILDER (Lite only, from template CIF)
# ═══════════════════════════════════════════════════════════════════════

def build_template_graph(cif_path):
    """Build graph from template CIF. Returns static arrays.

    In fixed-lattice MC, only node features change (atom swaps).
    Edge features (distance-based shell one-hot) and neighbor indices
    are invariant → computed once and cached.
    """
    crystal = Structure.from_file(cif_path)

    # Exclude species (e.g., Sr in STFO)
    if EXCLUDE_Z:
        keep_idx = [i for i, site in enumerate(crystal)
                    if site.specie.Z not in EXCLUDE_Z]
        crystal = Structure.from_sites([crystal[i] for i in keep_idx])

    n_at = len(crystal)

    # Neighbor search
    all_nbrs = crystal.get_all_neighbors(CUTOFF, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

    nbr_fea_idx = np.zeros((n_at, MAX_NUM_NBR), dtype=np.int32)
    nbr_dists   = np.full((n_at, MAX_NUM_NBR), CUTOFF + 1.0, dtype=np.float64)

    for i, nbr in enumerate(all_nbrs):
        n_nbr = min(len(nbr), MAX_NUM_NBR)
        for j in range(n_nbr):
            nbr_fea_idx[i, j] = nbr[j][2]
            nbr_dists[i, j]   = nbr[j][1]

    # Shell one-hot — v4: vectorized (no nested for-loop)
    valid_mask = nbr_dists < CUTOFF
    if SHELL_EDGES is not None:
        _edges = np.array(SHELL_EDGES)
    else:
        min_dist = nbr_dists[valid_mask].min() if valid_mask.any() else 0.1
        _edges = np.linspace(min_dist * 0.99, CUTOFF, N_SHELLS + 1)

    shell_idx = np.clip(np.digitize(nbr_dists, _edges) - 1, 0, N_SHELLS - 1)
    shell_oh = np.zeros((n_at, MAX_NUM_NBR, N_SHELLS), dtype=np.float32)
    rows, cols = np.where(valid_mask)
    shell_oh[rows, cols, shell_idx[rows, cols]] = 1.0

    # Species assignment from template
    species_arr = np.zeros(n_at, dtype=np.int32)
    for i, site in enumerate(crystal):
        z = site.specie.Z
        if z in SPECIES_MAP:
            species_arr[i] = SPECIES_MAP[z]

    # Sublattice masks: which sites belong to which sublattice
    sublattice_masks = {}
    for sub in SUBLATTICES:
        mask = np.zeros(n_at, dtype=bool)
        for sp in sub['species']:
            mask |= np.isin(species_arr, sp)
        sublattice_masks[sub['name']] = mask

    # Spin species mask
    spin_mask = np.zeros(n_at, dtype=bool)
    for sp in SPIN_SPECIES:
        spin_mask |= (species_arr == sp)

    print(f"  Template: {n_at} atoms (after exclude)")
    for sub in SUBLATTICES:
        m = sublattice_masks[sub['name']]
        print(f"    Sublattice '{sub['name']}': {m.sum()} sites, species {sub['species']}")
    if HAS_SPIN_FLIP:
        print(f"    Spin sites: {spin_mask.sum()}")

    # Sublattice site indices (for composition init)
    sublattice_site_idx = {}
    for sub in SUBLATTICES:
        sublattice_site_idx[sub['name']] = np.where(sublattice_masks[sub['name']])[0]

    return {
        'crystal': crystal,
        'n_atoms': n_at,
        'species': species_arr,
        'nbr_fea': jnp.array(shell_oh),
        'nbr_fea_idx': jnp.array(nbr_fea_idx),
        'sublattice_masks': {k: jnp.array(v.astype(np.float32))
                             for k, v in sublattice_masks.items()},
        'sublattice_site_idx': sublattice_site_idx,
        'spin_mask': jnp.array(spin_mask.astype(np.float32)),
    }


# ═══════════════════════════════════════════════════════════════════════
# 2. MODEL LOADING (v4: reuse cached checkpoint)
# ═══════════════════════════════════════════════════════════════════════

def load_model():
    """Load model and params from cached checkpoint (_CKPT)."""
    hp = _CKPT['hp']
    params = _CKPT['params']
    model_name = _CKPT.get('model_name', MODEL_TYPE)

    sisj = is_sisj_model(model_name)
    use_spin = is_spin_model(model_name)
    n_shells = hp['n_shells']
    edge_dim = n_shells + 1 if sisj else n_shells

    kwargs = {
        'atom_fea_len': hp['atom_fea_len'],
        'nbr_fea_len': edge_dim,
        'n_conv': hp['n_conv'],
        'h_fea_len': hp['h_fea_len'],
    }
    if use_spin:
        kwargs['odd_fea_len'] = hp['odd_fea_len']

    model = create_neuralce(model_type=model_name, pool_mode='fixed',
                            readout_type='sum', **kwargs)

    n_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"  Model: {model_name}, params: {n_params:,}")
    print(f"  HP: {hp}")

    return model, params, model_name, use_spin


# ═══════════════════════════════════════════════════════════════════════
# 3. ENERGY FUNCTION (vmap single-crystal)
# ═══════════════════════════════════════════════════════════════════════

def make_energy_fn(model, params, graph, use_spin):
    """Create single-crystal energy function.

    Returns:
        single_energy: (species, spins) → scalar
        batched_energy: (species_batch, spins_batch) → (n_replicas,)

    Edge features are FIXED (captured in closure) — only node features change.
    """
    nbr_fea     = graph['nbr_fea']       # (N, M, edge_dim) — static
    nbr_fea_idx = graph['nbr_fea_idx']   # (N, M) — static
    n_atoms     = graph['n_atoms']

    def single_energy(species, spins):
        atom_fea = jnp.eye(N_SPECIES, dtype=jnp.float32)[species]  # (N, N_SPECIES)
        kw = {
            'atom_fea': atom_fea,
            'nbr_fea': nbr_fea,
            'nbr_fea_idx': nbr_fea_idx,
            'batch_size': 1,
            'n_atoms_per_crystal': n_atoms,
        }
        if use_spin:
            kw['atom_spins'] = spins
        return model.apply(params, **kw).squeeze()

    batched_energy = vmap(single_energy, in_axes=(0, 0))

    return single_energy, batched_energy


# ═══════════════════════════════════════════════════════════════════════
# 4. MC MOVE FUNCTIONS (Gumbel trick, branch-free)
# ═══════════════════════════════════════════════════════════════════════

def make_swap_fn(sublattice_mask, species_list):
    """Create a swap function for a sublattice.

    sublattice_mask: (N,) float, 1.0 for sites in this sublattice
    species_list: list of 2+ species indices that can be swapped

    Picks two sites with different species and swaps them.
    Uses Gumbel trick for branch-free random selection.
    """
    species_a, species_b = species_list[0], species_list[1]

    def swap_step(key, species, spins):
        k1, k2 = random.split(key)
        n_at = species.shape[0]

        # Mask: sites that are species_a
        is_a = ((species == species_a) * sublattice_mask).astype(jnp.float32)
        # Mask: sites that are species_b
        is_b = ((species == species_b) * sublattice_mask).astype(jnp.float32)

        # Gumbel trick: pick random site from each group
        gumbel_a = random.gumbel(k1, (n_at,))
        gumbel_b = random.gumbel(k2, (n_at,))
        idx_a = jnp.argmax(jnp.where(is_a > 0, gumbel_a, -jnp.inf))
        idx_b = jnp.argmax(jnp.where(is_b > 0, gumbel_b, -jnp.inf))

        # Swap species
        new_species = species.at[idx_a].set(species_b).at[idx_b].set(species_a)

        # Swap spins too (if Fe moves, its spin follows)
        spin_a = spins[idx_a]
        spin_b = spins[idx_b]
        new_spins = spins.at[idx_a].set(spin_b).at[idx_b].set(spin_a)

        return new_species, new_spins

    return swap_step


def make_spin_flip_fn(spin_mask):
    """Create spin flip function.

    spin_mask: (N,) float, 1.0 for sites that carry spin.
    Picks one spin site and flips its spin (σ → -σ).
    """
    def flip_step(key, species, spins):
        n_at = species.shape[0]
        gumbel = random.gumbel(key, (n_at,))
        # Only pick from spin-carrying sites
        masked = jnp.where(spin_mask > 0, gumbel, -jnp.inf)
        flip_idx = jnp.argmax(masked)
        new_spins = spins.at[flip_idx].set(-spins[flip_idx])
        return species, new_spins

    return flip_step


# ═══════════════════════════════════════════════════════════════════════
# 5. VECTORIZED REPLICA EXCHANGE (v4: pure JAX, JIT-compatible)
# ═══════════════════════════════════════════════════════════════════════

def make_exchange_fn(n_replicas):
    """Create vectorized replica exchange function.

    Adjacent non-overlapping pairs are swapped simultaneously.
    Even pairs (0,1),(2,3),... and odd pairs (1,2),(3,4),... alternate
    based on step index (even step → even pairs, odd step → odd pairs).

    All pairs within one offset are independent → fully parallel.
    """
    # Precompute pair indices for both offsets (fixed shapes for JIT)
    # Even offset: pairs (0,1), (2,3), (4,5), ...
    n_even = (n_replicas) // 2         # number of even pairs
    n_odd  = (n_replicas - 1) // 2     # number of odd pairs
    n_max  = max(n_even, n_odd)        # pad to fixed shape

    even_i = jnp.arange(0, 2 * n_even, 2)  # [0, 2, 4, ...]
    odd_i  = jnp.arange(1, 1 + 2 * n_odd, 2)  # [1, 3, 5, ...]

    # Pad to same length for lax.cond compatibility
    even_i_pad = jnp.pad(even_i, (0, n_max - n_even), constant_values=0)
    odd_i_pad  = jnp.pad(odd_i,  (0, n_max - n_odd),  constant_values=0)
    even_mask  = jnp.arange(n_max) < n_even
    odd_mask   = jnp.arange(n_max) < n_odd

    def exchange(key, species_b, spins_b, E_b, betas, step_idx):
        """Vectorized replica exchange for one step.

        Args:
            key: PRNG key
            species_b: (n_replicas, n_atoms) int32
            spins_b: (n_replicas, n_atoms, 1) float32
            E_b: (n_replicas,) float32
            betas: (n_replicas,) float32
            step_idx: scalar int — used to alternate even/odd offset

        Returns:
            species_b, spins_b, E_b after exchange
        """
        key, k_accept = random.split(key)

        # Alternate even/odd based on step index
        use_even = (step_idx % 2 == 0)
        all_i = jnp.where(use_even, even_i_pad, odd_i_pad)
        valid = jnp.where(use_even, even_mask, odd_mask)
        all_j = all_i + 1

        # Compute acceptance for all pairs at once
        dBeta = betas[all_j] - betas[all_i]      # (n_max,)
        dE    = E_b[all_j] - E_b[all_i]           # (n_max,)
        log_prob = dE * dBeta
        u = random.uniform(k_accept, (n_max,))
        accept = (u < jnp.exp(jnp.minimum(log_prob, 0.0))) & valid  # (n_max,)

        # Conditional swap — all pairs simultaneously
        # Species: (n_max, n_atoms)
        acc_sp = accept[:, None]  # broadcast over atoms
        old_sp_i = species_b[all_i]
        old_sp_j = species_b[all_j]
        new_sp_i = jnp.where(acc_sp, old_sp_j, old_sp_i)
        new_sp_j = jnp.where(acc_sp, old_sp_i, old_sp_j)
        species_b = species_b.at[all_i].set(new_sp_i)
        species_b = species_b.at[all_j].set(new_sp_j)

        # Spins: (n_max, n_atoms, 1)
        acc_spin = accept[:, None, None]  # broadcast over atoms and spin dim
        old_spin_i = spins_b[all_i]
        old_spin_j = spins_b[all_j]
        new_spin_i = jnp.where(acc_spin, old_spin_j, old_spin_i)
        new_spin_j = jnp.where(acc_spin, old_spin_i, old_spin_j)
        spins_b = spins_b.at[all_i].set(new_spin_i)
        spins_b = spins_b.at[all_j].set(new_spin_j)

        # Energies: (n_max,)
        old_E_i = E_b[all_i]
        old_E_j = E_b[all_j]
        new_E_i = jnp.where(accept, old_E_j, old_E_i)
        new_E_j = jnp.where(accept, old_E_i, old_E_j)
        E_b = E_b.at[all_i].set(new_E_i)
        E_b = E_b.at[all_j].set(new_E_j)

        return key, species_b, spins_b, E_b

    return exchange


# ═══════════════════════════════════════════════════════════════════════
# 6. PT-MCMC ENGINE (v4: scan(vmap(step) + exchange))
# ═══════════════════════════════════════════════════════════════════════

def run_pt_mcmc():
    # ── Load everything ───────────────────────────────────────────────
    print("\n[1] Loading model & graph...")
    graph = build_template_graph(CIF_TEMPLATE)
    model, params, model_name, use_spin = load_model()
    single_energy, batched_energy = make_energy_fn(model, params, graph, use_spin)

    n_atoms = graph['n_atoms']

    # ── Build move functions ──────────────────────────────────────────
    move_fns = []
    move_names = []

    for sub in SUBLATTICES:
        mask = graph['sublattice_masks'][sub['name']]
        fn = make_swap_fn(mask, sub['species'])
        move_fns.append(fn)
        move_names.append(f"swap_{sub['name']}")

    if HAS_SPIN_FLIP:
        fn = make_spin_flip_fn(graph['spin_mask'])
        move_fns.append(fn)
        move_names.append("spin_flip")

    n_moves = len(move_fns)
    assert n_moves > 0, "No move types configured!"
    print(f"  Move types ({n_moves}): {move_names}")

    # ── Vectorized replica exchange ───────────────────────────────────
    exchange_fn = make_exchange_fn(N_REPLICAS)

    # ── Temperature schedule ──────────────────────────────────────────
    temperatures = np.geomspace(T_MIN, T_MAX, N_REPLICAS).astype(np.float32)
    kB = 8.617333e-5  # eV/K
    betas = jnp.array(1.0 / (kB * temperatures))
    print(f"  Temperatures: {T_MIN:.0f} → {T_MAX:.0f} K ({N_REPLICAS} replicas)")

    # ── Initial configuration ─────────────────────────────────────────
    print("\n[2] Initializing replicas...")

    rng = random.PRNGKey(SEED)

    if COMPOSITION is not None:
        # ── Composition-based init: per-replica random permutation ─────
        # COMPOSITION is a dict: {sublattice_name: {species_idx: count, ...}}
        #
        # Site lookup: union of ALL species mentioned in the composition entry.
        # This handles ternary single-sublattice (Fe-Ni-Cr) correctly:
        #   FCC: {0: 12, 1: 16, 2: 4} → sites where species ∈ {0,1,2} = all FCC sites
        #
        # For multi-sublattice (STFO):
        #   B_site: {0: 28, 1: 4} → sites where species ∈ {0,1} = B-sites
        #   O_site: {2: 94, 3: 2} → sites where species ∈ {2,3} = O-sites
        print(f"  Composition: {COMPOSITION}")

        base_species = np.zeros(n_atoms, dtype=np.int32)
        template_species = graph['species']

        sublattice_templates = {}
        sublattice_sites = {}

        for sub_name, spec_counts in COMPOSITION.items():
            # Find sites by union of all species in this composition entry
            all_sp = [int(s) for s in spec_counts.keys()]
            site_mask = np.zeros(n_atoms, dtype=bool)
            for sp in all_sp:
                site_mask |= (template_species == sp)

            # Also include sites from other species that share the same
            # physical sublattice. Check if a sublattice with this name exists.
            if sub_name in graph['sublattice_site_idx']:
                # Use the sublattice's full site list
                sub_sites = graph['sublattice_site_idx'][sub_name]
                site_mask[sub_sites] = True

            sites = np.where(site_mask)[0]
            n_sites = len(sites)

            # Build species template for this sublattice
            pieces = []
            total = 0
            for sp_str, count in spec_counts.items():
                sp = int(sp_str)
                pieces.append(jnp.full(count, sp, dtype=jnp.int32))
                total += count
            template_arr = jnp.concatenate(pieces)

            assert total == n_sites, (
                f"Composition '{sub_name}': sum={total} != n_sites={n_sites}. "
                f"Species in template at these sites: "
                f"{dict(zip(*np.unique(template_species[sites], return_counts=True)))}")

            sublattice_templates[sub_name] = template_arr
            sublattice_sites[sub_name] = jnp.array(sites)
            print(f"    {sub_name}: {dict(spec_counts)} ({n_sites} sites)")

        # Per-replica random permutation for each sublattice
        species_batch = jnp.tile(jnp.array(base_species)[None, :], (N_REPLICAS, 1))

        for sub_name in sublattice_templates:
            tmpl = sublattice_templates[sub_name]
            sites = sublattice_sites[sub_name]
            rng, key_perm = random.split(rng)
            keys = random.split(key_perm, N_REPLICAS)
            permed = vmap(lambda k, arr: random.permutation(k, arr),
                          in_axes=(0, None))(keys, tmpl)
            species_batch = species_batch.at[:, sites].set(permed)

    else:
        # ── Template-based init: all replicas identical ───────────────
        init_species = jnp.array(graph['species'])
        species_batch = jnp.tile(init_species[None, :], (N_REPLICAS, 1))

    # v4: vectorized spin initialization (reproducible via explicit RNG)
    # When composition is set, spin_mask must be recomputed per-replica
    # since species differ across replicas. Use replica 0 as representative
    # (all replicas have same species counts, just different positions).
    rng, key_spin = random.split(rng)
    if HAS_SPIN_FLIP:
        if COMPOSITION is not None:
            # Spin sites depend on actual species placement — but since
            # all replicas have the same species set (just permuted),
            # the NUMBER of spin sites is fixed. Each replica's spin sites
            # are wherever its spin-carrying species ended up.
            # For spin init, we assign random ±1 to ALL potential spin species
            # positions per replica.
            # spin_mask per replica: sites where species is in SPIN_SPECIES
            spin_species_set = jnp.array(SPIN_SPECIES)
            # (N_REPLICAS, n_atoms) bool
            per_replica_spin_mask = jnp.isin(species_batch, spin_species_set)
            n_spin_per_replica = int(per_replica_spin_mask[0].sum())
            # Generate random spins for each replica
            spin_vals = random.choice(
                key_spin, jnp.array([-1.0, 1.0]),
                shape=(N_REPLICAS, n_spin_per_replica))
            init_spins = jnp.zeros((N_REPLICAS, n_atoms, 1), dtype=jnp.float32)
            # For each replica, scatter spin_vals into spin sites
            def _set_spins(sp_batch_row, mask_row, vals_row):
                idx = jnp.where(mask_row, size=n_spin_per_replica)[0]
                return jnp.zeros((n_atoms, 1)).at[idx, 0].set(vals_row)
            spins_batch = vmap(_set_spins)(species_batch, per_replica_spin_mask, spin_vals)
        else:
            spin_mask_np = np.array(graph['spin_mask'])
            spin_sites = spin_mask_np > 0
            n_spin = spin_sites.sum()
            spin_vals = random.choice(
                key_spin, jnp.array([-1.0, 1.0]),
                shape=(N_REPLICAS, n_spin))
            init_spins = jnp.zeros((N_REPLICAS, n_atoms, 1), dtype=jnp.float32)
            spin_idx = jnp.where(jnp.array(spin_sites))[0]
            init_spins = init_spins.at[:, spin_idx, 0].set(spin_vals)
            spins_batch = init_spins
    else:
        spins_batch = jnp.zeros((N_REPLICAS, n_atoms, 1), dtype=jnp.float32)

    # Initial energies
    current_E = batched_energy(species_batch, spins_batch)
    print(f"  Initial E range: [{float(current_E.min()):.2f}, {float(current_E.max()):.2f}]")

    # ── MC kernel: single step for one replica ────────────────────────
    # v4: single_energy reused (no duplicate model.apply code)
    def local_step(key, species, spins, E, beta):
        """Single MC step for one replica. Returns updated state."""
        key, k_move, k_accept = random.split(key, 3)

        # Cycle through move types
        move_idx = key[0] % n_moves  # cheap deterministic cycling from key bits

        # Apply move (lax.switch for multiple move types)
        branches = []
        for fn in move_fns:
            branches.append(lambda k, s, sp, _fn=fn: _fn(k, s, sp))
        new_species, new_spins = lax.switch(move_idx, branches, k_move, species, spins)

        # v4: reuse single_energy (deduplicated)
        E_new = single_energy(new_species, new_spins)

        # Metropolis acceptance
        dE = E_new - E
        log_prob = -dE * beta
        accept = random.uniform(k_accept) < jnp.exp(jnp.minimum(log_prob, 0.0))

        species = jnp.where(accept, new_species, species)
        spins   = jnp.where(accept, new_spins, spins)
        E       = jnp.where(accept, E_new, E)

        return key, species, spins, E

    # ── v4 hybrid: vmap(lax.scan(step)) + vectorized exchange between chunks
    # This recovers v3 speed (XLA optimizes replica-internal scan) while
    # keeping the vectorized exchange from v4 (no Python for-loop).
    #
    # SWAP_INTERVAL is now in chunks (exchange every N chunks).
    # With small chunk_size (e.g., 100), this gives fine-grained control.

    def replica_chunk(keys, species_b, spins_b, E_b, betas):
        """Run CHUNK_SIZE local MC steps for all replicas (no exchange)."""
        step_indices = jnp.arange(CHUNK_SIZE)

        def run_one_replica(key, species, spins, E, beta):
            def scan_step(carry, _):
                key, species, spins, E, beta = carry
                key, species, spins, E = local_step(key, species, spins, E, beta)
                return (key, species, spins, E, beta), None

            (key, species, spins, E, _), _ = lax.scan(
                scan_step, (key, species, spins, E, beta), step_indices)
            return key, species, spins, E

        keys, species_b, spins_b, E_b = vmap(
            run_one_replica)(keys, species_b, spins_b, E_b, betas)
        return keys, species_b, spins_b, E_b

    # JIT compile MC chunk and exchange separately
    replica_chunk_jit = jax.jit(replica_chunk)

    @jax.jit
    def exchange_step(ex_key, species_b, spins_b, E_b, betas, step_idx):
        """One vectorized replica exchange."""
        return exchange_fn(ex_key, species_b, spins_b, E_b, betas, step_idx)

    # ── Production run ────────────────────────────────────────────────
    n_chunks = N_STEPS // CHUNK_SIZE
    # SWAP_INTERVAL: exchange every N chunks (1 = every chunk boundary)
    swap_every = max(1, SWAP_INTERVAL)
    print(f"\n[3] Production: {n_chunks} chunks × {CHUNK_SIZE} steps")
    print(f"    Burn-in: {BURNIN_CHUNKS} chunks ({BURNIN_STEPS} steps)")
    print(f"    Replica exchange: every {swap_every} chunk(s) = {swap_every * CHUNK_SIZE} steps")

    # Per-replica RNG keys
    rng, *replica_keys = random.split(rng, N_REPLICAS + 1)
    replica_keys = jnp.stack(replica_keys)
    rng, ex_key = random.split(rng)

    storage_species, storage_spins, storage_E = [], [], []

    # ── JIT warmup (first chunk triggers compilation) ─────────────────
    print("\n    [JIT] Compiling... ", end="", flush=True)
    t_jit_start = time.time()
    replica_keys, species_batch, spins_batch, current_E = replica_chunk_jit(
        replica_keys, species_batch, spins_batch, current_E, betas)
    current_E.block_until_ready()
    t_jit = time.time() - t_jit_start
    print(f"done in {t_jit:.1f}s")

    # ── Timed production (excluding JIT) ──────────────────────────────
    t_prod_start = time.time()
    global_step = CHUNK_SIZE  # chunk 0 done in warmup

    for chunk in range(1, n_chunks):
        # MC steps (vmap + lax.scan, no exchange)
        replica_keys, species_batch, spins_batch, current_E = replica_chunk_jit(
            replica_keys, species_batch, spins_batch, current_E, betas)

        global_step += CHUNK_SIZE

        # Vectorized replica exchange every swap_every chunks
        if chunk % swap_every == 0:
            ex_key, species_batch, spins_batch, current_E = exchange_step(
                ex_key, species_batch, spins_batch, current_E, betas, global_step)

        # v4: CHUNK_SIZE = save interval (no separate SAVE_EVERY)
        if chunk >= BURNIN_CHUNKS:
            storage_species.append(np.array(species_batch))
            storage_spins.append(np.array(spins_batch))
            storage_E.append(np.array(current_E))

        # Progress
        if (chunk + 1) % max(1, n_chunks // 20) == 0 or (chunk + 1) == n_chunks:
            current_E.block_until_ready()
            elapsed_prod = time.time() - t_prod_start
            steps_done = (chunk + 1) * CHUNK_SIZE
            steps_prod = steps_done - CHUNK_SIZE  # exclude warmup chunk
            speed = steps_prod / elapsed_prod if elapsed_prod > 0 else 0
            eta = (N_STEPS - steps_done) / speed if speed > 0 else 0
            n_saved = len(storage_E)
            E_low = float(current_E[0])  # lowest T replica
            print(f"    Chunk {chunk+1:>5d}/{n_chunks} | "
                  f"{speed:.0f} steps/s | "
                  f"E_low={E_low:.2f} | "
                  f"Saved: {n_saved} | "
                  f"ETA: {eta/60:.1f} min")

    t_prod_end = time.time()
    t_prod = t_prod_end - t_prod_start
    t_total = t_prod_end - t_jit_start

    # ── Save results ──────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT) or '.', exist_ok=True)
    print(f"\n[4] Saving → {OUTPUT}")
    if storage_E:
        all_species  = np.stack(storage_species)
        all_spins    = np.stack(storage_spins)
        all_energies = np.stack(storage_E)
    else:
        all_species  = np.array(species_batch)[None]
        all_spins    = np.array(spins_batch)[None]
        all_energies = np.array(current_E)[None]

    np.savez(
        OUTPUT,
        temperatures=np.array(temperatures),
        sampled_species=all_species,
        sampled_spins=all_spins,
        sampled_energies=all_energies,
        final_species=np.array(species_batch),
        final_spins=np.array(spins_batch),
        final_energy=np.array(current_E),
    )
    print(f"Snapshots: {all_energies.shape[0]}, "
          f"Species: {all_species.shape}, Spins: {all_spins.shape}")

    # ── Benchmark summary ─────────────────────────────────────────────
    gpu_name = "unknown"
    gpu_mem  = "unknown"
    try:
        devs = jax.devices()
        if devs:
            gpu_name = devs[0].device_kind
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=name,memory.total',
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    parts = result.stdout.strip().split(', ')
                    gpu_name = parts[0]
                    gpu_mem  = f"{int(parts[1])} MB"
            except Exception:
                pass
    except Exception:
        pass

    prod_steps = (n_chunks - 1) * CHUNK_SIZE
    prod_speed = prod_steps / t_prod if t_prod > 0 else 0
    steps_per_replica = prod_speed / N_REPLICAS if N_REPLICAS > 0 else 0

    benchmark = {
        'dataset': DATASET_NAME,
        'model_type': MODEL_TYPE,
        'n_params': sum(p.size for p in jax.tree.leaves(params)),
        'n_atoms': n_atoms,
        'n_replicas': N_REPLICAS,
        'n_steps': N_STEPS,
        'chunk_size': CHUNK_SIZE,
        'swap_interval': SWAP_INTERVAL,
        'n_moves': n_moves,
        'move_names': move_names,
        'gpu': gpu_name,
        'gpu_memory': gpu_mem,
        'jit_compile_sec': round(t_jit, 2),
        'production_sec': round(t_prod, 2),
        'total_sec': round(t_total, 2),
        'production_steps_per_sec': round(prod_speed, 1),
        'steps_per_replica_per_sec': round(steps_per_replica, 2),
        'total_mc_steps': N_STEPS * N_REPLICAS,
    }

    print(f"\n{'═' * 70}")
    print(f"  BENCHMARK SUMMARY")
    print(f"{'═' * 70}")
    print(f"  GPU:              {gpu_name} ({gpu_mem})")
    print(f"  System:           {DATASET_NAME} ({n_atoms} atoms)")
    print(f"  Model:            {MODEL_TYPE} ({benchmark['n_params']:,} params)")
    print(f"  Replicas:         {N_REPLICAS}")
    print(f"  Total MC steps:   {N_STEPS * N_REPLICAS:,} ({N_STEPS} × {N_REPLICAS})")
    print(f"  Moves:            {move_names}")
    print(f"  Swap interval:    every {SWAP_INTERVAL} step(s)")
    print(f"  ──────────────────────────────────────────────")
    print(f"  JIT compile:      {t_jit:.1f} s")
    print(f"  Production:       {t_prod:.1f} s")
    print(f"  Total:            {t_total:.1f} s")
    print(f"  Speed:            {prod_speed:,.0f} steps/s (all replicas)")
    print(f"                    {steps_per_replica:.1f} steps/replica/s")
    print(f"{'═' * 70}")

    # Save benchmark as JSON alongside npz
    bench_path = OUTPUT.replace('.npz', '_benchmark.json')
    with open(bench_path, 'w') as f:
        json.dump(benchmark, f, indent=2)
    print(f"  Benchmark → {bench_path}")


if __name__ == '__main__':
    run_pt_mcmc()
