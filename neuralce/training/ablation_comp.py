"""
ablation.py — 5-Model Lite Ablation + Optuna HP Search
JAX/Flax. Works for any fixed-lattice system (perovskites, alloys, etc.)

Models (Lite — one-hot node + shell edge):
  0. ising_lite              — No spin (baseline)
  1. neuralce_evenodd_lite   — Product backbone + EvenOdd (main)
  2. neuralce_sisj_lite      — Product backbone + σᵢσⱼ edge
  3. gnn_sisj_lite           — Concat-MLP backbone + σᵢσⱼ edge
  4. gnn_evenodd_lite        — Concat-MLP backbone + EvenOdd

Ablation table:
  ┌─────────────┬──────────────────────┬──────────────────────┐
  │             │ EvenOdd              │ SiSj (σᵢσⱼ edge)    │
  ├─────────────┼──────────────────────┼──────────────────────┤
  │ Product(CE) │ neuralce_evenodd_lite│ neuralce_sisj_lite   │
  │ Concat-MLP  │ gnn_evenodd_lite     │ gnn_sisj_lite        │
  └─────────────┴──────────────────────┴──────────────────────┘
  + ising_lite (no spin baseline)

Graph construction:
  If config contains graph.shell_edges, uses gap-based boundaries from
  analyze_cutoffs.py (recommended). Otherwise falls back to linspace.

Usage:
  Set CONFIG_PATH env var or edit below to point to your YAML config.
  Run: python ablation.py
  Or:  CONFIG_PATH=./configs/my_system.yaml python ablation.py
"""

import os, time, pickle, json, copy, random as pyrandom, re
import numpy as np
from scipy.stats import spearmanr

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("⚠ optuna not installed — will run single HP only")

import pandas as pd
from pymatgen.core.structure import Structure

from neuralce.models.NeuralCE_jax import (create_neuralce, is_spin_model, is_sisj_model,
                                          needs_spin, LITE_MODELS)


# ═══════════════════════════════════════════════════════════════════════
# CONFIG — yaml 경로만 바꾸면 됨
# ═══════════════════════════════════════════════════════════════════════
CONFIG_PATH = os.environ.get('CONFIG_PATH', './configs/stfo_spin_ablation.yaml')

import yaml
with open(CONFIG_PATH) as f:
    _cfg = yaml.safe_load(f)

# Mode check
_mode = _cfg.get('mode', 'ablation')
if _mode != 'ablation':
    raise ValueError(f"This script requires mode: ablation, got '{_mode}'. "
                     f"Use pt_mcmc.py for mode: pt_mcmc.")

# Common: dataset
DATASET_NAME = _cfg['dataset_name']
CIF_DIR      = _cfg['cif_dir']
DETAILED_CSV = _cfg['csv_path']
SPIN_PKL     = _cfg.get('spin_pkl', None)
ID_COL       = _cfg.get('id_col', 'cif_id')
COMP_REGEX   = _cfg.get('comp_regex', r'_(\d+)')
SPECIES_MAP  = {int(k): v for k, v in _cfg['species_map'].items()}
N_SPECIES    = len(SPECIES_MAP)
N_ATOMS      = _cfg['n_atoms']              # will be overwritten if exclude_species
N_ATOMS_ORIG = _cfg['n_atoms']              # original count for per-atom metrics
HAS_SPIN     = _cfg.get('has_spin', False)

# Species exclusion: remove constant-site atoms from graph
# e.g. exclude_species: [38]  removes Sr (Z=38) from STFO
EXCLUDE_Z    = set(_cfg.get('exclude_species', []))

# Graph config (from analyze_cutoffs.py or manual)
# Two modes:
#   (A) Single fixed cutoff:  graph.cutoff + graph.shell_edges
#   (B) Multiple candidates:  graph.candidates: {cutoff: {n_shells, shell_edges}, ...}
#   (C) Neither set:          Optuna searches cutoff/n_shells with linspace
_graph       = _cfg.get('graph', {})
GRAPH_MAX_NUM_NBR = _graph.get('max_num_nbr', 12)

_candidates = _graph.get('candidates', None)
if _candidates is not None:
    # Mode B: multiple cutoff candidates for Optuna to choose from
    GRAPH_CANDIDATES = {}
    for cut_str, info in _candidates.items():
        cut = float(cut_str)
        GRAPH_CANDIDATES[cut] = {
            'n_shells': info['n_shells'],
            'shell_edges': info['shell_edges'],
        }
    GRAPH_MODE = 'candidates'
    print(f"Graph config: {len(GRAPH_CANDIDATES)} cutoff candidates "
          f"{sorted(GRAPH_CANDIDATES.keys())} (Optuna will choose)")
elif _graph.get('cutoff') is not None:
    # Mode A: single fixed cutoff
    GRAPH_CANDIDATES = {
        _graph['cutoff']: {
            'n_shells': _graph['n_shells'],
            'shell_edges': _graph.get('shell_edges'),
        }
    }
    GRAPH_MODE = 'fixed'
    print(f"Graph config: fixed cutoff={_graph['cutoff']}, "
          f"n_shells={_graph['n_shells']}, shell_edges={_graph.get('shell_edges')}")
else:
    # Mode C: Optuna searches everything
    GRAPH_CANDIDATES = None
    GRAPH_MODE = 'search'
    print(f"Graph config: not set → Optuna will search cutoff/n_shells")

# Ablation-specific
_abl         = _cfg.get('ablation', {})
SEED         = _abl.get('seed', 42)
VAL_FRAC     = _abl.get('val_frac', 0.15)
TEST_FRAC    = _abl.get('test_frac', 0.15)
MAX_EPOCHS   = _abl.get('max_epochs', 3000)
PATIENCE     = _abl.get('patience', 80)
BATCH_SIZE   = _abl.get('batch_size', 32)
N_TRIALS     = _abl.get('n_trials', 30)
OPTUNA_TIMEOUT = _abl.get('optuna_timeout', None)
OUTPUT_DIR     = _abl.get('output_dir', '.')

# Search space from config (with defaults)
_ss = _abl.get('search_space', {})
SS_ATOM_FEA_LEN = _ss.get('atom_fea_len', [8, 16, 24, 32, 48])
SS_N_CONV       = _ss.get('n_conv', [2, 5])           # [min, max]
SS_H_FEA_LEN    = _ss.get('h_fea_len', [16, 32, 64, 128])
SS_LR           = _ss.get('lr', [1.0e-4, 5.0e-3])     # [min, max] log scale
SS_ODD_FEA_LEN  = _ss.get('odd_fea_len', [1, 2, 4, 8, 16])

print(f"Search space: atom_fea_len={SS_ATOM_FEA_LEN}, n_conv={SS_N_CONV}, "
      f"h_fea_len={SS_H_FEA_LEN}, lr={SS_LR}")
print(f"Training: max_epochs={MAX_EPOCHS}, patience={PATIENCE}, "
      f"n_trials={N_TRIALS}, batch_size={BATCH_SIZE}")

RUN_MODELS   = _abl.get('run_models', [
    'ising_lite',
    'neuralce_evenodd_lite',
    'neuralce_sisj_lite',
    'gnn_sisj_lite',
    'gnn_evenodd_lite',
])

# Auto-filter: spin 없으면 spin 모델 자동 제외
if not HAS_SPIN:
    _before = len(RUN_MODELS)
    RUN_MODELS = [m for m in RUN_MODELS if not is_spin_model(m)]
    if len(RUN_MODELS) < _before:
        print(f"⚠ HAS_SPIN=False → spin models excluded. Running: {RUN_MODELS}")
    if not RUN_MODELS:
        raise ValueError("No models left after filtering spin models.")


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_data(cif_dir, csv_path, spin_pkl_path, id_col, comp_regex,
              exclude_z=None):
    """Load dataset → list of dicts.
    
    spin_pkl_path=None → all spins zero.
    exclude_z: set of atomic numbers to remove from graph (e.g. {38} for Sr).
               Energy target is unchanged — the excluded atoms are structurally
               constant and their contribution is absorbed into the learned bias.
    """
    if exclude_z is None:
        exclude_z = set()

    df = pd.read_csv(csv_path)
    if id_col not in df.columns:
        # fallback: try both common names
        id_col = 'id' if 'id' in df.columns else 'cif_id'
    energy_map = dict(zip(df[id_col].astype(str), df['total_energy'].values))

    # Spin: load if available, else empty dict → fallback to zeros
    if spin_pkl_path and os.path.exists(spin_pkl_path):
        spin_df = pd.read_pickle(spin_pkl_path)
        spin_map = dict(zip(spin_df['cif_id'].astype(str),
                            spin_df['spin_states'].values))
        print(f"  Spin data loaded: {len(spin_map)} entries")
    else:
        spin_map = {}
        print(f"  No spin data → all spins set to 0")

    structures = []
    cif_files = sorted([f for f in os.listdir(cif_dir) if f.endswith('.cif')])
    n_skip = 0

    for cif_file in cif_files:
        cif_id = cif_file.replace('.cif', '')
        if cif_id not in energy_map:
            n_skip += 1
            continue

        crystal = Structure.from_file(os.path.join(cif_dir, cif_file))
        spins = spin_map.get(cif_id, [0] * len(crystal))
        spins = np.array(spins, dtype=np.float32)

        # --- Filter excluded species ---
        if exclude_z:
            keep_idx = [i for i, site in enumerate(crystal)
                        if site.specie.Z not in exclude_z]
            crystal = Structure.from_sites([crystal[i] for i in keep_idx])
            spins = spins[keep_idx]

        comp_match = re.search(comp_regex, cif_id)
        comp_code = int(comp_match.group(1)) if comp_match else 0

        structures.append({
            'cif_id': cif_id, 'total_energy': energy_map[cif_id],
            'crystal': crystal,
            'spin_states': spins,
            'comp_code': comp_code, 'n_atoms': len(crystal),
        })

    print(f"Loaded {len(structures)} structures ({n_skip} skipped)")
    if exclude_z:
        z_names = {38: 'Sr', 56: 'Ba', 20: 'Ca', 39: 'Y'}  # common A-site
        excluded = [z_names.get(z, f'Z={z}') for z in sorted(exclude_z)]
        print(f"  Excluded species: {excluded}")
        n_at_set = set(s['n_atoms'] for s in structures)
        print(f"  Atoms per crystal after exclusion: {sorted(n_at_set)}")
    comp_counts = {}
    for s in structures:
        comp_counts[s['comp_code']] = comp_counts.get(s['comp_code'], 0) + 1
    print(f"Compositions: {dict(sorted(comp_counts.items()))}")
    return structures


# ═══════════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ═══════════════════════════════════════════════════════════════════════

def build_graph_lite(struct, cutoff, n_shells, max_num_nbr=12, include_sisj=False,
                     shell_edges=None):
    """Build graph from pymatgen Structure.
    
    Node:  [Ti, Fe, O, Vo] one-hot (4-dim), Sr→zeros
    Edge:  shell one-hot (n_shells-dim)
           if include_sisj: append σᵢσⱼ → (n_shells+1)-dim
    Spin:  (n_atoms, 1) scalar

    shell_edges: if provided, use these boundaries for shell assignment
                 (from analyze_cutoffs.py). Otherwise linspace fallback.
    """
    crystal = struct['crystal']
    n_at = len(crystal)
    spins = struct['spin_states']

    # Node features
    atom_fea = np.zeros((n_at, N_SPECIES), dtype=np.float32)
    for i, site in enumerate(crystal):
        z = site.specie.Z
        if z in SPECIES_MAP:
            atom_fea[i, SPECIES_MAP[z]] = 1.0

    # Neighbor search
    all_nbrs = crystal.get_all_neighbors(cutoff, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

    nbr_fea_idx = np.zeros((n_at, max_num_nbr), dtype=np.int32)
    nbr_dists   = np.full((n_at, max_num_nbr), cutoff + 1.0, dtype=np.float64)

    for i, nbr in enumerate(all_nbrs):
        n_nbr = min(len(nbr), max_num_nbr)
        for j in range(n_nbr):
            nbr_fea_idx[i, j] = nbr[j][2]
            nbr_dists[i, j]   = nbr[j][1]

    # Shell one-hot
    valid_mask = nbr_dists < cutoff
    if shell_edges is not None:
        # Gap-based boundaries from analyze_cutoffs.py
        _edges = np.array(shell_edges)
    else:
        # Fallback: linspace (legacy behavior)
        min_dist = nbr_dists[valid_mask].min() if valid_mask.any() else 0.1
        _edges = np.linspace(min_dist * 0.99, cutoff, n_shells + 1)

    shell_idx = np.clip(np.digitize(nbr_dists, _edges) - 1, 0, n_shells - 1)

    shell_oh = np.zeros((n_at, max_num_nbr, n_shells), dtype=np.float32)
    for i in range(n_at):
        for j in range(max_num_nbr):
            if nbr_dists[i, j] < cutoff:
                shell_oh[i, j, shell_idx[i, j]] = 1.0

    # Edge features
    if include_sisj:
        # σᵢσⱼ for each edge
        sisj = np.zeros((n_at, max_num_nbr, 1), dtype=np.float32)
        for i in range(n_at):
            for j in range(max_num_nbr):
                if nbr_dists[i, j] < cutoff:
                    sisj[i, j, 0] = spins[i] * spins[nbr_fea_idx[i, j]]
        nbr_fea = np.concatenate([shell_oh, sisj], axis=-1)  # (n_at, M, n_shells+1)
    else:
        nbr_fea = shell_oh  # (n_at, M, n_shells)

    spin_fea = spins[:n_at].reshape(-1, 1).astype(np.float32)
    return atom_fea, nbr_fea, nbr_fea_idx, spin_fea


def build_all_graphs(structures, cutoff, n_shells, max_num_nbr=12,
                     include_sisj=False, shell_edges=None):
    """Build and batch all graphs."""
    lists = {'atom_fea': [], 'nbr_fea': [], 'nbr_fea_idx': [],
             'spin_fea': [], 'energies': [], 'comps': []}

    for s in structures:
        af, nf, nfi, sf = build_graph_lite(
            s, cutoff, n_shells, max_num_nbr, include_sisj,
            shell_edges=shell_edges)
        lists['atom_fea'].append(af)
        lists['nbr_fea'].append(nf)
        lists['nbr_fea_idx'].append(nfi)
        lists['spin_fea'].append(sf)
        lists['energies'].append(s['total_energy'])
        lists['comps'].append(s['comp_code'])

    return {
        'atom_fea':    jnp.array(np.stack(lists['atom_fea'])),
        'nbr_fea':     jnp.array(np.stack(lists['nbr_fea'])),
        'nbr_fea_idx': jnp.array(np.stack(lists['nbr_fea_idx'])),
        'spin_fea':    jnp.array(np.stack(lists['spin_fea'])),
        'energies':    jnp.array(lists['energies']),
        'comps':       lists['comps'],
    }


# ═══════════════════════════════════════════════════════════════════════
# MODEL CREATION
# ═══════════════════════════════════════════════════════════════════════

def create_model(model_name, hp):
    """Create Lite model from HP dict."""
    sisj = is_sisj_model(model_name)
    n_shells = hp['n_shells']
    edge_dim = n_shells + 1 if sisj else n_shells

    kwargs = {
        'atom_fea_len': hp['atom_fea_len'],
        'nbr_fea_len':  edge_dim,
        'n_conv':       hp['n_conv'],
        'h_fea_len':    hp['h_fea_len'],
    }
    if is_spin_model(model_name):
        kwargs['odd_fea_len'] = hp['odd_fea_len']

    return create_neuralce(model_type=model_name, pool_mode='fixed',
                           readout_type='sum', **kwargs)


# ═══════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════

def init_train_state(model, hp, rng_key, dataset, model_name):
    """Initialize model params + optimizer."""
    use_spin = is_spin_model(model_name)
    dummy_af  = dataset['atom_fea'][0]
    dummy_nf  = dataset['nbr_fea'][0]
    dummy_nfi = dataset['nbr_fea_idx'][0]

    init_kwargs = {
        'atom_fea': dummy_af, 'nbr_fea': dummy_nf, 'nbr_fea_idx': dummy_nfi,
        'batch_size': 1, 'n_atoms_per_crystal': N_ATOMS,
    }
    if use_spin:
        init_kwargs['atom_spins'] = dataset['spin_fea'][0]

    params = model.init(rng_key, **init_kwargs)

    lr = hp['lr']
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.1, peak_value=lr,
        warmup_steps=50, decay_steps=MAX_EPOCHS, end_value=lr * 0.01)
    optimizer = optax.chain(optax.clip_by_global_norm(5.0), optax.adamw(schedule))

    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer)


def _forward(state, af, nf, nfi, sf, use_spin):
    """Flat forward for a batch."""
    B = af.shape[0]
    af_flat = af.reshape(-1, af.shape[-1])
    offsets = jnp.arange(B)[:, None, None] * N_ATOMS
    nfi_flat = (nfi + offsets).reshape(-1, nfi.shape[-1])
    nf_flat  = nf.reshape(-1, nf.shape[-2], nf.shape[-1])

    kw = {'atom_fea': af_flat, 'nbr_fea': nf_flat, 'nbr_fea_idx': nfi_flat,
           'batch_size': B, 'n_atoms_per_crystal': N_ATOMS}
    if use_spin:
        kw['atom_spins'] = sf.reshape(-1, sf.shape[-1])

    return state.apply_fn(state.params, **kw).squeeze(-1)


def loss_fn(params, state, batch, use_spin):
    af, nf, nfi, sf, targets = batch
    B = af.shape[0]
    af_flat = af.reshape(-1, af.shape[-1])
    offsets = jnp.arange(B)[:, None, None] * N_ATOMS
    nfi_flat = (nfi + offsets).reshape(-1, nfi.shape[-1])
    nf_flat  = nf.reshape(-1, nf.shape[-2], nf.shape[-1])

    kw = {'atom_fea': af_flat, 'nbr_fea': nf_flat, 'nbr_fea_idx': nfi_flat,
           'batch_size': B, 'n_atoms_per_crystal': N_ATOMS}
    if use_spin:
        kw['atom_spins'] = sf.reshape(-1, sf.shape[-1])

    pred = state.apply_fn(params, **kw).squeeze(-1)
    return jnp.mean((pred - targets) ** 2)


@jax.jit
def train_step_spin(state, batch):
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params, state, batch, True)
    return state.apply_gradients(grads=grads), loss

@jax.jit
def train_step_nospin(state, batch):
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params, state, batch, False)
    return state.apply_gradients(grads=grads), loss


def eval_epoch(state, dataset, indices, use_spin, return_per_comp_srcc=False):
    """Evaluate on a subset.
    
    If return_per_comp_srcc=True, also compute mean per-composition SRCC
    using dataset['comps'].
    """
    all_preds, all_targets = [], []
    total_loss, n_total = 0.0, 0

    for start in range(0, len(indices), BATCH_SIZE):
        idx = indices[start:start+BATCH_SIZE]
        af = dataset['atom_fea'][idx]
        nf = dataset['nbr_fea'][idx]
        nfi = dataset['nbr_fea_idx'][idx]
        sf = dataset['spin_fea'][idx]
        targets = dataset['energies'][idx]
        B = af.shape[0]

        pred = _forward(state, af, nf, nfi, sf, use_spin)
        mse = float(jnp.mean((pred - targets) ** 2))
        total_loss += mse * B
        n_total += B
        all_preds.append(np.array(pred))
        all_targets.append(np.array(targets))

    val_loss = total_loss / max(n_total, 1)
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    if not return_per_comp_srcc:
        return val_loss, preds, targets

    # Per-composition SRCC
    comps = [dataset['comps'][i] for i in indices]
    comp_srccs = []
    for c in sorted(set(comps)):
        mask = [i for i, cc in enumerate(comps) if cc == c]
        if len(mask) >= 3:
            rho = spearmanr(targets[mask], preds[mask]).correlation
            if not np.isnan(rho):
                comp_srccs.append(rho)
    mean_comp_srcc = np.mean(comp_srccs) if comp_srccs else 0.0

    return val_loss, preds, targets, mean_comp_srcc


def train_model(model_name, hp, dataset, train_idx, val_idx, trial=None):
    """Full training loop.
    
    Best model is selected by mean per-composition SRCC (highest = best).
    Returns negative mean_comp_srcc for Optuna minimization.
    """
    use_spin = is_spin_model(model_name)
    model = create_model(model_name, hp)
    rng = jax.random.PRNGKey(SEED)
    state = init_train_state(model, hp, rng, dataset, model_name)
    n_params = sum(p.size for p in jax.tree.leaves(state.params))

    step_fn = train_step_spin if use_spin else train_step_nospin

    best_score = -float('inf')  # mean per-comp SRCC (higher is better)
    best_val_mse = float('inf')
    best_params = None
    patience_ctr = 0
    best_epoch = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        perm = np.array(jax.random.permutation(
            jax.random.PRNGKey(SEED + epoch), jnp.array(train_idx)))

        epoch_loss, n_bat = 0.0, 0
        for start in range(0, len(perm), BATCH_SIZE):
            idx = perm[start:start+BATCH_SIZE]
            if len(idx) < 2:
                continue
            batch = (dataset['atom_fea'][idx], dataset['nbr_fea'][idx],
                     dataset['nbr_fea_idx'][idx], dataset['spin_fea'][idx],
                     dataset['energies'][idx])
            state, loss = step_fn(state, batch)
            epoch_loss += float(loss)
            n_bat += 1
        epoch_loss /= max(n_bat, 1)

        val_loss, _, _, mean_comp_srcc = eval_epoch(
            state, dataset, val_idx, use_spin, return_per_comp_srcc=True)

        if mean_comp_srcc > best_score:
            best_score = mean_comp_srcc
            best_val_mse = val_loss
            best_params = copy.deepcopy(state.params)
            patience_ctr = 0
            best_epoch = epoch
        else:
            patience_ctr += 1

        if patience_ctr >= PATIENCE:
            break

        if trial is not None and HAS_OPTUNA:
            # Optuna minimizes, so report negative SRCC
            trial.report(-mean_comp_srcc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if epoch % 500 == 0:
            print(f"    [epoch {epoch:5d}] train_MSE={epoch_loss:.6f} "
                  f"val_MSE={val_loss:.6f} mean_comp_SRCC={mean_comp_srcc:.4f}")

    # Return negative SRCC for Optuna (minimize), plus val_mse for logging
    return -best_score, best_params, best_epoch, n_params, best_val_mse


# ═══════════════════════════════════════════════════════════════════════
# OPTUNA OBJECTIVE
# ═══════════════════════════════════════════════════════════════════════

def make_objective(model_name, structures, graph_cache, train_idx, val_idx):
    def objective(trial):
        atom_fea_len = trial.suggest_categorical('atom_fea_len', SS_ATOM_FEA_LEN)
        n_conv       = trial.suggest_int('n_conv', SS_N_CONV[0], SS_N_CONV[1])
        h_fea_len    = trial.suggest_categorical('h_fea_len', SS_H_FEA_LEN)
        lr           = trial.suggest_float('lr', SS_LR[0], SS_LR[1], log=True)

        # Graph params: 3 modes
        if GRAPH_MODE == 'candidates':
            cutoff_keys = sorted(GRAPH_CANDIDATES.keys())
            cutoff = trial.suggest_categorical('cutoff', cutoff_keys)
            info = GRAPH_CANDIDATES[cutoff]
            n_shells    = info['n_shells']
            shell_edges = info['shell_edges']
            max_num_nbr = GRAPH_MAX_NUM_NBR
        elif GRAPH_MODE == 'fixed':
            cutoff = list(GRAPH_CANDIDATES.keys())[0]
            info = GRAPH_CANDIDATES[cutoff]
            n_shells    = info['n_shells']
            shell_edges = info['shell_edges']
            max_num_nbr = GRAPH_MAX_NUM_NBR
        else:  # 'search'
            cutoff      = trial.suggest_float('cutoff', 3.0, 7.0, step=0.5)
            n_shells    = trial.suggest_int('n_shells', 2, 6)
            max_num_nbr = trial.suggest_categorical('max_num_nbr', [8, 12, 16])
            shell_edges = None

        hp = {
            'atom_fea_len': atom_fea_len, 'n_conv': n_conv,
            'h_fea_len': h_fea_len, 'lr': lr,
            'cutoff': cutoff, 'n_shells': n_shells,
            'max_num_nbr': max_num_nbr,
        }
        if is_spin_model(model_name):
            hp['odd_fea_len'] = trial.suggest_categorical('odd_fea_len', SS_ODD_FEA_LEN)

        sisj = is_sisj_model(model_name)
        cache_key = (cutoff, n_shells, max_num_nbr, sisj)
        if cache_key not in graph_cache:
            print(f"    Building graphs: cutoff={cutoff} n_shells={n_shells} "
                  f"max_nbr={max_num_nbr} sisj={sisj}"
                  f"{' (gap-based edges)' if shell_edges else ' (linspace edges)'}")
            graph_cache[cache_key] = build_all_graphs(
                structures, cutoff, n_shells, max_num_nbr,
                include_sisj=sisj, shell_edges=shell_edges)
        dataset = graph_cache[cache_key]

        print(f"  Trial {trial.number}: {hp}")
        neg_srcc, best_params, best_epoch, n_params, best_val_mse = train_model(
            model_name, hp, dataset, train_idx, val_idx, trial=trial)

        trial.set_user_attr('best_params', best_params)
        trial.set_user_attr('best_epoch', best_epoch)
        trial.set_user_attr('n_params', n_params)
        trial.set_user_attr('hp', hp)
        trial.set_user_attr('val_mse', best_val_mse)

        mean_srcc = -neg_srcc
        print(f"    → mean_comp_SRCC={mean_srcc:.4f} (val_MSE={best_val_mse:.6f}) "
              f"@ epoch {best_epoch} | params={n_params:,}")
        return neg_srcc
    return objective


# ═══════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════

def compute_metrics(targets, preds, comps_subset=None):
    rmse = np.sqrt(np.mean((targets - preds) ** 2))
    mae  = np.mean(np.abs(targets - preds))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-12)
    srcc = spearmanr(targets, preds).correlation

    per_comp = {}
    if comps_subset is not None:
        for c in sorted(set(comps_subset)):
            mask = [i for i, cc in enumerate(comps_subset) if cc == c]
            if len(mask) >= 3:
                per_comp[c] = spearmanr(targets[mask], preds[mask]).correlation

    return {'rmse': rmse, 'mae': mae, 'r2': r2,
            'srcc_global': srcc, 'srcc_per_comp': per_comp}


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    pyrandom.seed(SEED)
    np.random.seed(SEED)

    print(f"Dataset: {DATASET_NAME} (has_spin={HAS_SPIN})")
    if EXCLUDE_Z:
        print(f"Excluding atomic numbers: {sorted(EXCLUDE_Z)}")
    print(f"Loading data from {CIF_DIR} ...")
    structures = load_data(CIF_DIR, DETAILED_CSV, SPIN_PKL, ID_COL, COMP_REGEX,
                           exclude_z=EXCLUDE_Z)

    # Recompute N_ATOMS after exclusion (must be uniform for FixedPool)
    n_atoms_set = set(s['n_atoms'] for s in structures)
    if len(n_atoms_set) != 1:
        raise ValueError(
            f"Variable atom counts after exclusion: {sorted(n_atoms_set)}. "
            f"FixedPool requires uniform count. Check exclude_species config.")
    N_ATOMS = n_atoms_set.pop()
    print(f"N_ATOMS = {N_ATOMS} (after exclusion)" if EXCLUDE_Z else f"N_ATOMS = {N_ATOMS}")

    from sklearn.model_selection import train_test_split
    indices = list(range(len(structures)))
    trainval_idx, test_idx = train_test_split(indices, test_size=TEST_FRAC, random_state=SEED)
    val_rel = VAL_FRAC / (1 - TEST_FRAC)
    train_idx, val_idx = train_test_split(trainval_idx, test_size=val_rel, random_state=SEED)
    train_idx, val_idx, test_idx = np.array(train_idx), np.array(val_idx), np.array(test_idx)
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    all_results = []
    graph_cache = {}

    for model_name in RUN_MODELS:
        print(f"\n{'='*70}")
        print(f"  MODEL: {model_name}")
        print(f"{'='*70}")

        if HAS_OPTUNA and N_TRIALS > 1:
            study = optuna.create_study(
                direction='minimize',
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=50))
            objective = make_objective(model_name, structures, graph_cache,
                                       train_idx, val_idx)
            study.optimize(objective, n_trials=N_TRIALS, timeout=OPTUNA_TIMEOUT)

            bt = study.best_trial
            best_hp     = bt.user_attrs['hp']
            best_params = bt.user_attrs['best_params']
            best_epoch  = bt.user_attrs['best_epoch']
            n_params    = bt.user_attrs['n_params']
            best_val_mse = bt.user_attrs.get('val_mse', float('nan'))
            best_mean_srcc = -bt.value
            print(f"\n  Best: {best_hp}")
            print(f"  mean_comp_SRCC={best_mean_srcc:.4f} (val_MSE={best_val_mse:.6f}) "
                  f"@ epoch {best_epoch} | params={n_params:,}")
        else:
            # Single run with default HP — use first candidate if available
            if GRAPH_CANDIDATES:
                _first_cut = sorted(GRAPH_CANDIDATES.keys())[0]
                _first_info = GRAPH_CANDIDATES[_first_cut]
                _def_cutoff = _first_cut
                _def_nshells = _first_info['n_shells']
                _def_shell_edges = _first_info['shell_edges']
            else:
                _def_cutoff = 5.5
                _def_nshells = 4
                _def_shell_edges = None

            best_hp = {
                'atom_fea_len': 16, 'n_conv': 2, 'h_fea_len': 32,
                'odd_fea_len': 4, 'lr': 1e-3,
                'cutoff': _def_cutoff,
                'n_shells': _def_nshells,
                'max_num_nbr': GRAPH_MAX_NUM_NBR,
            }
            sisj = is_sisj_model(model_name)
            ck = (best_hp['cutoff'], best_hp['n_shells'], best_hp['max_num_nbr'], sisj)
            if ck not in graph_cache:
                graph_cache[ck] = build_all_graphs(
                    structures, best_hp['cutoff'], best_hp['n_shells'],
                    best_hp['max_num_nbr'], include_sisj=sisj,
                    shell_edges=_def_shell_edges)
            dataset = graph_cache[ck]

            print(f"  Single run: {best_hp}")
            neg_srcc, best_params, best_epoch, n_params, best_val_mse = train_model(
                model_name, best_hp, dataset, train_idx, val_idx)
            best_mean_srcc = -neg_srcc
            print(f"  mean_comp_SRCC={best_mean_srcc:.4f} (val_MSE={best_val_mse:.6f}) "
                  f"@ epoch {best_epoch} | params={n_params:,}")

        # --- Test evaluation ---
        sisj = is_sisj_model(model_name)
        ck = (best_hp['cutoff'], best_hp['n_shells'], best_hp['max_num_nbr'], sisj)
        if ck not in graph_cache:
            # Resolve shell_edges for this cutoff
            _se = None
            if GRAPH_CANDIDATES and best_hp['cutoff'] in GRAPH_CANDIDATES:
                _se = GRAPH_CANDIDATES[best_hp['cutoff']]['shell_edges']
            graph_cache[ck] = build_all_graphs(
                structures, best_hp['cutoff'], best_hp['n_shells'],
                best_hp['max_num_nbr'], include_sisj=sisj,
                shell_edges=_se)
        dataset = graph_cache[ck]

        use_spin = is_spin_model(model_name)
        model = create_model(model_name, best_hp)
        rng = jax.random.PRNGKey(SEED)
        eval_state = init_train_state(model, best_hp, rng, dataset, model_name)
        eval_state = eval_state.replace(params=best_params)

        _, preds_test, targets_test = eval_epoch(eval_state, dataset, test_idx, use_spin)

        test_comps = [dataset['comps'][i] for i in test_idx]
        metrics = compute_metrics(targets_test, preds_test, test_comps)
        metrics_pa = compute_metrics(targets_test / N_ATOMS_ORIG,
                                     preds_test / N_ATOMS_ORIG, test_comps)

        print(f"\n  TEST (total): RMSE={metrics['rmse']:.4f} eV | SRCC={metrics['srcc_global']:.4f}")
        print(f"  TEST (per atom): RMSE={metrics_pa['rmse']:.6f} eV/atom | "
              f"MAE={metrics_pa['mae']:.6f} | SRCC={metrics_pa['srcc_global']:.4f}")
        if metrics_pa['srcc_per_comp']:
            print(f"  Per-comp SRCC:")
            for c, s in sorted(metrics_pa['srcc_per_comp'].items()):
                print(f"    x={c/1000:.3f}: {s:.4f}")

        all_results.append({
            'name': model_name, 'hp': best_hp, 'n_params': n_params,
            'val_mse': best_val_mse, 'val_mean_comp_srcc': best_mean_srcc,
            'best_epoch': best_epoch,
            'test_targets': targets_test, 'test_preds': preds_test,
            'test_metrics': metrics, 'test_metrics_pa': metrics_pa,
        })

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        ckpt_path = os.path.join(OUTPUT_DIR, f'best_{DATASET_NAME}_{model_name}.pkl')
        with open(ckpt_path, 'wb') as f:
            pickle.dump({'params': best_params, 'hp': best_hp}, f)
        print(f"  Saved → {ckpt_path}")

    # --- Summary ---
    ROLES = {
        'ising_lite':              'No spin baseline',
        'neuralce_evenodd_lite':   'Product+EvenOdd (main)',
        'neuralce_sisj_lite':      'Product+SiSj',
        'gnn_sisj_lite':           'Concat+SiSj',
        'gnn_evenodd_lite':        'Concat+EvenOdd',
    }
    excl_tag = f", excl={sorted(EXCLUDE_Z)}" if EXCLUDE_Z else ""
    print(f"\n{'='*70}")
    print(f"  ABLATION SUMMARY ({DATASET_NAME} Lite{excl_tag})")
    print(f"  graph atoms={N_ATOMS}  orig atoms={N_ATOMS_ORIG}")
    print(f"{'='*70}")
    print(f"{'#':<3} {'Model':<26} {'Role':<24} {'Params':>7} "
          f"{'RMSE(eV/at)':>12} {'MAE(eV/at)':>12} {'SRCC':>7} {'CompSRCC':>8}")
    print(f"{'-'*3} {'-'*26} {'-'*24} {'-'*7} {'-'*12} {'-'*12} {'-'*7} {'-'*8}")
    for i, r in enumerate(all_results):
        m = r['test_metrics_pa']
        role = ROLES.get(r['name'], '')
        # Compute test mean per-comp SRCC
        comp_vals = list(m['srcc_per_comp'].values())
        test_comp_srcc = np.mean(comp_vals) if comp_vals else float('nan')
        print(f"{i:<3} {r['name']:<26} {role:<24} {r['n_params']:>7,} "
              f"{m['rmse']:>12.6f} {m['mae']:>12.6f} {m['srcc_global']:>7.4f} "
              f"{test_comp_srcc:>8.4f}")

    # --- Save ---
    meta = {
        'dataset': DATASET_NAME, 'has_spin': HAS_SPIN,
        'exclude_species': sorted(EXCLUDE_Z),
        'n_atoms_graph': N_ATOMS, 'n_atoms_orig': N_ATOMS_ORIG,
        'graph_mode': GRAPH_MODE,
        'graph_candidates': {str(k): v for k, v in GRAPH_CANDIDATES.items()}
            if GRAPH_CANDIDATES else None,
        'max_num_nbr': GRAPH_MAX_NUM_NBR,
    }
    log = []
    for r in all_results:
        log.append({
            'name': r['name'], 'hp': r['hp'], 'n_params': r['n_params'],
            'val_mse': float(r['val_mse']), 'best_epoch': r['best_epoch'],
            'test_rmse_pa': float(r['test_metrics_pa']['rmse']),
            'test_mae_pa': float(r['test_metrics_pa']['mae']),
            'test_srcc': float(r['test_metrics_pa']['srcc_global']),
            'test_srcc_per_comp': {str(k): float(v)
                for k, v in r['test_metrics_pa']['srcc_per_comp'].items()},
        })
    if EXCLUDE_Z:
        z_names = {38: 'Sr', 56: 'Ba', 20: 'Ca', 39: 'Y', 57: 'La'}
        suffix = '_no' + '_'.join(z_names.get(z, f'Z{z}') for z in sorted(EXCLUDE_Z))
    else:
        suffix = ''
    outname = os.path.join(OUTPUT_DIR, f'ablation_{DATASET_NAME}{suffix}_results.json')
    with open(outname, 'w') as f:
        json.dump({'meta': meta, 'results': log}, f, indent=2, default=str)
    print(f"\nSaved → {outname}")


if __name__ == '__main__':
    main()
