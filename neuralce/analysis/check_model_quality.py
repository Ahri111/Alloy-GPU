"""
check_model_quality.py — Per-composition parity plots + metrics

Loads the retrained checkpoint, runs inference on ALL data (train+val+test combined),
and produces per-composition parity subplots with SRCC, MAE, R² annotated.

Usage (Colab):
  CONFIG_PATH = './configs/stfo_wo_spin.yaml'
  CKPT_PATH = './best_pkl/retrained/retrained_stfo_wo_spin_ising_lite.pkl'
  %run check_model_quality.py
"""

import os, pickle, re
import numpy as np
import jax
import jax.numpy as jnp
import yaml
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from pymatgen.core.structure import Structure
import pandas as pd

from neuralce.models.NeuralCE_jax import create_neuralce, is_spin_model, is_sisj_model

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════
CONFIG_PATH = os.environ.get('CONFIG_PATH', './configs/stfo_wo_spin.yaml')
CKPT_PATH = os.environ.get('CKPT_PATH', './best_pkl/retrained/retrained_stfo_wo_spin_ising_lite.pkl')

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

# ═══════════════════════════════════════════════════════════════════════
# 1. LOAD DATA (same as retrain.py)
# ═══════════════════════════════════════════════════════════════════════
cif_dir    = cfg['cif_dir']
csv_path   = cfg['csv_path']
spin_pkl   = cfg.get('spin_pkl')
id_col     = cfg.get('id_col', 'id')
comp_regex = cfg.get('comp_regex', r'_(\d+)')
species_map = {int(k): v for k, v in cfg['species_map'].items()}
n_species   = len(species_map)
exclude_z   = set(cfg.get('exclude_species', []))
n_atoms_orig = cfg['n_atoms']

df = pd.read_csv(csv_path)
if id_col not in df.columns:
    id_col = 'id' if 'id' in df.columns else 'cif_id'
energy_map = dict(zip(df[id_col].astype(str), df['total_energy'].values))

if spin_pkl and os.path.exists(spin_pkl):
    spin_df = pd.read_pickle(spin_pkl)
    spin_map = dict(zip(spin_df['cif_id'].astype(str), spin_df['spin_states'].values))
else:
    spin_map = {}

print(f"Loading structures from {cif_dir}...")
structures = []
cif_files = sorted([f for f in os.listdir(cif_dir) if f.endswith('.cif')])

for cif_file in cif_files:
    cif_id = cif_file.replace('.cif', '')
    if cif_id not in energy_map:
        continue
    crystal = Structure.from_file(os.path.join(cif_dir, cif_file))
    spins = spin_map.get(cif_id, [0] * len(crystal))
    spins = np.array(spins, dtype=np.float32)

    if exclude_z:
        keep_idx = [i for i, site in enumerate(crystal)
                    if site.specie.Z not in exclude_z]
        crystal = Structure.from_sites([crystal[i] for i in keep_idx])
        spins = spins[keep_idx]

    comp_match = re.search(comp_regex, cif_id)
    comp_code = int(comp_match.group(1)) if comp_match else 0

    structures.append({
        'cif_id': cif_id, 'total_energy': energy_map[cif_id],
        'crystal': crystal, 'spin_states': spins,
        'comp_code': comp_code, 'n_atoms': len(crystal),
    })

print(f"  Loaded {len(structures)} structures")

# ═══════════════════════════════════════════════════════════════════════
# 2. LOAD MODEL
# ═══════════════════════════════════════════════════════════════════════
with open(CKPT_PATH, 'rb') as f:
    ckpt = pickle.load(f)

hp = ckpt['hp']
params = ckpt['params']
model_name = ckpt.get('model_name', 'ising_lite')

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

# Resolve shell edges
_graph = cfg.get('graph', {})
_candidates = _graph.get('candidates', {})
shell_edges = None
for ck, cv in _candidates.items():
    if abs(float(ck) - hp['cutoff']) < 1e-6:
        shell_edges = cv.get('shell_edges')
        break

# ═══════════════════════════════════════════════════════════════════════
# 3. BUILD GRAPHS & RUN INFERENCE ON ALL STRUCTURES
# ═══════════════════════════════════════════════════════════════════════
print("Building graphs & running inference...")

cutoff = hp['cutoff']
max_num_nbr = hp.get('max_num_nbr', 12)

targets_all = []
preds_all = []
comps_all = []

for s in structures:
    crystal = s['crystal']
    n_at = len(crystal)
    spins_arr = s['spin_states']

    # Atom features
    atom_fea = np.zeros((n_at, n_species), dtype=np.float32)
    for i, site in enumerate(crystal):
        z = site.specie.Z
        if z in species_map:
            atom_fea[i, species_map[z]] = 1.0

    # Neighbor graph
    all_nbrs = crystal.get_all_neighbors(cutoff, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

    nbr_fea_idx = np.zeros((n_at, max_num_nbr), dtype=np.int32)
    nbr_dists = np.full((n_at, max_num_nbr), cutoff + 1.0, dtype=np.float64)
    for i, nbr in enumerate(all_nbrs):
        n_nbr = min(len(nbr), max_num_nbr)
        for j in range(n_nbr):
            nbr_fea_idx[i, j] = nbr[j][2]
            nbr_dists[i, j] = nbr[j][1]

    valid_mask = nbr_dists < cutoff
    if shell_edges is not None:
        edges = np.array(shell_edges)
    else:
        min_dist = nbr_dists[valid_mask].min() if valid_mask.any() else 0.1
        edges = np.linspace(min_dist * 0.99, cutoff, n_shells + 1)

    shell_idx = np.clip(np.digitize(nbr_dists, edges) - 1, 0, n_shells - 1)
    shell_oh = np.zeros((n_at, max_num_nbr, n_shells), dtype=np.float32)
    rows, cols = np.where(valid_mask)
    shell_oh[rows, cols, shell_idx[rows, cols]] = 1.0

    if sisj:
        sisj_fea = np.zeros((n_at, max_num_nbr, 1), dtype=np.float32)
        for i in range(n_at):
            for j in range(max_num_nbr):
                if nbr_dists[i, j] < cutoff:
                    sisj_fea[i, j, 0] = spins_arr[i] * spins_arr[nbr_fea_idx[i, j]]
        nbr_fea = np.concatenate([shell_oh, sisj_fea], axis=-1)
    else:
        nbr_fea = shell_oh

    # Forward pass
    kw = {
        'atom_fea': jnp.array(atom_fea),
        'nbr_fea': jnp.array(nbr_fea),
        'nbr_fea_idx': jnp.array(nbr_fea_idx),
        'batch_size': 1,
        'n_atoms_per_crystal': n_at,
    }
    if use_spin:
        kw['atom_spins'] = jnp.array(spins_arr.reshape(-1, 1))

    pred = float(model.apply(params, **kw).squeeze())

    targets_all.append(s['total_energy'])
    preds_all.append(pred)
    comps_all.append(s['comp_code'])

targets_all = np.array(targets_all)
preds_all = np.array(preds_all)
comps_all = np.array(comps_all)

# Per-atom
targets_pa = targets_all / n_atoms_orig
preds_pa = preds_all / n_atoms_orig

print(f"  Inference done: {len(targets_all)} structures")

# ═══════════════════════════════════════════════════════════════════════
# 4. GLOBAL METRICS
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 60}")
print(f"  GLOBAL METRICS (all data, eV/atom)")
print(f"{'═' * 60}")
global_srcc = spearmanr(targets_pa, preds_pa).correlation
global_mae = np.mean(np.abs(targets_pa - preds_pa))
global_r2 = r2_score(targets_pa, preds_pa)
global_rmse = np.sqrt(np.mean((targets_pa - preds_pa) ** 2))
print(f"  SRCC = {global_srcc:.4f}")
print(f"  R²   = {global_r2:.4f}")
print(f"  MAE  = {global_mae:.6f} eV/atom")
print(f"  RMSE = {global_rmse:.6f} eV/atom")

# ═══════════════════════════════════════════════════════════════════════
# 5. PER-COMPOSITION METRICS & PARITY PLOTS
# ═══════════════════════════════════════════════════════════════════════
unique_comps = sorted(set(comps_all))
n_comps = len(unique_comps)

print(f"\n{'═' * 60}")
print(f"  PER-COMPOSITION METRICS (eV/atom)")
print(f"{'═' * 60}")

comp_metrics = {}
for c in unique_comps:
    mask = comps_all == c
    t = targets_pa[mask]
    p = preds_pa[mask]
    n = mask.sum()
    if n < 3:
        continue
    srcc = spearmanr(t, p).correlation
    mae = np.mean(np.abs(t - p))
    r2 = r2_score(t, p) if n >= 2 else float('nan')
    comp_metrics[c] = {'srcc': srcc, 'mae': mae, 'r2': r2, 'n': n}
    print(f"  x={c/1000:.3f} (n={n:3d}): SRCC={srcc:.4f}  MAE={mae:.6f}  R²={r2:.4f}")

# ── Plot: per-composition parity subplots ─────────────────────────────
comps_to_plot = [c for c in unique_comps if c in comp_metrics]
n_plot = len(comps_to_plot)
ncols = min(4, n_plot)
nrows = (n_plot + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), squeeze=False)

for idx, c in enumerate(comps_to_plot):
    row, col = idx // ncols, idx % ncols
    ax = axes[row, col]

    mask = comps_all == c
    t = targets_pa[mask]
    p = preds_pa[mask]
    m = comp_metrics[c]

    ax.scatter(t, p, s=12, alpha=0.6, color='steelblue', edgecolors='none')

    # Parity line
    lo = min(t.min(), p.min())
    hi = max(t.max(), p.max())
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            'k--', linewidth=0.8, alpha=0.5)

    ax.set_xlim(lo - margin, hi + margin)
    ax.set_ylim(lo - margin, hi + margin)
    ax.set_aspect('equal')
    ax.set_xlabel('DFT (eV/atom)', fontsize=9)
    ax.set_ylabel('Predicted (eV/atom)', fontsize=9)
    ax.set_title(f'x = {c/1000:.3f}  (n={m["n"]})', fontsize=10, fontweight='bold')

    # Annotate metrics
    text = f'SRCC={m["srcc"]:.3f}\nMAE={m["mae"]:.5f}\nR²={m["r2"]:.3f}'
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

    ax.grid(True, alpha=0.2)

# Hide empty subplots
for idx in range(n_plot, nrows * ncols):
    row, col = idx // ncols, idx % ncols
    axes[row, col].set_visible(False)

fig.suptitle(f'{model_name} — Per-composition parity (all data, eV/atom)\n'
             f'Global: SRCC={global_srcc:.4f}, R²={global_r2:.4f}, '
             f'MAE={global_mae:.6f} eV/atom',
             fontsize=12, fontweight='bold')
plt.tight_layout()

save_dir = os.path.dirname(CKPT_PATH) or '.'
plot_path = os.path.join(save_dir, f'parity_per_comp_{model_name}.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"\n  Plot → {plot_path}")
plt.show()

print(f"\n{'═' * 60}")
print(f"  DONE")
print(f"{'═' * 60}")
