"""
check_model_quality.py — Per-composition parity plots + metrics
 
Supports both Heavy (atom_init + Gaussian) and Lite (one-hot + shell) models.
Correctly uses n_atoms_orig for per-atom normalization (supercell size, not graph atoms).
 
Usage (Colab - as module):
    from neuralce.analysis.check_model_quality import run_check
    r = run_check('./configs/stfo_wo_spin.yaml', './best_pkl/retrained/retrained.pkl')
    r = run_check('./configs/stfo_wo_spin.yaml', './best_pkl/retrained/retrained.pkl',
                  n_atoms_orig=160)
 
Usage (Colab - %run):
    CONFIG_PATH = './configs/stfo_wo_spin.yaml'
    CKPT_PATH = './best_pkl/retrained/retrained_stfo_wo_spin_ising_lite.pkl'
    %run check_model_quality.py
"""
 
import os, pickle, re, json
import numpy as np
import jax
import jax.numpy as jnp
import yaml
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from pymatgen.core.structure import Structure
from neuralce.utils.cif_utils import load_cif_safe, get_specie_number
import pandas as pd
 
 
def _detect_model_family(ckpt):
    model_name = ckpt.get('model_name', '')
    hp = ckpt.get('hp', {})
    if 'lite' in model_name:
        return 'lite'
    if ckpt.get('model_config', {}).get('radius') is not None:
        return 'heavy'
    if hp.get('cutoff') is not None and hp.get('n_shells') is not None:
        return 'lite'
    return 'heavy'
 
 
def _load_and_infer_lite(ckpt, cfg):
    from neuralce.models.NeuralCE_jax import create_neuralce, is_spin_model, is_sisj_model
 
    hp = ckpt['hp']
    params = ckpt['params']
    model_name = ckpt.get('model_name', 'ising_lite')
 
    cif_dir = cfg['cif_dir']
    csv_path = cfg['csv_path']
    spin_pkl = cfg.get('spin_pkl')
    id_col = cfg.get('id_col', 'id')
    comp_regex = cfg.get('comp_regex', r'_(\d+)')
    species_map = {int(k): v for k, v in cfg['species_map'].items()}
    n_species = len(species_map)
    exclude_z = set(cfg.get('exclude_species', []))
 
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
 
    _graph = cfg.get('graph', {})
    _candidates = _graph.get('candidates', {})
    shell_edges = None
    for ck, cv in _candidates.items():
        if abs(float(ck) - hp['cutoff']) < 1e-6:
            shell_edges = cv.get('shell_edges')
            break
 
    df = pd.read_csv(csv_path)
    if id_col not in df.columns:
        id_col = 'id' if 'id' in df.columns else 'cif_id'
    energy_map = dict(zip(df[id_col].astype(str), df['total_energy'].values))
 
    spin_map = {}
    if spin_pkl and os.path.exists(spin_pkl):
        spin_df = pd.read_pickle(spin_pkl)
        spin_map = dict(zip(spin_df['cif_id'].astype(str), spin_df['spin_states'].values))
 
    cutoff = hp['cutoff']
    max_num_nbr = hp.get('max_num_nbr', 12)
 
    targets_all, preds_all, comps_all = [], [], []
    cif_files = sorted([f for f in os.listdir(cif_dir) if f.endswith('.cif')])
 
    for cif_file in cif_files:
        cif_id = cif_file.replace('.cif', '')
        if cif_id not in energy_map:
            continue
        crystal = load_cif_safe(os.path.join(cif_dir, cif_file))
        spins = spin_map.get(cif_id, [0] * len(crystal))
        spins = np.array(spins, dtype=np.float32)
 
        if exclude_z:
            keep_idx = [i for i, site in enumerate(crystal)
                        if get_specie_number(site.specie) not in exclude_z]
            crystal = Structure.from_sites([crystal[i] for i in keep_idx])
            spins = spins[keep_idx]
 
        comp_match = re.search(comp_regex, cif_id)
        comp_code = int(comp_match.group(1)) if comp_match else 0
 
        n_at = len(crystal)
        atom_fea = np.zeros((n_at, n_species), dtype=np.float32)
        for i, site in enumerate(crystal):
            z = get_specie_number(site.specie)
            if z in species_map:
                atom_fea[i, species_map[z]] = 1.0
 
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
                        sisj_fea[i, j, 0] = spins[i] * spins[nbr_fea_idx[i, j]]
            nbr_fea = np.concatenate([shell_oh, sisj_fea], axis=-1)
        else:
            nbr_fea = shell_oh
 
        kw = {
            'atom_fea': jnp.array(atom_fea),
            'nbr_fea': jnp.array(nbr_fea),
            'nbr_fea_idx': jnp.array(nbr_fea_idx),
            'batch_size': 1,
            'n_atoms_per_crystal': n_at,
        }
        if use_spin:
            kw['atom_spins'] = jnp.array(spins.reshape(-1, 1))
 
        pred = float(model.apply(params, **kw).squeeze())
 
        targets_all.append(energy_map[cif_id])
        preds_all.append(pred)
        comps_all.append(comp_code)
 
    return np.array(targets_all), np.array(preds_all), np.array(comps_all), model_name
 
 
import flax.linen as nn
 
 
class _CEInteractionLayerLegacy(nn.Module):
    """Legacy heavy conv: post-LayerNorm (matches old ckpt param tree)."""
    atom_fea_len: int
 
    @nn.compact
    def __call__(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        atom_nbr_fea = jnp.take(atom_in_fea, nbr_fea_idx, axis=0)
        atom_center_fea = jnp.tile(atom_in_fea[:, None, :], (1, M, 1))
        phi_center = nn.Dense(self.atom_fea_len)(atom_center_fea)
        phi_nbr = nn.Dense(self.atom_fea_len)(atom_nbr_fea)
        phi_edge = nn.Dense(self.atom_fea_len)(nbr_fea)
        interaction = phi_center * phi_nbr * phi_edge
        gate = nn.sigmoid(nn.Dense(self.atom_fea_len)(interaction))
        magnitude = nn.softplus(nn.Dense(self.atom_fea_len)(interaction))
        nbr_sumed = jnp.sum(gate * magnitude, axis=1)
        return atom_in_fea + nn.LayerNorm()(nbr_sumed)
 
 
class _NeuralCE_Ising_Legacy(nn.Module):
    """Legacy heavy ising: post-LN conv + inline 2-layer readout.
    Param tree: Dense_0(embed), CEInteractionLayer_0..N, Dense_1(out), Dense_2(hidden).
    Flax instantiates outer Dense(1) first -> Dense_1, inner Dense(h) -> Dense_2.
    """
    atom_fea_len: int = 64
    n_conv: int = 3
    h_fea_len: int = 128
 
    @nn.compact
    def __call__(self, atom_fea, nbr_fea, nbr_fea_idx, **kwargs):
        atom_fea = nn.Dense(self.atom_fea_len)(atom_fea)
        for i in range(self.n_conv):
            atom_fea = _CEInteractionLayerLegacy(
                self.atom_fea_len, name=f'CEInteractionLayer_{i}'
            )(atom_fea, nbr_fea, nbr_fea_idx)
        E_site = nn.Dense(1)(nn.softplus(nn.Dense(self.h_fea_len)(atom_fea)))
        B = kwargs['batch_size']
        N = kwargs['n_atoms_per_crystal']
        return jnp.sum(E_site.reshape(B, N), axis=1, keepdims=True)
 
 
def _load_and_infer_heavy(ckpt, cfg):
    from neuralce.data.data_jax import GaussianDistance, load_atom_embeddings, process_crystal
 
    mc = ckpt.get('model_config', {})
    params = ckpt['params']
 
    radius = mc.get('radius', 8.0)
    max_num_nbr = mc.get('max_num_nbr', 12)
    n_atoms_pad = mc.get('n_atoms', 160)
 
    cif_dir = cfg['cif_dir']
    csv_path = cfg['csv_path']
    id_col = cfg.get('id_col', 'id')
    comp_regex = cfg.get('comp_regex', r'_(\d+)')
 
    atom_init_path = os.path.join(cif_dir, 'atom_init.json')
    gdf = GaussianDistance(dmin=0.0, dmax=radius, step=0.2)
    atom_emb = load_atom_embeddings(atom_init_path)
 
    model = _NeuralCE_Ising_Legacy(
        atom_fea_len=mc.get('atom_fea_len', 64),
        n_conv=mc.get('n_conv', 3),
        h_fea_len=mc.get('h_fea_len', 128),
    )
    model_name = mc.get('model_type', 'ising_heavy')
 
    df = pd.read_csv(csv_path)
    if id_col not in df.columns:
        id_col = 'id' if 'id' in df.columns else 'cif_id'
    energy_map = dict(zip(df[id_col].astype(str), df['total_energy'].values))
 
    comp_col = cfg.get('comp_col', 'comp')
    comp_map = None
    if comp_col in df.columns:
        comp_map = dict(zip(df[id_col].astype(str), df[comp_col].values))
 
    variables = {'params': params}
    targets_all, preds_all, comps_all = [], [], []
    cif_files = sorted([f for f in os.listdir(cif_dir) if f.endswith('.cif')])
 
    for cif_file in cif_files:
        cif_id = cif_file.replace('.cif', '')
        if cif_id not in energy_map:
            continue
        graph = process_crystal(os.path.join(cif_dir, cif_file),
                                atom_emb, gdf, max_num_nbr, radius)
        if graph['n_atoms'] != n_atoms_pad:
            continue
 
        pred = float(model.apply(variables,
            atom_fea=jnp.array(graph['atom_fea']),
            nbr_fea=jnp.array(graph['nbr_fea']),
            nbr_fea_idx=jnp.array(graph['nbr_fea_idx']),
            batch_size=1,
            n_atoms_per_crystal=graph['n_atoms'],
        ).squeeze())
 
        if comp_map is not None:
            comps_all.append(int(comp_map[cif_id]))
        else:
            m = re.search(comp_regex, cif_id)
            comps_all.append(int(m.group(1)) if m else 0)
        targets_all.append(float(energy_map[cif_id]))
        preds_all.append(pred)
 
    return np.array(targets_all), np.array(preds_all), np.array(comps_all), model_name
 
 
def run_check(config_path, ckpt_path, n_atoms_orig=None, plot=True,
              comp_scale=1000):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
 
    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)
 
    family = _detect_model_family(ckpt)
    print(f"Detected model family: {family}")
 
    if family == 'lite':
        targets_all, preds_all, comps_all, model_name = _load_and_infer_lite(ckpt, cfg)
    else:
        targets_all, preds_all, comps_all, model_name = _load_and_infer_heavy(ckpt, cfg)
 
    if n_atoms_orig is None:
        n_atoms_orig = cfg.get('n_atoms_orig', cfg.get('n_atoms', 160))
        if cfg.get('exclude_species') and n_atoms_orig == cfg.get('n_atoms'):
            print(f"  WARNING: n_atoms={n_atoms_orig} may be post-exclusion count. "
                  f"Pass n_atoms_orig explicitly if this is wrong.")
 
    targets_pa = targets_all / n_atoms_orig
    preds_pa = preds_all / n_atoms_orig
 
    print(f"\nInference done: {len(targets_all)} structures, n_atoms_orig={n_atoms_orig}")
 
    global_srcc = spearmanr(targets_pa, preds_pa).correlation
    global_mae = np.mean(np.abs(targets_pa - preds_pa))
    global_r2 = r2_score(targets_pa, preds_pa)
    global_rmse = np.sqrt(np.mean((targets_pa - preds_pa) ** 2))
 
    print(f"\n{'=' * 60}")
    print(f"  GLOBAL METRICS (all data, eV/atom, /{ n_atoms_orig})")
    print(f"{'=' * 60}")
    print(f"  SRCC = {global_srcc:.4f}")
    print(f"  R²   = {global_r2:.4f}")
    print(f"  MAE  = {global_mae:.6f} eV/atom")
    print(f"  RMSE = {global_rmse:.6f} eV/atom")
 
    unique_comps = sorted(set(comps_all))
    comp_metrics = {}
 
    print(f"\n{'=' * 60}")
    print(f"  PER-COMPOSITION METRICS (eV/atom)")
    print(f"{'=' * 60}")
 
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
        print(f"  x={c/comp_scale:.3f} (n={n:3d}): SRCC={srcc:.4f}  MAE={mae:.6f}  R²={r2:.4f}")
 
    if plot:
        _plot_parity(targets_pa, preds_pa, comps_all, comp_metrics, unique_comps,
                     model_name, global_srcc, global_r2, global_mae,
                     n_atoms_orig, comp_scale, ckpt_path)
 
    return {
        'targets_all': targets_all,
        'preds_all': preds_all,
        'targets_pa': targets_pa,
        'preds_pa': preds_pa,
        'comps_all': comps_all,
        'comp_metrics': comp_metrics,
        'global_metrics': {
            'srcc': global_srcc, 'r2': global_r2,
            'mae': global_mae, 'rmse': global_rmse,
        },
        'model_name': model_name,
        'n_atoms_orig': n_atoms_orig,
    }
 
 
def _plot_parity(targets_pa, preds_pa, comps_all, comp_metrics, unique_comps,
                 model_name, global_srcc, global_r2, global_mae,
                 n_atoms_orig, comp_scale, ckpt_path):
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
        ax.set_title(f'x = {c/comp_scale:.3f}  (n={m["n"]})', fontsize=10, fontweight='bold')
 
        text = f'SRCC={m["srcc"]:.3f}\nMAE={m["mae"]:.5f}\nR²={m["r2"]:.3f}'
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))
        ax.grid(True, alpha=0.2)
 
    for idx in range(n_plot, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].set_visible(False)
 
    fig.suptitle(f'{model_name} — Per-composition parity (all data, eV/atom, /{n_atoms_orig})\n'
                 f'Global: SRCC={global_srcc:.4f}, R²={global_r2:.4f}, '
                 f'MAE={global_mae:.6f} eV/atom',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
 
    save_dir = os.path.dirname(ckpt_path) or '.'
    plot_path = os.path.join(save_dir, f'parity_per_comp_{model_name}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n  Plot -> {plot_path}")
    plt.show()
 
 
if __name__ == '__main__':
    CONFIG_PATH = os.environ.get('CONFIG_PATH', './configs/stfo_wo_spin.yaml')
    CKPT_PATH = os.environ.get('CKPT_PATH', './best_pkl/retrained/retrained_stfo_wo_spin_ising_lite.pkl')
    N_ATOMS_ORIG = int(os.environ.get('N_ATOMS_ORIG', '0')) or None
    run_check(CONFIG_PATH, CKPT_PATH, n_atoms_orig=N_ATOMS_ORIG)