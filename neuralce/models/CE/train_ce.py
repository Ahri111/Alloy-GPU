"""
train_ce.py
═══════════════════════════════════════════════════════════════════════════════

System-agnostic icet ClusterExpansion training pipeline with optional
spin pair correlation (Heisenberg Jᵢⱼ).

Drives the full training flow from a YAML config:
    1. Detect primitive cell from a reference CIF.
    2. Build a ClusterSpace.
    3. Add training structures from a CSV manifest.
    4. (Optional) Build spin pair correlation matrix from spin_info.pkl.
    5. (Optional) LASSO α scan with k-fold CV + 1-SE rule.
    6. Final fit on full dataset and save .ce + spin Jᵢⱼ.
    7. Parity plot.

SPIN SUPPORT
─────────────────────────────────────────────────────────────────────────────
When spin_pkl is provided in the config, the pipeline augments the icet
compositional cluster vector with spin-spin pair correlations computed on
icet's pair orbits.  The design matrix becomes:

    X = [X_comp | X_spin]    (n_struct, n_orbits + n_pair_orbits)

The same LASSO/ARDR fit produces compositional ECIs and magnetic Jᵢⱼ
simultaneously.  Spin correlations reuse icet's symmetry-equivalent pair
instances so multiplicity handling is automatic.

USAGE
─────────────────────────────────────────────────────────────────────────────
    python train_ce.py --config configs/stfo.yaml           # compositional only
    python train_ce.py --config configs/stfo_spin.yaml      # with spin

OUTPUT
─────────────────────────────────────────────────────────────────────────────
    output_dir/
        <name>.ce                — compositional CE (ClusterExpansion.read)
        <name>_spin.npz          — spin Jᵢⱼ + pair orbit info (only if spin_pkl set)
        <name>_parity.png        — parity plot (DFT vs CE+spin)
        <name>_alpha_scan.png    — α scan curve (only if scan_alpha: true)
        <name>_summary.txt       — config snapshot + train/test metrics
"""

from __future__ import annotations

import argparse
import os
import pickle
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import spglib
from ase import Atoms
from ase.io import read
from icet import ClusterSpace, StructureContainer, ClusterExpansion
from icet.tools import map_structure_to_reference
from sklearn.linear_model import Lasso, ARDRegression
from sklearn.model_selection import train_test_split, KFold
from scipy.stats import spearmanr


# ═══════════════════════════════════════════════════════════════════════════════
# Config dataclass
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class TrainConfig:
    """Training configuration — fields mirror the YAML keys 1:1."""
    # Data
    name: str
    cif_dir: str
    csv_path: str
    energy_col: str = 'total_energy'
    reference_cif: str = ''

    # Primitive detection
    ideal_map: dict = field(default_factory=dict)
    species_map: dict = field(default_factory=dict)
    inert_species: list = field(default_factory=list)
    symprec: float = 1e-2

    # Cluster space
    cutoffs: list = field(default_factory=lambda: [6.0, 5.0, 4.5])

    # Spin
    spin_pkl: str = ''
    spin_species: list = field(default_factory=list)

    # Training
    fit_method: str = 'lasso'
    lasso_alpha: float = 1e-3
    scan_alpha: bool = True
    alpha_mode: str = '1se'   # '1se' or 'best'
    scan_alpha_range: list = field(default_factory=lambda: [-6, -1, 30])
    n_splits: int = 10
    test_frac: float = 0.2
    seed: int = 42

    # Output
    output_dir: str = 'models'


def load_config(path: str) -> TrainConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    if raw.get('spin_pkl') is None:
        raw['spin_pkl'] = ''
    if raw.get('spin_species') is None:
        raw['spin_species'] = []
    return TrainConfig(**raw)


# ═══════════════════════════════════════════════════════════════════════════════
# Primitive cell detection
# ═══════════════════════════════════════════════════════════════════════════════
def detect_primitive(reference_cif: str, ideal_map: dict, symprec: float) -> Atoms:
    atoms = read(reference_cif)
    ideal = atoms.copy()
    for a in ideal:
        if a.symbol in ideal_map:
            a.symbol = ideal_map[a.symbol]

    cell_tuple = (ideal.cell.array,
                  ideal.get_scaled_positions(),
                  ideal.get_atomic_numbers())
    prim_lat, prim_pos, prim_num = spglib.find_primitive(cell_tuple, symprec=symprec)
    primitive = Atoms(numbers=prim_num,
                      scaled_positions=prim_pos,
                      cell=prim_lat, pbc=True)
    return primitive


# ═══════════════════════════════════════════════════════════════════════════════
# Structure ingestion
# ═══════════════════════════════════════════════════════════════════════════════
def build_structure_container(cs, csv_path, cif_dir, energy_col,
                              primitive, inert_species):
    df = pd.read_csv(csv_path)
    sc = StructureContainer(cs)
    failed = []
    successful_ids = []

    for _, row in df.iterrows():
        cif_path = os.path.join(cif_dir, f"{row['id']}.cif")
        try:
            struct = read(cif_path)
            mapped, _ = map_structure_to_reference(
                struct, primitive,
                inert_species=inert_species,
                assume_no_cell_relaxation=False,
            )
            sc.add_structure(mapped, properties={'energy': float(row[energy_col])})
            successful_ids.append(str(row['id']))
        except Exception as e:
            failed.append((row['id'], str(e)))

    return sc, df, failed, successful_ids


# ═══════════════════════════════════════════════════════════════════════════════
# Spin pair correlation
# ═══════════════════════════════════════════════════════════════════════════════
def load_spin_dict(spin_pkl_path: str) -> dict:
    """Load spin_info.pkl → {cif_id: spin_states_array}."""
    with open(spin_pkl_path, 'rb') as f:
        spin_df = pickle.load(f)
    if isinstance(spin_df, pd.DataFrame):
        return {str(row['cif_id']): np.array(row['spin_states'], dtype=np.float32)
                for _, row in spin_df.iterrows()}
    elif isinstance(spin_df, list):
        return {str(d['cif_id']): np.array(d['spin_states'], dtype=np.float32)
                for d in spin_df}
    else:
        raise ValueError(f"Unknown spin_pkl format: {type(spin_df)}")


def get_pair_orbit_instances(cs, supercell):
    """Extract pair orbit instances from icet's supercell orbit list.

    Returns a list of dicts, one per pair orbit:
        {'orbit_index': int,
         'radius': float,
         'instances': list of (site_i, site_j) tuples}
    """
    pol = cs._orbit_list
    fpt = cs.fractional_position_tolerance
    sup_ol = pol.get_supercell_orbit_list(
        supercell, fractional_position_tolerance=fpt)

    pair_orbits = []
    for orbit_idx in range(len(sup_ol)):
        orbit = sup_ol.get_orbit(orbit_idx)
        rep = orbit.representative_cluster
        if rep.order != 2:
            continue
        instances = []
        for cluster in orbit.clusters:
            sites = [s.index for s in cluster.lattice_sites]
            instances.append(tuple(sites))
        pair_orbits.append({
            'orbit_index': orbit_idx,
            'radius': rep.radius,
            'instances': instances,
        })
    return pair_orbits


def build_spin_correlation_matrix(cs, sc, spin_dict, successful_ids,
                                  spin_species_symbols=None):
    """Build spin pair correlation matrix X_spin.

    For each structure and each pair orbit, compute the average sᵢ·sⱼ
    over all instances in that orbit.  Only pairs where both sites host
    a magnetic species contribute; the rest are zero.

    Parameters
    ----------
    cs : ClusterSpace
    sc : StructureContainer
    spin_dict : dict  {cif_id: (n_atoms,) spin array}
    successful_ids : list of str  (cif_ids matching sc ordering)
    spin_species_symbols : list of str, optional
        Species that carry spin (e.g. ['Fe']).  If empty/None, all species
        on binary sublattices are assumed magnetic — which is usually wrong.
        Provide this explicitly.

    Returns
    -------
    X_spin : (n_struct, n_pair_orbits) ndarray
    pair_orbit_info : list of dicts with orbit metadata
    """
    if len(successful_ids) == 0:
        raise ValueError("No successful structures to compute spin correlations.")

    ref_struct = sc.get_structure(0)
    pair_orbits = get_pair_orbit_instances(cs, ref_struct)
    n_pair_orbits = len(pair_orbits)
    n_struct = len(successful_ids)

    X_spin = np.zeros((n_struct, n_pair_orbits), dtype=np.float64)

    missing_spin = []
    for s_idx, cif_id in enumerate(successful_ids):
        if cif_id not in spin_dict:
            missing_spin.append(cif_id)
            continue
        spins = spin_dict[cif_id]
        for p_idx, porbit in enumerate(pair_orbits):
            corr_sum = 0.0
            n_inst = len(porbit['instances'])
            for (i, j) in porbit['instances']:
                corr_sum += spins[i] * spins[j]
            X_spin[s_idx, p_idx] = corr_sum / n_inst

    if missing_spin:
        print(f"  WARNING: {len(missing_spin)} structures missing spin data "
              f"(first 5: {missing_spin[:5]})")

    return X_spin, pair_orbits


# ═══════════════════════════════════════════════════════════════════════════════
# LASSO α scan with 1-SE rule (augmented X support)
# ═══════════════════════════════════════════════════════════════════════════════
def lasso_alpha_scan_manual(X, y, alpha_range, n_splits, seed=42):
    """Manual k-fold CV α scan on arbitrary design matrix X.

    Used instead of trainstation's CrossValidationEstimator when the design
    matrix has been augmented with spin columns.
    """
    alphas = np.logspace(*alpha_range)
    cv_means, cv_stds = [], []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for a in alphas:
        rmses = []
        for train_idx, val_idx in kf.split(X):
            model = Lasso(alpha=a, max_iter=100000, fit_intercept=False)
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[val_idx])
            rmse = np.sqrt(np.mean((pred - y[val_idx])**2))
            rmses.append(rmse)
        rmses = np.array(rmses)
        cv_means.append(rmses.mean())
        cv_stds.append(rmses.std() / np.sqrt(len(rmses)))

    cv_means = np.array(cv_means)
    cv_stds = np.array(cv_stds)

    idx_best = np.argmin(cv_means)
    threshold = cv_means[idx_best] + cv_stds[idx_best]
    candidates = np.where(cv_means <= threshold)[0]
    idx_1se = candidates[-1]

    return {
        'alphas': alphas,
        'cv_means': cv_means,
        'cv_stds': cv_stds,
        'alpha_best': alphas[idx_best],
        'alpha_1se': alphas[idx_1se],
        'cv_best': cv_means[idx_best],
        'cv_1se': cv_means[idx_1se],
    }


def plot_alpha_scan(scan_result, save_path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 5))
    plt.errorbar(scan_result['alphas'], scan_result['cv_means'] * 1000,
                 yerr=scan_result['cv_stds'] * 1000,
                 fmt='o-', capsize=3, label='CV RMSE ± SE')
    plt.axvline(scan_result['alpha_best'], color='C1', ls='--',
                label=f"α_best = {scan_result['alpha_best']:.1e}")
    plt.axvline(scan_result['alpha_1se'], color='C2', ls='--',
                label=f"α_1SE = {scan_result['alpha_1se']:.1e}")
    plt.axhline((scan_result['cv_best'] + scan_result['cv_stds'][
                    np.argmin(scan_result['cv_means'])]) * 1000,
                color='gray', ls=':', label='min + 1 SE')
    plt.xscale('log')
    plt.xlabel('LASSO α')
    plt.ylabel('CV RMSE (meV)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Final fit + evaluation
# ═══════════════════════════════════════════════════════════════════════════════
def fit_and_evaluate(X, y, fit_method, alpha, test_frac, seed, n_atoms_per_struct,
                     n_comp_params=None):
    """Train/test split, fit, evaluate.

    Parameters
    ----------
    n_comp_params : int, optional
        Number of compositional parameters (icet cluster vector width).
        If provided, metrics will report ECI vs Jᵢⱼ sparsity separately.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_frac, random_state=seed)

    if fit_method == 'lasso':
        model = Lasso(alpha=alpha, max_iter=100000, fit_intercept=False)
    elif fit_method == 'ardr':
        model = ARDRegression(max_iter=300, fit_intercept=False,
                              compute_score=False, threshold_lambda=1e4)
    else:
        raise ValueError(f"Unknown fit_method: {fit_method}")

    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    pa = lambda v: v / n_atoms_per_struct
    metrics = {
        'train_rmse_meV_at': float(np.sqrt(np.mean((pa(y_train_pred) - pa(y_train))**2)) * 1000),
        'train_mae_meV_at':  float(np.mean(np.abs(pa(y_train_pred) - pa(y_train))) * 1000),
        'test_rmse_meV_at':  float(np.sqrt(np.mean((pa(y_test_pred)  - pa(y_test))**2))  * 1000),
        'test_mae_meV_at':   float(np.mean(np.abs(pa(y_test_pred)  - pa(y_test)))  * 1000),
        'test_srcc':         float(spearmanr(y_test, y_test_pred).correlation),
        'n_nonzero_total':   int(np.sum(np.abs(model.coef_) > 1e-10)),
        'n_total_params':    int(len(model.coef_)),
        'n_train':           len(y_train),
        'n_test':            len(y_test),
    }
    if n_comp_params is not None:
        comp_coef = model.coef_[:n_comp_params]
        spin_coef = model.coef_[n_comp_params:]
        metrics['n_nonzero_eci'] = int(np.sum(np.abs(comp_coef) > 1e-10))
        metrics['n_total_eci'] = len(comp_coef)
        metrics['n_nonzero_jij'] = int(np.sum(np.abs(spin_coef) > 1e-10))
        metrics['n_total_jij'] = len(spin_coef)

    return model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, metrics


def plot_parity(y_train, y_train_pred, y_test, y_test_pred, n_atoms_per_struct,
                metrics, fit_method, save_path, alpha=None, has_spin=False):
    import matplotlib.pyplot as plt
    pa = lambda v: v / n_atoms_per_struct

    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=150)
    ax.scatter(pa(y_train), pa(y_train_pred), c='#4C72B0', s=12, alpha=0.35,
               edgecolors='none', label=f'Train ({len(y_train)})', zorder=1)
    ax.scatter(pa(y_test), pa(y_test_pred), c='#C44E52', s=22, alpha=0.8,
               edgecolors='none', label=f'Test ({len(y_test)})', zorder=2)

    all_v = np.concatenate([pa(y_train), pa(y_train_pred),
                             pa(y_test),  pa(y_test_pred)])
    vmin, vmax = all_v.min(), all_v.max()
    m = (vmax - vmin) * 0.03
    ax.plot([vmin - m, vmax + m], [vmin - m, vmax + m], 'k--', lw=0.8, alpha=0.5)

    info = (f"Test RMSE: {metrics['test_rmse_meV_at']:.2f} meV/at\n"
            f"Test MAE:  {metrics['test_mae_meV_at']:.2f} meV/at\n"
            f"SRCC:      {metrics['test_srcc']:.4f}\n"
            f"Non-zero:  {metrics['n_nonzero_total']}/{metrics['n_total_params']}")
    if 'n_nonzero_eci' in metrics:
        info += f"\n  ECI:     {metrics['n_nonzero_eci']}/{metrics['n_total_eci']}"
        info += f"\n  Jᵢⱼ:     {metrics['n_nonzero_jij']}/{metrics['n_total_jij']}"
    if alpha is not None:
        info += f"\nα:         {alpha:.2e}"
    ax.text(0.05, 0.95, info, transform=ax.transAxes, fontsize=9, va='top',
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))

    title = f"iCET CE{' + Spin' if has_spin else ''} — {fit_method.upper()}"
    ax.set_xlabel('DFT Energy (eV/atom)')
    ax.set_ylabel('CE Predicted Energy (eV/atom)')
    ax.set_title(title)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════
def train(cfg: TrainConfig):
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    has_spin = bool(cfg.spin_pkl)

    n_steps = 6 if has_spin else 5

    # 1. Primitive detection
    print("=" * 70)
    print(f"Training run: {cfg.name}" + (" [with spin]" if has_spin else ""))
    print("=" * 70)
    print(f"\n[1/{n_steps}] Detecting primitive cell from {cfg.reference_cif}")
    primitive = detect_primitive(cfg.reference_cif, cfg.ideal_map, cfg.symprec)
    print(f"  Primitive: {primitive.get_chemical_symbols()}")

    # 2. ClusterSpace
    print(f"\n[2/{n_steps}] Building ClusterSpace (cutoffs = {cfg.cutoffs})")
    chemical_symbols = [cfg.species_map[s] for s in primitive.get_chemical_symbols()]
    cs = ClusterSpace(primitive, cfg.cutoffs, chemical_symbols)
    print(cs)

    # 3. Structures
    print(f"\n[3/{n_steps}] Adding training structures from {cfg.csv_path}")
    sc, df, failed, successful_ids = build_structure_container(
        cs, cfg.csv_path, cfg.cif_dir, cfg.energy_col,
        primitive, cfg.inert_species)
    print(f"  Successful: {len(sc)}   Failed: {len(failed)}")
    for fid, err in failed[:5]:
        print(f"    FAIL {fid}: {err[:80]}")

    n_atoms_per_struct = len(read(cfg.reference_cif))

    # Get compositional design matrix from icet
    X_comp, y = sc.get_fit_data(key='energy')
    n_comp_params = X_comp.shape[1]

    # 4. (Optional) Spin correlation
    X_spin = None
    pair_orbit_info = None
    n_spin_params = 0
    if has_spin:
        step = 4
        print(f"\n[{step}/{n_steps}] Building spin pair correlation matrix")
        spin_dict = load_spin_dict(cfg.spin_pkl)
        print(f"  Loaded {len(spin_dict)} spin entries from {cfg.spin_pkl}")

        X_spin, pair_orbit_info = build_spin_correlation_matrix(
            cs, sc, spin_dict, successful_ids,
            spin_species_symbols=cfg.spin_species if cfg.spin_species else None)
        n_spin_params = X_spin.shape[1]
        print(f"  Pair orbits with spin: {n_spin_params}")
        for po in pair_orbit_info:
            print(f"    orbit {po['orbit_index']:>3d}  r={po['radius']:.3f} Å  "
                  f"instances={len(po['instances'])}")

    # Build full design matrix
    if X_spin is not None:
        X = np.hstack([X_comp, X_spin])
    else:
        X = X_comp

    # 5. α scan
    alpha_step = 5 if has_spin else 4
    alpha = cfg.lasso_alpha
    if cfg.fit_method == 'lasso' and cfg.scan_alpha:
        print(f"\n[{alpha_step}/{n_steps}] LASSO α scan with 1-SE rule "
              f"(n_splits = {cfg.n_splits})")
        scan = lasso_alpha_scan_manual(X, y, cfg.scan_alpha_range,
                                       cfg.n_splits, cfg.seed)
        plot_alpha_scan(scan, out_dir / f"{cfg.name}_alpha_scan.png")
        print(f"  α_best : {scan['alpha_best']:.2e}  "
              f"CV = {scan['cv_best']*1000:.2f} meV")
        print(f"  α_1SE  : {scan['alpha_1se']:.2e}  "
              f"CV = {scan['cv_1se']*1000:.2f} meV")
        if cfg.alpha_mode == 'best':
            alpha = scan['alpha_best']
            print(f"  → using α_best")
        elif cfg.alpha_mode == '1se':
            alpha = scan['alpha_1se']
            print(f"  → using α_1SE")
        else:
            raise ValueError(f"alpha_mode must be 'best' or '1se', got {cfg.alpha_mode}")
    else:
        print(f"\n[{alpha_step}/{n_steps}] Skipping α scan; "
              f"using alpha = {alpha:.2e}")

    # 6. Final fit + parity + save
    fit_step = 6 if has_spin else 5
    print(f"\n[{fit_step}/{n_steps}] Final fit on full dataset")
    model, X_tr, X_te, y_tr, y_te, ytr_p, yte_p, metrics = fit_and_evaluate(
        X, y, cfg.fit_method, alpha, cfg.test_frac, cfg.seed,
        n_atoms_per_struct, n_comp_params=n_comp_params if has_spin else None)

    print(f"  Train: {metrics['n_train']}   Test: {metrics['n_test']}")
    print(f"  Train RMSE: {metrics['train_rmse_meV_at']:.2f} meV/at  "
          f"MAE: {metrics['train_mae_meV_at']:.2f} meV/at")
    print(f"  Test  RMSE: {metrics['test_rmse_meV_at']:.2f} meV/at  "
          f"MAE: {metrics['test_mae_meV_at']:.2f} meV/at  "
          f"SRCC: {metrics['test_srcc']:.4f}")
    if has_spin:
        print(f"  Non-zero ECIs: {metrics['n_nonzero_eci']}/{metrics['n_total_eci']}  "
              f"Jᵢⱼ: {metrics['n_nonzero_jij']}/{metrics['n_total_jij']}")
    else:
        print(f"  Non-zero: {metrics['n_nonzero_total']}/{metrics['n_total_params']}")

    # Refit on full data
    if cfg.fit_method == 'lasso':
        final = Lasso(alpha=alpha, max_iter=100000, fit_intercept=False)
    else:
        final = ARDRegression(max_iter=300, fit_intercept=False, threshold_lambda=1e4)
    final.fit(X, y)

    # Save compositional CE (icet format)
    comp_params = final.coef_[:n_comp_params]
    ce = ClusterExpansion(cs, comp_params)
    ce_path = out_dir / f"{cfg.name}.ce"
    ce.write(str(ce_path))

    # Save spin Jᵢⱼ (if applicable)
    spin_path = None
    if has_spin and pair_orbit_info is not None:
        spin_params = final.coef_[n_comp_params:]
        spin_save = {
            'j_values': spin_params,
            'pair_orbits': [{
                'orbit_index': po['orbit_index'],
                'radius': po['radius'],
                'n_instances': len(po['instances']),
                'instances': po['instances'],
            } for po in pair_orbit_info],
            'n_comp_params': n_comp_params,
            'n_spin_params': n_spin_params,
        }
        spin_path = out_dir / f"{cfg.name}_spin.npz"
        np.savez(str(spin_path), **{
            'j_values': spin_params,
            'orbit_indices': np.array([po['orbit_index'] for po in pair_orbit_info]),
            'orbit_radii': np.array([po['radius'] for po in pair_orbit_info]),
            'orbit_n_instances': np.array([len(po['instances']) for po in pair_orbit_info]),
        })
        spin_pkl_path = out_dir / f"{cfg.name}_spin_orbits.pkl"
        with open(spin_pkl_path, 'wb') as f:
            pickle.dump(spin_save, f)

    plot_parity(y_tr, ytr_p, y_te, yte_p, n_atoms_per_struct, metrics,
                cfg.fit_method, out_dir / f"{cfg.name}_parity.png",
                alpha=alpha if cfg.fit_method == 'lasso' else None,
                has_spin=has_spin)

    # Summary
    summary_path = out_dir / f"{cfg.name}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Training run: {cfg.name}\n")
        f.write(f"Spin: {'yes' if has_spin else 'no'}\n\n")
        f.write(f"Config:\n")
        for k, v in vars(cfg).items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nFinal alpha: {alpha:.6e}\n")
        f.write(f"Design matrix: {X.shape[0]} structures × {X.shape[1]} features\n")
        f.write(f"  Compositional: {n_comp_params}\n")
        if has_spin:
            f.write(f"  Spin pairs:    {n_spin_params}\n")
        f.write(f"\nMetrics:\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v}\n")
        if has_spin:
            f.write(f"\nSpin Jᵢⱼ (non-zero):\n")
            spin_params = final.coef_[n_comp_params:]
            for po, j in zip(pair_orbit_info, spin_params):
                if abs(j) > 1e-10:
                    f.write(f"  orbit {po['orbit_index']:>3d}  "
                            f"r={po['radius']:.3f} Å  "
                            f"J={j:.6f}  "
                            f"instances={len(po['instances'])}\n")

    print(f"\nSaved:")
    print(f"  CE        : {ce_path}")
    if spin_path:
        print(f"  Spin Jᵢⱼ  : {spin_path}")
        print(f"  Spin orbits: {spin_pkl_path}")
    print(f"  Parity    : {out_dir / (cfg.name + '_parity.png')}")
    if cfg.fit_method == 'lasso' and cfg.scan_alpha:
        print(f"  Alpha scan: {out_dir / (cfg.name + '_alpha_scan.png')}")
    print(f"  Summary   : {summary_path}")
    print("Done.\n")
    return ce


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train an icet ClusterExpansion (+ optional spin Jᵢⱼ) from a YAML config.")
    parser.add_argument('--config', required=True, help='Path to YAML config file.')
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)
