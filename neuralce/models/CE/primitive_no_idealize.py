"""
primitive_no_idealize.py
═══════════════════════════════════════════════════════════════════════════════

Drop-in replacement for `train_ce.detect_primitive` that avoids spglib's
silent idealisation of fractional coordinates.

Phase-1 diagnostic showed:
    spglib.find_primitive(cell_tuple, symprec=1e-2)
returns a primitive whose 6 oxygen sites are off the raw cif's actual
oxygen coordinates by a sub-1e-4 amount that, after the FCC×2
supercell tiling icet's `map_structure_to_reference` performs, becomes
a uniform 0.5068 Å shift on 64 of 96 oxygens (drmax=0.5068, dravg=0.20).
Strain tensor is exactly zero — the lattice itself is unchanged; only
the atomic positions get symmetrised.

Replacing find_primitive with
    spglib.standardize_cell(cell_tuple,
                            to_primitive=True,
                            no_idealize=True,
                            symprec=symprec)
keeps the raw cif's actual fractional coordinates and yields
160/160 PBC matching with drmax ≈ 4e-15 (numerical zero).

This module exposes:
    detect_primitive_no_idealize(reference_cif, ideal_map, symprec)
        — drop-in for detect_primitive
    train_no_idealize(cfg)
        — drop-in for train(), substitutes the primitive but otherwise
          uses train_ce.py's own helpers (build_structure_container,
          lasso_alpha_scan_manual, fit_and_evaluate, plot_*) so the fit,
          α-scan, parity plot and summary file are bit-identical to the
          existing pipeline aside from the primitive geometry.

Constraints honoured:
    • train_ce.py is imported, never modified.
    • map_structure_to_reference is reused unchanged.
    • No monkey-patching — the substitution is an explicit function call.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import spglib
from ase import Atoms
from ase.io import read
from icet import ClusterSpace, ClusterExpansion
from sklearn.linear_model import Lasso, ARDRegression

from neuralce.models.CE.train_ce import (
    TrainConfig,
    build_structure_container,
    fit_and_evaluate,
    lasso_alpha_scan_manual,
    plot_alpha_scan,
    plot_parity,
)


def detect_primitive_no_idealize(reference_cif: str,
                                 ideal_map: dict,
                                 symprec: float) -> Atoms:
    """Return the primitive cell of `reference_cif` after applying
    `ideal_map`, using spglib.standardize_cell with no_idealize=True.

    Atomic positions are NOT symmetrised — they keep the raw cif's
    fractional coordinates exactly.
    """
    atoms = read(reference_cif)
    ideal = atoms.copy()
    for a in ideal:
        if a.symbol in ideal_map:
            a.symbol = ideal_map[a.symbol]

    cell_tuple = (ideal.cell.array,
                  ideal.get_scaled_positions(),
                  ideal.get_atomic_numbers())
    res = spglib.standardize_cell(cell_tuple,
                                  to_primitive=True,
                                  no_idealize=True,
                                  symprec=symprec)
    if res is None:
        raise RuntimeError(
            f"spglib.standardize_cell returned None for {reference_cif} "
            f"(symprec={symprec}, ideal_map={ideal_map}).")
    prim_lat, prim_pos, prim_num = res
    return Atoms(numbers=prim_num,
                 scaled_positions=prim_pos,
                 cell=prim_lat,
                 pbc=True)


def train_no_idealize(cfg: TrainConfig) -> ClusterExpansion:
    """Replicates train_ce.train() but uses
    detect_primitive_no_idealize for the primitive cell.

    Output .ce path is `<output_dir>/<name>_noideal.ce`.
    Parity / alpha-scan / summary use the same `<name>_noideal` prefix.
    Spin support is preserved (the same path as train_ce.train()).
    """
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    has_spin = bool(cfg.spin_pkl)
    out_name = f"{cfg.name}_noideal"

    print("=" * 70)
    print(f"train_no_idealize: {out_name}"
          + (" [with spin]" if has_spin else ""))
    print("=" * 70)

    # 1. Primitive (no-idealize variant)
    print(f"\n[1] Primitive from {cfg.reference_cif} via "
          f"standardize_cell(no_idealize=True)")
    primitive = detect_primitive_no_idealize(
        cfg.reference_cif, cfg.ideal_map, cfg.symprec)
    print(f"    primitive species: {primitive.get_chemical_symbols()}")

    # 2. ClusterSpace
    print(f"\n[2] ClusterSpace (cutoffs = {cfg.cutoffs})")
    chemical_symbols = [cfg.species_map[s]
                        for s in primitive.get_chemical_symbols()]
    cs = ClusterSpace(primitive, cfg.cutoffs, chemical_symbols)
    print(cs)

    # 3. StructureContainer
    print(f"\n[3] Adding training structures from {cfg.csv_path}")
    sc, df, failed, successful_ids = build_structure_container(
        cs, cfg.csv_path, cfg.cif_dir, cfg.energy_col,
        primitive, cfg.inert_species)
    print(f"    Successful: {len(sc)}   Failed: {len(failed)}")
    for fid, err in failed[:5]:
        print(f"      FAIL {fid}: {err[:80]}")

    n_atoms_per_struct = len(read(cfg.reference_cif))
    X_comp, y = sc.get_fit_data(key='energy')
    n_comp_params = X_comp.shape[1]

    # 4. Spin (optional)
    X_spin = None
    pair_orbit_info = None
    n_spin_params = 0
    if has_spin:
        from neuralce.models.CE.train_ce import (
            load_spin_dict, build_spin_correlation_matrix)
        print(f"\n[4] Spin pair correlation")
        spin_dict = load_spin_dict(cfg.spin_pkl)
        X_spin, pair_orbit_info = build_spin_correlation_matrix(
            cs, sc, spin_dict, successful_ids,
            spin_species_symbols=cfg.spin_species or None)
        n_spin_params = X_spin.shape[1]
        print(f"    Pair orbits with spin: {n_spin_params}")

    X = np.hstack([X_comp, X_spin]) if X_spin is not None else X_comp

    # 5. α scan (LASSO only)
    alpha = cfg.lasso_alpha
    if cfg.fit_method == 'lasso' and cfg.scan_alpha:
        print(f"\n[5] LASSO α scan ({cfg.n_splits}-fold)")
        scan = lasso_alpha_scan_manual(X, y, cfg.scan_alpha_range,
                                       cfg.n_splits, cfg.seed)
        plot_alpha_scan(scan, out_dir / f"{out_name}_alpha_scan.png")
        print(f"    α_best = {scan['alpha_best']:.2e}  "
              f"CV = {scan['cv_best']*1000:.2f} meV")
        print(f"    α_1SE  = {scan['alpha_1se']:.2e}  "
              f"CV = {scan['cv_1se']*1000:.2f} meV")
        if cfg.alpha_mode == 'best':
            alpha = scan['alpha_best']
        elif cfg.alpha_mode == '1se':
            alpha = scan['alpha_1se']
        else:
            raise ValueError(f"alpha_mode must be 'best' or '1se', "
                             f"got {cfg.alpha_mode}")
        print(f"    using {cfg.alpha_mode} → α = {alpha:.2e}")
    else:
        print(f"\n[5] α scan skipped — α = {alpha:.2e}")

    # 6. Fit + evaluate
    print(f"\n[6] Fit ({cfg.fit_method}) + evaluate")
    model, X_tr, X_te, y_tr, y_te, ytr_p, yte_p, metrics = fit_and_evaluate(
        X, y, cfg.fit_method, alpha, cfg.test_frac, cfg.seed,
        n_atoms_per_struct,
        n_comp_params=n_comp_params if has_spin else None)

    print(f"    train RMSE {metrics['train_rmse_meV_at']:.2f} meV/at  "
          f"MAE {metrics['train_mae_meV_at']:.2f}")
    print(f"    test  RMSE {metrics['test_rmse_meV_at']:.2f} meV/at  "
          f"MAE {metrics['test_mae_meV_at']:.2f}  "
          f"SRCC {metrics['test_srcc']:.4f}")

    # Refit on full data
    if cfg.fit_method == 'lasso':
        final = Lasso(alpha=alpha, max_iter=100000, fit_intercept=False)
    elif cfg.fit_method == 'ardr':
        final = ARDRegression(max_iter=300, fit_intercept=False,
                              threshold_lambda=1e4)
    else:
        raise ValueError(f"Unknown fit_method: {cfg.fit_method}")
    final.fit(X, y)

    comp_params = final.coef_[:n_comp_params]
    ce = ClusterExpansion(cs, comp_params)
    ce_path = out_dir / f"{out_name}.ce"
    ce.write(str(ce_path))

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
        spin_path = out_dir / f"{out_name}_spin.npz"
        np.savez(str(spin_path), **{
            'j_values': spin_params,
            'orbit_indices': np.array(
                [po['orbit_index'] for po in pair_orbit_info]),
            'orbit_radii': np.array(
                [po['radius'] for po in pair_orbit_info]),
            'orbit_n_instances': np.array(
                [len(po['instances']) for po in pair_orbit_info]),
        })
        with open(out_dir / f"{out_name}_spin_orbits.pkl", 'wb') as f:
            pickle.dump(spin_save, f)

    plot_parity(y_tr, ytr_p, y_te, yte_p, n_atoms_per_struct, metrics,
                cfg.fit_method, out_dir / f"{out_name}_parity.png",
                alpha=alpha if cfg.fit_method == 'lasso' else None,
                has_spin=has_spin)

    summary_path = out_dir / f"{out_name}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"train_no_idealize: {out_name}\n")
        f.write(f"Spin: {'yes' if has_spin else 'no'}\n\n")
        f.write("Config:\n")
        for k, v in vars(cfg).items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nFinal alpha: {alpha:.6e}\n")
        f.write(f"Design matrix: {X.shape[0]} structures × "
                f"{X.shape[1]} features\n")
        f.write(f"  Compositional: {n_comp_params}\n")
        if has_spin:
            f.write(f"  Spin pairs:    {n_spin_params}\n")
        f.write("\nMetrics:\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v}\n")

    print(f"\nSaved:")
    print(f"  CE      : {ce_path}")
    if spin_path:
        print(f"  Spin    : {spin_path}")
    print(f"  Parity  : {out_dir / (out_name + '_parity.png')}")
    if cfg.fit_method == 'lasso' and cfg.scan_alpha:
        print(f"  α scan  : {out_dir / (out_name + '_alpha_scan.png')}")
    print(f"  Summary : {summary_path}")
    return ce
