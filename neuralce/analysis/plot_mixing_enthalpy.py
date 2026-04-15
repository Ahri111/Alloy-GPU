"""
plot_mixing_enthalpy.py — Mixing Enthalpy Plot from retrain checkpoint

Reads a retrained checkpoint (.pkl) and plots ΔH_mix vs composition.
No model inference — uses saved predictions and targets directly.

Mixing enthalpy:
    ΔH_mix(x) = E(x)/N - [(1-x)·E_ref(0)/N₀ + x·E_ref(1)/N₁]

where E_ref are the lowest-energy structures at each endpoint composition.
For intermediate compositions, x is extracted from comp_code.

Usage:
  python plot_mixing_enthalpy.py --checkpoint retrained_best_stfo_wo_spin_ising_lite.pkl
  python plot_mixing_enthalpy.py --checkpoint retrained_best_feni_cr_neuralce_evenodd_lite.pkl \\
      --comp_scale 100 --xlabel "Fe fraction" --title "Fe-Ni-Cr Mixing Enthalpy"

  # Custom endpoint compositions (default: auto-detect min/max)
  python plot_mixing_enthalpy.py --checkpoint retrained.pkl --endpoints 0,1000

  # Save without displaying
  python plot_mixing_enthalpy.py --checkpoint retrained.pkl --save mixing.png --no-show
"""

import argparse, pickle, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Plot mixing enthalpy from retrain checkpoint.")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Retrained .pkl checkpoint with preds/targets/comps")
    p.add_argument("--comp_scale", type=float, default=None,
                   help="Divisor for comp_code → fraction x. "
                        "Default: auto-detect from max comp_code (e.g. 1000 for STFO)")
    p.add_argument("--endpoints", type=str, default=None,
                   help="Comma-separated endpoint comp_codes for reference line. "
                        "Default: auto min,max")
    p.add_argument("--xlabel", type=str, default=None,
                   help="X-axis label. Default: 'Composition x'")
    p.add_argument("--title", type=str, default=None,
                   help="Plot title. Default: auto from checkpoint name")
    p.add_argument("--save", type=str, default=None,
                   help="Save path (png/pdf). Default: <checkpoint>_mixing_enthalpy.png")
    p.add_argument("--no-show", action="store_true",
                   help="Don't display plot (for headless/batch)")
    p.add_argument("--figsize", type=str, default="8,5",
                   help="Figure size as 'w,h'. Default: 8,5")
    p.add_argument("--e_ref_low", type=float, default=-1209.4888,
                   help="Low-endpoint reference total energy (eV/supercell). "
                        "Default: STO -1209.4888")
    p.add_argument("--e_ref_high", type=float, default=-937.79694,
                   help="High-endpoint reference total energy (eV/supercell). "
                        "Default: SFO G-AFM -937.79694")
    p.add_argument("--ref_n_atoms", type=int, default=160,
                   help="Atom count used to per-atom normalize targets + "
                        "reference. Default: 160 (STFO supercell)")
    p.add_argument("--no_override", action="store_true",
                   help="Ignore --e_ref_low/--e_ref_high and use in-data mean.")
    return p.parse_args()


def load_checkpoint(path, ref_n_atoms=160):
    """Load retrain checkpoint and extract all splits.

    ref_n_atoms: divisor for per-atom normalization. Default 160 (STFO supercell).
                 Overrides ckpt['n_atoms_orig']. targets and reference must share
                 the same divisor so ΔH_mix is on the correct scale.
    """
    with open(path, 'rb') as f:
        ckpt = pickle.load(f)

    required = ['targets_train', 'preds_train', 'comps_train',
                'targets_val', 'preds_val', 'comps_val',
                'targets_test', 'preds_test', 'comps_test']
    for key in required:
        if key not in ckpt:
            raise KeyError(f"Checkpoint missing '{key}'. "
                           f"Re-run retrain.py to generate full checkpoint.")

    # Merge all splits
    targets = np.concatenate([ckpt['targets_train'], ckpt['targets_val'], ckpt['targets_test']])
    preds   = np.concatenate([ckpt['preds_train'], ckpt['preds_val'], ckpt['preds_test']])
    comps   = list(ckpt['comps_train']) + list(ckpt['comps_val']) + list(ckpt['comps_test'])
    comps   = np.array(comps)
    splits  = (['train'] * len(ckpt['targets_train']) +
               ['val']   * len(ckpt['targets_val']) +
               ['test']  * len(ckpt['targets_test']))

    print(f"  Per-atom divisor (ref_n_atoms): {ref_n_atoms}")
    divisors = np.full(len(targets), ref_n_atoms, dtype=np.float64)

    targets_pa = targets / divisors
    preds_pa   = preds / divisors

    return targets_pa, preds_pa, comps, np.array(splits), ckpt


def compute_mixing_enthalpy(targets_pa, preds_pa, comps, comp_scale,
                            endpoint_comps=None,
                            e_ref_low_override=None, e_ref_high_override=None,
                            ref_n_atoms=160):
    """Compute mixing enthalpy.

    ΔH_mix(x) = E_pa(x) - [(1-x)*E_ref_low + x*E_ref_high]

    e_ref_{low,high}_override: external reference total energies (eV/supercell).
        If given, divided by ref_n_atoms to get per-atom and used instead of
        the in-data mean. Ideal for fixed STO/SFO end-members.
    ref_n_atoms: divisor for the override energies (should match the divisor
        used on targets_pa — default 160 for STFO).

    If overrides are None → fall back to mean of data at endpoint comps.
    x-axis: simple fraction `comps / comp_scale`.
    """
    unique_comps = sorted(set(comps))

    if endpoint_comps is None:
        endpoint_comps = (min(unique_comps), max(unique_comps))
    c_low, c_high = endpoint_comps

    mask_low  = comps == c_low
    mask_high = comps == c_high

    # Low endpoint reference
    if e_ref_low_override is not None:
        e_ref_low = e_ref_low_override / ref_n_atoms
        print(f"  E_ref_low  (override): {e_ref_low_override:.4f} eV / {ref_n_atoms} = {e_ref_low:.6f} eV/atom")
    else:
        if not mask_low.any():
            raise ValueError(f"No structures at endpoint comp_code={c_low}")
        e_ref_low = np.mean(targets_pa[mask_low])
        print(f"  E_ref_low  (data mean @ x={c_low/comp_scale:.4f}): "
              f"{e_ref_low:.6f} eV/atom ({mask_low.sum()} structures)")

    # High endpoint reference
    if e_ref_high_override is not None:
        e_ref_high = e_ref_high_override / ref_n_atoms
        print(f"  E_ref_high (override): {e_ref_high_override:.4f} eV / {ref_n_atoms} = {e_ref_high:.6f} eV/atom")
    else:
        if not mask_high.any():
            raise ValueError(f"No structures at endpoint comp_code={c_high}")
        e_ref_high = np.mean(targets_pa[mask_high])
        print(f"  E_ref_high (data mean @ x={c_high/comp_scale:.4f}): "
              f"{e_ref_high:.6f} eV/atom ({mask_high.sum()} structures)")

    # Fraction: simple comp/comp_scale
    x = comps / comp_scale

    # Linear interpolation reference (at x values of each structure)
    e_ref_interp = (1 - x) * e_ref_low + x * e_ref_high

    # Mixing enthalpy (meV/atom for readability)
    dh_dft  = (targets_pa - e_ref_interp) * 1000
    dh_pred = (preds_pa   - e_ref_interp) * 1000

    return x, dh_dft, dh_pred


def plot_mixing_enthalpy(x, dh_dft, dh_pred, splits, comps, comp_scale,
                         xlabel=None, title=None, figsize=(8, 5)):
    """Plot ΔH_mix vs composition with per-composition statistics."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [2, 1]})

    # ── Left panel: scatter plot ──────────────────────────────────────
    # Color by split
    colors = {'train': '#aaaaaa', 'val': '#5DA5DA', 'test': '#F15854'}
    alphas = {'train': 0.3, 'val': 0.6, 'test': 0.8}
    zorders = {'train': 1, 'val': 2, 'test': 3}
    labels = {'train': 'Train', 'val': 'Val', 'test': 'Test'}

    for split in ['train', 'val', 'test']:
        mask = splits == split
        if not mask.any():
            continue
        ax1.scatter(x[mask], dh_dft[mask], c=colors[split], alpha=alphas[split],
                   s=15, label=f'{labels[split]} (DFT)', zorder=zorders[split],
                   edgecolors='none')
        ax1.scatter(x[mask], dh_pred[mask], c=colors[split], alpha=alphas[split],
                   s=15, marker='x', label=f'{labels[split]} (Pred)',
                   zorder=zorders[split], linewidths=0.8)

    # Per-composition mean curves
    unique_x = sorted(set(x))
    mean_dft  = [np.mean(dh_dft[x == xi])  for xi in unique_x]
    mean_pred = [np.mean(dh_pred[x == xi]) for xi in unique_x]

    ax1.plot(unique_x, mean_dft, 'k-', linewidth=1.5, label='DFT mean', zorder=5)
    ax1.plot(unique_x, mean_pred, 'r--', linewidth=1.5, label='Pred mean', zorder=5)

    # Reference line at 0
    ax1.axhline(0, color='gray', linewidth=0.5, linestyle=':')

    ax1.set_xlabel(xlabel or 'Composition x', fontsize=11)
    ax1.set_ylabel('ΔH$_{mix}$ (meV/atom)', fontsize=11)
    ax1.legend(fontsize=7, ncol=2, loc='best', framealpha=0.8)
    if title:
        ax1.set_title(title, fontsize=12)

    # ── Right panel: parity plot (DFT vs Pred mixing enthalpy) ────────
    for split in ['train', 'val', 'test']:
        mask = splits == split
        if not mask.any():
            continue
        ax2.scatter(dh_dft[mask], dh_pred[mask], c=colors[split], alpha=alphas[split],
                   s=12, label=labels[split], zorder=zorders[split], edgecolors='none')

    # Parity line
    all_vals = np.concatenate([dh_dft, dh_pred])
    vmin, vmax = np.min(all_vals), np.max(all_vals)
    margin = (vmax - vmin) * 0.05
    ax2.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin],
             'k-', linewidth=0.8, alpha=0.5)

    # Metrics on test set
    test_mask = splits == 'test'
    if test_mask.any():
        residuals = dh_pred[test_mask] - dh_dft[test_mask]
        rmse = np.sqrt(np.mean(residuals ** 2))
        mae  = np.mean(np.abs(residuals))
        ax2.text(0.05, 0.95, f'Test RMSE: {rmse:.1f} meV/at\nTest MAE: {mae:.1f} meV/at',
                transform=ax2.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax2.set_xlabel('DFT ΔH$_{mix}$ (meV/atom)', fontsize=10)
    ax2.set_ylabel('Pred ΔH$_{mix}$ (meV/atom)', fontsize=10)
    ax2.set_title('Parity', fontsize=11)
    ax2.legend(fontsize=8, loc='lower right')
    ax2.set_aspect('equal', adjustable='datalim')

    plt.tight_layout()
    return fig


def main():
    args = parse_args()

    if args.no_show:
        matplotlib.use('Agg')

    print(f"Loading checkpoint: {args.checkpoint}")
    targets_pa, preds_pa, comps, splits, ckpt = load_checkpoint(
        args.checkpoint, ref_n_atoms=args.ref_n_atoms)
    print(f"  Total structures: {len(targets_pa)}")
    print(f"  Unique compositions: {sorted(set(comps))}")

    # Determine comp_scale
    unique_comps = sorted(set(comps))
    if args.comp_scale is not None:
        comp_scale = args.comp_scale
    else:
        max_comp = max(unique_comps)
        if max_comp > 100:
            comp_scale = max_comp  # e.g. 1000 for STFO
        elif max_comp > 1:
            comp_scale = 100       # e.g. percentage
        else:
            comp_scale = 1.0       # already fraction
    print(f"  comp_scale: {comp_scale}")

    # Endpoints
    if args.endpoints:
        ep = [int(e) for e in args.endpoints.split(',')]
        endpoint_comps = (ep[0], ep[1])
    else:
        endpoint_comps = None  # auto min/max

    # Compute mixing enthalpy
    e_low  = None if args.no_override else args.e_ref_low
    e_high = None if args.no_override else args.e_ref_high
    x, dh_dft, dh_pred = compute_mixing_enthalpy(
        targets_pa, preds_pa, comps, comp_scale,
        endpoint_comps=endpoint_comps,
        e_ref_low_override=e_low, e_ref_high_override=e_high,
        ref_n_atoms=args.ref_n_atoms)

    # Print per-composition summary
    print(f"\n  {'Comp':>8} {'x':>6} {'N':>5} {'DFT mean':>10} {'Pred mean':>10} {'Diff':>8}")
    print(f"  {'-'*8} {'-'*6} {'-'*5} {'-'*10} {'-'*10} {'-'*8}")
    for c in sorted(set(comps)):
        mask = comps == c
        xi = x[mask][0]
        n = mask.sum()
        dm = np.mean(dh_dft[mask])
        pm = np.mean(dh_pred[mask])
        print(f"  {c:>8} {xi:>6.3f} {n:>5} {dm:>10.2f} {pm:>10.2f} {pm-dm:>8.2f}")

    # Plot
    figw, figh = [float(v) for v in args.figsize.split(',')]

    # Auto title
    model_name = ckpt.get('model_name', '')
    dataset = args.checkpoint.replace('.pkl', '').replace('retrained_', '')
    default_title = f'Mixing Enthalpy — {model_name or dataset}'
    title = args.title or default_title

    fig = plot_mixing_enthalpy(x, dh_dft, dh_pred, splits, comps, comp_scale,
                               xlabel=args.xlabel, title=title,
                               figsize=(figw, figh))

    # Save
    save_path = args.save or args.checkpoint.replace('.pkl', '_mixing_enthalpy.png')
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\n  Saved → {save_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
