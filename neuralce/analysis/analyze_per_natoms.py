"""
analyze_per_natoms.py — n_atoms 그룹별 metric 분석

Retrained ckpt를 로드해서 n_atoms 별로 (32, 108 등) SRCC / R² / MAE / RMSE
를 계산합니다. variable n_atoms 데이터셋(예: feni_cr)에서 작은 셀과 큰 셀의
일반화 성능을 분리해서 보기 위한 도구입니다.

Usage:
    python -m neuralce.analysis.analyze_per_natoms \\
        --checkpoint ./best_pkl/retrained_unified/retrained_feni_cr_evenodd.pkl \\
        [--save ./best_pkl/feni_cr_variable/per_natoms_metrics.json]

ckpt에 다음 키가 있어야 합니다:
    targets_all, preds_all, comps_all, splits_all, n_atoms_list (또는 n_atoms_orig)
"""

import os, argparse, pickle, json
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, help='Retrained .pkl 경로')
    p.add_argument('--save', default=None, help='결과 JSON 저장 경로 (기본: ckpt 옆)')
    p.add_argument('--per_atom', action='store_true',
                   help='per-atom 단위로도 metric 계산 (n_atoms로 나눔)')
    p.add_argument('--no_plot', action='store_true', help='figure 생성 생략')
    p.add_argument('--fig_dir', default=None,
                   help='figure 저장 디렉토리 (기본: ckpt 옆 figures/)')
    return p.parse_args()


def _plot_parity_per_natoms(targets, preds, n_arr, splits, split_names,
                            unique_n, ckpt_name, fig_dir, per_atom=False):
    """n_atoms 그룹별 parity plot (1행 N열)."""
    n_groups = len(unique_n)
    fig, axes = plt.subplots(1, n_groups, figsize=(4.5 * n_groups, 4.2),
                              squeeze=False)
    color = {0: '#aaaaaa', 1: '#5DA5DA', 2: '#F15854'}
    label = {0: 'train', 1: 'val', 2: 'test'}

    for j, n in enumerate(unique_n):
        ax = axes[0, j]
        mask_n = n_arr == n
        if per_atom:
            t_g = targets[mask_n] / n
            p_g = preds[mask_n] / n
            unit = 'eV/atom'
        else:
            t_g = targets[mask_n]
            p_g = preds[mask_n]
            unit = 'eV'

        for sidx in sorted(set(splits.tolist())):
            m = mask_n & (splits == sidx)
            if not m.any():
                continue
            ts = targets[m] / (n if per_atom else 1)
            ps = preds[m] / (n if per_atom else 1)
            ax.scatter(ts, ps, s=12, alpha=0.6, color=color.get(sidx, 'k'),
                       label=split_names[sidx] if sidx < len(split_names) else f's{sidx}',
                       edgecolors='none')

        if len(t_g) >= 3:
            lo, hi = float(min(t_g.min(), p_g.min())), float(max(t_g.max(), p_g.max()))
            margin = (hi - lo) * 0.05
            ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                    'k--', lw=0.8, alpha=0.5)
            rho = spearmanr(t_g, p_g).correlation
            r2 = r2_score(t_g, p_g)
            mae = np.mean(np.abs(t_g - p_g))
            ax.text(0.05, 0.95,
                    f'SRCC={rho:.4f}\nR²={r2:.4f}\nMAE={mae:.4f}',
                    transform=ax.transAxes, fontsize=9, va='top',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
        ax.set_xlabel(f'DFT ({unit})', fontsize=10)
        ax.set_ylabel(f'Pred ({unit})', fontsize=10)
        ax.set_title(f'n_atoms = {n} (count={int(mask_n.sum())})',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(alpha=0.25)
        ax.set_aspect('equal', adjustable='datalim')

    fig.suptitle(f'{ckpt_name} — parity per n_atoms ({unit})',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    out = os.path.join(fig_dir,
        f'parity_per_natoms{"_pa" if per_atom else ""}_{ckpt_name}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot → {out}")
    return out


def _plot_metric_bar(per_natoms_metrics, ckpt_name, fig_dir):
    """n_atoms × metric bar chart (SRCC / R² / MAE / RMSE 4-panel)."""
    if not per_natoms_metrics:
        return None
    ns = sorted(per_natoms_metrics.keys())
    metrics_list = ['srcc', 'r2', 'mae', 'rmse']
    titles = ['SRCC', 'R²', 'MAE (eV)', 'RMSE (eV)']

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for ax, key, title in zip(axes, metrics_list, titles):
        vals = [per_natoms_metrics[n][key] for n in ns]
        bars = ax.bar([str(n) for n in ns], vals,
                      color=['steelblue', 'coral', 'mediumseagreen', 'gold'][:len(ns)])
        ax.set_xlabel('n_atoms', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(alpha=0.25, axis='y')
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v, f'{v:.4f}',
                    ha='center', va='bottom', fontsize=8)
    fig.suptitle(f'{ckpt_name} — metrics by n_atoms',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    out = os.path.join(fig_dir, f'metrics_per_natoms_{ckpt_name}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot → {out}")
    return out


def _metrics(t, p):
    """단일 그룹의 metric 4종."""
    if len(t) < 3:
        return None
    rho = spearmanr(t, p).correlation
    r2 = r2_score(t, p)
    mae = float(np.mean(np.abs(t - p)))
    rmse = float(np.sqrt(np.mean((t - p) ** 2)))
    return {'srcc': float(rho), 'r2': float(r2), 'mae': mae, 'rmse': rmse, 'n': len(t)}


def main():
    args = parse_args()
    with open(args.checkpoint, 'rb') as f:
        ckpt = pickle.load(f)

    targets = np.asarray(ckpt['targets_all'], dtype=np.float64)
    preds   = np.asarray(ckpt['preds_all'],   dtype=np.float64)
    splits  = np.asarray(ckpt['splits_all'])  # 0=train, 1=val, 2=test
    split_names = ckpt.get('split_names', ['train', 'val', 'test'])

    n_list = ckpt.get('n_atoms_list')
    if n_list is None:
        n_orig = ckpt.get('n_atoms_orig') or ckpt.get('n_atoms')
        if n_orig is None:
            raise KeyError('ckpt에 n_atoms_list 또는 n_atoms_orig가 필요합니다.')
        n_arr = np.full(len(targets), int(n_orig))
        print(f"  n_atoms_list 없음 → 전부 {n_orig}으로 가정")
    else:
        n_arr = np.asarray(n_list)
        if len(n_arr) != len(targets):
            raise ValueError(
                f"n_atoms_list({len(n_arr)}) != targets_all({len(targets)})")

    print(f"\n{'=' * 70}")
    print(f"  Per n_atoms metric — {os.path.basename(args.checkpoint)}")
    print(f"{'=' * 70}")
    print(f"  Total: {len(targets)} structures")
    print(f"  Split: {dict(zip(*np.unique(splits, return_counts=True)))}")
    unique_n = sorted(set(n_arr.tolist()))
    print(f"  n_atoms 그룹: {unique_n}")

    out = {'checkpoint': args.checkpoint, 'global': {}, 'per_natoms': {},
           'per_natoms_per_split': {}}

    # Global
    g = _metrics(targets, preds)
    out['global'] = g
    print(f"\n  GLOBAL: SRCC={g['srcc']:.4f}  R²={g['r2']:.4f}  "
          f"MAE={g['mae']:.4f}  RMSE={g['rmse']:.4f}  (n={g['n']})")

    if args.per_atom:
        ga = _metrics(targets / n_arr, preds / n_arr)
        out['global_per_atom'] = ga
        print(f"  GLOBAL/atom: SRCC={ga['srcc']:.4f}  R²={ga['r2']:.4f}  "
              f"MAE={ga['mae']:.6f}  RMSE={ga['rmse']:.6f} eV/atom")

    # Per n_atoms (전체)
    print(f"\n  Per n_atoms (all splits):")
    print(f"    {'n_atoms':>8} {'count':>6} {'SRCC':>8} {'R²':>8} "
          f"{'MAE':>10} {'RMSE':>10}")
    for n in unique_n:
        mask = n_arr == n
        m = _metrics(targets[mask], preds[mask])
        if m is None:
            print(f"    {n:>8} {int(mask.sum()):>6}  (n<3, skip)")
            continue
        out['per_natoms'][int(n)] = m
        print(f"    {n:>8} {m['n']:>6} {m['srcc']:>8.4f} {m['r2']:>8.4f} "
              f"{m['mae']:>10.4f} {m['rmse']:>10.4f}")

    # Per n_atoms × split
    print(f"\n  Per n_atoms × split:")
    print(f"    {'n_atoms':>8} {'split':>6} {'count':>6} {'SRCC':>8} "
          f"{'R²':>8} {'MAE':>10} {'RMSE':>10}")
    for n in unique_n:
        for sidx, sname in enumerate(split_names):
            mask = (n_arr == n) & (splits == sidx)
            if not mask.any():
                continue
            m = _metrics(targets[mask], preds[mask])
            if m is None:
                print(f"    {n:>8} {sname:>6} {int(mask.sum()):>6}  (n<3)")
                continue
            out['per_natoms_per_split'].setdefault(int(n), {})[sname] = m
            print(f"    {n:>8} {sname:>6} {m['n']:>6} {m['srcc']:>8.4f} "
                  f"{m['r2']:>8.4f} {m['mae']:>10.4f} {m['rmse']:>10.4f}")

    # Save JSON
    save_path = args.save or args.checkpoint.replace('.pkl', '_per_natoms.json')
    with open(save_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved JSON → {save_path}")

    # Figures
    if not args.no_plot:
        fig_dir = args.fig_dir or os.path.join(
            os.path.dirname(args.checkpoint) or '.', 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        ckpt_name = os.path.basename(args.checkpoint).replace('.pkl', '')
        out['figures'] = []
        # Parity (raw eV)
        out['figures'].append(
            _plot_parity_per_natoms(targets, preds, n_arr, splits, split_names,
                                    unique_n, ckpt_name, fig_dir, per_atom=False))
        # Parity (per-atom)
        if args.per_atom:
            out['figures'].append(
                _plot_parity_per_natoms(targets, preds, n_arr, splits, split_names,
                                        unique_n, ckpt_name, fig_dir, per_atom=True))
        # Metric bar chart
        if out['per_natoms']:
            out['figures'].append(
                _plot_metric_bar(out['per_natoms'], ckpt_name, fig_dir))
        # JSON 다시 저장 (figures 경로 포함)
        with open(save_path, 'w') as f:
            json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()
