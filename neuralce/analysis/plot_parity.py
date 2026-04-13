"""
plot_parity.py — Parity Plot for NeuralCE retrained models

Standalone module. Import and call from Colab/Jupyter:

    from plot_parity import plot_parity
    plot_parity("retrained_best_stfo_wo_spin_ising_lite.pkl")
    plot_parity("retrained_best_stfo_wo_spin_ising_lite.pkl", save_png="parity.png")
    plot_parity("retrained_best_stfo_wo_spin_ising_lite.pkl", save_html="parity.html")

Or run from CLI:
    python plot_parity.py retrained_best_stfo_wo_spin_ising_lite.pkl
    python plot_parity.py retrained_best_stfo_wo_spin_ising_lite.pkl --html
"""

import pickle
import numpy as np
from scipy.stats import spearmanr


def _load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        ckpt = pickle.load(f)

    required = ['preds_test', 'targets_test']
    for k in required:
        if k not in ckpt:
            raise KeyError(
                f"'{k}' not found in checkpoint. "
                f"Re-run retrain.py (updated version) to include plot data.\n"
                f"  Available keys: {list(ckpt.keys())}")
    return ckpt


def plot_parity(pkl_path, save_png=None, save_html=None, show=True,
                per_atom=True, figsize=(7, 6.5), dpi=200):
    """Plot DFT vs Predicted parity plot from retrain checkpoint.

    Args:
        pkl_path:  Path to retrained .pkl checkpoint.
        save_png:  If str, save matplotlib figure to this path.
        save_html: If str, save interactive plotly figure to this path.
        show:      If True, display inline (Colab/Jupyter).
        per_atom:  If True, plot per-atom energy (eV/atom).
        figsize:   Matplotlib figure size.
        dpi:       PNG resolution.
    """
    ckpt = _load_pkl(pkl_path)

    n_orig = ckpt.get('n_atoms_orig', 1)
    scale = n_orig if per_atom else 1
    unit = 'eV/atom' if per_atom else 'eV'

    model_name = ckpt.get('model_name', 'model')

    splits = {}
    for split in ['train', 'val', 'test']:
        pk = f'preds_{split}'
        tk = f'targets_{split}'
        ck = f'comps_{split}'
        if pk in ckpt and tk in ckpt:
            splits[split] = {
                'preds': np.array(ckpt[pk]) / scale,
                'targets': np.array(ckpt[tk]) / scale,
                'comps': ckpt.get(ck, None),
            }

    if not splits:
        raise ValueError("No prediction data found in checkpoint.")

    # ── Metrics (test) ────────────────────────────────────────────
    test = splits.get('test', list(splits.values())[-1])
    t, p = test['targets'], test['preds']
    rmse = np.sqrt(np.mean((t - p) ** 2))
    mae = np.mean(np.abs(t - p))
    srcc = spearmanr(t, p).correlation

    # Per-comp SRCC
    comp_srcc = {}
    if test['comps'] is not None:
        for c in sorted(set(test['comps'])):
            mask = [i for i, cc in enumerate(test['comps']) if cc == c]
            if len(mask) >= 3:
                comp_srcc[c] = spearmanr(t[mask], p[mask]).correlation

    # ── Matplotlib figure ─────────────────────────────────────────
    import matplotlib
    if save_png and not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    COLORS = {
        'train': '#4C72B0',
        'val':   '#DD8452',
        'test':  '#C44E52',
    }
    LABELS = {
        'train': 'Train',
        'val':   'Val',
        'test':  'Test',
    }

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    all_vals = []
    for split_name in ['train', 'val', 'test']:
        if split_name not in splits:
            continue
        s = splits[split_name]
        color = COLORS[split_name]
        alpha = 0.25 if split_name == 'train' else 0.55
        size  = 12 if split_name == 'train' else 22

        ax.scatter(s['targets'], s['preds'],
                   c=color, s=size, alpha=alpha, edgecolors='none',
                   label=f"{LABELS[split_name]} ({len(s['targets'])})", zorder=2)
        all_vals.extend(s['targets'])
        all_vals.extend(s['preds'])

    # y=x line
    margin = (max(all_vals) - min(all_vals)) * 0.05
    lo = min(all_vals) - margin
    hi = max(all_vals) + margin
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.0, alpha=0.5, zorder=1)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')

    ax.set_xlabel(f'DFT Energy ({unit})', fontsize=13)
    ax.set_ylabel(f'Predicted Energy ({unit})', fontsize=13)

    # Metrics text box
    metrics_lines = [
        f'RMSE = {rmse:.4f} {unit}',
        f'MAE  = {mae:.4f} {unit}',
        f'SRCC = {srcc:.4f}',
    ]
    if comp_srcc:
        avg_comp_srcc = np.mean(list(comp_srcc.values()))
        metrics_lines.append(f'SRCC (per-comp avg) = {avg_comp_srcc:.4f}')

    metrics_text = '\n'.join(metrics_lines)
    ax.text(0.04, 0.96, metrics_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#cccccc', alpha=0.9))

    ax.set_title(f'{model_name}', fontsize=14, fontweight='bold', pad=12)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)

    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()

    if save_png:
        fig.savefig(save_png, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"  PNG saved → {save_png}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    # ── Plotly HTML (optional) ────────────────────────────────────
    if save_html:
        _save_plotly_html(splits, model_name, unit, rmse, mae, srcc,
                          comp_srcc, save_html)

    return {'rmse': rmse, 'mae': mae, 'srcc': srcc,
            'srcc_per_comp': comp_srcc}


def _save_plotly_html(splits, model_name, unit, rmse, mae, srcc,
                      comp_srcc, html_path):
    """Interactive plotly parity plot with hover info."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  ⚠ plotly not installed — skipping HTML export")
        return

    COLORS = {
        'train': 'rgba(76, 114, 176, 0.3)',
        'val':   'rgba(221, 132, 82, 0.6)',
        'test':  'rgba(196, 78, 82, 0.7)',
    }
    SIZES = {'train': 4, 'val': 7, 'test': 8}

    fig = go.Figure()
    all_vals = []

    for split_name in ['train', 'val', 'test']:
        if split_name not in splits:
            continue
        s = splits[split_name]
        comps = s['comps']

        hover_text = []
        for i in range(len(s['targets'])):
            comp_str = f"comp={comps[i]}" if comps is not None else ""
            hover_text.append(
                f"DFT: {s['targets'][i]:.4f}<br>"
                f"Pred: {s['preds'][i]:.4f}<br>"
                f"Err: {s['preds'][i]-s['targets'][i]:.4f}<br>"
                f"{comp_str}")

        fig.add_trace(go.Scattergl(
            x=s['targets'], y=s['preds'],
            mode='markers',
            marker=dict(size=SIZES[split_name], color=COLORS[split_name]),
            name=f"{split_name.capitalize()} ({len(s['targets'])})",
            text=hover_text, hoverinfo='text',
        ))
        all_vals.extend(s['targets'])
        all_vals.extend(s['preds'])

    margin = (max(all_vals) - min(all_vals)) * 0.05
    lo, hi = min(all_vals) - margin, max(all_vals) + margin

    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode='lines',
        line=dict(dash='dash', color='black', width=1),
        showlegend=False))

    comp_str = ""
    if comp_srcc:
        avg = np.mean(list(comp_srcc.values()))
        comp_str = f" | SRCC(comp-avg)={avg:.4f}"

    fig.update_layout(
        title=f"{model_name}  —  RMSE={rmse:.4f}  MAE={mae:.4f}  SRCC={srcc:.4f}{comp_str}",
        xaxis_title=f"DFT Energy ({unit})",
        yaxis_title=f"Predicted Energy ({unit})",
        xaxis=dict(range=[lo, hi], scaleanchor='y'),
        yaxis=dict(range=[lo, hi]),
        width=700, height=650,
        template='plotly_white',
        legend=dict(x=0.72, y=0.05),
    )

    fig.write_html(html_path, include_plotlyjs='cdn')
    print(f"  HTML saved → {html_path}")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description="Parity plot from retrain checkpoint")
    p.add_argument("pkl", type=str, help="Path to retrained .pkl checkpoint")
    p.add_argument("--png", type=str, default=None,
                   help="Save PNG path. Default: <pkl>_parity.png")
    p.add_argument("--html", action="store_true",
                   help="Also save interactive HTML")
    p.add_argument("--total", action="store_true",
                   help="Plot total energy (eV) instead of per-atom")
    p.add_argument("--no-show", action="store_true",
                   help="Don't display (for headless servers)")
    args = p.parse_args()

    png_path = args.png or args.pkl.replace('.pkl', '_parity.png')
    html_path = args.pkl.replace('.pkl', '_parity.html') if args.html else None

    plot_parity(args.pkl,
                save_png=png_path,
                save_html=html_path,
                show=not args.no_show,
                per_atom=not args.total)
