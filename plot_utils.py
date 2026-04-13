"""
plot_utils.py — Plotting utilities for NeuralCE results

Usage (Colab):
    from plot_utils import plot_parity_mixing_enthalpy, plot_parity, plot_loss_curve
    
    # Mixing enthalpy parity
    plot_parity_mixing_enthalpy("retrained_stfo_wo_spin_ising_lite.pkl")
    
    # Raw energy parity (split colored)
    plot_parity("retrained_stfo_wo_spin_ising_lite.pkl")
    
    # Composition colored parity
    plot_parity("retrained_stfo_wo_spin_ising_lite.pkl", color_by='comp')
    
    # Loss curve
    plot_loss_curve("retrained_stfo_wo_spin_ising_lite.pkl")
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.stats import spearmanr


def _load_ckpt(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# ═══════════════════════════════════════════════════════════════════════
# 1. Mixing Enthalpy Parity Plot
# ═══════════════════════════════════════════════════════════════════════

def plot_parity_mixing_enthalpy(ckpt_path, save=None, figsize=(5, 5), dpi=200):
    """Mixing enthalpy parity plot from retrained checkpoint.
    
    Args:
        ckpt_path: path to retrained .pkl
        save: save path (png/pdf). None = don't save
        figsize: figure size
        dpi: resolution
    Returns:
        fig, ax
    """
    ckpt = _load_ckpt(ckpt_path)
    n_orig = ckpt['n_atoms_orig']

    targets_pa = ckpt['targets_all'] / n_orig
    preds_pa   = ckpt['preds_all']   / n_orig
    comps      = ckpt['comps_all']
    splits     = ckpt['splits_all']  # 0=train, 1=val, 2=test

    # Endpoint reference
    comp_scale = comps.max()
    x = comps / comp_scale
    e_ref_low  = np.mean(targets_pa[comps == comps.min()])
    e_ref_high = np.mean(targets_pa[comps == comps.max()])
    e_interp   = (1 - x) * e_ref_low + x * e_ref_high

    dh_dft  = (targets_pa - e_interp) * 1000  # meV/atom
    dh_pred = (preds_pa   - e_interp) * 1000

    # Test metrics
    test = splits == 2
    rmse = np.sqrt(np.mean((dh_dft[test] - dh_pred[test])**2))
    mae  = np.mean(np.abs(dh_dft[test] - dh_pred[test]))
    srcc = spearmanr(dh_dft[test], dh_pred[test]).correlation

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    _scatter_splits(ax, dh_dft, dh_pred, splits)
    _add_parity_line(ax, dh_dft, dh_pred)

    ax.text(0.05, 0.95,
            f'Test RMSE: {rmse:.1f} meV/at\nTest MAE: {mae:.1f} meV/at\nSRCC: {srcc:.4f}',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

    ax.set_xlabel(r'DFT $\Delta H_{mix}$ (meV/atom)', fontsize=11)
    ax.set_ylabel(r'Pred $\Delta H_{mix}$ (meV/atom)', fontsize=11)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_aspect('equal', adjustable='datalim')
    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches='tight')
        print(f"Saved → {save}")

    plt.show()
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════
# 2. Raw Energy Parity Plot
# ═══════════════════════════════════════════════════════════════════════

def plot_parity(ckpt_path, color_by='split', save=None, figsize=(5, 5), dpi=200):
    """Energy parity plot (eV/atom).
    
    Args:
        ckpt_path: path to retrained .pkl
        color_by: 'split' or 'comp'
        save: save path
    Returns:
        fig, ax
    """
    ckpt = _load_ckpt(ckpt_path)
    n_orig = ckpt['n_atoms_orig']

    targets_pa = ckpt['targets_all'] / n_orig
    preds_pa   = ckpt['preds_all']   / n_orig
    comps      = ckpt['comps_all']
    splits     = ckpt['splits_all']

    # Test metrics
    test = splits == 2
    t_test, p_test = targets_pa[test], preds_pa[test]
    rmse = np.sqrt(np.mean((t_test - p_test)**2)) * 1000
    mae  = np.mean(np.abs(t_test - p_test)) * 1000
    srcc = spearmanr(t_test, p_test).correlation

    fig, ax = plt.subplots(figsize=figsize)

    if color_by == 'comp':
        comp_scale = comps.max() if comps.max() > 100 else (100 if comps.max() > 1 else 1.0)
        fe_ratios = comps / comp_scale
        norm = Normalize(vmin=0.0, vmax=fe_ratios.max())
        sc = ax.scatter(targets_pa, preds_pa, c=fe_ratios, cmap='viridis', norm=norm,
                        s=20, alpha=0.7, edgecolors='none')
        plt.colorbar(sc, ax=ax, label='Fe ratio x', pad=0.02)

        # Per-comp SRCC
        unique_comps = sorted(set(comps))
        lines = []
        for c in unique_comps:
            m = comps == c
            if m.sum() >= 3:
                s = spearmanr(targets_pa[m], preds_pa[m]).correlation
                lines.append(f'x={c/comp_scale:.3f}: {s:.4f}')
        if lines:
            ax.text(0.95, 0.05, 'Per-comp SRCC:\n' + '\n'.join(lines),
                    transform=ax.transAxes, fontsize=7, va='bottom', ha='right',
                    family='monospace',
                    bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.7))
    else:
        _scatter_splits(ax, targets_pa, preds_pa, splits)
        ax.legend(fontsize=9, loc='lower right')

    _add_parity_line(ax, targets_pa, preds_pa)

    ax.text(0.05, 0.95,
            f'Test RMSE: {rmse:.1f} meV/at\nTest MAE: {mae:.1f} meV/at\nSRCC: {srcc:.4f}',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

    ax.set_xlabel('DFT Energy (eV/atom)', fontsize=11)
    ax.set_ylabel('Predicted Energy (eV/atom)', fontsize=11)
    ax.set_aspect('equal', adjustable='datalim')
    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches='tight')
        print(f"Saved → {save}")

    plt.show()
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════
# 3. Loss Curve
# ═══════════════════════════════════════════════════════════════════════

def plot_loss_curve(ckpt_path, save=None, figsize=(7, 4), dpi=200):
    """Training loss curve from retrained checkpoint.
    
    Returns:
        fig, ax
    """
    ckpt = _load_ckpt(ckpt_path)
    history = ckpt['history']
    best_epoch = ckpt['best_epoch']
    model_name = ckpt.get('model_name', '')

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(history['epoch'], history['train_loss'], color='#888888',
            alpha=0.6, linewidth=0.5, label='Train')
    ax.plot(history['epoch'], history['val_loss'], color='#5DA5DA',
            alpha=0.8, linewidth=0.8, label='Val')
    ax.axvline(best_epoch, color='#F15854', linestyle='--', linewidth=0.8,
               alpha=0.7, label=f'Best @ {best_epoch}')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss (eV²)')
    ax.set_title(f'{model_name} — Loss Curve')
    ax.legend(fontsize=9)
    ax.set_yscale('log')
    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches='tight')
        print(f"Saved → {save}")

    plt.show()
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════
# 4. Error Distribution
# ═══════════════════════════════════════════════════════════════════════

def plot_error_dist(ckpt_path, save=None, figsize=(6, 4), dpi=200):
    """Error distribution histogram per split.
    
    Returns:
        fig, ax
    """
    ckpt = _load_ckpt(ckpt_path)
    n_orig = ckpt['n_atoms_orig']
    splits = ckpt['splits_all']
    errors_all = (ckpt['preds_all'] - ckpt['targets_all']) / n_orig * 1000  # meV/atom

    colors = {0: '#888888', 1: '#5DA5DA', 2: '#F15854'}
    labels = {0: 'Train', 1: 'Val', 2: 'Test'}

    fig, ax = plt.subplots(figsize=figsize)
    for s in [0, 1, 2]:
        m = splits == s
        e = errors_all[m]
        ax.hist(e, bins=30, alpha=0.5, color=colors[s],
                label=f'{labels[s]} (σ={np.std(e):.1f})', density=True)

    ax.set_xlabel('Error (meV/atom)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(fontsize=9)
    ax.axvline(0, color='k', linewidth=0.5, linestyle=':')
    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches='tight')
        print(f"Saved → {save}")

    plt.show()
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════

def _scatter_splits(ax, x_data, y_data, splits):
    colors = {0: '#aaaaaa', 1: '#5DA5DA', 2: '#F15854'}
    labels = {0: 'Train', 1: 'Val', 2: 'Test'}
    alphas = {0: 0.25, 1: 0.7, 2: 0.8}
    sizes  = {0: 10, 1: 20, 2: 20}
    for s in [0, 1, 2]:
        m = splits == s
        ax.scatter(x_data[m], y_data[m], c=colors[s], alpha=alphas[s],
                   s=sizes[s], edgecolors='none', label=labels[s], zorder=s+1)


def _add_parity_line(ax, x_data, y_data):
    vmin = min(x_data.min(), y_data.min())
    vmax = max(x_data.max(), y_data.max())
    pad = (vmax - vmin) * 0.03
    ax.plot([vmin-pad, vmax+pad], [vmin-pad, vmax+pad], 'k-', lw=0.8, alpha=0.5)
