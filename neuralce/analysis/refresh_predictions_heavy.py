"""
refresh_predictions_heavy.py — Re-run inference for legacy heavy ckpts

Notebook:
    from neuralce.training.refresh_predictions_heavy import refresh_predictions_heavy
    refresh_predictions_heavy(
        ckpt_path='./best_pkl/ising_heavy/best_model.pkl',
        cif_dir='./data/processed/stfo_wo_spin',
        csv_path='./data/processed/stfo_wo_spin/detailed_info.csv',
        atom_init_path='./data/processed/stfo_wo_spin/atom_init.json',
    )
"""

import os, pickle, re
import numpy as np
import jax, jax.numpy as jnp
import flax.linen as nn
import pandas as pd

from neuralce.data.data_jax import GaussianDistance, load_atom_embeddings, process_crystal


class _CEInteractionLayerLegacy(nn.Module):
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


def refresh_predictions_heavy(ckpt_path, cif_dir, csv_path, atom_init_path,
                               id_col='id', prop_name='total_energy',
                               comp_col='comp', verbose=True):
    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)

    cfg = ckpt['model_config']
    variables = {'params': ckpt['params']}
    radius = cfg['radius']
    max_num_nbr = cfg['max_num_nbr']
    n_atoms_pad = cfg['n_atoms']

    gdf = GaussianDistance(dmin=0.0, dmax=radius, step=0.2)
    atom_emb = load_atom_embeddings(atom_init_path)

    df = pd.read_csv(csv_path)
    if id_col not in df.columns:
        id_col = 'id' if 'id' in df.columns else 'cif_id'
    energy_map = dict(zip(df[id_col].astype(str), df[prop_name].values))
    comp_map = (dict(zip(df[id_col].astype(str), df[comp_col].values))
                if comp_col in df.columns else None)

    model = _NeuralCE_Ising_Legacy(
        atom_fea_len=cfg['atom_fea_len'],
        n_conv=cfg['n_conv'],
        h_fea_len=cfg['h_fea_len'],
    )
    if verbose:
        print(f"Model: legacy ising_heavy, radius={radius}, n_atoms={n_atoms_pad}")

    preds, targets, comps = [], [], []
    for cif_file in sorted(f for f in os.listdir(cif_dir) if f.endswith('.cif')):
        cif_id = cif_file.replace('.cif', '')
        if cif_id not in energy_map:
            continue
        graph = process_crystal(os.path.join(cif_dir, cif_file),
                                atom_emb, gdf, max_num_nbr, radius)
        n_at = graph['n_atoms']
        if n_at != n_atoms_pad:
            if verbose:
                print(f"  skip {cif_id}: n_atoms={n_at} != {n_atoms_pad}")
            continue

        pred = float(model.apply(variables,
            atom_fea=jnp.array(graph['atom_fea']),
            nbr_fea=jnp.array(graph['nbr_fea']),
            nbr_fea_idx=jnp.array(graph['nbr_fea_idx']),
            batch_size=1,
            n_atoms_per_crystal=n_at,
        ).squeeze())

        if comp_map is not None:
            comps.append(int(comp_map[cif_id]))
        else:
            m = re.match(r'^(\d+)_', cif_id)
            comps.append(int(m.group(1)) if m else 0)
        preds.append(pred)
        targets.append(float(energy_map[cif_id]))

    preds = np.array(preds); targets = np.array(targets); comps = np.array(comps)

    if verbose:
        mae = np.mean(np.abs(preds - targets))
        print(f"\n  {len(preds)} structures, {len(set(comps.tolist()))} compositions")
        print(f"  Target: [{targets.min():.2f}, {targets.max():.2f}]")
        print(f"  Pred:   [{preds.min():.2f}, {preds.max():.2f}]")
        print(f"  MAE:    {mae:.4f} eV")

    return preds, targets, comps