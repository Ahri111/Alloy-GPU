"""
retrain.py — Retrain best model from ablation with full epochs, no early stopping.
Fixed pool mode with auto-padding for variable atom counts.

Usage:
  python retrain.py \
      --config ./configs/stfo_wo_spin.yaml \
      --checkpoint best_stfo_wo_spin_ising_lite.pkl \
      --epochs 3000

  # Or with custom output name:
  python retrain.py \
      --config ./configs/stfo_wo_spin.yaml \
      --checkpoint best_stfo_wo_spin_ising_lite.pkl \
      --epochs 5000 \
      --output retrained_ising_lite.pkl
"""

import os, argparse, pickle, copy, re, json
import numpy as np
import yaml

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import pandas as pd
from scipy.stats import spearmanr
from pymatgen.core.structure import Structure

from NeuralCE_jax import (create_neuralce, is_spin_model, is_sisj_model,
                           needs_spin, LITE_MODELS)


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Retrain best model without early stopping.")
    p.add_argument("--config", type=str, required=True,
                   help="YAML config (same one used for ablation)")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Checkpoint .pkl from ablation (e.g. best_stfo_wo_spin_ising_lite.pkl)")
    p.add_argument("--epochs", type=int, default=3000,
                   help="Number of epochs to train. Default: 3000")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Batch size. Default: from config")
    p.add_argument("--output", type=str, default=None,
                   help="Output checkpoint path. Default: retrained_<checkpoint>")
    p.add_argument("--model", type=str, default=None,
                   help="Model name override (e.g. hegnn_lite). "
                        "Auto-inferred from checkpoint if not given.")
    p.add_argument("--no-resume", dest='resume', action='store_false', default=True,
                   help="Random init instead of resuming from checkpoint params. "
                        "Default: resume from checkpoint.")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# DATA & GRAPH
# ═══════════════════════════════════════════════════════════════════════

def load_data(cif_dir, csv_path, spin_pkl_path, id_col, comp_regex, exclude_z=None):
    if exclude_z is None:
        exclude_z = set()

    df = pd.read_csv(csv_path)
    if id_col not in df.columns:
        id_col = 'id' if 'id' in df.columns else 'cif_id'
    energy_map = dict(zip(df[id_col].astype(str), df['total_energy'].values))

    # Spin: try (1) spin_pkl, (2) CSV 'spins' column, (3) zeros
    spin_map = {}
    if spin_pkl_path and os.path.exists(spin_pkl_path):
        spin_df = pd.read_pickle(spin_pkl_path)
        spin_map = dict(zip(spin_df['cif_id'].astype(str),
                            spin_df['spin_states'].values))
        print(f"  Spin data loaded from PKL: {len(spin_map)} entries")
    elif 'spins' in df.columns:
        for _, row in df.iterrows():
            sid = str(row[id_col])
            try:
                spin_map[sid] = json.loads(row['spins'])
            except (json.JSONDecodeError, TypeError):
                pass
        print(f"  Spin data loaded from CSV: {len(spin_map)} entries")
    else:
        print(f"  No spin data → all spins set to 0")

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

    return structures


def build_graph_lite(struct, cutoff, n_shells, max_num_nbr, species_map,
                     include_sisj=False, shell_edges=None):
    crystal = struct['crystal']
    n_at = len(crystal)
    spins = struct['spin_states']
    n_species = len(species_map)

    atom_fea = np.zeros((n_at, n_species), dtype=np.float32)
    for i, site in enumerate(crystal):
        z = site.specie.Z
        if z in species_map:
            atom_fea[i, species_map[z]] = 1.0

    all_nbrs = crystal.get_all_neighbors(cutoff, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

    nbr_fea_idx = np.zeros((n_at, max_num_nbr), dtype=np.int32)
    nbr_dists = np.full((n_at, max_num_nbr), cutoff + 1.0, dtype=np.float64)

    for i, nbr in enumerate(all_nbrs):
        n_nbr = min(len(nbr), max_num_nbr)
        if i == 0 and len(nbr) > max_num_nbr:
            import warnings
            warnings.warn(
                f"Atom 0 has {len(nbr)} neighbors within cutoff={cutoff} Å "
                f"but max_num_nbr={max_num_nbr}. "
                f"{len(nbr)-max_num_nbr} neighbors truncated per atom.",
                stacklevel=3)
        for j in range(n_nbr):
            nbr_fea_idx[i, j] = nbr[j][2]
            nbr_dists[i, j] = nbr[j][1]

    valid_mask = nbr_dists < cutoff
    if shell_edges is not None:
        _edges = np.array(shell_edges)
    else:
        min_dist = nbr_dists[valid_mask].min() if valid_mask.any() else 0.1
        _edges = np.linspace(min_dist * 0.99, cutoff, n_shells + 1)

    shell_idx = np.clip(np.digitize(nbr_dists, _edges) - 1, 0, n_shells - 1)

    shell_oh = np.zeros((n_at, max_num_nbr, n_shells), dtype=np.float32)
    for i in range(n_at):
        for j in range(max_num_nbr):
            if nbr_dists[i, j] < cutoff:
                shell_oh[i, j, shell_idx[i, j]] = 1.0

    if include_sisj:
        sisj = np.zeros((n_at, max_num_nbr, 1), dtype=np.float32)
        for i in range(n_at):
            for j in range(max_num_nbr):
                if nbr_dists[i, j] < cutoff:
                    sisj[i, j, 0] = spins[i] * spins[nbr_fea_idx[i, j]]
        nbr_fea = np.concatenate([shell_oh, sisj], axis=-1)
    else:
        nbr_fea = shell_oh

    spin_fea = spins[:n_at].reshape(-1, 1).astype(np.float32)
    return atom_fea, nbr_fea, nbr_fea_idx, spin_fea


def build_all_graphs(structures, cutoff, n_shells, max_num_nbr, species_map,
                     n_atoms_pad, include_sisj=False, shell_edges=None):
    """Build all graphs, pad to n_atoms_pad, stack."""
    raw = {'atom_fea': [], 'nbr_fea': [], 'nbr_fea_idx': [],
           'spin_fea': [], 'energies': [], 'comps': [], 'n_atoms_list': []}

    for s in structures:
        af, nf, nfi, sf = build_graph_lite(
            s, cutoff, n_shells, max_num_nbr, species_map,
            include_sisj, shell_edges)
        raw['atom_fea'].append(af)
        raw['nbr_fea'].append(nf)
        raw['nbr_fea_idx'].append(nfi)
        raw['spin_fea'].append(sf)
        raw['energies'].append(s['total_energy'])
        raw['comps'].append(s['comp_code'])
        raw['n_atoms_list'].append(af.shape[0])

    n_atoms_arr = np.array(raw['n_atoms_list'])
    n_species = raw['atom_fea'][0].shape[-1]
    edge_dim  = raw['nbr_fea'][0].shape[-1]

    padded = {'atom_fea': [], 'nbr_fea': [], 'nbr_fea_idx': [],
              'spin_fea': [], 'atom_mask': []}
    n_padded = 0
    for i in range(len(raw['atom_fea'])):
        n_at = raw['atom_fea'][i].shape[0]
        if n_at == n_atoms_pad:
            padded['atom_fea'].append(raw['atom_fea'][i])
            padded['nbr_fea'].append(raw['nbr_fea'][i])
            padded['nbr_fea_idx'].append(raw['nbr_fea_idx'][i])
            padded['spin_fea'].append(raw['spin_fea'][i])
            padded['atom_mask'].append(np.ones((n_atoms_pad,), dtype=np.float32))
        elif n_at < n_atoms_pad:
            n_pad = n_atoms_pad - n_at
            padded['atom_fea'].append(np.concatenate([raw['atom_fea'][i],
                np.zeros((n_pad, n_species), dtype=np.float32)], axis=0))
            padded['nbr_fea'].append(np.concatenate([raw['nbr_fea'][i],
                np.zeros((n_pad, max_num_nbr, edge_dim), dtype=np.float32)], axis=0))
            padded['nbr_fea_idx'].append(np.concatenate([raw['nbr_fea_idx'][i],
                np.zeros((n_pad, max_num_nbr), dtype=np.int32)], axis=0))
            padded['spin_fea'].append(np.concatenate([raw['spin_fea'][i],
                np.zeros((n_pad, 1), dtype=np.float32)], axis=0))
            padded['atom_mask'].append(np.concatenate([
                np.ones((n_at,), dtype=np.float32),
                np.zeros((n_pad,), dtype=np.float32)], axis=0))
            n_padded += 1
        else:
            raise ValueError(f"Structure {i} has {n_at} atoms > n_atoms_pad={n_atoms_pad}")

    if n_padded > 0:
        print(f"    Padded {n_padded}/{len(raw['atom_fea'])} structures to {n_atoms_pad} atoms")

    return {
        'atom_fea':    jnp.array(np.stack(padded['atom_fea'])),
        'nbr_fea':     jnp.array(np.stack(padded['nbr_fea'])),
        'nbr_fea_idx': jnp.array(np.stack(padded['nbr_fea_idx'])),
        'spin_fea':    jnp.array(np.stack(padded['spin_fea'])),
        'atom_mask':   jnp.array(np.stack(padded['atom_mask'])),
        'energies':    jnp.array(raw['energies']),
        'comps':       raw['comps'],
        'n_atoms_list': n_atoms_arr,
    }


# ═══════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════

def create_model(model_name, hp):
    sisj = is_sisj_model(model_name)
    n_shells = hp['n_shells']
    edge_dim = n_shells + 1 if sisj else n_shells

    kwargs = {
        'atom_fea_len': hp['atom_fea_len'],
        'nbr_fea_len': edge_dim,
        'n_conv': hp['n_conv'],
        'h_fea_len': hp['h_fea_len'],
    }
    if is_spin_model(model_name):
        kwargs['odd_fea_len'] = hp['odd_fea_len']

    return create_neuralce(model_type=model_name, pool_mode='padded',
                           readout_type='sum', **kwargs)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── Load config ───────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cif_dir    = cfg['cif_dir']
    csv_path   = cfg['csv_path']
    spin_pkl   = cfg.get('spin_pkl')
    id_col     = cfg.get('id_col', 'id')
    comp_regex = cfg.get('comp_regex', r'_(\d+)')
    species_map = {int(k): v for k, v in cfg['species_map'].items()}
    exclude_z  = set(cfg.get('exclude_species', []))

    _n_atoms_cfg = cfg.get('n_atoms', 'auto')

    _abl = cfg.get('ablation', {})
    seed = _abl.get('seed', 42)
    batch_size = args.batch_size or _abl.get('batch_size', 32)
    val_frac  = _abl.get('val_frac', 0.15)
    test_frac = _abl.get('test_frac', 0.15)

    # ── Load checkpoint ───────────────────────────────────────────────
    with open(args.checkpoint, 'rb') as f:
        ckpt = pickle.load(f)
    hp = ckpt['hp']

    # Infer model name: (1) checkpoint key, (2) CLI --model, (3) run_models, (4) dataset prefix
    model_name = ckpt.get('model_name') or args.model
    if model_name is None:
        ckpt_base = os.path.basename(args.checkpoint).replace('.pkl', '')
        run_models = _abl.get('run_models', [])
        for m in sorted(run_models, key=len, reverse=True):  # longest match first
            if m in ckpt_base:
                model_name = m
                break
    if model_name is None:
        ckpt_base = os.path.basename(args.checkpoint).replace('.pkl', '')
        dataset_name = cfg.get('dataset_name', '')
        prefix = f'best_{dataset_name}_'
        if ckpt_base.startswith(prefix):
            model_name = ckpt_base[len(prefix):]
        else:
            model_name = '_'.join(ckpt_base.split('_')[2:])
    if model_name is None:
        raise ValueError(
            "Cannot infer model name. Use --model <name> "
            f"(available: {sorted(LITE_MODELS)})")

    # ── Resolve graph config ──────────────────────────────────────────
    _graph = cfg.get('graph', {})
    _candidates = _graph.get('candidates')
    shell_edges = None
    max_num_nbr = hp['max_num_nbr']

    if _candidates and hp['cutoff'] in {float(k) for k in _candidates}:
        info = _candidates[hp['cutoff']] if hp['cutoff'] in _candidates else _candidates[str(hp['cutoff'])]
        shell_edges = info.get('shell_edges')
        if 'max_num_nbr' in info:
            max_num_nbr = info['max_num_nbr']
    elif _graph.get('shell_edges'):
        shell_edges = _graph['shell_edges']

    # ── Load data ─────────────────────────────────────────────────────
    print(f"{'═' * 70}")
    print(f"  RETRAIN: {model_name}")
    print(f"  HP: {hp}")
    print(f"  Epochs: {args.epochs} (no early stopping)")
    print(f"  Init: {'resume from ablation' if args.resume else 'random init'}")
    print(f"{'═' * 70}")

    print("Loading data ...")
    structures = load_data(cif_dir, csv_path, spin_pkl, id_col, comp_regex, exclude_z)

    # ── Determine N_ATOMS (auto-padding for variable counts) ──────────
    n_atoms_set = set(s['n_atoms'] for s in structures)

    if len(n_atoms_set) == 1:
        N_ATOMS = n_atoms_set.pop()
        n_atoms_orig = N_ATOMS
    else:
        N_ATOMS = max(n_atoms_set)
        if isinstance(_n_atoms_cfg, int) and _n_atoms_cfg >= N_ATOMS:
            N_ATOMS = _n_atoms_cfg
        n_atoms_orig = None  # per-structure for per-atom metrics
        print(f"Mixed atom counts {sorted(n_atoms_set)} → padding to {N_ATOMS}")

    print(f"Loaded {len(structures)} structures, N_ATOMS={N_ATOMS}")

    # ── Build graphs ──────────────────────────────────────────────────
    sisj = is_sisj_model(model_name)
    use_spin = is_spin_model(model_name)
    dataset = build_all_graphs(
        structures, hp['cutoff'], hp['n_shells'], max_num_nbr,
        species_map, n_atoms_pad=N_ATOMS,
        include_sisj=sisj, shell_edges=shell_edges)

    # ── Split ─────────────────────────────────────────────────────────
    from sklearn.model_selection import train_test_split
    indices = list(range(len(structures)))
    trainval_idx, test_idx = train_test_split(indices, test_size=test_frac, random_state=seed)
    val_rel = val_frac / (1 - test_frac)
    train_idx, val_idx = train_test_split(trainval_idx, test_size=val_rel, random_state=seed)
    train_idx, val_idx, test_idx = np.array(train_idx), np.array(val_idx), np.array(test_idx)
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    # ── Create model & init ───────────────────────────────────────────
    model = create_model(model_name, hp)
    rng = jax.random.PRNGKey(seed)

    dummy_af  = dataset['atom_fea'][0]
    dummy_nf  = dataset['nbr_fea'][0]
    dummy_nfi = dataset['nbr_fea_idx'][0]
    dummy_mask = dataset['atom_mask'][0]
    init_kw = {
        'atom_fea': dummy_af, 'nbr_fea': dummy_nf, 'nbr_fea_idx': dummy_nfi,
        'batch_size': 1, 'n_atoms_per_crystal': N_ATOMS,
        'atom_mask': dummy_mask[None, :],
    }
    if use_spin:
        init_kw['atom_spins'] = dataset['spin_fea'][0]

    params = model.init(rng, **init_kw)
    n_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"Params: {n_params:,}")

    # Resume from checkpoint params (default) or random init (--no-resume)
    if args.resume and 'params' in ckpt:
        params = ckpt['params']
        print(f"  Resumed params from ablation checkpoint")
    else:
        print(f"  Random init ({'--no-resume' if not args.resume else 'no params in checkpoint'})")

    lr = hp['lr']
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.1, peak_value=lr,
        warmup_steps=50, decay_steps=args.epochs, end_value=lr * 0.01)
    wd = hp.get('weight_decay', 0.0)
    optimizer = optax.chain(optax.clip_by_global_norm(5.0), optax.adamw(schedule, weight_decay=wd))

    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer)

    # ── Forward helpers ───────────────────────────────────────────────
    def _forward(state, af, nf, nfi, sf, mask):
        B = af.shape[0]
        af_flat = af.reshape(-1, af.shape[-1])
        offsets = jnp.arange(B)[:, None, None] * N_ATOMS
        nfi_flat = (nfi + offsets).reshape(-1, nfi.shape[-1])
        nf_flat = nf.reshape(-1, nf.shape[-2], nf.shape[-1])
        kw = {'atom_fea': af_flat, 'nbr_fea': nf_flat, 'nbr_fea_idx': nfi_flat,
              'batch_size': B, 'n_atoms_per_crystal': N_ATOMS, 'atom_mask': mask}
        if use_spin:
            kw['atom_spins'] = sf.reshape(-1, sf.shape[-1])
        return state.apply_fn(state.params, **kw).squeeze(-1)

    def loss_fn(params, state, batch):
        af, nf, nfi, sf, mask, targets = batch
        B = af.shape[0]
        af_flat = af.reshape(-1, af.shape[-1])
        offsets = jnp.arange(B)[:, None, None] * N_ATOMS
        nfi_flat = (nfi + offsets).reshape(-1, nfi.shape[-1])
        nf_flat = nf.reshape(-1, nf.shape[-2], nf.shape[-1])
        kw = {'atom_fea': af_flat, 'nbr_fea': nf_flat, 'nbr_fea_idx': nfi_flat,
              'batch_size': B, 'n_atoms_per_crystal': N_ATOMS, 'atom_mask': mask}
        if use_spin:
            kw['atom_spins'] = sf.reshape(-1, sf.shape[-1])
        pred = state.apply_fn(params, **kw).squeeze(-1)
        return jnp.mean((pred - targets) ** 2)

    @jax.jit
    def train_step(state, batch):
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params, state, batch)
        return state.apply_gradients(grads=grads), loss

    def eval_set(state, indices):
        all_preds, all_targets = [], []
        total_loss, n_total = 0.0, 0
        for start in range(0, len(indices), batch_size):
            idx = indices[start:start+batch_size]
            af = dataset['atom_fea'][idx]
            nf = dataset['nbr_fea'][idx]
            nfi = dataset['nbr_fea_idx'][idx]
            sf = dataset['spin_fea'][idx]
            mask = dataset['atom_mask'][idx]
            targets = dataset['energies'][idx]
            B = af.shape[0]
            pred = _forward(state, af, nf, nfi, sf, mask)
            mse = float(jnp.mean((pred - targets) ** 2))
            total_loss += mse * B
            n_total += B
            all_preds.append(np.array(pred))
            all_targets.append(np.array(targets))
        return (total_loss / max(n_total, 1),
                np.concatenate(all_preds), np.concatenate(all_targets))

    # ── Training loop (no patience) ───────────────────────────────────
    best_val = float('inf')
    best_params = None
    best_epoch = 0
    history = {'epoch': [], 'train_loss': [], 'val_loss': []}

    for epoch in range(1, args.epochs + 1):
        perm = np.array(jax.random.permutation(
            jax.random.PRNGKey(seed + epoch), jnp.array(train_idx)))

        epoch_loss, n_bat = 0.0, 0
        for start in range(0, len(perm), batch_size):
            idx = perm[start:start+batch_size]
            if len(idx) < 2:
                continue
            batch = (dataset['atom_fea'][idx], dataset['nbr_fea'][idx],
                     dataset['nbr_fea_idx'][idx], dataset['spin_fea'][idx],
                     dataset['atom_mask'][idx],
                     dataset['energies'][idx])
            state, loss = train_step(state, batch)
            epoch_loss += float(loss)
            n_bat += 1
        epoch_loss /= max(n_bat, 1)

        val_loss, _, _ = eval_set(state, val_idx)

        history['epoch'].append(epoch)
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_params = copy.deepcopy(state.params)
            best_epoch = epoch

        if epoch % 100 == 0:
            print(f"  [epoch {epoch:5d}] train={epoch_loss:.6f} "
                  f"val={val_loss:.6f} best={best_val:.6f}@{best_epoch}")

    print(f"\n  Training done. Best val_MSE={best_val:.6f} @ epoch {best_epoch}")

    # ── Evaluate ALL splits ───────────────────────────────────────────
    state = state.replace(params=best_params)

    split_names = ['train', 'val', 'test']
    split_indices = {'train': train_idx, 'val': val_idx, 'test': test_idx}
    all_preds, all_targets, all_comps = {}, {}, {}

    for sname in split_names:
        idx = split_indices[sname]
        _, preds_s, targets_s = eval_set(state, idx)
        comps_s = np.array([dataset['comps'][i] for i in idx])
        all_preds[sname] = preds_s
        all_targets[sname] = targets_s
        all_comps[sname] = comps_s

    # Concatenated arrays for convenience
    preds_all   = np.concatenate([all_preds[s] for s in split_names])
    targets_all = np.concatenate([all_targets[s] for s in split_names])
    comps_all   = np.concatenate([all_comps[s] for s in split_names])
    splits_all  = np.concatenate([np.full(len(all_preds[s]), i)
                                  for i, s in enumerate(split_names)])

    # ── Print test metrics ────────────────────────────────────────────
    preds_test   = all_preds['test']
    targets_test = all_targets['test']
    test_comps   = all_comps['test']

    rmse = np.sqrt(np.mean((targets_test - preds_test) ** 2))
    mae = np.mean(np.abs(targets_test - preds_test))
    srcc = spearmanr(targets_test, preds_test).correlation

    # Per-atom metrics: use per-structure n_atoms if variable
    if n_atoms_orig is not None:
        divisors_test = np.full(len(test_idx), n_atoms_orig, dtype=np.float64)
    else:
        divisors_test = np.array([structures[i]['n_atoms'] for i in test_idx], dtype=np.float64)

    rmse_pa = np.sqrt(np.mean(((targets_test - preds_test) / divisors_test) ** 2))
    mae_pa = np.mean(np.abs((targets_test - preds_test) / divisors_test))

    print(f"\n  TEST (total):    RMSE={rmse:.4f} eV | SRCC={srcc:.4f}")
    print(f"  TEST (per atom): RMSE={rmse_pa:.6f} eV/atom | MAE={mae_pa:.6f}")

    # Per-comp SRCC
    for c in sorted(set(test_comps)):
        mask_c = [i for i, cc in enumerate(test_comps) if cc == c]
        if len(mask_c) >= 3:
            s = spearmanr(targets_test[mask_c], preds_test[mask_c]).correlation
            print(f"    x={c/1000:.3f}: SRCC={s:.4f}")

    # ── Save ──────────────────────────────────────────────────────────
    out_path = args.output or f"retrained_{os.path.basename(args.checkpoint)}"
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump({
            'params': best_params,
            'hp': hp,
            'model_name': model_name,
            'best_epoch': best_epoch,
            'best_val_mse': float(best_val),
            'test_rmse': float(rmse),
            'test_mae': float(mae),
            'test_srcc': float(srcc),
            'retrain_epochs': args.epochs,
            'n_atoms': N_ATOMS,
            'n_atoms_orig': n_atoms_orig,
            'n_atoms_list': dataset['n_atoms_list'],
            # ── Per-split predictions ──
            'preds_train':   all_preds['train'],
            'targets_train': all_targets['train'],
            'comps_train':   all_comps['train'],
            'preds_val':     all_preds['val'],
            'targets_val':   all_targets['val'],
            'comps_val':     all_comps['val'],
            'preds_test':    all_preds['test'],
            'targets_test':  all_targets['test'],
            'comps_test':    all_comps['test'],
            # ── Concatenated (all splits) ──
            'preds_all':   preds_all,
            'targets_all': targets_all,
            'comps_all':   comps_all,
            'splits_all':  splits_all,   # 0=train, 1=val, 2=test
            'split_names': split_names,
            # ── Training history ──
            'history': history,
        }, f)
    print(f"\n  Saved → {out_path}")
    print(f"  Keys: params, hp, preds/targets/comps per split, *_all, history")

    # ── Plot results ──────────────────────────────────────────────────
    fig_dir = os.path.join(os.path.dirname(out_path) or '.', 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    prefix = os.path.basename(out_path).replace('.pkl', '')

    # For plotting per-atom: use n_atoms_orig if fixed, else per-structure
    plot_n_atoms = n_atoms_orig if n_atoms_orig is not None else dataset['n_atoms_list']
    plot_results(history, all_preds, all_targets, all_comps,
                 plot_n_atoms, structures, split_indices,
                 model_name, best_epoch, fig_dir, prefix)


# ═══════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════

def plot_results(history, all_preds, all_targets, all_comps,
                 n_atoms_div, structures, split_indices,
                 model_name, best_epoch, fig_dir, prefix):
    """Generate and save all result figures.

    n_atoms_div: int (fixed) or np.array of per-structure atom counts.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    split_colors = {'train': '#888888', 'val': '#5DA5DA', 'test': '#F15854'}
    split_labels = {'train': 'Train', 'val': 'Val', 'test': 'Test'}

    def _per_atom(arr, split_name):
        """Divide by per-atom divisor."""
        if isinstance(n_atoms_div, (int, float)):
            return arr / n_atoms_div
        else:
            idx = split_indices[split_name]
            return arr / np.array([n_atoms_div[i] for i in idx], dtype=np.float64)

    # ── 1. Loss curve ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = history['epoch']
    ax.plot(epochs, history['train_loss'], color='#888888', alpha=0.6,
            linewidth=0.5, label='Train')
    ax.plot(epochs, history['val_loss'], color='#5DA5DA', alpha=0.8,
            linewidth=0.8, label='Val')
    ax.axvline(best_epoch, color='#F15854', linestyle='--', linewidth=0.8,
               alpha=0.7, label=f'Best @ {best_epoch}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss (eV²)')
    ax.set_title(f'{model_name} — Loss Curve')
    ax.legend(fontsize=9)
    ax.set_yscale('log')
    plt.tight_layout()
    path = os.path.join(fig_dir, f'{prefix}_loss.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [fig] {path}")

    # ── 2. Parity plot (eV/atom, all splits) ──────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 5))

    for split in ['train', 'val', 'test']:
        t = _per_atom(all_targets[split], split)
        p = _per_atom(all_preds[split], split)
        alpha = 0.25 if split == 'train' else 0.7
        sz = 10 if split == 'train' else 20
        ax.scatter(t, p, c=split_colors[split], alpha=alpha, s=sz,
                   edgecolors='none', label=split_labels[split],
                   zorder=1 if split == 'train' else 2)

    all_t = np.concatenate([_per_atom(all_targets[s], s) for s in ['train', 'val', 'test']])
    all_p = np.concatenate([_per_atom(all_preds[s], s) for s in ['train', 'val', 'test']])
    vmin = min(all_t.min(), all_p.min())
    vmax = max(all_t.max(), all_p.max())
    margin = (vmax - vmin) * 0.03
    ax.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin],
            'k-', linewidth=0.8, alpha=0.5)

    # Test metrics
    t_test = _per_atom(all_targets['test'], 'test')
    p_test = _per_atom(all_preds['test'], 'test')
    rmse_pa = np.sqrt(np.mean((t_test - p_test) ** 2)) * 1000
    mae_pa  = np.mean(np.abs(t_test - p_test)) * 1000
    srcc    = spearmanr(t_test, p_test).correlation
    ax.text(0.05, 0.95,
            f'Test RMSE: {rmse_pa:.2f} meV/at\nTest MAE: {mae_pa:.2f} meV/at\nSRCC: {srcc:.4f}',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

    ax.set_xlabel('DFT Energy (eV/atom)', fontsize=11)
    ax.set_ylabel('Predicted Energy (eV/atom)', fontsize=11)
    ax.set_title(f'{model_name} — Parity (per atom)')
    ax.legend(fontsize=9)
    ax.set_aspect('equal', adjustable='datalim')
    plt.tight_layout()
    path = os.path.join(fig_dir, f'{prefix}_parity.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [fig] {path}")

    # ── 3. Parity plot colored by composition ─────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 5))

    comps_concat = np.concatenate([all_comps[s] for s in ['train', 'val', 'test']])
    max_comp = comps_concat.max()
    comp_scale = max_comp if max_comp > 100 else (100 if max_comp > 1 else 1.0)
    fe_ratios = comps_concat / comp_scale

    norm = Normalize(vmin=0.0, vmax=fe_ratios.max())
    sc = ax.scatter(all_t, all_p, c=fe_ratios, cmap='viridis', norm=norm,
                    s=20, alpha=0.7, edgecolors='none')
    ax.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin],
            'k-', linewidth=0.8, alpha=0.5)

    ax.set_xlabel('DFT Energy (eV/atom)', fontsize=11)
    ax.set_ylabel('Predicted Energy (eV/atom)', fontsize=11)
    ax.set_title(f'{model_name} — Parity (by composition)')
    plt.colorbar(sc, ax=ax, label='Fe ratio x', pad=0.02)

    # Per-comp SRCC text
    unique_comps = sorted(set(comps_concat))
    comp_srcc_lines = []
    for c in unique_comps:
        cmask = comps_concat == c
        if cmask.sum() >= 3:
            s = spearmanr(all_t[cmask], all_p[cmask]).correlation
            comp_srcc_lines.append(f'x={c/comp_scale:.3f}: {s:.4f}')
    if comp_srcc_lines:
        ax.text(0.95, 0.05, 'Per-comp SRCC:\n' + '\n'.join(comp_srcc_lines),
                transform=ax.transAxes, fontsize=7, va='bottom', ha='right',
                family='monospace',
                bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.7))

    ax.set_aspect('equal', adjustable='datalim')
    plt.tight_layout()
    path = os.path.join(fig_dir, f'{prefix}_parity_comp.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [fig] {path}")

    # ── 4. Error distribution ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    for split in ['train', 'val', 'test']:
        errors = (_per_atom(all_preds[split], split)
                  - _per_atom(all_targets[split], split)) * 1000
        ax.hist(errors, bins=30, alpha=0.5, color=split_colors[split],
                label=f'{split_labels[split]} (σ={np.std(errors):.1f})', density=True)
    ax.set_xlabel('Error (meV/atom)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'{model_name} — Error Distribution')
    ax.legend(fontsize=9)
    ax.axvline(0, color='k', linewidth=0.5, linestyle=':')
    plt.tight_layout()
    path = os.path.join(fig_dir, f'{prefix}_error_dist.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [fig] {path}")


if __name__ == "__main__":
    main()
