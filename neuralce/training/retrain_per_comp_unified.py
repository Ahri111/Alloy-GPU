"""
retrain_per_comp.py — Composition-specific retrain with warm-start

Strategy:
  1. Full data ablation → best HP
  2. Full data retrain  → best params (warm-start source)
  3. This script: filter by composition → warm-start from (2) → fine-tune

Usage:
  # Single composition
  python retrain_per_comp.py \
      --config ./configs/tuning/stfo_wo_spin.yaml \
      --checkpoint ./retrained_stfo_wo_spin_ising_lite.pkl \
      --comp 250 \
      --epochs 3000

  # Multiple compositions in one run
  python retrain_per_comp.py \
      --config ./configs/tuning/stfo_wo_spin.yaml \
      --checkpoint ./retrained_stfo_wo_spin_ising_lite.pkl \
      --comp 250 500 750 \
      --epochs 3000

  # Custom LR scale (default: 0.1× of checkpoint LR)
  python retrain_per_comp.py \
      --config ... --checkpoint ... --comp 250 \
      --lr_scale 0.05 --epochs 5000

  # No warm-start (random init, for comparison)
  python retrain_per_comp.py \
      --config ... --checkpoint ... --comp 250 \
      --no-resume --epochs 3000
"""

import os, argparse, pickle, copy, re
import numpy as np
import yaml

import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from pymatgen.core.structure import Structure
from neuralce.utils.cif_utils import load_cif_safe, get_specie_number

from neuralce.models.NeuralCE_jax import (create_neuralce, is_spin_model, is_sisj_model,
                                          LITE_MODELS)


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Retrain per composition with warm-start from full retrain.")
    p.add_argument("--config", type=str, required=True,
                   help="YAML config (same as ablation/retrain)")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Checkpoint from full retrain (warm-start source)")
    p.add_argument("--comp", type=int, nargs='+', required=True,
                   help="Composition code(s) to train. e.g., 250 500 750")
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr_scale", type=float, default=0.1,
                   help="LR multiplier relative to checkpoint LR (default: 0.1)")
    p.add_argument("--no-resume", action="store_true",
                   help="Random init instead of warm-start")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory for per-comp checkpoints")
    p.add_argument("--patience", type=int, default=None,
                   help="Early stopping patience (default: no early stopping)")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# DATA & GRAPH (from retrain.py)
# ═══════════════════════════════════════════════════════════════════════

def load_data(cif_dir, csv_path, spin_pkl_path, id_col, comp_regex, exclude_z=None):
    if exclude_z is None:
        exclude_z = set()

    df = pd.read_csv(csv_path)
    if id_col not in df.columns:
        id_col = 'id' if 'id' in df.columns else 'cif_id'
    energy_map = dict(zip(df[id_col].astype(str), df['total_energy'].values))

    if spin_pkl_path and os.path.exists(spin_pkl_path):
        spin_df = pd.read_pickle(spin_pkl_path)
        spin_map = dict(zip(spin_df['cif_id'].astype(str),
                            spin_df['spin_states'].values))
    else:
        spin_map = {}

    structures = []
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
                     include_sisj=False, shell_edges=None):
    lists = {'atom_fea': [], 'nbr_fea': [], 'nbr_fea_idx': [],
             'spin_fea': [], 'energies': [], 'comps': []}

    for s in structures:
        af, nf, nfi, sf = build_graph_lite(
            s, cutoff, n_shells, max_num_nbr, species_map,
            include_sisj, shell_edges)
        lists['atom_fea'].append(af)
        lists['nbr_fea'].append(nf)
        lists['nbr_fea_idx'].append(nfi)
        lists['spin_fea'].append(sf)
        lists['energies'].append(s['total_energy'])
        lists['comps'].append(s['comp_code'])

    return {
        'atom_fea':    jnp.array(np.stack(lists['atom_fea'])),
        'nbr_fea':     jnp.array(np.stack(lists['nbr_fea'])),
        'nbr_fea_idx': jnp.array(np.stack(lists['nbr_fea_idx'])),
        'spin_fea':    jnp.array(np.stack(lists['spin_fea'])),
        'energies':    jnp.array(lists['energies']),
        'comps':       lists['comps'],
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

    return create_neuralce(model_type=model_name, pool_mode='fixed',
                           readout_type='sum', **kwargs)


# ═══════════════════════════════════════════════════════════════════════
# TRAIN ONE COMPOSITION
# ═══════════════════════════════════════════════════════════════════════

def train_one_comp(comp_code, structures, cfg, hp, model_name, ckpt_params,
                   args, species_map, shell_edges):
    """Train on a single composition subset."""

    # ── Filter by composition ─────────────────────────────────────────
    comp_structs = [s for s in structures if s['comp_code'] == comp_code]
    if len(comp_structs) < 5:
        print(f"  [!] comp={comp_code}: only {len(comp_structs)} structures, skip")
        return None

    n_atoms_set = set(s['n_atoms'] for s in comp_structs)
    if len(n_atoms_set) != 1:
        print(f"  [!] comp={comp_code}: variable atom counts {sorted(n_atoms_set)}, skip")
        return None
    N_ATOMS = n_atoms_set.pop()

    n_atoms_orig = cfg['n_atoms']
    _abl = cfg.get('ablation', {})
    seed = _abl.get('seed', 42)
    batch_size = args.batch_size or _abl.get('batch_size', 32)

    print(f"\n{'═' * 70}")
    print(f"  COMPOSITION: {comp_code} ({len(comp_structs)} structures)")
    print(f"  Warm-start: {'NO (random init)' if args.no_resume else 'YES'}")
    print(f"  LR: {hp['lr']} × {args.lr_scale} = {hp['lr'] * args.lr_scale:.6f}")
    print(f"{'═' * 70}")

    # ── Build graphs ──────────────────────────────────────────────────
    sisj = is_sisj_model(model_name)
    dataset = build_all_graphs(
        comp_structs, hp['cutoff'], hp['n_shells'], hp['max_num_nbr'],
        species_map, include_sisj=sisj, shell_edges=shell_edges)

    # ── Split (80/20 for small datasets) ──────────────────────────────
    from sklearn.model_selection import train_test_split
    indices = list(range(len(comp_structs)))

    # Use 80/20 train/val split (no separate test — test is the entire comp)
    val_frac = 0.2 if len(comp_structs) > 20 else 0.15
    train_idx, val_idx = train_test_split(
        indices, test_size=val_frac, random_state=seed)
    train_idx, val_idx = np.array(train_idx), np.array(val_idx)
    print(f"  Train: {len(train_idx)} | Val: {len(val_idx)}")

    # ── Create model ──────────────────────────────────────────────────
    use_spin = is_spin_model(model_name)
    model = create_model(model_name, hp)
    rng = jax.random.PRNGKey(seed)

    dummy_af = dataset['atom_fea'][0]
    dummy_nf = dataset['nbr_fea'][0]
    dummy_nfi = dataset['nbr_fea_idx'][0]

    init_kw = {
        'atom_fea': dummy_af, 'nbr_fea': dummy_nf, 'nbr_fea_idx': dummy_nfi,
        'batch_size': 1, 'n_atoms_per_crystal': N_ATOMS,
    }
    if use_spin:
        init_kw['atom_spins'] = dataset['spin_fea'][0]

    params = model.init(rng, **init_kw)

    # Warm-start: load params from full retrain
    if not args.no_resume and ckpt_params is not None:
        params = ckpt_params
        print(f"  Warm-start from checkpoint params")
    else:
        print(f"  Random init")

    n_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"  Params: {n_params:,}")

    # ── Optimizer (reduced LR for fine-tune) ──────────────────────────
    lr = hp['lr'] * args.lr_scale
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.1, peak_value=lr,
        warmup_steps=30, decay_steps=args.epochs, end_value=lr * 0.01)
    optimizer = optax.chain(optax.clip_by_global_norm(5.0), optax.adamw(schedule))

    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer)

    # ── Forward helpers ───────────────────────────────────────────────
    def _forward(state, af, nf, nfi, sf):
        B = af.shape[0]
        af_flat = af.reshape(-1, af.shape[-1])
        offsets = jnp.arange(B)[:, None, None] * N_ATOMS
        nfi_flat = (nfi + offsets).reshape(-1, nfi.shape[-1])
        nf_flat = nf.reshape(-1, nf.shape[-2], nf.shape[-1])
        kw = {'atom_fea': af_flat, 'nbr_fea': nf_flat, 'nbr_fea_idx': nfi_flat,
              'batch_size': B, 'n_atoms_per_crystal': N_ATOMS}
        if use_spin:
            kw['atom_spins'] = sf.reshape(-1, sf.shape[-1])
        return state.apply_fn(state.params, **kw).squeeze(-1)

    def loss_fn(params, state, batch):
        af, nf, nfi, sf, targets = batch
        B = af.shape[0]
        af_flat = af.reshape(-1, af.shape[-1])
        offsets = jnp.arange(B)[:, None, None] * N_ATOMS
        nfi_flat = (nfi + offsets).reshape(-1, nfi.shape[-1])
        nf_flat = nf.reshape(-1, nf.shape[-2], nf.shape[-1])
        kw = {'atom_fea': af_flat, 'nbr_fea': nf_flat, 'nbr_fea_idx': nfi_flat,
              'batch_size': B, 'n_atoms_per_crystal': N_ATOMS}
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
            idx = indices[start:start + batch_size]
            af = dataset['atom_fea'][idx]
            nf = dataset['nbr_fea'][idx]
            nfi = dataset['nbr_fea_idx'][idx]
            sf = dataset['spin_fea'][idx]
            targets = dataset['energies'][idx]
            B = af.shape[0]
            pred = _forward(state, af, nf, nfi, sf)
            mse = float(jnp.mean((pred - targets) ** 2))
            total_loss += mse * B
            n_total += B
            all_preds.append(np.array(pred))
            all_targets.append(np.array(targets))
        return (total_loss / max(n_total, 1),
                np.concatenate(all_preds), np.concatenate(all_targets))

    # ── Training loop ─────────────────────────────────────────────────
    # Best model = max (SRCC + R²) / 2 on val set
    best_score = -float('inf')
    best_val = float('inf')
    best_params = None
    best_epoch = 0
    patience_ctr = 0

    for epoch in range(1, args.epochs + 1):
        perm = np.array(jax.random.permutation(
            jax.random.PRNGKey(seed + epoch), jnp.array(train_idx)))

        epoch_loss, n_bat = 0.0, 0
        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            if len(idx) < 2:
                continue
            batch = (dataset['atom_fea'][idx], dataset['nbr_fea'][idx],
                     dataset['nbr_fea_idx'][idx], dataset['spin_fea'][idx],
                     dataset['energies'][idx])
            state, loss = train_step(state, batch)
            epoch_loss += float(loss)
            n_bat += 1
        epoch_loss /= max(n_bat, 1)

        val_loss, val_preds, val_targets = eval_set(state, val_idx)

        if len(val_targets) >= 3:
            rho = spearmanr(val_targets, val_preds).correlation
            r2 = r2_score(val_targets, val_preds)
            if np.isnan(rho) or np.isnan(r2):
                val_score = -float('inf')
            else:
                val_score = (rho + r2) / 2.0
        else:
            val_score = -float('inf')

        if val_score > best_score:
            best_score = val_score
            best_val = val_loss
            best_params = copy.deepcopy(state.params)
            best_epoch = epoch
            patience_ctr = 0
        else:
            patience_ctr += 1

        if args.patience and patience_ctr >= args.patience:
            print(f"  Early stop @ epoch {epoch} (patience={args.patience})")
            break

        if epoch % 200 == 0:
            print(f"  [epoch {epoch:5d}] train={epoch_loss:.6f} "
                  f"val={val_loss:.6f} val_score={val_score:.4f} "
                  f"best_score={best_score:.4f}@{best_epoch}")

    print(f"  Training done. Best val_score=(SRCC+R²)/2={best_score:.4f} "
          f"(val_MSE={best_val:.6f}) @ epoch {best_epoch}")

    # ── Evaluate on ALL data for this composition ─────────────────────
    state = state.replace(params=best_params)
    all_idx = np.arange(len(comp_structs))
    _, preds_all, targets_all = eval_set(state, all_idx)

    rmse = np.sqrt(np.mean((targets_all - preds_all) ** 2))
    mae = np.mean(np.abs(targets_all - preds_all))
    srcc = spearmanr(targets_all, preds_all).correlation
    rmse_pa = np.sqrt(np.mean(((targets_all - preds_all) / n_atoms_orig) ** 2))
    mae_pa = np.mean(np.abs((targets_all - preds_all) / n_atoms_orig))

    print(f"\n  ALL DATA (comp={comp_code}):")
    print(f"    RMSE={rmse:.4f} eV | MAE={mae:.4f} | SRCC={srcc:.4f}")
    print(f"    RMSE={rmse_pa:.6f} eV/at | MAE={mae_pa:.6f} eV/at")

    # ── Save ──────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    dataset_name = cfg['dataset_name']
    out_name = f'{dataset_name}_comp{comp_code}_{model_name}.pkl'
    out_path = os.path.join(args.output_dir, out_name)

    with open(out_path, 'wb') as f:
        pickle.dump({
            'params': best_params,
            'hp': hp,
            'model_name': model_name,
            'comp_code': comp_code,
            'n_structures': len(comp_structs),
            'best_epoch': best_epoch,
            'best_val_mse': float(best_val),
            'best_val_score': float(best_score),
            'all_rmse': float(rmse),
            'all_mae': float(mae),
            'all_srcc': float(srcc),
            'retrain_epochs': args.epochs,
            'warm_start': not args.no_resume,
            'lr_scale': args.lr_scale,
        }, f)
    print(f"  Saved → {out_path}")

    return {
        'comp_code': comp_code, 'n_structures': len(comp_structs),
        'srcc': srcc, 'rmse_pa': rmse_pa, 'mae_pa': mae_pa,
        'best_epoch': best_epoch, 'path': out_path,
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── Load config ───────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    args.output_dir = (args.output_dir
                       or cfg.get('per_comp', {}).get('output_dir')
                       or './best_pkl/per_comp')

    cif_dir    = cfg['cif_dir']
    csv_path   = cfg['csv_path']
    spin_pkl   = cfg.get('spin_pkl')
    id_col     = cfg.get('id_col', 'id')
    comp_regex = cfg.get('comp_regex', r'_(\d+)')
    species_map = {int(k): v for k, v in cfg['species_map'].items()}
    exclude_z  = set(cfg.get('exclude_species', []))

    # ── Load checkpoint ───────────────────────────────────────────────
    with open(args.checkpoint, 'rb') as f:
        ckpt = pickle.load(f)
    hp = ckpt['hp']
    ckpt_params = ckpt.get('params')

    # Infer model name
    model_name = ckpt.get('model_name')
    if model_name is None:
        ckpt_base = os.path.basename(args.checkpoint).replace('.pkl', '')
        model_name = '_'.join(ckpt_base.split('_')[2:])
        _abl = cfg.get('ablation', {})
        if model_name not in LITE_MODELS:
            for m in _abl.get('run_models', []):
                if m in ckpt_base:
                    model_name = m
                    break

    print(f"{'═' * 70}")
    print(f"  RETRAIN PER COMPOSITION")
    print(f"  Model: {model_name}")
    print(f"  HP: {hp}")
    print(f"  Compositions: {args.comp}")
    print(f"  Epochs: {args.epochs}")
    print(f"  LR scale: {args.lr_scale}")
    print(f"  Warm-start: {not args.no_resume}")
    print(f"{'═' * 70}")

    # ── Resolve shell_edges ───────────────────────────────────────────
    _graph = cfg.get('graph', {})
    _candidates = _graph.get('candidates')
    shell_edges = None

    if _candidates and hp['cutoff'] in {float(k) for k in _candidates}:
        info = _candidates.get(hp['cutoff']) or _candidates.get(str(hp['cutoff']))
        shell_edges = info.get('shell_edges') if info else None
    elif _graph.get('shell_edges'):
        shell_edges = _graph['shell_edges']

    # ── Load ALL data (filter per comp inside loop) ───────────────────
    print("\nLoading data ...")
    structures = load_data(cif_dir, csv_path, spin_pkl, id_col, comp_regex, exclude_z)
    print(f"Total: {len(structures)} structures")

    comp_counts = {}
    for s in structures:
        comp_counts[s['comp_code']] = comp_counts.get(s['comp_code'], 0) + 1
    print(f"Available compositions: {dict(sorted(comp_counts.items()))}")

    # ── Train each composition ────────────────────────────────────────
    results = []
    for comp_code in args.comp:
        if comp_code not in comp_counts:
            print(f"\n  [!] comp={comp_code} not found in data, skip")
            continue
        r = train_one_comp(
            comp_code, structures, cfg, hp, model_name, ckpt_params,
            args, species_map, shell_edges)
        if r:
            results.append(r)

    # ── Summary ───────────────────────────────────────────────────────
    if results:
        print(f"\n{'═' * 70}")
        print(f"  SUMMARY")
        print(f"{'═' * 70}")
        print(f"  {'Comp':>6} {'N':>5} {'SRCC':>7} {'RMSE(eV/at)':>12} "
              f"{'MAE(eV/at)':>12} {'Epoch':>6}")
        print(f"  {'-'*6} {'-'*5} {'-'*7} {'-'*12} {'-'*12} {'-'*6}")
        for r in results:
            print(f"  {r['comp_code']:>6} {r['n_structures']:>5} "
                  f"{r['srcc']:>7.4f} {r['rmse_pa']:>12.6f} "
                  f"{r['mae_pa']:>12.6f} {r['best_epoch']:>6}")
        print(f"\n  Checkpoints → {args.output_dir}/")
        print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
