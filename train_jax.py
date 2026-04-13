
"""
CGFormer Training with JAX/Flax (LayerNorm version)

Changes from original:
- batch_stats 관리 로직 제거 (LayerNorm은 상태 없음)
- Normalizer 비활성화 옵션 (use_normalizer=False)
- Total energy 학습 지원 (energy_scale=n_atoms)
"""

import argparse
import os
import pickle
import time
from typing import Optional, Dict
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from flax.training import train_state
import optax

from CGFormer_jax import CrystalGraphConvNet
from data_jax import load_dataset_variable, load_dataset_fixed

print("=" * 60)
print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print("=" * 60)


# =============================================================================
# Normalizer (비활성화 가능)
# =============================================================================

class Normalizer:
    """Normalize targets. Can be disabled by setting enabled=False."""

    def __init__(self, data=None, enabled=True):
        self.enabled = enabled
        if enabled and data is not None:
            self.mean = float(jnp.mean(data))
            self.std = float(jnp.std(data))
        else:
            self.mean = 0.0
            self.std = 1.0

    def norm(self, x):
        if not self.enabled:
            return x
        return (x - self.mean) / self.std

    def denorm(self, x):
        if not self.enabled:
            return x
        return x * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std, 'enabled': self.enabled}

    def load_state_dict(self, d):
        self.mean = d['mean']
        self.std = d['std']
        self.enabled = d.get('enabled', True)


# =============================================================================
# Training State (batch_stats 제거)
# =============================================================================

class TrainState(train_state.TrainState):
    """Simple TrainState without batch_stats."""
    pass


def create_train_state(
    model: nn.Module,
    rng: random.PRNGKey,
    dummy_batch: Dict,
    pool_mode: str,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> TrainState:
    """Create initial training state."""
    
    init_kwargs = {
        'atom_fea': dummy_batch['atom_fea'],
        'nbr_fea': dummy_batch['nbr_fea'],
        'nbr_fea_idx': dummy_batch['nbr_fea_idx'],
        'mode': pool_mode,
        'train': False,
    }
    
    if pool_mode == 'fixed':
        init_kwargs['batch_size'] = dummy_batch['batch_size']
        init_kwargs['n_atoms_per_crystal'] = dummy_batch['n_atoms_per_crystal']
    else:
        init_kwargs['segment_ids'] = dummy_batch['segment_ids']
        init_kwargs['num_crystals'] = dummy_batch['num_crystals']
    
    variables = model.init(rng, **init_kwargs)
    params = variables['params']
    # LayerNorm: batch_stats 없음

    tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


# =============================================================================
# Training / Eval Steps
# =============================================================================

@partial(jax.jit, static_argnames=['pool_mode', 'batch_size', 'n_atoms_per_crystal', 'num_crystals'])
def train_step(state, batch, norm_params, pool_mode: str, 
               batch_size=None, n_atoms_per_crystal=None, num_crystals=None,
               dropout_key=None):
    """JIT Compiled Training Step (no batch_stats)."""

    def loss_fn(params):
        variables = {'params': params}

        kwargs = {
            'atom_fea': batch['atom_fea'],
            'nbr_fea': batch['nbr_fea'],
            'nbr_fea_idx': batch['nbr_fea_idx'],
            'mode': pool_mode,
            'train': True,
        }
        
        if pool_mode == 'fixed':
            kwargs['batch_size'] = batch_size
            kwargs['n_atoms_per_crystal'] = n_atoms_per_crystal
        else:
            kwargs['segment_ids'] = batch['segment_ids']
            kwargs['num_crystals'] = num_crystals

        rngs = {}
        if dropout_key is not None:
            rngs['dropout'] = dropout_key

        outputs = state.apply_fn(variables, **kwargs, rngs=rngs)

        # Loss calculation
        targets = batch['target']
        if targets.ndim > 1:
            targets = targets.reshape(-1)
        
        # Normalize if enabled (norm_params['std'] != 1.0)
        if norm_params['std'] != 1.0:
            targets_norm = (targets - norm_params['mean']) / norm_params['std']
        else:
            targets_norm = targets
            
        loss = jnp.mean((outputs.reshape(-1) - targets_norm) ** 2)

        return loss, outputs

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, outputs), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)

    # MAE (denormalized)
    if norm_params['std'] != 1.0:
        preds = outputs.reshape(-1) * norm_params['std'] + norm_params['mean']
    else:
        preds = outputs.reshape(-1)
        
    targets = batch['target']
    if targets.ndim > 1:
        targets = targets.reshape(-1)
    mae = jnp.mean(jnp.abs(preds - targets))

    return state, loss, mae


@partial(jax.jit, static_argnames=['pool_mode', 'batch_size', 'n_atoms_per_crystal', 'num_crystals'])
def eval_step(state, batch, norm_params, pool_mode: str,
              batch_size=None, n_atoms_per_crystal=None, num_crystals=None):
    """JIT Compiled Evaluation Step."""
    variables = {'params': state.params}

    kwargs = {
        'atom_fea': batch['atom_fea'],
        'nbr_fea': batch['nbr_fea'],
        'nbr_fea_idx': batch['nbr_fea_idx'],
        'mode': pool_mode,
        'train': False,
    }
    
    if pool_mode == 'fixed':
        kwargs['batch_size'] = batch_size
        kwargs['n_atoms_per_crystal'] = n_atoms_per_crystal
    else:
        kwargs['segment_ids'] = batch['segment_ids']
        kwargs['num_crystals'] = num_crystals

    outputs = state.apply_fn(variables, **kwargs)

    # Metrics
    targets = batch['target']
    if targets.ndim > 1:
        targets = targets.reshape(-1)
    
    if norm_params['std'] != 1.0:
        targets_norm = (targets - norm_params['mean']) / norm_params['std']
    else:
        targets_norm = targets
        
    loss = jnp.mean((outputs.reshape(-1) - targets_norm) ** 2)

    if norm_params['std'] != 1.0:
        preds = outputs.reshape(-1) * norm_params['std'] + norm_params['mean']
    else:
        preds = outputs.reshape(-1)
        
    mae = jnp.mean(jnp.abs(preds - targets))

    return loss, mae, preds


# =============================================================================
# Save / Load Checkpoint
# =============================================================================

def save_checkpoint(
    path: str,
    state: TrainState,
    normalizer: Normalizer,
    epoch: int,
    pool_mode: str,
    model_config: Dict,
    n_atoms: Optional[int] = None,
):
    """Save checkpoint."""
    checkpoint = {
        'params': jax.device_get(state.params),
        'opt_state': jax.device_get(state.opt_state),
        'step': int(state.step),
        'normalizer': normalizer.state_dict(),
        'epoch': epoch,
        'pool_mode': pool_mode,
        'model_config': model_config,
        'n_atoms': n_atoms,  # MCMC에서 필요할 수 있음
    }
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"Saved: {path}")


def load_checkpoint(path: str) -> Dict:
    """Load checkpoint."""
    with open(path, 'rb') as f:
        checkpoint = pickle.load(f)
    print(f"Loaded: {path}")
    return checkpoint


# =============================================================================
# Training Loop
# =============================================================================

def train(args):
    """Full training loop."""
    print("\n" + "=" * 60)
    print(f"JAX Training (pool_mode={args.pool_mode}, use_normalizer={args.use_normalizer})")
    print("=" * 60)

    # Dataset
    print(f"\nLoading data from: {args.data_dir}")
    if args.pool_mode == 'fixed':
        if args.n_atoms is None:
            raise ValueError("--n_atoms required for fixed mode")
        dataset = load_dataset_fixed(
            args.data_dir,
            n_atoms_per_crystal=args.n_atoms,
            max_num_nbr=args.max_num_nbr,
            radius=args.radius,
            seed=args.seed,
        )
    else:
        dataset = load_dataset_variable(
            args.data_dir,
            max_num_nbr=args.max_num_nbr,
            radius=args.radius,
            seed=args.seed,
        )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Atom feature dim: {dataset.orig_atom_fea_len}")
    print(f"Neighbor feature dim: {dataset.nbr_fea_len}")

    # Split
    train_idx, val_idx, test_idx = dataset.get_split_indices(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Energy scaling (per-atom → total energy)
    energy_scale = args.n_atoms if args.scale_to_total else 1
    if energy_scale != 1:
        print(f"Scaling energy by {energy_scale} (per-atom → total)")
        # 데이터셋의 target을 스케일링
        dataset.scale_targets(energy_scale)

    # Normalizer
    train_targets = jnp.array([dataset[i]['target'][0] for i in train_idx])
    normalizer = Normalizer(train_targets, enabled=args.use_normalizer)
    print(f"Target stats: mean={float(jnp.mean(train_targets)):.4f}, std={float(jnp.std(train_targets)):.4f}")
    print(f"Normalizer: enabled={normalizer.enabled}, mean={normalizer.mean:.4f}, std={normalizer.std:.4f}")

    # Model config
    model_config = {
        'orig_atom_fea_len': dataset.orig_atom_fea_len,
        'nbr_fea_len': dataset.nbr_fea_len,
        'atom_fea_len': args.atom_fea_len,
        'n_conv': args.n_conv,
        'h_fea_len': args.h_fea_len,
        'n_h': args.n_h,
        'graphormer_layers': args.graphormer_layers,
        'num_heads': args.num_heads,
    }
    
    model = CrystalGraphConvNet(**model_config)

    # Initialize
    rng = random.PRNGKey(args.seed)
    rng, init_rng = random.split(rng)
    
    dummy_batch = dataset.get_batch(train_idx[:args.batch_size].tolist())
    
    state = create_train_state(
        model, init_rng, dummy_batch, args.pool_mode,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )
    
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(state.params))
    print(f"Model params: {n_params:,}")

    # Training
    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_val_mae = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_losses, train_maes = [], []
        rng, epoch_rng = random.split(rng)
        norm_params = normalizer.state_dict()
        
        for batch in dataset.iterate_batches(train_idx, args.batch_size, shuffle=True, rng=epoch_rng):
            rng, step_rng = random.split(rng)
            
            if args.pool_mode == 'fixed':
                state, loss, mae = train_step(
                    state, batch, norm_params, args.pool_mode,
                    batch_size=batch['batch_size'],
                    n_atoms_per_crystal=batch['n_atoms_per_crystal'],
                    dropout_key=step_rng
                )
            else:
                state, loss, mae = train_step(
                    state, batch, norm_params, args.pool_mode,
                    num_crystals=batch['num_crystals'],
                    dropout_key=step_rng
                )
            
            train_losses.append(float(loss))
            train_maes.append(float(mae))

        train_loss = np.mean(train_losses)
        train_mae = np.mean(train_maes)

        # Validation
        val_losses, val_maes = [], []
        for batch in dataset.iterate_batches(val_idx, args.batch_size, shuffle=False):
            if args.pool_mode == 'fixed':
                loss, mae, _ = eval_step(
                    state, batch, norm_params, args.pool_mode,
                    batch_size=batch['batch_size'],
                    n_atoms_per_crystal=batch['n_atoms_per_crystal']
                )
            else:
                loss, mae, _ = eval_step(
                    state, batch, norm_params, args.pool_mode,
                    num_crystals=batch['num_crystals']
                )
            val_losses.append(float(loss))
            val_maes.append(float(mae))

        val_loss = np.mean(val_losses)
        val_mae = np.mean(val_maes)
        
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {train_loss:.4f} MAE: {train_mae:.4f} | "
              f"Val Loss: {val_loss:.4f} MAE: {val_mae:.4f} | "
              f"Time: {epoch_time:.1f}s")

        # Checkpoint
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            save_checkpoint(
                os.path.join(args.ckpt_dir, 'best.pkl'),
                state, normalizer, epoch, args.pool_mode, model_config,
                n_atoms=args.n_atoms
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Test
    print("\n" + "=" * 60)
    print("Testing")
    print("=" * 60)

    ckpt = load_checkpoint(os.path.join(args.ckpt_dir, 'best.pkl'))
    state = state.replace(params=ckpt['params'])
    normalizer.load_state_dict(ckpt['normalizer'])

    test_maes = []
    norm_params = normalizer.state_dict()
    for batch in dataset.iterate_batches(test_idx, args.batch_size, shuffle=False):
        if args.pool_mode == 'fixed':
            loss, mae, _ = eval_step(
                state, batch, norm_params, args.pool_mode,
                batch_size=batch['batch_size'],
                n_atoms_per_crystal=batch['n_atoms_per_crystal']
            )
        else:
            loss, mae, _ = eval_step(
                state, batch, norm_params, args.pool_mode,
                num_crystals=batch['num_crystals']
            )
        test_maes.append(float(mae))

    test_mae = np.mean(test_maes)
    print(f"Test MAE: {test_mae:.4f}")

    return state, normalizer


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict'])
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints')
    parser.add_argument('--ckpt', type=str, default=None)

    # Pool mode
    parser.add_argument('--pool_mode', type=str, default='variable', choices=['variable', 'fixed'])
    parser.add_argument('--n_atoms', type=int, default=None)

    # Energy scaling
    parser.add_argument('--scale_to_total', action='store_true',
                        help='Scale per-atom energy to total energy (multiply by n_atoms)')
    parser.add_argument('--use_normalizer', action='store_true',
                        help='Enable target normalization')

    # Data
    parser.add_argument('--max_num_nbr', type=int, default=12)
    parser.add_argument('--radius', type=float, default=8.0)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)

    # Model
    parser.add_argument('--atom_fea_len', type=int, default=64)
    parser.add_argument('--n_conv', type=int, default=3)
    parser.add_argument('--h_fea_len', type=int, default=128)
    parser.add_argument('--n_h', type=int, default=1)
    parser.add_argument('--graphormer_layers', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=4)

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)


if __name__ == '__main__':
    main()