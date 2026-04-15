import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, Any, Optional


class FixedSumPool(nn.Module):
    @nn.compact
    def __call__(self, site_energy, **kwargs):
        batch_size = kwargs['batch_size']
        n_atoms = kwargs['n_atoms_per_crystal']
        site_E = site_energy.reshape(batch_size, n_atoms)
        return jnp.sum(site_E, axis=1, keepdims=True)


class FixedMeanPool(nn.Module):
    @nn.compact
    def __call__(self, site_energy, **kwargs):
        batch_size = kwargs['batch_size']
        n_atoms = kwargs['n_atoms_per_crystal']
        site_E = site_energy.reshape(batch_size, n_atoms)
        return jnp.mean(site_E, axis=1, keepdims=True)


class VariableSumPool(nn.Module):
    @nn.compact
    def __call__(self, site_energy, **kwargs):
        segment_ids = kwargs['segment_ids']
        num_crystals = kwargs['num_crystals']
        return jax.ops.segment_sum(
            site_energy.squeeze(-1),
            segment_ids,
            num_segments=num_crystals
        )[:, None]


class VariableMeanPool(nn.Module):
    @nn.compact
    def __call__(self, site_energy, **kwargs):
        segment_ids = kwargs['segment_ids']
        num_crystals = kwargs['num_crystals']
        flat_E = site_energy.squeeze(-1)
        total_E = jax.ops.segment_sum(flat_E, segment_ids, num_segments=num_crystals)
        ones = jnp.ones_like(flat_E)
        counts = jax.ops.segment_sum(ones, segment_ids, num_segments=num_crystals)
        mean_E = total_E / jnp.maximum(counts, 1.0)
        return mean_E[:, None]


class CEInteractionLayer(nn.Module):
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


class NeuralCE_Ising(nn.Module):
    pooler_cls: Callable
    atom_fea_len: int = 64
    nbr_fea_len: int = 41
    n_conv: int = 3
    h_fea_len: int = 128

    @nn.compact
    def __call__(self, atom_fea, nbr_fea, nbr_fea_idx, **kwargs):
        atom_fea = nn.Dense(self.atom_fea_len)(atom_fea)
        for _ in range(self.n_conv):
            atom_fea = CEInteractionLayer(self.atom_fea_len)(atom_fea, nbr_fea, nbr_fea_idx)
        E_site = nn.Dense(1)(nn.softplus(nn.Dense(self.h_fea_len)(atom_fea)))
        return self.pooler_cls()(E_site, **kwargs)


def create_neuralce(model_type='ising', pool_mode='fixed', readout_type='sum', **kwargs):
    pooler_map = {
        ('fixed', 'sum'): FixedSumPool,
        ('fixed', 'mean'): FixedMeanPool,
        ('variable', 'sum'): VariableSumPool,
        ('variable', 'mean'): VariableMeanPool,
    }
    pooler = pooler_map[(pool_mode, readout_type)]
    valid_fields = {'atom_fea_len', 'nbr_fea_len', 'n_conv', 'h_fea_len'}
    filtered = {k: v for k, v in kwargs.items() if k in valid_fields}
    return NeuralCE_Ising(pooler_cls=pooler, **filtered)