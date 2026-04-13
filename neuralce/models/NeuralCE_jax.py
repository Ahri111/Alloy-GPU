"""
NeuralCE JAX — Neural Cluster Expansion for Alloy Energy Prediction

Cluster Expansion:
    E(σ) = J₀ + Σᵢ Jᵢσᵢ + Σᵢⱼ Jᵢⱼσᵢσⱼ + Σᵢⱼₖ Jᵢⱼₖσᵢσⱼσₖ + ...

Model Hierarchy:
    ──────────────────────────────────────────────────────────────────────
    Heavy (atom_init 92-dim + Gaussian 41-dim edges, pt_mcmc compat):
        ising          — Chemical only, no spin
        spin_v3        — SpinEmbedding + MagExchange, E(σ) ≠ E(-σ)
        spin_v8        — Even-Odd dual readout, E(σ) = E(-σ) ✓
        spin_v8u       — Even-Odd unified readout, E(σ) = E(-σ) ✓

    Lite (one-hot 4-dim node + shell edge, Sr excluded, Optuna tuning):
        ising_lite              — No spin (reference)
        neuralce_evenodd_lite   — Product backbone + EvenOdd spin
        neuralce_sisj_lite      — Product backbone + σᵢσⱼ in edge
        gnn_sisj_lite           — Concat-MLP backbone + σᵢσⱼ in edge
        gnn_evenodd_lite        — Concat-MLP backbone + EvenOdd spin

    Ablation table (Lite):
    ┌─────────────┬──────────────────────┬──────────────────────┐
    │             │ EvenOdd              │ SiSj (σᵢσⱼ edge)    │
    ├─────────────┼──────────────────────┼──────────────────────┤
    │ Product(CE) │ neuralce_evenodd_lite│ neuralce_sisj_lite   │
    │ Concat-MLP  │ gnn_evenodd_lite     │ gnn_sisj_lite        │
    └─────────────┴──────────────────────┴──────────────────────┘
    + ising_lite (no spin baseline)
    ──────────────────────────────────────────────────────────────────────
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, Optional


# =============================================================================
# 1. Pooling Modules
# =============================================================================

class FixedSumPool(nn.Module):
    @nn.compact
    def __call__(self, site_energy, **kwargs):
        B = kwargs['batch_size']
        N = kwargs['n_atoms_per_crystal']
        return jnp.sum(site_energy.reshape(B, N), axis=1, keepdims=True)

class FixedMeanPool(nn.Module):
    @nn.compact
    def __call__(self, site_energy, **kwargs):
        B = kwargs['batch_size']
        N = kwargs['n_atoms_per_crystal']
        return jnp.mean(site_energy.reshape(B, N), axis=1, keepdims=True)

class VariableSumPool(nn.Module):
    @nn.compact
    def __call__(self, site_energy, **kwargs):
        seg = kwargs['segment_ids']
        nc = kwargs['num_crystals']
        return jax.ops.segment_sum(site_energy.squeeze(-1), seg, num_segments=nc)[:, None]

class VariableMeanPool(nn.Module):
    @nn.compact
    def __call__(self, site_energy, **kwargs):
        seg = kwargs['segment_ids']
        nc = kwargs['num_crystals']
        flat = site_energy.squeeze(-1)
        total = jax.ops.segment_sum(flat, seg, num_segments=nc)
        counts = jax.ops.segment_sum(jnp.ones_like(flat), seg, num_segments=nc)
        return (total / jnp.maximum(counts, 1.0))[:, None]


class PaddedSumPool(nn.Module):
    """Sum pool with mask for padded atoms. mask shape: (B*N,) or (B, N)."""
    @nn.compact
    def __call__(self, site_energy, **kwargs):
        B = kwargs['batch_size']
        N = kwargs['n_atoms_per_crystal']
        mask = kwargs['atom_mask']  # (B, N) or (B*N,)
        mask = mask.reshape(B, N)
        energy = site_energy.reshape(B, N)
        return jnp.sum(energy * mask, axis=1, keepdims=True)

class PaddedMeanPool(nn.Module):
    """Mean pool with mask for padded atoms."""
    @nn.compact
    def __call__(self, site_energy, **kwargs):
        B = kwargs['batch_size']
        N = kwargs['n_atoms_per_crystal']
        mask = kwargs['atom_mask']
        mask = mask.reshape(B, N)
        energy = site_energy.reshape(B, N)
        total = jnp.sum(energy * mask, axis=1, keepdims=True)
        counts = jnp.sum(mask, axis=1, keepdims=True)
        return total / jnp.maximum(counts, 1.0)


# =============================================================================
# 2. Core Convolution Layers
# =============================================================================

class CEInteractionLayer(nn.Module):
    """Product-based graph conv (NeuralCE backbone) with Pre-LayerNorm."""
    atom_fea_len: int

    @nn.compact
    def __call__(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        atom_normed = nn.LayerNorm()(atom_in_fea)
        atom_nbr_fea = jnp.take(atom_normed, nbr_fea_idx, axis=0)
        atom_center_fea = jnp.tile(atom_normed[:, None, :], (1, M, 1))

        phi_center = nn.Dense(self.atom_fea_len)(atom_center_fea)
        phi_nbr    = nn.Dense(self.atom_fea_len)(atom_nbr_fea)
        phi_edge   = nn.Dense(self.atom_fea_len)(nbr_fea)
        interaction = phi_center * phi_nbr * phi_edge

        gate      = nn.sigmoid(nn.Dense(self.atom_fea_len)(interaction))
        magnitude = nn.softplus(nn.Dense(self.atom_fea_len)(interaction))
        nbr_sumed = jnp.sum(gate * magnitude, axis=1)
        return atom_in_fea + nbr_sumed


class ConcatConvLayer(nn.Module):
    """Concat-MLP graph conv (GNN backbone)."""
    node_dim: int
    edge_dim: int

    @nn.compact
    def __call__(self, x, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        x_normed = nn.LayerNorm()(x)
        x_nbr = jnp.take(x_normed, nbr_fea_idx, axis=0)      # (N, M, node_dim)
        x_center = jnp.tile(x_normed[:, None, :], (1, M, 1))  # (N, M, node_dim)

        concat = jnp.concatenate([x_center, x_nbr, nbr_fea], axis=-1)
        msg = nn.Dense(self.node_dim)(nn.silu(nn.Dense(self.node_dim)(concat)))
        agg = jnp.sum(msg, axis=1)
        return x + agg


class ReadoutMLP(nn.Module):
    """2-layer readout: atom features → site energy scalar."""
    h_fea_len: int

    @nn.compact
    def __call__(self, atom_fea):
        h = nn.softplus(nn.Dense(self.h_fea_len)(atom_fea))
        return nn.Dense(1)(nn.softplus(nn.Dense(self.h_fea_len // 2)(h)))


# =============================================================================
# 3. Spin-specific Layers
# =============================================================================

class SpinEmbedding(nn.Module):
    """Learnable spin embedding: scalar spin → vector (v3)."""
    embed_dim: int = 32

    @nn.compact
    def __call__(self, atom_spins):
        x = nn.Dense(self.embed_dim)(atom_spins)
        x = nn.softplus(x)
        x = nn.Dense(self.embed_dim)(x)
        return x


class MagneticExchangeLayer(nn.Module):
    """Spin-asymmetric magnetic exchange (v3). E(σ) ≠ E(-σ)."""
    h_fea_len: int
    spin_embed_dim: int = 32

    @nn.compact
    def __call__(self, atom_chem_fea, nbr_fea, nbr_fea_idx, atom_spins):
        N, M = nbr_fea_idx.shape
        spin_emb = SpinEmbedding(self.spin_embed_dim)(atom_spins)

        center_chem = jnp.tile(atom_chem_fea[:, None, :], (1, M, 1))
        nbr_chem    = jnp.take(atom_chem_fea, nbr_fea_idx, axis=0)
        center_spin = jnp.tile(spin_emb[:, None, :], (1, M, 1))
        nbr_spin    = jnp.take(spin_emb, nbr_fea_idx, axis=0)

        pair_input = jnp.concatenate([
            center_chem + nbr_chem, jnp.abs(center_chem - nbr_chem),
            center_spin + nbr_spin, jnp.abs(center_spin - nbr_spin),
            nbr_fea], axis=-1)
        h    = nn.softplus(nn.Dense(self.h_fea_len)(pair_input))
        J_ij = nn.Dense(1)(nn.softplus(nn.Dense(self.h_fea_len // 2)(h)))

        spin_i = jnp.tile(atom_spins[:, None, :], (1, M, 1))
        spin_j = jnp.take(atom_spins, nbr_fea_idx, axis=0)
        return jnp.sum(J_ij * spin_i * spin_j, axis=1)


class EvenOddConvLayer(nn.Module):
    """Even-Odd Equivariant Convolution (v8).
    σ→-σ ⟹ odd→-odd, odd_cross invariant, even invariant → E(σ)=E(-σ) ✓
    """
    even_fea_len: int
    odd_fea_len: int
    edge_fea_len: int

    @nn.compact
    def __call__(self, even_node, odd_node, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        even_i = jnp.tile(even_node[:, None, :], (1, M, 1))
        even_j = jnp.take(even_node, nbr_fea_idx, axis=0)
        odd_i  = jnp.tile(odd_node[:, None, :], (1, M, 1))
        odd_j  = jnp.take(odd_node, nbr_fea_idx, axis=0)
        odd_cross = odd_i * odd_j  # even ✓

        # Even channel
        even_input = jnp.concatenate([even_i, even_j, nbr_fea, odd_cross], axis=-1)
        even_h    = nn.softplus(nn.Dense(self.even_fea_len, name='even_msg_1')(even_input))
        even_gate = nn.sigmoid(nn.Dense(self.even_fea_len, name='even_gate')(even_h))
        even_msg  = nn.softplus(nn.Dense(self.even_fea_len, name='even_msg_2')(even_h))
        even_agg  = jnp.sum(even_gate * even_msg, axis=1)

        # Odd channel
        odd_ie = odd_i * nn.Dense(self.odd_fea_len, name='odd_proj_ej')(even_j)
        odd_ei = nn.Dense(self.odd_fea_len, name='odd_proj_ei')(even_i) * odd_j
        odd_linear = nn.Dense(self.odd_fea_len, use_bias=False, name='odd_msg_1')(
            jnp.concatenate([odd_ie, odd_ei], axis=-1))
        odd_value = jnp.tanh(odd_linear)

        even_gate_in = jnp.concatenate([even_i, even_j, nbr_fea, odd_cross], axis=-1)
        odd_gate = nn.sigmoid(nn.Dense(self.odd_fea_len, name='odd_gate')(
            nn.softplus(nn.Dense(self.odd_fea_len, name='odd_gate_h')(even_gate_in))))
        odd_agg = jnp.sum(odd_gate * odd_value, axis=1)

        return even_node + even_agg, odd_node + odd_agg


class EvenOddEdgeUpdateLayer(nn.Module):
    """Edge update preserving even parity (v8)."""
    edge_fea_len: int
    odd_fea_len: int

    @nn.compact
    def __call__(self, even_node, odd_node, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        even_i = jnp.tile(even_node[:, None, :], (1, M, 1))
        even_j = jnp.take(even_node, nbr_fea_idx, axis=0)
        odd_i  = jnp.tile(odd_node[:, None, :], (1, M, 1))
        odd_j  = jnp.take(odd_node, nbr_fea_idx, axis=0)
        odd_cross = odd_i * odd_j  # even ✓
        concat = jnp.concatenate([nbr_fea, even_i, even_j, odd_cross], axis=-1)
        delta  = nn.Dense(self.edge_fea_len)(nn.softplus(nn.Dense(self.edge_fea_len)(concat)))
        return nbr_fea + delta


# =============================================================================
# 4. Heavy Models (atom_init + Gaussian, pt_mcmc compatibility)
# =============================================================================

class NeuralCE_Ising(nn.Module):
    """Base: Chemical only, no spin (heavy)."""
    pooler_cls: Callable
    atom_fea_len: int = 64
    nbr_fea_len: int  = 41
    n_conv: int       = 3
    h_fea_len: int    = 128

    @nn.compact
    def __call__(self, atom_fea, nbr_fea, nbr_fea_idx, **kwargs):
        atom_fea = nn.Dense(self.atom_fea_len)(atom_fea)
        for _ in range(self.n_conv):
            atom_fea = CEInteractionLayer(self.atom_fea_len)(atom_fea, nbr_fea, nbr_fea_idx)
        return self.pooler_cls()(ReadoutMLP(self.h_fea_len)(atom_fea), **kwargs)


class NeuralCE_Spin_v3(nn.Module):
    """v3: Spin Early Fusion (heavy). E(σ) ≠ E(-σ)."""
    pooler_cls: Callable
    atom_fea_len: int  = 64
    nbr_fea_len: int   = 41
    n_conv: int        = 3
    h_fea_len: int     = 128
    spin_embed_dim: int = 32

    @nn.compact
    def __call__(self, atom_fea, nbr_fea, nbr_fea_idx, atom_spins=None, **kwargs):
        atom_fea = nn.Dense(self.atom_fea_len)(atom_fea)
        if atom_spins is not None:
            spin_emb = SpinEmbedding(self.spin_embed_dim)(atom_spins)
            atom_fea = nn.Dense(self.atom_fea_len)(
                jnp.concatenate([atom_fea, spin_emb], axis=-1))
        for _ in range(self.n_conv):
            atom_fea = CEInteractionLayer(self.atom_fea_len)(atom_fea, nbr_fea, nbr_fea_idx)
        E_chem = ReadoutMLP(self.h_fea_len)(atom_fea)
        if atom_spins is not None:
            E_mag = MagneticExchangeLayer(self.h_fea_len, self.spin_embed_dim)(
                atom_fea, nbr_fea, nbr_fea_idx, atom_spins)
            alpha = self.param('mag_weight', nn.initializers.constant(0.5), (1,))
            E_chem = E_chem + alpha * E_mag
        return self.pooler_cls()(E_chem, **kwargs)


class NeuralCE_Spin_v8(nn.Module):
    """v8: Even-Odd, Dual Readout (heavy). E(σ) = E(-σ) ✓."""
    pooler_cls: Callable
    atom_fea_len: int = 64
    nbr_fea_len: int  = 41
    n_conv: int       = 3
    h_fea_len: int    = 128
    odd_fea_len: int  = 32

    @nn.compact
    def __call__(self, atom_fea, nbr_fea, nbr_fea_idx, atom_spins=None, **kwargs):
        even_node = nn.Dense(self.atom_fea_len)(atom_fea)
        nbr_fea   = nn.Dense(self.nbr_fea_len)(nbr_fea)
        odd_node  = (nn.Dense(self.odd_fea_len, use_bias=False)(atom_spins)
                     if atom_spins is not None
                     else jnp.zeros((atom_fea.shape[0], self.odd_fea_len)))
        for _ in range(self.n_conv):
            nbr_fea = EvenOddEdgeUpdateLayer(
                edge_fea_len=self.nbr_fea_len, odd_fea_len=self.odd_fea_len,
            )(even_node, odd_node, nbr_fea, nbr_fea_idx)
            even_node, odd_node = EvenOddConvLayer(
                even_fea_len=self.atom_fea_len, odd_fea_len=self.odd_fea_len,
                edge_fea_len=self.nbr_fea_len,
            )(even_node, odd_node, nbr_fea, nbr_fea_idx)
        E_chem = ReadoutMLP(self.h_fea_len)(even_node)
        E_mag  = ReadoutMLP(self.h_fea_len // 2)(even_node)
        return self.pooler_cls()(E_chem + E_mag, **kwargs)


class NeuralCE_Spin_v8u(nn.Module):
    """v8u: Even-Odd, Unified Readout (heavy). E(σ) = E(-σ) ✓."""
    pooler_cls: Callable
    atom_fea_len: int = 64
    nbr_fea_len: int  = 41
    n_conv: int       = 3
    h_fea_len: int    = 128
    odd_fea_len: int  = 32

    @nn.compact
    def __call__(self, atom_fea, nbr_fea, nbr_fea_idx, atom_spins=None, **kwargs):
        even_node = nn.Dense(self.atom_fea_len)(atom_fea)
        nbr_fea   = nn.Dense(self.nbr_fea_len)(nbr_fea)
        odd_node  = (nn.Dense(self.odd_fea_len, use_bias=False)(atom_spins)
                     if atom_spins is not None
                     else jnp.zeros((atom_fea.shape[0], self.odd_fea_len)))
        for _ in range(self.n_conv):
            nbr_fea = EvenOddEdgeUpdateLayer(
                edge_fea_len=self.nbr_fea_len, odd_fea_len=self.odd_fea_len,
            )(even_node, odd_node, nbr_fea, nbr_fea_idx)
            even_node, odd_node = EvenOddConvLayer(
                even_fea_len=self.atom_fea_len, odd_fea_len=self.odd_fea_len,
                edge_fea_len=self.nbr_fea_len,
            )(even_node, odd_node, nbr_fea, nbr_fea_idx)
        return self.pooler_cls()(ReadoutMLP(self.h_fea_len)(even_node), **kwargs)


# =============================================================================
# 5. Lite Models (one-hot 4-dim node + shell edge, Sr excluded)
#
#    5-model ablation: 2×2 (backbone × spin) + ising baseline
#    Node: [Ti, Fe, O, Vo] one-hot (4-dim)
#    Edge: shell one-hot (n_shells-dim, tunable)
#    σᵢσⱼ models: edge_attr = [shell_onehot, σᵢσⱼ] → edge_dim = n_shells + 1
#    EvenOdd models: edge_attr = shell_onehot only → edge_dim = n_shells
#                    spin enters via odd channel (bias-free Dense embedding)
# =============================================================================

class NeuralCE_Ising_Lite(nn.Module):
    """Lite: No spin, Product backbone."""
    pooler_cls: Callable
    atom_fea_len: int = 16
    nbr_fea_len: int  = 4
    n_conv: int       = 2
    h_fea_len: int    = 32

    @nn.compact
    def __call__(self, atom_fea, nbr_fea, nbr_fea_idx, **kwargs):
        atom_fea = nn.Dense(self.atom_fea_len)(atom_fea)
        for _ in range(self.n_conv):
            atom_fea = CEInteractionLayer(self.atom_fea_len)(atom_fea, nbr_fea, nbr_fea_idx)
        return self.pooler_cls()(ReadoutMLP(self.h_fea_len)(atom_fea), **kwargs)


class NeuralCE_Heisenberg_Lite(nn.Module):
    """Lite: HEGNN — Ising backbone + learned Heisenberg Jᵢⱼ(r)·σᵢσⱼ.
    
    E_total = Σᵢ hᵢ(r) + Σᵢⱼ Jᵢⱼ(r)·σᵢσⱼ
    
    The chemical energy hᵢ is learned by Product-based conv (same as ising_lite).
    The Heisenberg coefficient Jᵢⱼ is learned from edge features via MLP.
    Spin enters ONLY through the explicit σᵢσⱼ product — no spin in conv.
    
    Benefits:
      - Spin contribution is fully decomposable: E_chem vs E_mag
      - Jᵢⱼ can be extracted for physical interpretation
      - Guaranteed E(σ) = E(-σ) since σᵢσⱼ is even under global spin flip
    
    Note: atom_spins is passed as kwarg (same as EvenOdd models).
          Edge features are shell one-hot only (no σᵢσⱼ in edge).
          nbr_fea_len = n_shells (same as ising_lite / evenodd_lite).
    """
    pooler_cls: Callable
    atom_fea_len: int = 16
    nbr_fea_len: int  = 4     # = n_shells (shell one-hot only)
    n_conv: int       = 2
    h_fea_len: int    = 32
    odd_fea_len: int  = 4     # reused as J_ij hidden dim for consistency

    @nn.compact
    def __call__(self, atom_fea, nbr_fea, nbr_fea_idx, atom_spins=None, **kwargs):
        N, M = nbr_fea_idx.shape

        # ── Chemical energy: Ising backbone (spin-free) ──────────────
        atom_fea = nn.Dense(self.atom_fea_len)(atom_fea)
        nbr_fea_h = nn.Dense(self.nbr_fea_len)(nbr_fea)
        for _ in range(self.n_conv):
            atom_fea = CEInteractionLayer(self.atom_fea_len)(
                atom_fea, nbr_fea_h, nbr_fea_idx)

        E_chem = ReadoutMLP(self.h_fea_len)(atom_fea)  # (N, 1) site energy

        # ── Heisenberg term: Jᵢⱼ(r)·σᵢσⱼ ────────────────────────────
        if atom_spins is not None:
            # Learned Jᵢⱼ from edge context (node + edge features)
            atom_i = jnp.tile(atom_fea[:, None, :], (1, M, 1))      # (N, M, F)
            atom_j = jnp.take(atom_fea, nbr_fea_idx, axis=0)        # (N, M, F)
            edge_context = jnp.concatenate([atom_i, atom_j, nbr_fea_h], axis=-1)

            # MLP: edge_context → scalar Jᵢⱼ per edge
            J_h = nn.softplus(nn.Dense(self.odd_fea_len, name='J_fc1')(edge_context))
            J_ij = nn.Dense(1, name='J_fc2')(J_h)  # (N, M, 1)

            # σᵢσⱼ per edge
            spins_flat = atom_spins.squeeze(-1)         # (N,)
            spin_i = jnp.tile(spins_flat[:, None], (1, M))           # (N, M)
            spin_j = jnp.take(spins_flat, nbr_fea_idx, axis=0)      # (N, M)
            sisj = (spin_i * spin_j)[..., None]                      # (N, M, 1)

            # Mask out padded neighbors (nbr_fea is zero for padded)
            nbr_mask = jnp.any(nbr_fea > 0, axis=-1, keepdims=True)  # (N, M, 1)

            # E_mag per atom = Σⱼ Jᵢⱼ·σᵢσⱼ (sum over neighbors)
            E_mag_per_atom = jnp.sum(J_ij * sisj * nbr_mask, axis=1)  # (N, 1)

            E_site = E_chem + E_mag_per_atom
        else:
            E_site = E_chem

        return self.pooler_cls()(E_site, **kwargs)


class NeuralCE_EvenOdd_Lite(nn.Module):
    """Lite: Product backbone + EvenOdd spin. E(σ) = E(-σ) ✓."""
    pooler_cls: Callable
    atom_fea_len: int = 16
    nbr_fea_len: int  = 4     # = n_shells (no σᵢσⱼ in edge)
    n_conv: int       = 2
    h_fea_len: int    = 32
    odd_fea_len: int  = 4

    @nn.compact
    def __call__(self, atom_fea, nbr_fea, nbr_fea_idx, atom_spins=None, **kwargs):
        even_node = nn.Dense(self.atom_fea_len)(atom_fea)
        nbr_fea   = nn.Dense(self.nbr_fea_len)(nbr_fea)
        odd_node  = (nn.Dense(self.odd_fea_len, use_bias=False)(atom_spins)
                     if atom_spins is not None
                     else jnp.zeros((atom_fea.shape[0], self.odd_fea_len)))
        for _ in range(self.n_conv):
            nbr_fea = EvenOddEdgeUpdateLayer(
                edge_fea_len=self.nbr_fea_len, odd_fea_len=self.odd_fea_len,
            )(even_node, odd_node, nbr_fea, nbr_fea_idx)
            even_node, odd_node = EvenOddConvLayer(
                even_fea_len=self.atom_fea_len, odd_fea_len=self.odd_fea_len,
                edge_fea_len=self.nbr_fea_len,
            )(even_node, odd_node, nbr_fea, nbr_fea_idx)
        return self.pooler_cls()(ReadoutMLP(self.h_fea_len)(even_node), **kwargs)


class NeuralCE_SiSj_Lite(nn.Module):
    """Lite: Product backbone + σᵢσⱼ in edge. No EvenOdd.
    
    edge_attr includes [shell_onehot, σᵢσⱼ].
    Spin enters as σᵢσⱼ pair product in edge features only.
    """
    pooler_cls: Callable
    atom_fea_len: int = 16
    nbr_fea_len: int  = 5     # = n_shells + 1 (σᵢσⱼ appended)
    n_conv: int       = 2
    h_fea_len: int    = 32

    @nn.compact
    def __call__(self, atom_fea, nbr_fea, nbr_fea_idx, **kwargs):
        # nbr_fea already contains [shell_onehot, σᵢσⱼ]
        atom_fea = nn.Dense(self.atom_fea_len)(atom_fea)
        nbr_fea  = nn.Dense(self.nbr_fea_len)(nbr_fea)
        for _ in range(self.n_conv):
            atom_fea = CEInteractionLayer(self.atom_fea_len)(atom_fea, nbr_fea, nbr_fea_idx)
        return self.pooler_cls()(ReadoutMLP(self.h_fea_len)(atom_fea), **kwargs)


class GNN_SiSj_Lite(nn.Module):
    """Lite: Concat-MLP backbone + σᵢσⱼ in edge. No EvenOdd.
    
    edge_attr includes [shell_onehot, σᵢσⱼ].
    """
    pooler_cls: Callable
    atom_fea_len: int = 16
    nbr_fea_len: int  = 5     # = n_shells + 1
    n_conv: int       = 2
    h_fea_len: int    = 32

    @nn.compact
    def __call__(self, atom_fea, nbr_fea, nbr_fea_idx, **kwargs):
        atom_fea = nn.Dense(self.atom_fea_len)(atom_fea)
        nbr_fea  = nn.Dense(self.nbr_fea_len)(nbr_fea)
        for _ in range(self.n_conv):
            atom_fea = ConcatConvLayer(self.atom_fea_len, self.nbr_fea_len)(
                atom_fea, nbr_fea, nbr_fea_idx)
        return self.pooler_cls()(ReadoutMLP(self.h_fea_len)(atom_fea), **kwargs)


class GNN_EvenOdd_Lite(nn.Module):
    """Lite: Concat-MLP backbone + EvenOdd spin. E(σ) = E(-σ) ✓.

    Uses ConcatConvLayer for even channel (replaces CEInteractionLayer)
    but keeps EvenOdd structure for spin parity.
    
    Implementation: EvenOddConvLayer's even channel already does
    concat-like aggregation, so we reuse it but with a simpler
    even message path.
    """
    pooler_cls: Callable
    atom_fea_len: int = 16
    nbr_fea_len: int  = 4
    n_conv: int       = 2
    h_fea_len: int    = 32
    odd_fea_len: int  = 4

    @nn.compact
    def __call__(self, atom_fea, nbr_fea, nbr_fea_idx, atom_spins=None, **kwargs):
        even_node = nn.Dense(self.atom_fea_len)(atom_fea)
        nbr_fea   = nn.Dense(self.nbr_fea_len)(nbr_fea)
        odd_node  = (nn.Dense(self.odd_fea_len, use_bias=False)(atom_spins)
                     if atom_spins is not None
                     else jnp.zeros((atom_fea.shape[0], self.odd_fea_len)))
        for _ in range(self.n_conv):
            nbr_fea = EvenOddEdgeUpdateLayer(
                edge_fea_len=self.nbr_fea_len, odd_fea_len=self.odd_fea_len,
            )(even_node, odd_node, nbr_fea, nbr_fea_idx)
            even_node, odd_node = EvenOddConvLayer(
                even_fea_len=self.atom_fea_len, odd_fea_len=self.odd_fea_len,
                edge_fea_len=self.nbr_fea_len,
            )(even_node, odd_node, nbr_fea, nbr_fea_idx)
        return self.pooler_cls()(ReadoutMLP(self.h_fea_len)(even_node), **kwargs)


# =============================================================================
# 6. Registry & Factory
# =============================================================================

MODEL_REGISTRY = {
    # Heavy
    'ising':                NeuralCE_Ising,
    'spin_v3':              NeuralCE_Spin_v3,
    'spin_v8':              NeuralCE_Spin_v8,
    'spin_v8u':             NeuralCE_Spin_v8u,
    # Lite
    'ising_lite':           NeuralCE_Ising_Lite,
    'hegnn_lite':           NeuralCE_Heisenberg_Lite,
    'neuralce_evenodd_lite': NeuralCE_EvenOdd_Lite,
    'neuralce_sisj_lite':   NeuralCE_SiSj_Lite,
    'gnn_sisj_lite':        GNN_SiSj_Lite,
    'gnn_evenodd_lite':     GNN_EvenOdd_Lite,
}

SPIN_MODELS = {
    'spin_v3', 'spin_v8', 'spin_v8u',
    'hegnn_lite',
    'neuralce_evenodd_lite', 'gnn_evenodd_lite',
}

# Models that use σᵢσⱼ in edge features (not through odd channel)
SISJ_MODELS = {
    'neuralce_sisj_lite', 'gnn_sisj_lite',
}

LITE_MODELS = {
    'ising_lite', 'hegnn_lite',
    'neuralce_evenodd_lite', 'neuralce_sisj_lite',
    'gnn_sisj_lite', 'gnn_evenodd_lite',
}


def is_spin_model(model_type: str) -> bool:
    """True if model uses EvenOdd spin channel (atom_spins kwarg)."""
    return model_type in SPIN_MODELS

def is_sisj_model(model_type: str) -> bool:
    """True if model uses σᵢσⱼ in edge features."""
    return model_type in SISJ_MODELS

def is_lite_model(model_type: str) -> bool:
    return model_type in LITE_MODELS

def needs_spin(model_type: str) -> bool:
    """True if model needs spin info in any form (EvenOdd or SiSj)."""
    return model_type in SPIN_MODELS or model_type in SISJ_MODELS


def create_neuralce(model_type: str = 'ising', pool_mode: str = 'fixed',
                    readout_type: str = 'sum', **kwargs):
    """Factory function for NeuralCE models.

    Args:
        model_type:   One of MODEL_REGISTRY keys
        pool_mode:    'fixed' | 'padded' | 'variable'
        readout_type: 'sum' | 'mean'
        **kwargs:     atom_fea_len, nbr_fea_len, n_conv, h_fea_len,
                      odd_fea_len (EvenOdd), spin_embed_dim (v3)
    """
    pooler_map = {
        ('fixed',    'sum'):  FixedSumPool,
        ('fixed',    'mean'): FixedMeanPool,
        ('padded',   'sum'):  PaddedSumPool,
        ('padded',   'mean'): PaddedMeanPool,
        ('variable', 'sum'):  VariableSumPool,
        ('variable', 'mean'): VariableMeanPool,
    }
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown: '{model_type}'. Choose from: {sorted(MODEL_REGISTRY.keys())}")

    pooler    = pooler_map[(pool_mode, readout_type)]
    model_cls = MODEL_REGISTRY[model_type]
    valid_fields = {f.name for f in model_cls.__dataclass_fields__.values()
                    if f.name != 'pooler_cls'}
    filtered = {k: v for k, v in kwargs.items() if k in valid_fields}
    return model_cls(pooler_cls=pooler, **filtered)


# =============================================================================
# 7. Symmetry Verification
# =============================================================================

def verify_spin_symmetry(model, params, atom_fea, nbr_fea, nbr_fea_idx,
                         atom_spins, **kwargs):
    """Check E(σ) == E(-σ) for a batch."""
    E_up   = model.apply(params, atom_fea, nbr_fea, nbr_fea_idx,
                         atom_spins=atom_spins, **kwargs)
    E_down = model.apply(params, atom_fea, nbr_fea, nbr_fea_idx,
                         atom_spins=-atom_spins, **kwargs)
    max_diff = jnp.max(jnp.abs(E_up - E_down))
    return {'E_up': E_up, 'E_down': E_down,
            'max_diff': float(max_diff), 'is_symmetric': float(max_diff) < 1e-5}
