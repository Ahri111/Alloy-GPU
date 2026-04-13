"""
analyze_ptmcmc.py — Single PT-MCMC NPZ Analysis

Produces:
  1. Top-K lowest energy CIF extraction
  2. E(T) — Mean energy vs temperature
  3. Cv(T) — Heat capacity + Tc detection
  4. SRO(T) — Warren-Cowley short-range order parameter

Usage:
  python analyze_ptmcmc.py result.npz \
      --template ./data/stfo_wo_spin/250_0001.cif \
      --species_map 22:0 26:1 8:2 54:3 \
      --exclude_z 38 \
      --sro_species 0 1 \
      --top_k 5

  # Minimal (no SRO, no CIF)
  python analyze_ptmcmc.py result.npz \
      --template ./data/stfo_wo_spin/250_0001.cif \
      --species_map 22:0 26:1 8:2 54:3 \
      --exclude_z 38 \
      --no_cif --no_sro
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from pymatgen.core.structure import Structure
from pymatgen.core import DummySpecies, Element


# ═══════════════════════════════════════════════════════════════════════
# ARGPARSE
# ═══════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(
    description='Analyze single PT-MCMC v4 result: CIF + E(T) + Cv(T) + SRO(T)')

parser.add_argument('npz', type=str, help='PT-MCMC result .npz file')
parser.add_argument('--template', type=str, required=True,
                    help='Template CIF file')
parser.add_argument('--species_map', type=str, nargs='+', required=True,
                    help='Z:idx pairs, e.g. 22:0 26:1 8:2 54:3')
parser.add_argument('--exclude_z', type=int, nargs='*', default=[],
                    help='Atomic numbers to exclude from template')
parser.add_argument('--vacancy_z', type=int, nargs='*', default=[54, 0],
                    help='Atomic numbers treated as vacancy (default: 54 0)')

# CIF extraction
parser.add_argument('--top_k', type=int, default=5,
                    help='Number of lowest energy CIFs to extract (default: 5)')
parser.add_argument('--no_cif', action='store_true',
                    help='Skip CIF extraction')
parser.add_argument('--no_vacancy', action='store_true',
                    help='Remove vacancy sites from CIF')

# SRO
parser.add_argument('--sro_species', type=int, nargs=2, default=None,
                    help='Species indices for SRO calculation, e.g. 0 1 (Ti Fe)')
parser.add_argument('--sro_cutoff', type=float, default=None,
                    help='Neighbor cutoff for SRO (default: auto from template)')
parser.add_argument('--no_sro', action='store_true',
                    help='Skip SRO calculation')

# Plot
parser.add_argument('--cv_sigma', type=int, default=10,
                    help='Gaussian smoothing sigma for Cv (default: 10)')
parser.add_argument('--output_dir', type=str, default=None,
                    help='Output directory (default: {npz_stem}_analysis/)')

args = parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# PARSE SPECIES MAP
# ═══════════════════════════════════════════════════════════════════════
species_map = {}
for pair in args.species_map:
    z_str, idx_str = pair.split(':')
    species_map[int(z_str)] = int(idx_str)

idx_to_z = {v: k for k, v in species_map.items()}

vacancy_z_set = set(args.vacancy_z)
idx_to_sym = {}
vacancy_idx = None
for idx, z in idx_to_z.items():
    if z in vacancy_z_set:
        idx_to_sym[idx] = 'Xe'
        vacancy_idx = idx
    else:
        idx_to_sym[idx] = Element.from_Z(z).symbol

exclude_z = set(args.exclude_z)
output_dir = args.output_dir or os.path.splitext(args.npz)[0] + '_analysis'
os.makedirs(output_dir, exist_ok=True)

print(f"{'═' * 70}")
print(f"  analyze_ptmcmc — {args.npz}")
print(f"  Template:  {args.template}")
print(f"  Species:   {idx_to_sym}")
print(f"  Output:    {output_dir}/")
print(f"{'═' * 70}")

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 14,
    'legend.fontsize': 11, 'figure.dpi': 150,
})


# ═══════════════════════════════════════════════════════════════════════
# 1. LOAD
# ═══════════════════════════════════════════════════════════════════════
data = np.load(args.npz)
temperatures     = data['temperatures']
sampled_energies = data['sampled_energies']
sampled_species  = data['sampled_species']

n_snapshots, n_replicas, n_atoms = sampled_species.shape

print(f"\n[1] Data loaded")
print(f"    Snapshots: {n_snapshots}, Replicas: {n_replicas}, Atoms: {n_atoms}")
print(f"    T: {temperatures[0]:.0f} → {temperatures[-1]:.0f} K")
print(f"    E: [{sampled_energies.min():.4f}, {sampled_energies.max():.4f}] eV")

# Composition check
ref_sp = sampled_species[0, 0]
ref_counts = {}
for idx in np.unique(ref_sp):
    ref_counts[idx_to_sym.get(int(idx), f'?{idx}')] = int(np.sum(ref_sp == idx))
print(f"    Composition: {ref_counts}")

kB = 8.617333262145e-5


# ═══════════════════════════════════════════════════════════════════════
# 2. TEMPLATE
# ═══════════════════════════════════════════════════════════════════════
template_full = Structure.from_file(args.template)
if exclude_z:
    keep = [i for i, s in enumerate(template_full) if s.specie.Z not in exclude_z]
    template = Structure.from_sites([template_full[i] for i in keep])
else:
    template = template_full

assert len(template) == n_atoms, \
    f"Template ({len(template)}) != NPZ ({n_atoms})"


# ═══════════════════════════════════════════════════════════════════════
# 3. E(T)
# ═══════════════════════════════════════════════════════════════════════
print(f"\n[2] E(T) plot...")

E_mean = sampled_energies.mean(axis=0) / n_atoms
E_std = sampled_energies.std(axis=0) / n_atoms

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(temperatures, E_mean, 'b-', lw=2, label='⟨E⟩')
ax.fill_between(temperatures, E_mean - E_std, E_mean + E_std,
                alpha=0.2, color='blue', label='±1σ')
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Energy (eV/atom)')
ax.set_title('Mean Energy vs Temperature')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'energy_vs_T.png'), dpi=200, bbox_inches='tight')
plt.show()


# ═══════════════════════════════════════════════════════════════════════
# 4. Cv(T)
# ═══════════════════════════════════════════════════════════════════════
print(f"[3] Cv(T) plot...")

E_var = sampled_energies.var(axis=0)
Cv = E_var / (kB * temperatures**2) / n_atoms
Cv_smooth = gaussian_filter1d(Cv, sigma=args.cv_sigma)

Tc_idx = np.argmax(Cv_smooth)
Tc = temperatures[Tc_idx]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(temperatures, Cv, 'r-', alpha=0.15, lw=0.5, label='Raw')
ax.plot(temperatures, Cv_smooth, 'r-', lw=2, label=f'Smoothed (σ={args.cv_sigma})')
ax.scatter(Tc, Cv_smooth[Tc_idx], color='white', edgecolors='red',
           s=80, zorder=5, linewidths=2)
ax.annotate(f'$T_c$ ≈ {Tc:.0f} K', xy=(Tc, Cv_smooth[Tc_idx]),
            xytext=(12, 8), textcoords='offset points',
            fontsize=11, fontweight='bold', color='red')
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('$C_v$ / atom (eV/K)')
ax.set_title('Heat Capacity vs Temperature')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'cv_vs_T.png'), dpi=200, bbox_inches='tight')
plt.show()

print(f"    Tc ≈ {Tc:.0f} K")


# ═══════════════════════════════════════════════════════════════════════
# 5. SRO(T)
# ═══════════════════════════════════════════════════════════════════════
if not args.no_sro and args.sro_species is not None:
    print(f"\n[4] SRO(T) plot...")

    sp_a, sp_b = args.sro_species
    sym_a = idx_to_sym.get(sp_a, f'?{sp_a}')
    sym_b = idx_to_sym.get(sp_b, f'?{sp_b}')

    # Determine SRO cutoff
    if args.sro_cutoff:
        sro_cutoff = args.sro_cutoff
    else:
        # Auto: use 1NN distance × 1.2
        all_nbrs = template.get_all_neighbors(8.0, include_index=True)
        min_dist = min(n[1] for nbrs in all_nbrs for n in nbrs)
        sro_cutoff = min_dist * 1.2
    print(f"    SRO pair: {sym_a}-{sym_b}, cutoff: {sro_cutoff:.2f} Å")

    # Neighbor list (fixed lattice)
    all_nbrs = template.get_all_neighbors(sro_cutoff, include_index=True)
    nbr_list = {}
    for i, nbrs in enumerate(all_nbrs):
        nbr_list[i] = [n[2] for n in sorted(nbrs, key=lambda x: x[1])]

    # Sublattice: sites that are sp_a or sp_b
    def get_sublattice(species_arr):
        mask = (species_arr == sp_a) | (species_arr == sp_b)
        return np.where(mask)[0]

    def compute_sro(species_arr, sub_idx):
        n_sub = len(sub_idx)
        x_b = np.sum(species_arr[sub_idx] == sp_b) / n_sub
        if x_b < 1e-10 or x_b > 1 - 1e-10:
            return 0.0

        a_centers = sub_idx[species_arr[sub_idx] == sp_a]
        if len(a_centers) == 0:
            return 0.0

        sub_set = set(sub_idx)
        b_count, total = 0, 0
        for i in a_centers:
            for j in nbr_list.get(i, []):
                if j in sub_set:
                    total += 1
                    if species_arr[j] == sp_b:
                        b_count += 1

        if total == 0:
            return 0.0
        return 1.0 - (b_count / total) / x_b

    # Compute per replica (average last N snapshots)
    n_avg = min(5, n_snapshots)
    sub_idx = get_sublattice(sampled_species[0, 0])

    sro_per_replica = np.zeros(n_replicas)
    for r in range(n_replicas):
        vals = []
        for s in range(n_snapshots - n_avg, n_snapshots):
            vals.append(compute_sro(sampled_species[s, r], sub_idx))
        sro_per_replica[r] = np.mean(vals)

    sro_smooth = gaussian_filter1d(sro_per_replica, sigma=args.cv_sigma)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(temperatures, sro_per_replica, 'g-', alpha=0.15, lw=0.5, label='Raw')
    ax.plot(temperatures, sro_smooth, 'g-', lw=2, label='Smoothed')
    ax.axhline(0, color='gray', ls='--', alpha=0.5, label='Random (α=0)')
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel(f'Warren-Cowley α₁ ({sym_a}-{sym_b})')
    ax.set_title('Short-Range Order vs Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sro_vs_T.png'), dpi=200, bbox_inches='tight')
    plt.show()

    print(f"    SRO at T_min: {sro_per_replica[0]:.4f}")
    print(f"    SRO at T_max: {sro_per_replica[-1]:.4f}")

elif not args.no_sro:
    print(f"\n[4] SRO skipped (--sro_species not provided)")


# ═══════════════════════════════════════════════════════════════════════
# 6. TOP-K LOWEST ENERGY CIF
# ═══════════════════════════════════════════════════════════════════════
if not args.no_cif:
    print(f"\n[5] Top-{args.top_k} lowest energy CIFs...")

    def species_to_structure(species_arr):
        new_species, new_coords = [], []
        for i, sp_idx in enumerate(species_arr):
            sp_idx = int(sp_idx)
            symbol = idx_to_sym.get(sp_idx, f'Unk{sp_idx}')
            if sp_idx == vacancy_idx:
                if args.no_vacancy:
                    continue
                new_species.append(DummySpecies('X', oxidation_state=0))
            else:
                new_species.append(Element(symbol))
            new_coords.append(template[i].frac_coords)

        struct = Structure(template.lattice, new_species, new_coords)
        counters = {}
        for site in struct:
            sym = site.specie.symbol if hasattr(site.specie, 'symbol') else 'X'
            counters[sym] = counters.get(sym, 0)
            site.label = f'{sym}{counters[sym]}'
            counters[sym] += 1
        return struct

    flat_species = sampled_species.reshape(-1, n_atoms)
    flat_energies = sampled_energies.reshape(-1)

    sorted_idx = np.argsort(flat_energies)
    seen = set()
    results = []

    for idx in sorted_idx:
        key = flat_species[idx].tobytes()
        if key in seen:
            continue
        seen.add(key)
        results.append((float(flat_energies[idx]), flat_species[idx]))
        if len(results) >= args.top_k:
            break

    cif_dir = os.path.join(output_dir, 'cifs')
    os.makedirs(cif_dir, exist_ok=True)

    for rank, (energy, sp_arr) in enumerate(results, 1):
        e_pa = energy / n_atoms
        counts = {}
        for idx in np.unique(sp_arr):
            counts[idx_to_sym.get(int(idx), '?')] = int(np.sum(sp_arr == idx))
        counts_str = ' '.join(f'{k}={v}' for k, v in counts.items())

        struct = species_to_structure(sp_arr)
        cif_name = f'top_{rank:02d}_E{energy:.4f}.cif'
        struct.to(filename=os.path.join(cif_dir, cif_name), fmt='cif')

        tag = ' ★' if rank == 1 else ''
        print(f"    #{rank}: {energy:.4f} eV ({e_pa:.6f} eV/at) {counts_str}{tag}")

    if results:
        best_struct = species_to_structure(results[0][1])
        best_struct.to(filename=os.path.join(cif_dir, 'lowest_energy.cif'), fmt='cif')


# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 70}")
print(f"  SUMMARY")
print(f"{'═' * 70}")
print(f"  Tc ≈ {Tc:.0f} K")
print(f"  E_min: {sampled_energies.min():.4f} eV ({sampled_energies.min()/n_atoms:.6f} eV/at)")
print(f"  Composition: {ref_counts}")
print(f"  Output: {output_dir}/")
print(f"    energy_vs_T.png")
print(f"    cv_vs_T.png")
if not args.no_sro and args.sro_species:
    print(f"    sro_vs_T.png")
if not args.no_cif:
    print(f"    cifs/lowest_energy.cif + top_01..{args.top_k:02d}")
print(f"{'═' * 70}")
