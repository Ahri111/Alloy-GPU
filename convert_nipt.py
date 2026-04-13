"""
convert_nipt.py — NiPt DFT 데이터 → NeuralCE 파이프라인 양식 변환

입력 구조:
  2000/
    run_000/
      POSCAR_INITIAL      ← unrelaxed 구조 (input)
      POSCAR_RELAXED
      CONTCAR
      RESULT              ← JSON (energy, n_atoms, converged, ...)
    run_001/
      ...

출력:
  data/CIF_NiPt/
    run_000.cif           ← POSCAR_INITIAL → CIF
    run_001.cif
    ...
    detailed_info.csv     ← id, total_energy, n_atoms, composition, n_ni, n_pt
    atom_init.json        ← atom embeddings (copied or generated)

  configs/
    nipt_ablation.yaml
    nipt_ptmcmc.yaml

Usage:
  python convert_nipt.py ./2000 --output_dir ./data/CIF_NiPt

  # 여러 폴더:
  python convert_nipt.py ./2000 ./3000 ./4000 --output_dir ./data/CIF_NiPt
"""

import os
import json
import argparse
import numpy as np
import csv
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Poscar


# ═══════════════════════════════════════════════════════════════════════
# ARGPARSE
# ═══════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(
    description='Convert NiPt DFT data to NeuralCE pipeline format')

parser.add_argument('data_dirs', type=str, nargs='+',
                    help='Root directories containing run_xxx subfolders')
parser.add_argument('--output_dir', type=str, default='./data/CIF_NiPt',
                    help='Output directory for CIF + CSV')
parser.add_argument('--poscar_name', type=str, default='POSCAR_INITIAL',
                    help='Which POSCAR to convert (default: POSCAR_INITIAL)')
parser.add_argument('--energy_key', type=str, default='final_energy_eV',
                    help='JSON key for energy target (default: final_energy_eV)')
parser.add_argument('--result_name', type=str, default='RESULT',
                    help='JSON result filename (default: RESULT)')
parser.add_argument('--only_converged', action='store_true',
                    help='Skip unconverged runs')
parser.add_argument('--gen_configs', action='store_true',
                    help='Generate YAML configs for ablation + pt_mcmc')
parser.add_argument('--config_dir', type=str, default='./configs',
                    help='Output directory for YAML configs')

args = parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# 1. SCAN & CONVERT
# ═══════════════════════════════════════════════════════════════════════
os.makedirs(args.output_dir, exist_ok=True)

print(f"{'═' * 70}")
print(f"  convert_nipt — DFT → NeuralCE pipeline")
print(f"  Data dirs:  {args.data_dirs}")
print(f"  POSCAR:     {args.poscar_name}")
print(f"  Energy key: {args.energy_key}")
print(f"  Output:     {args.output_dir}")
print(f"{'═' * 70}")

rows = []
n_skip = 0
n_unconverged = 0
n_err = 0
n_atoms_set = set()
comp_counts = {}  # {(n_ni, n_pt): count}

for data_dir in args.data_dirs:
    if not os.path.isdir(data_dir):
        print(f"  [!] Not a directory: {data_dir}")
        continue

    subdirs = sorted([d for d in os.listdir(data_dir)
                      if os.path.isdir(os.path.join(data_dir, d))])

    print(f"\n[1] Scanning {data_dir} ({len(subdirs)} subdirectories)...")

    for run_name in subdirs:
        run_path = os.path.join(data_dir, run_name)
        poscar_path = os.path.join(run_path, args.poscar_name)
        result_path = os.path.join(run_path, args.result_name)

        # Check files exist
        if not os.path.exists(poscar_path):
            n_skip += 1
            continue
        if not os.path.exists(result_path):
            n_skip += 1
            continue

        # Read RESULT JSON
        try:
            with open(result_path) as f:
                result = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  [!] {run_name}: JSON error — {e}")
            n_err += 1
            continue

        # Check convergence
        if args.only_converged and not result.get('converged', True):
            n_unconverged += 1
            continue

        # Get energy
        energy = result.get(args.energy_key)
        if energy is None:
            print(f"  [!] {run_name}: missing '{args.energy_key}' in RESULT")
            n_err += 1
            continue

        # Read POSCAR → Structure
        try:
            poscar = Poscar.from_file(poscar_path)
            struct = poscar.structure
        except Exception as e:
            print(f"  [!] {run_name}: POSCAR parse error — {e}")
            n_err += 1
            continue

        n_atoms = len(struct)
        n_atoms_set.add(n_atoms)

        # Count species
        species_count = {}
        for site in struct:
            sym = site.specie.symbol
            species_count[sym] = species_count.get(sym, 0) + 1

        n_ni = species_count.get('Ni', 0)
        n_pt = species_count.get('Pt', 0)
        comp_key = (n_ni, n_pt)
        comp_counts[comp_key] = comp_counts.get(comp_key, 0) + 1

        # Composition code: Pt count (for comp_regex matching)
        # e.g., Ni54Pt54 → comp_code=54, Ni81Pt27 → comp_code=27
        comp_code = n_pt

        # CIF ID: composition prefix + run name (matches STFO convention)
        # e.g., Ni54Pt54_run_149.cif
        cif_id = f"Ni{n_ni}Pt{n_pt}_{run_name}"

        # Save CIF
        cif_path = os.path.join(args.output_dir, f"{cif_id}.cif")
        struct.to(filename=cif_path, fmt='cif')

        # CSV row
        rows.append({
            'id': cif_id,
            'total_energy': energy,
            'n_atoms': n_atoms,
            'n_ni': n_ni,
            'n_pt': n_pt,
            'composition': f"Ni{n_ni}Pt{n_pt}",
            'comp_code': comp_code,
            'converged': result.get('converged', True),
            'energy_per_atom': result.get(
                'final_energy_per_atom_eV', energy / n_atoms),
            'source_dir': run_name,
        })

print(f"\n[2] Summary")
print(f"    Converted: {len(rows)}")
print(f"    Skipped (missing files): {n_skip}")
print(f"    Skipped (unconverged): {n_unconverged}")
print(f"    Errors: {n_err}")
print(f"    Atom counts: {sorted(n_atoms_set)}")
print(f"    Compositions ({len(comp_counts)}):")
for (n_ni, n_pt), cnt in sorted(comp_counts.items()):
    print(f"      Ni{n_ni}Pt{n_pt}: {cnt} structures")


# ═══════════════════════════════════════════════════════════════════════
# 2. SAVE CSV
# ═══════════════════════════════════════════════════════════════════════
csv_path = os.path.join(args.output_dir, 'detailed_info.csv')
fieldnames = ['id', 'total_energy', 'n_atoms', 'n_ni', 'n_pt',
              'composition', 'comp_code', 'converged', 'energy_per_atom',
              'source_dir']

with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\n[3] Saved: {csv_path} ({len(rows)} rows)")


# ═══════════════════════════════════════════════════════════════════════
# 4. YAML CONFIGS (optional)
# ═══════════════════════════════════════════════════════════════════════
if args.gen_configs:
    os.makedirs(args.config_dir, exist_ok=True)

    # Determine n_atoms (must be uniform for fixed pool)
    if len(n_atoms_set) == 1:
        n_atoms = n_atoms_set.pop()
    else:
        n_atoms = max(n_atoms_set)
        print(f"  ⚠ Variable atom counts {sorted(n_atoms_set)}, using max={n_atoms}")

    # --- Ablation YAML ---
    ablation_yaml = f"""# NiPt ablation config (auto-generated by convert_nipt.py)
mode: ablation

dataset_name: nipt
cif_dir:      {args.output_dir}
csv_path:     {args.output_dir}/detailed_info.csv
spin_pkl:     null
id_col:       id
comp_regex:   'Pt(\\d+)_'

species_map:
  28: 0   # Ni
  78: 1   # Pt
n_atoms:  {n_atoms}
has_spin: false

ablation:
  seed:       42
  val_frac:   0.15
  test_frac:  0.15
  max_epochs: 3000
  patience:   80
  batch_size: 32
  n_trials:   30
  run_models:
    - ising_lite
"""
    abl_path = os.path.join(args.config_dir, 'nipt_ablation.yaml')
    with open(abl_path, 'w') as f:
        f.write(ablation_yaml)
    print(f"\n[5] Saved: {abl_path}")

    # --- PT-MCMC YAML ---
    # Pick first CIF as template (or user can override)
    template_cif = os.path.join(args.output_dir, f"{rows[0]['id']}.cif")

    # Composition: determine from most common composition
    most_common = max(comp_counts, key=comp_counts.get)
    n_ni_mc, n_pt_mc = most_common

    ptmcmc_yaml = f"""# NiPt PT-MCMC config (auto-generated by convert_nipt.py)
mode: pt_mcmc

dataset_name: nipt
species_map:
  28: 0   # Ni
  78: 1   # Pt
n_atoms:  {n_atoms}
has_spin: false

pt_mcmc:
  model_ckpt:     ./best_nipt_ising_lite.pkl
  model_type:     ising_lite
  cif_template:   {template_cif}
  n_replicas:     3000
  n_steps:        100000
  t_min:          200.0
  t_max:          2000.0
  chunk_size:     200
  swap_interval:  10
  burnin_frac:    0.2
  seed:           42
  output:         ./pt_nipt_result.npz

  matmul_precision: float32
  enable_x64:      false

  sublattices:
    - name: FCC
      species: [0, 1]

  composition:
    FCC:
      0: {n_ni_mc}    # Ni
      1: {n_pt_mc}    # Pt

  spin_flip: false
  spin_species: []
"""
    mc_path = os.path.join(args.config_dir, 'nipt_ptmcmc.yaml')
    with open(mc_path, 'w') as f:
        f.write(ptmcmc_yaml)
    print(f"    Saved: {mc_path}")


# ═══════════════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 70}")
print(f"  Done! Pipeline-ready data in {args.output_dir}/")
print(f"  Next steps:")
print(f"    1. python ablation.py  (CONFIG_PATH=configs/nipt_ablation.yaml)")
print(f"    2. python retrain.py   (on best model)")
print(f"    3. python pt_mcmc.py   (CONFIG_PATH=configs/nipt_ptmcmc.yaml)")
print(f"    4. python extract_lowest_cif.py result.npz \\")
print(f"         --template <template.cif> --species_map 28:0 78:1")
print(f"{'═' * 70}")
