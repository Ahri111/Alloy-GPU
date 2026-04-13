"""
convert_dataset.py — Unified Data Converter
============================================
Converts various raw data formats into a unified folder structure:

  output_dir/
  ├── detailed_info.csv     # id, total_energy, comp, n_atoms, spins (JSON)
  ├── atom_init.json        # species → embedding (auto-generated)
  ├── 0001.cif
  ├── 0002.cif
  └── ...

Supported sources:
  --pkl         Trans_Set_*.pkl files (Channyung STFO format)
  --mag_data    mag_data text file (Tianyu Fe-Ni-Cr format)
  --poscar_dirs Trans_Set_*/ folders with POSCAR_VO + OUTCAR

Examples:
  # Single pkl
  python convert_dataset.py --pkl Trans_Set_125_G.pkl --output data/stfo_125G

  # Multiple pkls merged
  python convert_dataset.py \
      --pkl Trans_Set_125.pkl Trans_Set_125_G.pkl Trans_Set_250.pkl \
      --output data/stfo_mixed

  # mag_data
  python convert_dataset.py --mag_data mag_data --output data/feni_cr

  # POSCAR_VO folders
  python convert_dataset.py \
      --poscar_dirs Trans_Set_125/ Trans_Set_250/ \
      --output data/stfo_poscar

  # Mixed sources
  python convert_dataset.py \
      --pkl Trans_Set_125_G.pkl \
      --mag_data mag_data \
      --output data/mixed
"""

import os, sys, json, argparse, pickle, re, glob
import numpy as np
import csv
from collections import Counter


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Convert raw data sources into unified CIF + CSV format.")
    p.add_argument("--pkl", nargs='+', default=[],
                   help="Trans_Set_*.pkl files (Channyung format)")
    p.add_argument("--mag_data", nargs='+', default=[],
                   help="mag_data text files (Tianyu Fe-Ni-Cr format)")
    p.add_argument("--poscar_dirs", nargs='+', default=[],
                   help="Trans_Set_*/ folders with POSCAR_VO + OUTCAR")
    p.add_argument("--output", type=str, required=True,
                   help="Output directory")
    p.add_argument("--spin_threshold", type=float, default=3.5,
                   help="Magmom threshold for Ising spin assignment (POSCAR_VO). Default: 3.5")
    p.add_argument("--vo_element", type=str, default="Xe",
                   help="Dummy element for oxygen vacancy sites. Default: Xe")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# PARSER 1: Trans_Set_*.pkl  (Channyung STFO)
# ═══════════════════════════════════════════════════════════════════════
# PKL structure: {folder_name: {energy, spin_states, poscar_text}}

def parse_poscar_text(poscar_text, vo_element="Xe"):
    """Parse POSCAR_VO text → lattice, species_labels, frac_coords.
    
    VO sites are mapped to vo_element (default Xe) so pymatgen can handle them.
    """
    lines = poscar_text.strip().split('\n')

    scale = float(lines[1].strip())
    lattice = np.array([[float(x) for x in lines[i].split()] for i in range(2, 5)]) * scale

    species_names = lines[5].split()
    counts = [int(x) for x in lines[6].split()]

    # Map VO → dummy element
    mapped_species = []
    for sp in species_names:
        if sp.upper() == 'VO' or sp.upper() == 'V_O':
            mapped_species.append(vo_element)
        else:
            mapped_species.append(sp)

    labels = []
    for sp, cnt in zip(mapped_species, counts):
        labels.extend([sp] * cnt)

    idx = 7
    if lines[idx].strip()[0].lower() == 's':  # Selective dynamics
        idx += 1
    coord_type = lines[idx].strip()[0].lower()
    idx += 1

    n_atoms = sum(counts)
    coords = np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        parts = lines[idx + i].split()
        coords[i] = [float(parts[0]), float(parts[1]), float(parts[2])]

    if coord_type == 'c' or coord_type == 'k':
        # Cartesian → fractional
        inv_lat = np.linalg.inv(lattice)
        coords = coords @ inv_lat

    return lattice, labels, coords


def load_from_pkl(pkl_paths, vo_element="Xe"):
    """Load structures from Trans_Set_*.pkl files.
    
    Returns list of standardized dicts.
    """
    structures = []

    for pkl_path in pkl_paths:
        pkl_name = os.path.splitext(os.path.basename(pkl_path))[0]

        # Extract comp info from filename: Trans_Set_125_G → comp=125, suffix=G
        comp_match = re.search(r'(\d+)', pkl_name)
        comp_str = comp_match.group(1) if comp_match else pkl_name

        # Prefix for unique IDs
        prefix = pkl_name.replace('Trans_Set_', '').replace(' ', '_')

        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        print(f"  [{pkl_name}] {len(data)} structures, comp={comp_str}")

        for i, (folder_name, entry) in enumerate(data.items()):
            energy = entry['energy']
            spin_states = entry.get('spin_states', [])
            poscar_text = entry['poscar_text']

            lattice, labels, frac_coords = parse_poscar_text(poscar_text, vo_element)

            if spin_states:
                raw_spins = [int(s) for s in spin_states]
                # PKL spin excludes VO sites (e.g. 144 spins for 160 atoms)
                # Expand to full length by inserting 0 at VO positions
                if len(raw_spins) < len(labels):
                    spins = []
                    si = 0
                    for lab in labels:
                        if lab == vo_element:
                            spins.append(0)  # VO site → no spin
                        else:
                            spins.append(raw_spins[si] if si < len(raw_spins) else 0)
                            si += 1
                    if si != len(raw_spins):
                        print(f"  ⚠ Spin alignment mismatch for {folder_name}: "
                              f"used {si}/{len(raw_spins)} spins, {len(labels)} atoms")
                else:
                    spins = raw_spins
            else:
                spins = [0] * len(labels)

            struct_id = f"{prefix}_{i:04d}"

            structures.append({
                'id': struct_id,
                'energy': float(energy),
                'comp': comp_str,
                'lattice': lattice,
                'species': labels,
                'frac_coords': frac_coords,
                'spins': spins,
                'n_atoms': len(labels),
                'source': pkl_name,
            })

    return structures


# ═══════════════════════════════════════════════════════════════════════
# PARSER 2: mag_data text  (Tianyu Fe-Ni-Cr)
# ═══════════════════════════════════════════════════════════════════════

MAGDATA_SPECIES = {0: 'Fe', 1: 'Ni', 2: 'Cr'}

def load_from_mag_data(mag_data_paths):
    """Parse mag_data text file(s) → list of standardized dicts."""
    structures = []

    for mag_path in mag_data_paths:
        mag_name = os.path.splitext(os.path.basename(mag_path))[0]

        with open(mag_path) as f:
            lines = f.readlines()

        count = 0
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('# Fe Ni Cr'):
                header = lines[i+1].strip().split('\t')
                n_fe, n_ni, n_cr = int(header[0]), int(header[1]), int(header[2])
                n_atoms = n_fe + n_ni + n_cr
                struct_id_raw = header[3]
                energy = float(header[4])

                lattice = []
                for li in range(3):
                    lattice.append([float(x) for x in lines[i+2+li].strip().split('\t')])
                lattice = np.array(lattice)

                species_labels = []
                spins = []
                frac_coords = []
                for j in range(n_atoms):
                    parts = lines[i+5+j].strip().split('\t')
                    sp_idx = int(parts[1])
                    species_labels.append(MAGDATA_SPECIES[sp_idx])
                    spins.append(float(parts[2]))
                    frac_coords.append([float(parts[3]), float(parts[4]), float(parts[5])])

                comp = f"Fe{n_fe}Ni{n_ni}Cr{n_cr}"
                struct_id = f"{mag_name}_{count:04d}"

                structures.append({
                    'id': struct_id,
                    'energy': float(energy),
                    'comp': comp,
                    'lattice': lattice,
                    'species': species_labels,
                    'frac_coords': np.array(frac_coords),
                    'spins': [float(s) for s in spins],
                    'n_atoms': n_atoms,
                    'source': mag_name,
                })

                count += 1
                i += 5 + n_atoms
            else:
                i += 1

        print(f"  [{mag_name}] {count} structures")

    return structures


# ═══════════════════════════════════════════════════════════════════════
# PARSER 3: POSCAR_VO + OUTCAR folders  (Channyung raw)
# ═══════════════════════════════════════════════════════════════════════

def parse_outcar(path):
    """Parse OUTCAR → energy, magmoms."""
    with open(path) as f:
        lines = f.readlines()

    energy = None
    for line in reversed(lines):
        if 'free  energy   TOTEN' in line:
            energy = float(line.split()[-2])
            break

    magmoms = []
    for i in range(len(lines) - 1, -1, -1):
        if 'magnetization (x)' in lines[i]:
            j = i + 4
            while j < len(lines):
                stripped = lines[j].strip()
                if not stripped or '---' in stripped:
                    j += 1
                    if magmoms:
                        break
                    continue
                parts = stripped.split()
                if len(parts) >= 5:
                    try:
                        magmoms.append(float(parts[-1]))
                    except ValueError:
                        break
                else:
                    break
                j += 1
            break

    return energy, np.array(magmoms) if magmoms else None


def load_from_poscar_dirs(poscar_dirs, spin_threshold=3.5, vo_element="Xe"):
    """Load from Trans_Set_*/ folders containing POSCAR_VO + OUTCAR."""
    structures = []

    for base_dir in poscar_dirs:
        dir_name = os.path.basename(base_dir.rstrip('/'))
        comp_match = re.search(r'(\d+)', dir_name)
        comp_str = comp_match.group(1) if comp_match else dir_name
        prefix = dir_name.replace('Trans_Set_', '').replace(' ', '_')

        sub_dirs = sorted([
            d for d in glob.glob(os.path.join(base_dir, '*'))
            if os.path.isdir(d)
        ])

        count = 0
        for sd in sub_dirs:
            poscar_path = os.path.join(sd, 'POSCAR_VO')
            outcar_path = os.path.join(sd, 'OUTCAR')

            if not (os.path.isfile(poscar_path) and os.path.isfile(outcar_path)):
                continue

            # Parse POSCAR_VO
            with open(poscar_path) as f:
                poscar_text = f.read()
            lattice, labels, frac_coords = parse_poscar_text(poscar_text, vo_element)

            # Parse OUTCAR
            energy, magmoms = parse_outcar(outcar_path)
            if energy is None:
                continue

            # Map magmoms → Ising spins
            # OUTCAR has only real atoms (no VO), so map back
            real_idx = [i for i, l in enumerate(labels) if l != vo_element]
            spins = [0] * len(labels)

            if magmoms is not None and len(magmoms) == len(real_idx):
                for ri, mag in zip(real_idx, magmoms):
                    if labels[ri] == 'Fe' and abs(mag) > spin_threshold:
                        spins[ri] = int(np.sign(mag))
            elif magmoms is not None:
                print(f"  ⚠ Atom count mismatch in {sd}: "
                      f"POSCAR real={len(real_idx)}, OUTCAR={len(magmoms)}")
                continue

            struct_id = f"{prefix}_{count:04d}"
            structures.append({
                'id': struct_id,
                'energy': float(energy),
                'comp': comp_str,
                'lattice': lattice,
                'species': labels,
                'frac_coords': frac_coords,
                'spins': spins,
                'n_atoms': len(labels),
                'source': dir_name,
            })
            count += 1

        print(f"  [{dir_name}] {count} structures, comp={comp_str}")

    return structures


# ═══════════════════════════════════════════════════════════════════════
# CIF WRITER
# ═══════════════════════════════════════════════════════════════════════

def write_cif(path, lattice, species, frac_coords):
    """Write minimal CIF file from lattice + species + fractional coords.
    
    Uses pymatgen for reliable CIF output with proper symmetry handling.
    """
    from pymatgen.core.structure import Structure
    from pymatgen.core import Lattice, DummySpecies as DummySp, Element

    # Build species list for pymatgen
    pm_species = []
    for sp in species:
        try:
            pm_species.append(Element(sp))
        except ValueError:
            pm_species.append(DummySp(sp))

    lat = Lattice(lattice)
    struct = Structure(lat, pm_species, frac_coords)
    struct.to(filename=path)





# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    if not args.pkl and not args.mag_data and not args.poscar_dirs:
        print("Error: at least one data source required (--pkl, --mag_data, --poscar_dirs)")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # ── Collect from all sources ──────────────────────────────────────
    all_structures = []

    if args.pkl:
        print(f"\n=== Loading from PKL ({len(args.pkl)} files) ===")
        all_structures.extend(load_from_pkl(args.pkl, args.vo_element))

    if args.mag_data:
        print(f"\n=== Loading from mag_data ({len(args.mag_data)} files) ===")
        all_structures.extend(load_from_mag_data(args.mag_data))

    if args.poscar_dirs:
        print(f"\n=== Loading from POSCAR dirs ({len(args.poscar_dirs)} dirs) ===")
        all_structures.extend(load_from_poscar_dirs(
            args.poscar_dirs, args.spin_threshold, args.vo_element))

    if not all_structures:
        print("Error: no structures loaded from any source.")
        sys.exit(1)

    # ── Check for ID collisions ───────────────────────────────────────
    id_counts = Counter(s['id'] for s in all_structures)
    dupes = {k: v for k, v in id_counts.items() if v > 1}
    if dupes:
        print(f"\n⚠ Duplicate IDs detected, adding source prefix:")
        for s in all_structures:
            s['id'] = f"{s['source']}_{s['id']}"
        # Verify uniqueness after fix
        id_counts2 = Counter(s['id'] for s in all_structures)
        dupes2 = {k: v for k, v in id_counts2.items() if v > 1}
        if dupes2:
            print(f"  ERROR: still have duplicates: {dupes2}")
            sys.exit(1)

    # ── Write CIF files ───────────────────────────────────────────────
    print(f"\n=== Writing {len(all_structures)} CIF files → {args.output}/ ===")
    all_species = set()

    for i, s in enumerate(all_structures):
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(all_structures)}")

        cif_path = os.path.join(args.output, f"{s['id']}.cif")
        write_cif(cif_path, s['lattice'], s['species'], s['frac_coords'])
        all_species.update(s['species'])

    # ── Write detailed_info.csv ───────────────────────────────────────
    csv_path = os.path.join(args.output, 'detailed_info.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'total_energy', 'comp', 'n_atoms', 'spins'])
        for s in all_structures:
            writer.writerow([
                s['id'],
                f"{s['energy']:.8f}",
                s['comp'],
                s['n_atoms'],
                json.dumps(s['spins']),
            ])
    print(f"  CSV → {csv_path}")

    # ── Summary ───────────────────────────────────────────────────────
    comp_counts = Counter(s['comp'] for s in all_structures)
    source_counts = Counter(s['source'] for s in all_structures)
    n_atoms_set = set(s['n_atoms'] for s in all_structures)
    has_spin = any(any(sp != 0 for sp in s['spins']) for s in all_structures)

    print(f"\n{'='*60}")
    print(f"  CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"  Output:       {args.output}/")
    print(f"  Structures:   {len(all_structures)}")
    print(f"  Species:      {sorted(all_species)}")
    print(f"  Atoms/struct: {sorted(n_atoms_set)}")
    print(f"  Has spin:     {has_spin}")
    print(f"  Compositions: {dict(sorted(comp_counts.items()))}")
    print(f"  Sources:      {dict(sorted(source_counts.items()))}")
    print(f"\n  Files:")
    print(f"    {len(all_structures)} .cif files")
    print(f"    detailed_info.csv ({len(all_structures)} rows)")


if __name__ == '__main__':
    main()
