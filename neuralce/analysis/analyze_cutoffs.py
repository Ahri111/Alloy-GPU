"""
analyze_cutoffs.py — Optimal Cutoff & Shell Analyzer
=====================================================
Reads dataset info from a YAML config, analyzes pairwise distances,
finds natural distance clusters, and writes graph settings back to config.

Usage:
  python analyze_cutoffs.py --config ./configs/tuning/stfo_wo_spin.yaml
  python analyze_cutoffs.py --config ./configs/tuning/stfo_wo_spin.yaml --tier balanced --write
  python analyze_cutoffs.py --config ./configs/tuning/stfo_wo_spin.yaml --tier comprehensive --write --output ./configs/tuning/stfo_comp.yaml
"""

import os, sys, glob, argparse
import numpy as np
import yaml
from collections import defaultdict
from pymatgen.core.structure import Structure
from pymatgen.core import DummySpecies
from neuralce.utils.cif_utils import load_cif_safe, get_specie_number


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Analyze crystal distances and find optimal cutoff/shell config.")
    p.add_argument("--config", type=str, required=True,
                   help="Path to YAML config (e.g. ./configs/stfo_without_sr.yaml)")
    p.add_argument("--max_dist", type=float, default=8.0,
                   help="Maximum distance to scan (Å). Default: 8.0")
    p.add_argument("--sample_n", type=int, default=10,
                   help="Number of CIFs to sample. Default: 10")
    p.add_argument("--cluster_tol", type=float, default=0.10,
                   help="Gap threshold to merge into same cluster (Å). Default: 0.10")

    # Write modes (mutually exclusive)
    write_grp = p.add_mutually_exclusive_group()
    write_grp.add_argument("--write", action="store_true",
                           help="Write single cutoff (best of --tier) to config.")
    write_grp.add_argument("--write-candidates", action="store_true",
                           help="Write top-N cutoff candidates to config for Optuna.")

    p.add_argument("--tier", type=str, default=None,
                   choices=["minimal", "balanced", "comprehensive"],
                   help="For --write: select this tier. Default: auto.")
    p.add_argument("--top_n", type=int, default=3,
                   help="For --write-candidates: number of candidates. Default: 3")
    p.add_argument("--output", type=str, default=None,
                   help="Output YAML path. Default: overwrites input config.")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def get_label(site):
    if isinstance(site.specie, DummySpecies):
        return "Vo"
    return site.specie.symbol


def pair_key(s1, s2):
    return "-".join(sorted([s1, s2]))


def is_metal(symbol):
    non_metals = {'O', 'S', 'Se', 'Te', 'N', 'P', 'F', 'Cl', 'Br', 'I', 'Vo', 'H', 'C'}
    return symbol not in non_metals


# ═══════════════════════════════════════════════════════════════════════
# CORE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def collect_distances(cif_dir, max_dist, exclude_z, sample_n=10, round_dec=2):
    cif_files = sorted(glob.glob(os.path.join(cif_dir, "*.cif")))
    if not cif_files:
        raise FileNotFoundError(f"No CIF files in {cif_dir}")
    if sample_n and len(cif_files) > sample_n:
        step = max(1, len(cif_files) // sample_n)
        cif_files = cif_files[::step][:sample_n]

    print(f"Scanning {len(cif_files)} CIF files, max_dist={max_dist} Å")

    dist_pair = defaultdict(lambda: defaultdict(int))
    per_cif_dists = {}
    all_species = set()

    # cutoff → max per-atom neighbor count across all CIFs
    # key: cutoff (float), value: max neighbor count seen in the largest structure
    # We track per-CIF neighbor counts keyed by (cif_id, atom_i) to find
    # the worst-case (largest structure).
    # Strategy: for each CIF, record the max per-atom neighbor count at max_dist,
    # then store (n_atoms, nbr_counts) so we can later query any cutoff threshold.
    per_cif_nbr_counts = {}  # cif_id → list of (dist,) for each atom (sorted)

    for cif_path in cif_files:
        cif_id = os.path.basename(cif_path).replace('.cif', '')
        crystal = load_cif_safe(cif_path)
        if exclude_z:
            keep = [i for i, s in enumerate(crystal) if get_specie_number(s.specie) not in exclude_z]
            crystal = Structure.from_sites([crystal[i] for i in keep])

        labels = [get_label(s) for s in crystal]
        all_species.update(labels)
        cif_dists = set()

        all_nbrs = crystal.get_all_neighbors(max_dist, include_index=True)

        # Store all neighbor distances per atom for later cutoff queries
        atom_nbr_dists = []
        for i, nbrs in enumerate(all_nbrs):
            atom_dists = []
            for nbr in nbrs:
                d = round(nbr[1], round_dec)
                pk = pair_key(labels[i], labels[nbr[2]])
                dist_pair[d][pk] += 1
                cif_dists.add(d)
                atom_dists.append(nbr[1])  # raw float for cutoff filtering
            atom_nbr_dists.append(sorted(atom_dists))

        per_cif_dists[cif_id] = sorted(cif_dists)
        per_cif_nbr_counts[cif_id] = (len(crystal), atom_nbr_dists)

    return dist_pair, per_cif_dists, all_species, per_cif_nbr_counts


def compute_max_num_nbr(per_cif_nbr_counts, cutoff):
    """
    For a given cutoff, find the maximum per-atom neighbor count across all CIFs.
    Scans ALL structures (not just largest) to ensure true worst-case.
    """
    max_count = 0
    for cif_id, (n_atoms, atom_nbr_dists) in per_cif_nbr_counts.items():
        for atom_dists in atom_nbr_dists:
            count = sum(1 for d in atom_dists if d < cutoff)
            if count > max_count:
                max_count = count
    return max_count


def find_clusters(dist_pair, cluster_tol=0.10):
    sorted_d = np.array(sorted(dist_pair.keys()))
    if len(sorted_d) < 2:
        return [sorted_d.tolist()], np.array([])

    gaps = np.diff(sorted_d)
    clusters = []
    current = [sorted_d[0]]
    for i in range(len(gaps)):
        if gaps[i] < cluster_tol:
            current.append(sorted_d[i + 1])
        else:
            clusters.append(current)
            current = [sorted_d[i + 1]]
    clusters.append(current)

    inter_gaps = []
    for i in range(len(clusters) - 1):
        g = float(min(clusters[i + 1]) - max(clusters[i]))
        inter_gaps.append(round(g, 4))

    return clusters, np.array(inter_gaps)


def get_shell_info(clusters, dist_pair, b_species):
    shells = []
    for i, cluster in enumerate(clusters):
        total = sum(sum(dist_pair[d].values()) for d in cluster)
        pairs = defaultdict(int)
        for d in cluster:
            for p, c in dist_pair[d].items():
                pairs[p] += c

        has_bb, has_bo, has_oo = False, False, False
        for p in pairs:
            s1, s2 = p.split('-')
            m1, m2 = s1 in b_species, s2 in b_species
            if m1 and m2:
                has_bb = True
            elif m1 or m2:
                has_bo = True
            else:
                has_oo = True

        shells.append({
            'shell': i,
            'd_min': float(min(cluster)), 'd_max': float(max(cluster)),
            'n_dists': len(cluster),
            'total_edges': total,
            'pairs': dict(sorted(pairs.items())),
            'has_bb': has_bb, 'has_bo': has_bo, 'has_oo': has_oo,
            'n_unique_pairs': len(pairs),
        })
    return shells


def score_cutoffs(clusters, inter_gaps, shells):
    candidates = []
    for i in range(len(inter_gaps)):
        n_shells = i + 1
        gap = float(inter_gaps[i])
        cutoff = round(float(max(clusters[i])) + 0.05, 3)

        included = shells[:n_shells]

        # Interaction diversity
        all_pairs_seen = set()
        for s in included:
            all_pairs_seen.update(s['pairs'].keys())
        n_pair_types = len(all_pairs_seen)

        # Category coverage
        has_bb = any(s['has_bb'] for s in included)
        has_bo = any(s['has_bo'] for s in included)
        has_oo = any(s['has_oo'] for s in included)
        category_coverage = sum([has_bb, has_bo, has_oo])

        # Gap safety
        gap_safety = min(gap / 0.5, 1.0)

        # Marginal gain
        if n_shells >= 2:
            prev_pairs = set()
            for s in included[:-1]:
                prev_pairs.update(s['pairs'].keys())
            marginal_new = len(set(included[-1]['pairs'].keys()) - prev_pairs)
        else:
            marginal_new = len(all_pairs_seen)

        total_edges = sum(s['total_edges'] for s in included)

        # Scoring
        if n_shells < 2:
            score = -10.0
        else:
            score = n_pair_types * 0.8
            score += category_coverage * 1.5
            if has_bb:
                score += 2.0
            score += gap_safety * 1.0
            score -= max(0, n_shells - 4) * 0.5
            if marginal_new == 0:
                score -= 1.0

        # Tier assignment
        if n_shells < 2 or not has_bb:
            tier = "incomplete"
        elif n_shells <= 4:
            tier = "minimal" if n_shells <= 3 else "balanced"
        elif n_shells <= 6:
            tier = "balanced" if n_shells <= 5 else "comprehensive"
        else:
            tier = "comprehensive"

        # Boundaries
        boundaries = []
        for j in range(n_shells - 1):
            mid = round((float(max(clusters[j])) + float(min(clusters[j + 1]))) / 2, 3)
            boundaries.append(mid)

        candidates.append({
            'cutoff': cutoff,
            'n_shells': n_shells,
            'gap': gap,
            'gap_safety': round(gap_safety, 3),
            'n_pair_types': n_pair_types,
            'category_coverage': category_coverage,
            'has_bb': has_bb,
            'marginal_new': marginal_new,
            'total_edges': total_edges,
            'score': round(score, 2),
            'tier': tier,
            'boundaries': boundaries,
        })

    return candidates


# ═══════════════════════════════════════════════════════════════════════
# REPORT & WRITE
# ═══════════════════════════════════════════════════════════════════════

def print_report(shells, inter_gaps, candidates, cluster_tol, per_cif_nbr_counts=None):
    # Shells
    print(f"\n{'═' * 85}")
    print(f"  DISTANCE CLUSTERS (cluster_tol={cluster_tol} Å)")
    print(f"{'═' * 85}")
    for s in shells:
        d_range = (f"{s['d_min']:.2f}" if s['d_min'] == s['d_max']
                   else f"{s['d_min']:.2f}–{s['d_max']:.2f}")
        bb_mark = " ★B-B" if s['has_bb'] else ""
        print(f"\n  Shell {s['shell']}: {d_range} Å  "
              f"({s['n_dists']} dists, {s['total_edges']} edges){bb_mark}")
        sorted_pairs = sorted(s['pairs'].items(), key=lambda x: -x[1])
        pair_strs = [f"{p}:{c}" for p, c in sorted_pairs[:8]]
        print(f"         {', '.join(pair_strs)}")

    # Gaps
    print(f"\n{'═' * 85}")
    print(f"  INTER-CLUSTER GAPS")
    print(f"{'═' * 85}")
    for i, g in enumerate(inter_gaps):
        bar = "██████" if g > 0.4 else "████" if g > 0.2 else "██" if g > 0.1 else "█"
        print(f"  Shell {i} → {i+1}:  gap = {g:.3f} Å  {bar}")

    # Candidates
    print(f"\n{'═' * 85}")
    print(f"  ALL CUTOFF CANDIDATES (sorted by score)")
    print(f"{'═' * 85}")
    print(f"  {'Cut':>6} {'Shells':>6} {'Gap':>6} {'Pairs':>5} "
          f"{'Cat':>3} {'B-B':>4} {'New':>4} {'Score':>6}  Tier")
    print(f"  {'------':>6} {'------':>6} {'------':>6} {'-----':>5} "
          f"{'---':>3} {'----':>4} {'----':>4} {'------':>6}  ----")
    for c in sorted(candidates, key=lambda x: -x['score']):
        bb = "✓" if c['has_bb'] else "·"
        print(f"  {c['cutoff']:6.2f} {c['n_shells']:6d} {c['gap']:6.3f} "
              f"{c['n_pair_types']:5d} {c['category_coverage']:3d} "
              f"{bb:>4} {c['marginal_new']:4d} {c['score']:6.2f}  {c['tier']}")

    # Tier recommendations
    print(f"\n{'═' * 85}")
    print(f"  RECOMMENDATIONS BY TIER")
    print(f"{'═' * 85}")

    # Coordination summary table
    if per_cif_nbr_counts:
        print(f"\n  COORDINATION SUMMARY (max_num_nbr required per cutoff):")
        print(f"  {'Cutoff':>8} {'Shells':>6} {'max_num_nbr':>12} {'Tier':>15}")
        print(f"  {'--------':>8} {'------':>6} {'------------':>12} {'---------------':>15}")
        for c in sorted(candidates, key=lambda x: x['cutoff']):
            if c['score'] <= 0:
                continue
            mnbr = compute_max_num_nbr(per_cif_nbr_counts, c['cutoff'])
            trunc = " ⚠ TRUNCATED @12" if mnbr > 12 else ""
            print(f"  {c['cutoff']:8.2f} {c['n_shells']:6d} {mnbr:12d}  {c['tier']:>14}{trunc}")

    tier_bests = {}
    for tier_name in ["minimal", "balanced", "comprehensive"]:
        tier_cands = [c for c in candidates if c['tier'] == tier_name and c['score'] > 0]
        if not tier_cands:
            continue
        best = max(tier_cands, key=lambda x: x['score'])
        tier_bests[tier_name] = best
        edges = [0.0] + best['boundaries'] + [best['cutoff']]

        print(f"\n  [{tier_name.upper()}] cutoff={best['cutoff']:.3f} Å, "
              f"{best['n_shells']} shells, score={best['score']:.2f}")
        mnbr = compute_max_num_nbr(per_cif_nbr_counts, best['cutoff']) if per_cif_nbr_counts else 12
        print(f"    Gap: {best['gap']:.3f} Å | Pairs: {best['n_pair_types']} | "
              f"B-B: {'YES' if best['has_bb'] else 'NO'} | max_num_nbr: {mnbr}")
        print(f"    SHELL_EDGES = {[round(e, 3) for e in edges]}")

        for j in range(best['n_shells']):
            s = shells[j]
            d_range = (f"{s['d_min']:.2f}" if s['d_min'] == s['d_max']
                       else f"{s['d_min']:.2f}–{s['d_max']:.2f}")
            top3 = sorted(s['pairs'].items(), key=lambda x: -x[1])[:3]
            top3_str = ", ".join(f"{p}:{c}" for p, c in top3)
            bb = " ★" if s['has_bb'] else ""
            print(f"      Shell {j}: {d_range} Å → {top3_str}{bb}")

    return tier_bests


def write_config(cfg, tier_bests, tier, config_path, output_path, per_cif_nbr_counts):
    if tier not in tier_bests:
        print(f"\n⚠ Tier '{tier}' not available. Options: {list(tier_bests.keys())}")
        return

    chosen = tier_bests[tier]
    edges = [round(e, 3) for e in [0.0] + chosen['boundaries'] + [chosen['cutoff']]]
    max_num_nbr = compute_max_num_nbr(per_cif_nbr_counts, chosen['cutoff'])

    cfg['graph'] = {
        'cutoff': chosen['cutoff'],
        'n_shells': chosen['n_shells'],
        'shell_edges': edges,
        'max_num_nbr': max_num_nbr,
    }

    out_path = output_path or config_path
    with open(out_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)

    print(f"\n{'═' * 85}")
    print(f"  ✓ WRITTEN (single cutoff): {out_path}")
    print(f"{'═' * 85}")
    print(f"  graph:")
    for k, v in cfg['graph'].items():
        print(f"    {k}: {v}")


def write_candidates(cfg, candidates, top_n, config_path, output_path, per_cif_nbr_counts):
    """Write top-N cutoff candidates as graph.candidates dict to YAML."""
    # Filter: must have B-B, score > 0
    valid = [c for c in candidates if c['has_bb'] and c['score'] > 0]
    if not valid:
        print("\n⚠ No valid candidates with B-B pairs. Skipping write.")
        return

    # Sort by score, take top_n
    valid.sort(key=lambda x: -x['score'])
    selected = valid[:top_n]

    # Compute global max_num_nbr = max across all selected candidates
    # (single value covers all cutoffs in the candidate set)
    global_max_nbr = max(
        compute_max_num_nbr(per_cif_nbr_counts, c['cutoff'])
        for c in selected
    )

    # Build candidates dict — per-candidate max_num_nbr
    cand_dict = {}
    for c in selected:
        edges = [round(e, 3) for e in [0.0] + c['boundaries'] + [c['cutoff']]]
        mnbr = compute_max_num_nbr(per_cif_nbr_counts, c['cutoff'])
        entry = {
            'n_shells': c['n_shells'],
            'shell_edges': edges,
        }
        # Only add per-candidate max_num_nbr if it differs from global
        if mnbr != global_max_nbr:
            entry['max_num_nbr'] = mnbr
        cand_dict[c['cutoff']] = entry

    cfg['graph'] = {
        'max_num_nbr': global_max_nbr,
        'candidates': cand_dict,
    }

    out_path = output_path or config_path
    with open(out_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)

    print(f"\n{'═' * 85}")
    print(f"  ✓ WRITTEN ({len(selected)} candidates): {out_path}")
    print(f"{'═' * 85}")
    print(f"  graph:")
    print(f"    max_num_nbr: {global_max_nbr}")
    print(f"    candidates:")
    for cut, info in sorted(cand_dict.items()):
        mnbr = info.get('max_num_nbr', global_max_nbr)
        print(f"      {cut}:")
        print(f"        n_shells: {info['n_shells']}")
        if 'max_num_nbr' in info:
            print(f"        max_num_nbr: {info['max_num_nbr']}")
        print(f"        shell_edges: {info['shell_edges']}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # Read config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cif_dir   = cfg['cif_dir']
    exclude_z = set(cfg.get('exclude_species', []))
    if exclude_z:
        print(f"Excluding atomic numbers: {sorted(exclude_z)}")

    # Analyze
    dist_pair, per_cif_dists, all_species, per_cif_nbr_counts = collect_distances(
        cif_dir, args.max_dist, exclude_z, args.sample_n)

    b_species = {s for s in all_species if is_metal(s)}
    print(f"Species: {sorted(all_species)}, B-site: {sorted(b_species)}")

    dist_sets = [set(d) for d in per_cif_dists.values()]
    is_fixed = all(s == dist_sets[0] for s in dist_sets) if dist_sets else True
    print(f"Fixed lattice: {'YES ✓' if is_fixed else 'NO ✗'}")

    # Detect n_atoms from scanned CIFs
    atom_counts = set()
    for cif_id, (n_atoms, _) in per_cif_nbr_counts.items():
        atom_counts.add(n_atoms)
    if len(atom_counts) == 1:
        detected_n_atoms = atom_counts.pop()
        print(f"Atom count: {detected_n_atoms} (uniform)")
    else:
        detected_n_atoms = max(atom_counts)
        print(f"Atom counts: {sorted(atom_counts)} → n_atoms will be set to {detected_n_atoms} (max, for padding)")

    # Full scan for n_atoms if sample was small
    if args.sample_n and args.sample_n < 50:
        cif_files_all = sorted(glob.glob(os.path.join(cif_dir, "*.cif")))
        all_atom_counts = set()
        for cif_path in cif_files_all:
            crystal = load_cif_safe(cif_path)
            if exclude_z:
                keep = [i for i, s in enumerate(crystal) if get_specie_number(s.specie) not in exclude_z]
                crystal = Structure.from_sites([crystal[i] for i in keep])
            all_atom_counts.add(len(crystal))
        if all_atom_counts != atom_counts:
            detected_n_atoms = max(all_atom_counts)
            print(f"  Full scan: atom counts {sorted(all_atom_counts)} → n_atoms={detected_n_atoms}")

    clusters, inter_gaps = find_clusters(dist_pair, args.cluster_tol)
    shells = get_shell_info(clusters, dist_pair, b_species)
    candidates = score_cutoffs(clusters, inter_gaps, shells)

    # Report
    tier_bests = print_report(shells, inter_gaps, candidates, args.cluster_tol, per_cif_nbr_counts)

    # Write
    if args.write_candidates or args.write:
        # Auto-set n_atoms in config
        current_n_atoms = cfg.get('n_atoms')
        if current_n_atoms is None or str(current_n_atoms).lower() in ('variable', 'auto'):
            cfg['n_atoms'] = detected_n_atoms
            print(f"\n  Auto-set n_atoms: {detected_n_atoms}")
        elif isinstance(current_n_atoms, int) and len(atom_counts) > 1:
            if current_n_atoms < detected_n_atoms:
                cfg['n_atoms'] = detected_n_atoms
                print(f"\n  Updated n_atoms: {current_n_atoms} → {detected_n_atoms} (max atom count)")

    if args.write_candidates:
        write_candidates(cfg, candidates, args.top_n, args.config, args.output, per_cif_nbr_counts)
    elif args.write:
        if args.tier:
            tier = args.tier
        else:
            bb_cands = [c for c in candidates if c['has_bb'] and c['score'] > 0]
            if bb_cands:
                tier = max(bb_cands, key=lambda x: x['score'])['tier']
            else:
                print("\n⚠ No candidate with B-B pairs. Skipping write.")
                return
        write_config(cfg, tier_bests, tier, args.config, args.output, per_cif_nbr_counts)
    else:
        print(f"\n  Usage examples:")
        print(f"    Single cutoff:    python analyze_cutoffs.py --config {args.config} --tier balanced --write")
        print(f"    Top-3 candidates: python analyze_cutoffs.py --config {args.config} --write-candidates --top_n 3")


if __name__ == "__main__":
    main()
