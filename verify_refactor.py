"""
verify_refactor.py — Xe→DummySpecies + species_map 0:3 + data_jax 이동 검증

사용:
    conda activate neuralce2
    cd /u/jaejunl3/GPUAlloy
    python verify_refactor.py
"""

import os, sys, subprocess, yaml, tempfile
from collections import Counter

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

os.environ['CONFIG_PATH'] = f'{REPO}/configs/tuning_unified/stfo_wo_spin.yaml'

CIF = f'{REPO}/data/processed/stfo_wo_spin/125_0000.cif'


def step(n, msg):
    print(f"\n[{n}] {msg}")


def ok(msg):
    print(f"  ✓ {msg}")


def main():
    # ── 1. Imports ────────────────────────────────────────────────────
    step(1, "Import 검증")
    from neuralce.utils.cif_utils import load_cif_safe, get_specie_number
    from neuralce.data.data_jax import (
        GaussianDistance, load_atom_embeddings, process_crystal)
    from neuralce.analysis import check_model_quality, refresh_predictions_heavy
    from neuralce.training import (
        retrain_per_comp, retrain_per_comp_unified,
        ablation_unified, ablation_comp)
    import data_jax  # shim
    ok("모든 모듈 import 성공")
    ok("data_jax shim 동작")

    # ── 2. CIF 로딩 ────────────────────────────────────────────────────
    step(2, "CIF 로딩 및 Xe→DummySpecies 복원")
    from pymatgen.core import DummySpecies

    assert 'Xe' in open(CIF).read(), "raw CIF에 Xe 없음"
    s = load_cif_safe(CIF)
    species_str = Counter(str(x.specie) for x in s)
    z_counts = Counter(get_specie_number(x.specie) for x in s)
    print(f"      species: {dict(species_str)}")
    print(f"      Z     : {dict(z_counts)}")
    assert 'Xe' not in species_str, f"Xe 복원 안 됨: {species_str}"
    n_dummy = sum(1 for x in s if isinstance(x.specie, DummySpecies))
    assert n_dummy > 0, "DummySpecies 없음"
    assert z_counts[0] == n_dummy, f"Z=0 카운트({z_counts[0]}) != dummy({n_dummy})"
    ok(f"Xe {n_dummy}개 → DummySpecies(Z=0)")

    # ── 3. YAML species_map ───────────────────────────────────────────
    step(3, "YAML species_map 0→3 매핑")
    cfg = yaml.safe_load(open(os.environ['CONFIG_PATH']))
    smap = {int(k): v for k, v in cfg['species_map'].items()}
    print(f"      species_map: {smap}")
    assert smap.get(0) == 3, "species_map[0] != 3"
    n_vac = sum(1 for site in s if get_specie_number(site.specie) == 0)
    assert n_vac > 0
    ok(f"species_map[0]=3, vacancy {n_vac}개")

    # ── 4. 전역 치환 잔재 확인 ────────────────────────────────────────
    step(4, "전역 치환 잔재 확인")

    r1 = subprocess.run(['grep', '-rn', '-E', r'^\s*54:\s+3', f'{REPO}/configs'],
                        capture_output=True, text=True)
    assert not r1.stdout, f"54:3 남음:\n{r1.stdout}"
    ok("configs/ 아래 54:3 없음")

    r2 = subprocess.run(['grep', '-rln', 'Structure.from_file',
                         f'{REPO}/neuralce', f'{REPO}/analyze_cutoffs.py'],
                        capture_output=True, text=True)
    bad = [l for l in r2.stdout.splitlines() if 'cif_utils.py' not in l]
    assert not bad, f"Structure.from_file 남음:\n{chr(10).join(bad)}"
    ok("Structure.from_file 잔재 없음 (cif_utils.py 구현부만 허용)")

    # ── 5. 그래프 빌드 파이프라인 ────────────────────────────────────
    step(5, "그래프 빌드 (vacancy → atom_fea[:, 3])")
    import numpy as np
    from pymatgen.core.structure import Structure
    from neuralce.training.ablation_unified import (
        build_graph_lite, SPECIES_MAP, N_SPECIES)

    excl = set(cfg.get('exclude_species', []))
    keep = [i for i, site in enumerate(s)
            if get_specie_number(site.specie) not in excl]
    crystal2 = Structure.from_sites([s[i] for i in keep])
    struct2 = {'crystal': crystal2,
               'spin_states': np.zeros(len(crystal2), dtype=np.float32)}

    af, nf, nfi, sf = build_graph_lite(
        struct2, cutoff=4.08, n_shells=4, max_num_nbr=12,
        include_sisj=False,
        shell_edges=[0.0, 2.41, 3.135, 3.705, 4.08])

    n_vac_graph = int(af[:, SPECIES_MAP[0]].sum())
    print(f"      atom_fea shape={af.shape}, n_species={N_SPECIES}")
    print(f"      graph vacancy idx={SPECIES_MAP[0]}: {n_vac_graph}")
    assert n_vac_graph == n_dummy, f"{n_vac_graph} != {n_dummy}"
    ok(f"vacancy → atom_fea[:, {SPECIES_MAP[0]}] 정확히 인코딩")

    # ── 6. check_model_quality (기존 ckpt 추론) ──────────────────────
    step(6, "기존 ckpt로 추론 (선택)")
    ckpt = f'{REPO}/best_pkl/retrained/retrained_stfo_wo_spin_ising_lite_2.pkl'
    if not os.path.exists(ckpt):
        print(f"      SKIP: ckpt 없음 ({ckpt})")
    else:
        # YAML 내부 상대경로 절대화
        cfg2 = yaml.safe_load(open(f'{REPO}/configs/tuning/stfo_wo_spin.yaml'))
        for k in ('cif_dir', 'csv_path', 'spin_pkl'):
            v = cfg2.get(k)
            if v and v.startswith('./'):
                cfg2[k] = os.path.normpath(os.path.join(REPO, v))
        tmp = tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False)
        yaml.dump(cfg2, tmp)
        tmp.close()

        from neuralce.analysis.check_model_quality import run_check
        r = run_check(tmp.name, ckpt, plot=False)
        srcc = r['global_metrics']['srcc']
        r2 = r['global_metrics']['r2']
        print(f"      global SRCC={srcc:.4f}, R²={r2:.4f}")
        assert srcc > 0
        ok("기존 ckpt 추론 정상")

    print("\n" + "=" * 60)
    print("  전체 PASS — Xe→DummySpecies 전역 수정 검증 완료")
    print("=" * 60)


if __name__ == '__main__':
    main()
