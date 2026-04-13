"""
test_imports.py — 패키지 import 테스트

Usage:
    cd /u/jaejunl3/GPUAlloy
    pip install -e . -q
    python tests/test_imports.py
"""

import sys

PASS = "  PASS"
FAIL = "  FAIL"


def check(label, fn):
    try:
        fn()
        print(f"{PASS}  {label}")
        return True
    except Exception as e:
        print(f"{FAIL}  {label}")
        print(f"        {type(e).__name__}: {e}")
        return False


results = []

print("=" * 55)
print("  neuralce import tests")
print("=" * 55)

# models
results.append(check(
    "neuralce.models.NeuralCE_jax — create_neuralce, LITE_MODELS",
    lambda: __import__("neuralce.models.NeuralCE_jax", fromlist=["create_neuralce", "LITE_MODELS"])
))
results.append(check(
    "neuralce.models.module_octa_CE",
    lambda: __import__("neuralce.models.module_octa_CE", fromlist=["*"])
))

# data
results.append(check(
    "neuralce.data.convert_dataset",
    lambda: __import__("neuralce.data.convert_dataset", fromlist=["main"])
))
results.append(check(
    "neuralce.data.convert_nipt",
    lambda: __import__("neuralce.data.convert_nipt", fromlist=["*"])
))

# analysis
results.append(check(
    "neuralce.analysis.plot_utils — plot_results, plot_parity",
    lambda: __import__("neuralce.analysis.plot_utils", fromlist=["plot_results", "plot_parity"])
))
results.append(check(
    "neuralce.analysis.plot_parity",
    lambda: __import__("neuralce.analysis.plot_parity", fromlist=["*"])
))
results.append(check(
    "neuralce.analysis.plot_mixing_enthalpy",
    lambda: __import__("neuralce.analysis.plot_mixing_enthalpy", fromlist=["*"])
))
results.append(check(
    "neuralce.analysis.analyze_cutoffs — main",
    lambda: __import__("neuralce.analysis.analyze_cutoffs", fromlist=["main"])
))

# training (ablation은 CONFIG_PATH 필요 → 별도 테스트)
results.append(check(
    "neuralce.training.retrain — main",
    lambda: __import__("neuralce.training.retrain", fromlist=["main"])
))
results.append(check(
    "neuralce.training.retrain_per_comp — main",
    lambda: __import__("neuralce.training.retrain_per_comp", fromlist=["main"])
))

# mcmc (CONFIG_PATH 필요 → 별도 테스트)
# pt_mcmc은 모듈 레벨에서 config 안 읽으므로 import 가능
results.append(check(
    "neuralce.mcmc.pt_mcmc — run_pt_mcmc",
    lambda: __import__("neuralce.mcmc.pt_mcmc", fromlist=["run_pt_mcmc"])
))

print("=" * 55)
n_pass = sum(results)
n_fail = len(results) - n_pass
print(f"  결과: {n_pass}/{len(results)} PASS  ({n_fail} FAIL)")
print("=" * 55)

if n_fail:
    sys.exit(1)
