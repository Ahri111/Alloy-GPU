"""
test_ablation_import.py — CONFIG_PATH 필요한 스크립트 import 테스트

Usage:
    cd /u/jaejunl3/GPUAlloy
    python tests/test_ablation_import.py
"""

import sys
import os

PASS = "  PASS"
FAIL = "  FAIL"

CONFIG = "./configs/tuning/nipt_ablation.yaml"

if not os.path.exists(CONFIG):
    print(f"config 없음: {CONFIG}")
    sys.exit(1)

os.environ["CONFIG_PATH"] = CONFIG

print("=" * 55)
print("  CONFIG_PATH 필요한 스크립트 import 테스트")
print(f"  CONFIG_PATH={CONFIG}")
print("=" * 55)

results = []

def check(label, fn):
    try:
        fn()
        print(f"{PASS}  {label}")
        return True
    except Exception as e:
        print(f"{FAIL}  {label}")
        print(f"        {type(e).__name__}: {e}")
        return False

results.append(check(
    "neuralce.training.ablation — main",
    lambda: __import__("neuralce.training.ablation", fromlist=["main"])
))
results.append(check(
    "neuralce.training.ablation_comp — main",
    lambda: __import__("neuralce.training.ablation_comp", fromlist=["main"])
))

print("=" * 55)
n_pass = sum(results)
n_fail = len(results) - n_pass
print(f"  결과: {n_pass}/{len(results)} PASS  ({n_fail} FAIL)")
print("=" * 55)

if n_fail:
    sys.exit(1)
