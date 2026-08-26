#!/usr/bin/env bash
# Validate the plumbing before anyone spends money on it.
#
# Deliberately asserts on MECHANICS, never on pass rates. A smoke run on a cheap
# endpoint tells you nothing about whether saaga helps -- results from one model
# do not transfer to another. What it does tell you is whether the four arms are
# actually four different experiments, which is the failure that would otherwise
# be discovered only after the full spend.

set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PY:-./.venv/bin/python}"

echo "=============================================="
echo " 1. Unit checks: contamination + corpus + arm D"
echo "=============================================="
"$PY" -m pytest -q

echo
echo "=============================================="
echo " 2. Arms must be four distinct configurations"
echo "=============================================="
"$PY" - <<'PYEOF'
import sys
sys.path.insert(0, "src")
from saaga_eval.arms import ARMS, CORE_2X2, get_arm

seen = {}
for key in CORE_2X2:
    arm = get_arm(key)
    signature = (arm.plan_type, arm.remove_docs)
    if signature in seen:
        raise SystemExit(
            f"FAIL: arms {seen[signature]!r} and {key!r} resolve to the same "
            f"configuration {signature}. They would silently measure the same thing."
        )
    seen[signature] = key
    print(f"  ok  {key:20s} plan_type={arm.plan_type:16s} remove_docs={arm.remove_docs}")

saaga_arms = {k for k in CORE_2X2 if get_arm(k).has_saaga}
stripped = {k for k in CORE_2X2 if get_arm(k).remove_docs}
assert len(saaga_arms) == 2, "design needs exactly two saaga arms"
assert len(stripped) == 2, "design needs exactly two stripped arms"
assert saaga_arms & stripped, "design needs a saaga+stripped cell (arm D)"
print("  ok  2x2 is complete and balanced")
PYEOF

echo
echo "=============================================="
echo " 3. Contamination check fires on a poisoned doc"
echo "=============================================="
"$PY" - <<'PYEOF'
import sys
sys.path.insert(0, "src")
from saaga_eval.contamination import blocking, check_instance

patch = "+++ b/m.py\n+def reticulate_splines(x):\n+    return x\n"
clean = {"saaga-docs/a.md": "This module handles geometry."}
poisoned = {"saaga-docs/a.md": "Call reticulate_splines to normalise input."}

assert not blocking(check_instance(clean, patch, [])), "FAIL: false positive on clean docs"
assert blocking(check_instance(poisoned, patch, [])), "FAIL: leak not detected"
print("  ok  detects a leaked symbol, stays quiet on clean docs")
PYEOF

echo
echo "=============================================="
echo " 4. Every arm resolves through AGENTbench"
echo "=============================================="
if "$PY" -c "import configs" 2>/dev/null; then
  for arm in baseline saaga stripped_baseline saaga_substitution; do
    NVIDIA_NIM_API_KEY="${NVIDIA_NIM_API_KEY:-nvapi-dry-run}" \
      "$PY" scripts/run_arm.py --arm "$arm" \
        --exec-model "${SMOKE_MODEL:-nim:zai/glm-5.2}" \
        --slice-spec ":2" --workers 1 --dry-run >/dev/null
    echo "  ok  $arm resolves"
  done
else
  echo "  SKIP  AGENTbench not installed:"
  echo "        git clone https://github.com/eth-sri/agentbench vendor/agentbench"
  echo "        pip install -e vendor/agentbench"
fi

echo
echo "=============================================="
echo " Remaining checks require a live run"
echo "=============================================="
cat <<'EOF'
Run two instances on NIM's free tier, then verify by hand:

  export NVIDIA_NIM_API_KEY=nvapi-...
  python scripts/run_arm.py --arm stripped_baseline \
      --exec-model nim:zai/glm-5.2 --slice-spec ":2" --workers 1
  python scripts/run_arm.py --arm saaga_substitution \
      --exec-model nim:zai/glm-5.2 --slice-spec ":2" --workers 1

(C and D are the informative pair to smoke first: they exercise doc stripping,
which is where the silent failure lives. Copy the exact model id from
build.nvidia.com. Keep --workers at 1-2 if the endpoint is rate limited.)

Then confirm, in the run output:

  [ ] saaga-docs/ is present in the container for arms B and D
  [ ] the repo's own README/docs are GONE in arms C and D
  [ ] saaga-docs/ SURVIVED in arm D  (this is the one that silently breaks)
  [ ] the agent's first turns differ between arms (diff the trajectories)
  [ ] a patch was produced and evaluate.py ran the tests
  [ ] analyze.py emitted token and cost columns, not just pass/fail
  [ ] network egress was blocked during solve

Note on that last one: AGENTbench starts containers with --network=host, so an
agent can reach github.com and fetch the upstream fix. Git history is already
scrubbed by the harness (_clean_git_history), but the network route is not.
EOF

echo
echo "Smoke checks passed."
