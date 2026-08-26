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
        --exec-model "${SMOKE_MODEL:-nim:moonshotai/kimi-k3}" \
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
echo " 5. Arms differ in a real container"
echo "=============================================="
# This is the check that used to require a 30-minute agent run. Both planners
# ignore the model argument, so it needs no API key and no tokens -- it sets up
# the container, plans, strips docs, and looks at the filesystem.
if command -v docker >/dev/null && docker info >/dev/null 2>&1; then
  if [ -d "${CORPUS_ROOT:-corpora}" ]; then
    ARMS="baseline saaga stripped_baseline saaga_substitution"
  else
    ARMS="baseline stripped_baseline"
    echo "  note: no corpus yet, checking only the two arms that do not need one"
  fi
  # shellcheck disable=SC2086
  "$PY" scripts/verify_arms.py --repo "${SMOKE_REPO:-huggingface/smolagents}" --arms $ARMS
else
  echo "  SKIP  docker unavailable"
fi

echo
echo "=============================================="
echo " Remaining checks require a live run"
echo "=============================================="
cat <<'EOF'
Run two instances on NIM's free tier, then verify by hand:

  export NVIDIA_NIM_API_KEY=nvapi-...
  python scripts/run_arm.py --arm stripped_baseline \
      --exec-model nim:moonshotai/kimi-k3 --slice-spec ":2" --workers 1
  python scripts/run_arm.py --arm saaga_substitution \
      --exec-model nim:moonshotai/kimi-k3 --slice-spec ":2" --workers 1

(C and D are the informative pair to smoke first: they exercise doc stripping,
which is where the silent failure lives. Arm C needs no corpus at all, so it can
run before any saaga init -- do it first and prove the pipeline for free.

Verify the model before running; a catalogue listing does not mean it is
deployed, and reasoning models need a large completion budget or
reasoning_content consumes it before any answer appears:

  python scripts/check_endpoint.py --key-file secret --model moonshotai/kimi-k3

Keep --workers 1 on a free tier; NIM returns 429 on back-to-back requests.)

Then confirm, in the run output:

  [ ] a patch was produced and evaluate.py ran the tests
  [ ] analyze.py emitted number_steps_first_read, not just pass/fail
  [ ] network egress was blocked during solve

(Step 5 above already covers corpus presence, doc stripping, and corpus
survival in arm D -- those no longer need an agent run.)

Note on that last one: AGENTbench starts containers with --network=host, so an
agent can reach github.com and fetch the upstream fix. Git history is already
scrubbed by the harness (_clean_git_history), but the network route is not.
EOF

echo
echo "Smoke checks passed."
