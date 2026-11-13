#!/usr/bin/env bash
# FROZEN. One experiment = train.py + evaluate.py. Prints only the summary lines.
#   ./run_experiment.sh "short description of the hypothesis" [epochs] [seed]
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python}"
DESC="${1:?description required}"; EPOCHS="${2:-}"; SEED="${3:-0}"
mkdir -p runs
N=$(ls runs | grep -c '^r[0-9]' || true)
RUN="runs/r$(printf '%04d' $((N + 1)))"
mkdir -p "$RUN"
git diff HEAD -- train.py > "$RUN/train.diff" 2>/dev/null || true
cp train.py "$RUN/train.py.snapshot"
echo "[run] $RUN :: $DESC"
if ! "$PY" train.py --run_dir "$RUN" ${EPOCHS:+--epochs "$EPOCHS"} --seed "$SEED" > "$RUN/train.log" 2>&1; then
  echo "[run] TRAINING CRASHED. Last lines of $RUN/train.log:"; tail -n 25 "$RUN/train.log"
  printf '%s\t%s\tCRASH\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t%s\n' "$(date '+%Y-%m-%d %H:%M')" "$(basename $RUN)" "$DESC" >> results.tsv
  exit 1
fi
tail -n 2 "$RUN/train.log"
"$PY" evaluate.py --run_dir "$RUN" --desc "$DESC" 2>"$RUN/eval.err" | tee "$RUN/eval.log" | grep -E '^\[evaluate\] eval|^RESULT' \
  || { echo "[run] EVAL FAILED:"; tail -n 25 "$RUN/eval.err"; exit 1; }
