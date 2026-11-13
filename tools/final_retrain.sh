#!/usr/bin/env bash
# HUMAN-LED FINAL STAGE. Retrain the chosen ensemble configs on train+val (--full) at FIXED epoch counts
# (validated: short training acts as early stopping; longer runs r0028-r0030 were worse).
# Each member is trained from the exact train.py snapshot of its screening run.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python}"
mkdir -p runs_final
#      name            source run   epochs  seed
MEMBERS=(
  "xception_s0     r0019        4       10"
  "xception_s1     r0034        4       11"
  "densenet121     r0036        5       12"
  "b0_degrade      r0014        8       13"
)
for m in "${MEMBERS[@]}"; do
  read -r NAME SRC EP SEED <<< "$m"
  OUT="runs_final/$NAME"
  if [[ -f "$OUT/model.pt" ]]; then echo "[final] $NAME exists, skipping"; continue; fi
  cp "runs/$SRC/train.py.snapshot" "_final_${NAME}.py"
  echo "[final] $NAME <- $SRC snapshot | --full --epochs $EP --seed $SEED"
  "$PY" "_final_${NAME}.py" --run_dir "$OUT" --full --epochs "$EP" --seed "$SEED" > "runs_final/$NAME.log" 2>&1
  cp "_final_${NAME}.py" "$OUT/train.py.snapshot"; rm "_final_${NAME}.py"
  grep -E "epoch|saved" "runs_final/$NAME.log" | tail -2
done
echo "[final] done"
