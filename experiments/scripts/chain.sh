#!/bin/bash
# Wait for E2, then run E3/E4/E5 and E6 automatically.
cd /home/tensu-hiwi/Proposal_Real
while pgrep -f "e2_distill" >/dev/null; do sleep 30; done
echo "=== E2 finished at $(date +%H:%M:%S) ==="
if [ ! -f experiments/results/e2_distill_sst2.json ]; then
  echo "FATAL: e2_distill_sst2.json missing — E2 did not complete cleanly"; exit 1
fi
echo; echo "=== E3/E4/E5 ==="
.venv/bin/python experiments/scripts/e345_analyse.py --task sst2 2>&1 | grep -vE "Warning|warn"
echo; echo "=== E6 figures + tables ==="
.venv/bin/python experiments/scripts/e6_figures_tables.py --task sst2 2>&1 | grep -vE "Warning|warn"
echo; echo "=== PIPELINE COMPLETE $(date +%H:%M:%S) ==="
