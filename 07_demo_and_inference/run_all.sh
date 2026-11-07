#!/bin/bash
# Runs the whole demo in order. Read README.md first.
# Usage:  bash run_all.sh            (add sudo in front if tegrastats needs it)
cd "$(dirname "$0")"

echo "=== 0. the board ==="
cat /etc/nv_tegra_release 2>/dev/null | head -1
nvpmodel -q 2>/dev/null
python3 --version

echo; echo "=== 1. install (skip if already done) ==="
pip3 install -r requirements.txt

echo; echo "=== 2. inference on all 569 windows (writes to run_output/) ==="
python3 55_session_checkpoints.py --pack demo_sessions_pack --out run_output

echo; echo "=== 3. power: idle, running, idle ==="
python3 power_measure.py \
  --cmd "python3 55_session_checkpoints.py --pack demo_sessions_pack --no-explain --out power_run_output" \
  --windows 569 --repeat 5 --idle-seconds 60 --out run_output

echo; echo "=== done. Results are in run_output/ ==="
ls -l run_output
