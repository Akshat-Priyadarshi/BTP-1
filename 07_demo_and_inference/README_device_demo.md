# Final_Demo — stress scoring on a Jetson Orin Nano

Scores five recorded wrist-wearable sessions (569 windows of 60 s) with the
trained model, shows the latency of every window, and measures the power the
model costs: the board idle versus the board running the model.

No survey answers exist for these sessions, so there is no accuracy here.
This shows that the model runs on the device, gives the same numbers as the
laptop, and what it costs.

## What is in the folder

| Item | What it is |
|---|---|
| `55_session_checkpoints.py` | The inference. Reads each session, cuts 60 s windows every 30 s, scores every one (569), marks the 59 checkpoint windows (every 5 min), times each window, measures what each context column was worth, and compares with the laptop's scores. |
| `power_measure.py` | The power measurement. Reads the board's power from `tegrastats` while idle, running the model, and idle again; the difference is the model's cost. |
| `run_all.sh` | Runs the steps below in order. |
| `requirements.txt` | Python packages. `numpy<2` must stay. |
| `adastress/` | The model classes (encoder, gate, classifier). The saved model cannot be loaded without it. |
| `demo_sessions_pack/` | The data and the model. |
| `demo_sessions_pack/model.pth` | Classifier and gate (the 7-context model). |
| `demo_sessions_pack/encoder.pth` | The frozen encoder. |
| `demo_sessions_pack/scaler.pkl` | The scaling the model was trained with. |
| `demo_sessions_pack/sessions.csv` | The five sessions: start, length, checkpoint step. |
| `demo_sessions_pack/<session>/signals.csv` | 7 wrist signals at 1 Hz. |
| `demo_sessions_pack/<session>/context.csv` | 7 context columns every 30 s. |
| `demo_sessions_pack/reference_windows.csv` | The laptop's score for every window, to compare with. |
| `demo_sessions_pack/reference_scores.csv` | The laptop's score at the 59 checkpoints. |

## Commands, in order (on the lab computer connected to the Jetson)

```bash
# 1. unzip and go in
unzip Final_Demo.zip
cd Final_Demo

# 2. note the board and its power mode (it should say 25W)
cat /etc/nv_tegra_release | head -1
sudo nvpmodel -q

# 3. install the packages (needs internet once)
pip3 install -r requirements.txt

# 4. inference on all 569 windows: a line per window, '*' marks a checkpoint
python3 55_session_checkpoints.py --pack demo_sessions_pack --out run_output
#    slower, readable on screen / for a video:
#    python3 55_session_checkpoints.py --pack demo_sessions_pack --sleep 0.5

# 5. power: idle 60 s, run the model 5 times, idle 60 s again (about 4 min)
#    close other programs first and do not touch the board while it runs
python3 power_measure.py \
  --cmd "python3 55_session_checkpoints.py --pack demo_sessions_pack --no-explain --out power_run_output" \
  --windows 569 --repeat 5 --idle-seconds 60 --out run_output
#    if it says tegrastats gave no output, run the same line with sudo

# 6. look at the results
cat run_output/device_summary.txt
cat run_output/power_summary.txt
head run_output/window_scores_device.csv
```
Or all of it: `bash run_all.sh`

## What each output means

- `run_output/window_scores_device.csv` — one row per window (569): time, score
  (0 to 2), P(Low), P(Medium), P(High), state, latency in ms, memory, and how
  much each context column moved the score.
- `run_output/checkpoint_scores_device.csv` — the 59 checkpoint windows only.
- `run_output/device_summary.txt` — latency (median, first, worst), memory,
  model size, and the comparison with the laptop. Look for the line
  `RESULT : PASS`.
- `run_output/power_summary.txt` — idle power, running power and their
  difference, per rail, energy per window, and what it means for a worn
  device.
- `run_output/power_trace.csv` — every half-second power reading.

## The power formulas

    inference power  = running power - idle power            [mW]
    energy per window = inference power x time per window    [mJ]
    charge per window = energy / 1000 / V / 3.6               [mAh]  (V = 5.0)
    worn, a window every 30 s:  busy = time per window / 30
                                added power = inference power x busy

Idle is measured twice and the steadier reading is used, because the first
reading can still contain the previous program shutting down.

## If something goes wrong

- `No module named 'adastress'` — keep the `adastress/` folder beside the scripts.
- numpy errors — `pip3 install "numpy<2"`.
- `tegrastats gave no output` — run the power command with `sudo`.
- the two idle readings differ a lot — the board was busy; close other programs and rerun.
- `CHECK` instead of `PASS` — send the `device_summary.txt`.
