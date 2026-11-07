# ContextGate: stress detection from a wrist wearable — code showcase

Akshat Priyadarshi (23CS30003), IIT Kharagpur, CSE Dual Degree. Supervisor: Dr. Bivas Mitra.
Project: *Physiological Signal Understanding and Stress Level Detection using LLMs* (model: **ContextGate**).

This folder is the complete code of the project, arranged **in the order the pipeline runs**, together with
the trained encoder, the trained classifier, the demo sessions and the inference scripts. Section 1 is the
folder map, section 2 the pipeline, section 3 the architecture and the maths, section 4 how to run the demo,
section 5 the results and the honest caveats.

---

## 1. Folder structure

```
Mentor_Code_Showcase/
├── README.md                          <- this file (map, pipeline, architecture, how to run)
├── FILE_GUIDE.md                      <- one line per script: role and what it does
├── 00_setup_and_configs/              folder layout script + the YAML configurations
├── 01_wesad_preprocessing_and_audit/  WESAD: preprocessing, audit, leakage checks, 7-channel build
├── 02_wearer_data_pipeline/           raw EmbracePlus files -> 7 channels at 1 Hz, + derived context signals
├── 03_labels_and_context/             STAI-6 answers -> labels; the 7 context columns
├── 04_model_code_and_training/        the model (encoder, gate, classifier), SSL pre-training, training
├── 05_evaluation/                     scoring all windows, context ablation, actual-vs-detected
├── 06_trained_models/                 trained encoder (W7) and trained classifier (FINAL_ctx7_random)
├── 07_demo_and_inference/             demo sessions, inference scripts (laptop + Jetson), dashboards
├── 08_exploration_chronos/            Chronos-token experiments (explored at mentor's request; not in final model)
└── 09_results_and_labels/             label files, actual-vs-detected tables/figures, WESAD results, decisions
```

### 00_setup_and_configs
| File | What it is |
|---|---|
| `setup_stage0.sh` | Creates the fixed project layout (`runs/`, `docs/`, `data/`, `vendor/` read-only) so experiments never overwrite each other. |
| `configs/w7.yaml` | **Final model's configuration**: 7 wrist channels, 60 s windows, stride 30 s, SSL 20 epochs, τ = 0.5, 128 LSTM units, classifier 20 epochs, gate on. |
| `configs/w6.yaml` | Earlier 6-channel (no breathing) configuration. |
| `configs/w13.yaml`, `btp_13ch*.yaml` | 13-channel WESAD (chest + wrist) configurations used for the reproduction and the leakage tests (PANAS / condition labels, with and without the Task column). |

### 01_wesad_preprocessing_and_audit  (stage 1: the lab dataset)
| File | What it does |
|---|---|
| `s0_check_wesad_raw.py` | Checks every raw WESAD subject folder and pickle keys before anything else. |
| `01_prep_wesad.py` | Runnable copy of the repository's `Preproc.ipynb` (the notebook does not run as shipped). |
| `02_check_session_order.py` | WESAD mixes the task order between subjects; verifies which session is which. |
| `03_assemble_wesad.py` | Builds the unlabelled / labelled CSVs the training code expects. |
| `04_audit_wesad.py` | Audit: does the context give away the label? |
| `05_train.py`, `06_collect_results.py`, `09_package_results.py` | Pre-training + classifier on WESAD without the notebook's flaws; results collector/packager. |
| `20_build_wesad_13ch.py` | 13-channel 1 Hz WESAD input. |
| `21_train_subjectwise.py` | Re-measures the repository's 94.4 % with person-wise splits. |
| `22_floor_check.py` | Random-forest floor: is the model or the data the bottleneck? |
| `23_context_leak_check.py` | Each context column alone vs the label. **Task alone = 1.000** (the task *is* the session, the session *is* the label). |
| `38_wesad_resp.py` | Breathing rate from the wrist BVP, checked against the chest respiration belt (correlation 0.185). |
| `39_build_wesad_7ch.py` | **WESAD as the 7 channels the wearers can match** — the data the encoder is pre-trained on. |

### 02_wearer_data_pipeline  (stage 2: daily-life EmbracePlus data)
| File | What it does |
|---|---|
| `24_download_embraceplus.ipynb`, `24b_download_remaining_users.ipynb` | Download raw EmbracePlus CSVs (per person/day/sensor). **Credentials removed**; placeholders only. |
| `25_build_user_6ch.py` | Raw files → 6 wrist channels at 1 Hz (ACC x/y/z, BVP, EDA, Temp), chunked reading. |
| `27_derive_pr.py` | Pulse rate from the beat timestamps (40–180 bpm kept). |
| `27_merge_biomarkers.py` | Merges the per-metric Empatica biomarker CSVs into one per-minute file. |
| `30_derive_context.py` | Derives context signals at 30 s: pulse rate, breathing, activity count, **posture** (from wrist tilt/movement), time of day, … |
| `40_build_user_7ch.py` | Adds the derived breathing channel → **7 channels at 1 Hz**, matching the encoder. |
| `26_apply_to_users.py`, `33_check_outputs.py`, `s0_check_user1_context.py` | Range/flat-channel warnings, output inventory, first look at user1. |
| `07_prep_user1.py`, `08_infer_user1.py` | First transfer test of WESAD models on user1 (earlier experiment). |
| `originals/` | The study team's own download and derived-modalities notebooks (credentials removed). |

### 03_labels_and_context  (stage 3: ground truth and situation)
| File | What it does |
|---|---|
| `31_build_labels.py` | STAI-6 → labels (see §2.3). 91 answers, 88 matched, 5,119 windows. |
| `32_user_leak_check.py` | Each context column alone vs label, no model. |
| `41_build_context7.py`, `42_build_context8.py` | Earlier context sets (8th column was a survey item, dropped). |
| `44_build_context7_all.py` | **The 7 context columns for every window a person wore the device (~104,000).** |
| `43_check_scores.py` | Sanity checks on the written scores. |

### 04_model_code_and_training  (stage 4: the model)
| File | What it does |
|---|---|
| `adastress/models.py` | `Encoder` (Conv1d×3 + BiLSTM + mean pool), `ProjectionHead`, `ContextGateFusion`, `MLPClassifier`, `StressModel`. **The architecture lives here.** |
| `adastress/ssl_pretrain.py` | NT-Xent contrastive pre-training loop (no labels). |
| `adastress/datasets.py` | Windowing, scaler fit/apply, label and context alignment, datasets. |
| `adastress/train_classifier.py` | Generic classifier training loop. |
| `adastress/inference.py` | `StressPredictor`: one window in, probabilities + score out. |
| `adastress/config.py`, `utils.py`, `__init__.py`, `LICENSE` | Config loading, seeds/devices, package files. |
| `adastress_scripts/` | `pretrain_encoder.py`, `train_classifier.py`, `run_inference.py` — command-line entry points. |
| `34_train_users.py` | **Our training script**: trains ONLY the gate + classifier on the survey labels (encoder frozen), with 4 splits (`--split random / response / user / temporal`). |

The `adastress/` package is the published AdaStress code, used as shipped. Our contributions are the data
pipeline (01–03), the 7-channel + context design, `34_train_users.py`, the evaluation (05) and the device work (07).

### 05_evaluation
| File | What it does |
|---|---|
| `45_score_all_with_context.py` | Scores every window of every person with the trained model. |
| `46_context_ablation.py` | Drop-one / keep-one context columns, 5 seeds. |
| `70_actual_vs_detected.py` | Survey answer (actual) vs model (detected), 4 splits × with/without context; writes tables, xlsx, figures. |

### 06_trained_models
| File | What it is |
|---|---|
| `encoder_W7/encoder.pth` | Frozen encoder, contrastively pre-trained on WESAD (7 channels). 357,760 weights. |
| `encoder_W7/scaler.pkl` | Per-channel mean/std fitted on WESAD, applied unchanged to every wearer. |
| `encoder_W7/meta.json` | Channel order, window 60 s, stride 30 s, 1 Hz, embedding 256. |
| `classifier_FINAL_ctx7_random/model.pth` | Trained gate + context projection + classifier (+ context encoder settings). |
| `classifier_FINAL_ctx7_random/metrics.json` | Its training record (7 context columns, random 70/10/20 split, 5 seeds). |

### 07_demo_and_inference
| Path | What it is |
|---|---|
| `demo_sessions_pack/` | **The demo data**: 5 recorded sessions (`user1_0322`, `user1_0713`, `user12_0603`, `user13_0522`, `user14_0717`), each with `signals.csv` (7 channels at 1 Hz), `context.csv` (7 context columns per minute) and `pulse_rate.csv`; plus `sessions.csv`, `model.pth`, `encoder.pth`, `scaler.pkl`, and the laptop's `reference_scores.csv` / `reference_windows.csv`. |
| `adastress/` | Copy of the model classes, so the saved model loads on a device without the full repository. |
| `55_session_checkpoints.py` | **The inference run**: scores all 569 windows of the 5 sessions, times each, marks 59 checkpoints, computes each context column's weight, compares with the laptop (PASS/CHECK). |
| `power_measure.py` | Reads Jetson power (`tegrastats`) idle / running / idle; the difference is the model's cost. |
| `device_dashboard.py` | Streamlit dashboard for the Jetson results. |
| `run_all.sh`, `requirements.txt`, `README_device_demo.md` | One-shot runner, packages (numpy < 2), the device-demo notes. |
| `laptop_inference_and_dashboard/` | `dashboard.py` (6-page Streamlit app), `50_inference.py` (window-by-window like a deployed monitor), `51_make_sample.py`, `52_inference_demo.py` (per-window context contribution), `54_make_demo_pack.py` (builds the pack), `60_build_demo_sessions.py`, `61_demo_intervals.py`, `62_session_scores.py`, `inference.py` (reference script, keys removed). |

### 08_exploration_chronos
Chronos tokenisation and embeddings of the signals (the SensorLLM path without the language model). Explored at the
mentor's request; **not part of the final model**.

### 09_results_and_labels
| Path | What it is |
|---|---|
| `labels/responses_scored.csv`, `label_summary.json`, `labelled_windows_ctx7b.csv` | The 91 scored STAI-6 answers, cut points, and the 5,119 labelled windows. |
| `comparisons/` | `COMPARISONS.md` (read first), `Actual_vs_Detected_Stress.xlsx`, `csv/` tables, `figures/` (confusion, score vs actual, per person). |
| `wesad_results.md`, `wesad_findings.md`, `wesad_summary.csv` | WESAD reproduction and audit results. |
| `00_decisions.md`, `56_labels.md`, `57_user_leak_check.md`, `60_wesad_resp.md`, `64_context_ablation.md` | The written decision log and per-stage reports. |

---

## 2. The pipeline

```
 WESAD (15 people, lab)                      5 labelled wearers (EmbracePlus, daily life)
        │                                              │
  01: 38, 39                                     02: 25, 27, 30, 40
  7 channels @ 1 Hz                              7 channels @ 1 Hz
        │                                              │        03: 44 → 7 context columns
        ▼                                              │        03: 31 → STAI-6 labels
 04: contrastive pre-training (NO labels)              │
        │                                              │
        ▼                                              ▼
 06: frozen encoder  ───────────────►  04: 34_train_users.py trains ONLY gate + classifier
                                                       │
                          score = 1·P(Medium) + 2·P(High)      (0 … 2)
                                                       │
                       05: score all windows · ablation · actual vs detected
                                                       │
                       07: dashboard · demo sessions · Jetson Orin Nano inference
```

### 2.1 Signals → windows
Seven wrist channels at **1 Hz**: ACC x/y/z, BVP, EDA, Temp, and **RESP** (breathing rate, derived from the pulse;
on WESAD checked against the chest belt: correlation 0.185 — weak). Windows are **60 s** long with a **30 s** stride:
number of windows = ⌊(T − 60)/30⌋ + 1 (a 3600 s session → 119). A scaler fitted on WESAD is applied unchanged to
the wearers.

### 2.2 Context (7 columns → 19 numbers)
| Column | Where it comes from |
|---|---|
| `HOUR`, `IS_WEEKEND` | The IST timestamp. |
| `POSTURE` | Derived by us from the wrist accelerometer: tilt = arccos(\|g_z\|/\|g\|); movement std > 0.12 → walking, else tilt < 40° → standing, else sitting. |
| `ACT_COUNT` | ActiGraph-style activity counts from the accelerometer (agcounts; 0.25–2.5 Hz band-pass fallback). |
| `ACT_CLASS`, `BODY_POSITION`, `SLEEP` | **Empatica's own biomarkers** (one value per minute, held across two 30 s steps; "Unknown" when the watch is off or loose). Not derived by us. |

HOUR, IS_WEEKEND and ACT_COUNT stay numeric; POSTURE, ACT_CLASS, BODY_POSITION, SLEEP are one-hot (4 slots each) →
**19 numbers**, fitted on training rows only. Empatica's body position was unreliable (called sitting "standing" 33 of 56 times).

### 2.3 Labels (STAI-6)
Six items — calm, tense, upset, relaxed, content, worried — rated 1–4. Calm, relaxed and content are reversed (5 − x).
raw = sum (6 to 24); **STAI6 = raw × 20/6** (20 to 80). Cut points across everyone: **Low ≤ 46.67 (raw ≤ 14)**,
**Medium 46.67 to 56.67**, **High ≥ 56.67 (raw ≥ 17)**. Each answer labels the windows within **±15 min**.
91 answers, 88 matched, 5,119 windows (users 17, 18, 25, 26, 27 → 13, 31, 23, 18, 6 answers); classes 33 Low / 27 Medium / 31 High.

### 2.4 Training in two steps
1. **Pre-training (no labels, WESAD only)** — `ssl_pretrain.py`: two augmented views of each window (scale in [0.9, 1.1] + noise σ = 0.01),
   encoder + projection head (256→256→128, L2-normalised), **NT-Xent loss, τ = 0.5**, 20 epochs, batch 128, Adam 1e-3; loss 4.38 → 3.64. Then **frozen**.
2. **Supervised (wearers' STAI-6 labels)** — `34_train_users.py`: cross-entropy, Adam 1e-3, batch 64, 20 epochs; trains **only the context projection + gate + classifier**.

---

## 3. Architecture and maths

```
 window  X  (7 × 60)                                         context u (19)
    │                                                             │
 Conv1d 7→64  k7 · ReLU                                    c = W·u + b      (19 → 256)
 Conv1d 64→128 k5 · ReLU                                          │
 Conv1d 128→128 k3 · ReLU      ← receptive field 13 s              │
 BiLSTM 128 units/direction                                       │
 mean over time  →  h (256)  ─────────────┐                       │
                                          ▼                       ▼
                       g = σ( W_g [h ; c] + b_g )      (512 → 256, one gate per dimension)
                       fused = g ⊙ h + (1 − g) ⊙ c
                                          │
              Linear 256→128 · ReLU · Dropout 0.3 · Linear 128→64 · ReLU · Linear 64→3
                                          │ logits
                       P = softmax(logits) = (P_Low, P_Medium, P_High)
```

* **State** = argmax P. **Score** = P(Medium) + 2·P(High) (range 0 to 2). **Uncertainty** = 1 − max P.
* Cross-entropy training: the gradient on the logits is **P − truth**.
* **Weights:** encoder 357,760 (frozen) + trainable 177,795 (context projection + gate 136,448; classifier 41,347) = **535,555 weights, about 2.14 MB**.
* A trained gate averaged g ≈ 0.42–0.49: roughly half body, half context.
* Why this design: the same heart rate means different things when running or sitting; the gate lets the model decide, per dimension, how much to trust the body versus the situation.

---

## 4. How to run

Run scripts **from the project root** (the folder that holds `vendor/`, `data/`, `runs/`, `btp/`), because the numbered
scripts read `data/` and write `runs/`. The raw wearer data (8.4 GB) is **not** in this showcase.

**Inference demo (self-contained — needs only this folder):**
```bash
cd 07_demo_and_inference
pip install -r requirements.txt            # numpy < 2, torch, pandas, scikit-learn
python3 55_session_checkpoints.py --pack demo_sessions_pack --out run_output
python3 55_session_checkpoints.py --pack demo_sessions_pack --sleep 2          # live line every 2 s
python3 55_session_checkpoints.py --pack demo_sessions_pack --only user1_0322  # one session
streamlit run device_dashboard.py                                              # view the results
```
Output: `window_scores_device.csv`, `checkpoint_scores_device.csv`, `device_summary.txt` (latency, memory, laptop comparison).
Power on the Jetson: `python3 power_measure.py --cmd "python3 55_session_checkpoints.py --pack demo_sessions_pack" --windows 569 --repeat 5 --idle-seconds 60`.

**Demo sessions.** Five recordings (user1 ×2, user12, user13, user14; 30 to 91 min). They have **no survey answers**, so they show the
model running and agreeing with the laptop, not accuracy. Checkpoints: session > 130 min → every 15 min; ≤ 100 min → every 5 min; 101–130 → every 10.

**Re-training (from the project root, needs the data):**
```bash
python3 vendor/adastress/scripts/pretrain_encoder.py --config configs/w7.yaml
python3 btp/34_train_users.py ...  --split random|response|user|temporal
python3 btp/46_context_ablation.py ; python3 btp/70_actual_vs_detected.py
```
(Script file names are the same as in this folder; the showcase copies live in numbered folders.)
`70_actual_vs_detected.py` rebuilds `09_results_and_labels/comparisons/`.

---

## 5. Results and honest caveats

| Finding | Number | Where |
|---|---|---|
| WESAD, random split of windows | 0.935 accuracy | `01/05_train.py` |
| WESAD, overlapping windows kept together | 0.800 | `01/21` |
| WESAD, unseen people | 0.733 random forest; **0.567 deep model**; guessing 0.531 | `01/21, 22` |
| WESAD, Task as the only context | **1.000 — leakage** (the task is the session is the label) | `01/23` |
| Wearers, balanced accuracy, signals only → + context (per window) | random 0.540 → 0.836; response 0.436 → 0.360; user 0.240 → 0.244; temporal 0.524 → 0.488 | `04/34`, `05/46` |
| Wearers, per answer (used in the report) | random 0.595 → 0.976; answers kept whole 0.477 → 0.372; unseen person 0.150 → 0.139; temporal 0.558 → 0.500 | `09/comparisons` |
| Context gain is the clock | dropping HOUR −0.118, IS_WEEKEND −0.092 | `05/46` |
| Jetson Orin Nano (25 W) | 569/569 windows match the laptop; 58 ms median, 141 ms worst; 2.15 MB model | `07/55` |
| Jetson energy | one 58 ms forward pass ≈ 131 mJ; the measured demo run (9 forward passes per window, 0.417 s) ≈ **944 mJ/window** (+2,265 mW over idle 4,975 mW); ≈ **31 mW** average at one window per 30 s | `07/power_measure.py` |

**Read this honestly.**
* The pipeline runs end to end, on a laptop and on an edge device, with identical scores.
* The stress score has **not** been shown to match what people reported on answers or people the model had not seen; for an unseen person it is below chance. The 0.84 / 0.98 random-split numbers are recall — windows of the same answer sit on both sides — and the gain from context is mostly time of day and weekend.
* Five labelled wearers only (88 answers); one reported Low in all 18 answers, another never Low.
* Accelerometer units differ by about 64× between WESAD and EmbracePlus; the WESAD scaler left 4 of 6 channels flat on the wearers. **Unsolved.**
* Breathing is derived from the pulse and is a weak proxy (0.185 with the chest belt).
* The five demo recordings are not the five labelled wearers' survey sessions.
* Scores are indicative, not a clinical measurement.
* Some older documents (slides 7 and 9, the report's Jetson power lines) say "context is computed from raw signals" and "≈ 4 mW / 131 mJ"; the accurate statements are above (three context columns are derived by us, three are Empatica biomarkers, plus the timestamp; the measured energy is 944 mJ per demo window).

## 6. What is *not* in this folder
Raw wearer recordings and the 8.4 GB `data/` folder (privacy and size); download notebooks have credentials removed;
`Code_Showcase/` and `stress_bundle_v2/` in the project root are the earlier organised copies (`stress_bundle_v2` is the slim peer bundle).
