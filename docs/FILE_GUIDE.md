# File guide: what every script does

Per-file descriptions for the scripts in this showcase (companion to README.md, which has the folder map, pipeline and architecture).
Role legend: Final pipeline = used for the final model/results; Supporting check = data check or helper; Demo / device = dashboard, checkpoints, Jetson; Earlier experiment = tried, not in the final model.

Corrections to older wording: context is *not* all derived from raw signals - HOUR/IS_WEEKEND come from the timestamp, POSTURE/ACT_COUNT are derived by us from the wrist accelerometer,
and ACT_CLASS/BODY_POSITION/SLEEP are Empatica biomarkers. Measured Jetson energy is about 944 mJ per demo window (about 31 mW average at one window per 30 s).

## 00_setup_and_configs

The folder layout and the AdaStress configuration files. `w7.yaml` is the one the final model uses; the others belong to the WESAD reproduction.

| File | Role | What it does |
|---|---|---|
| `setup_stage0.sh` | Setup | Creates the fixed folder layout (runs/, docs/, data/, vendor/ read-only) so experiments never overwrite each other and the original repository code stays untouched. |
| `configs/w7.yaml` | Final pipeline | Configuration of the final model: the 7 wrist channels, 60 s windows every 30 s, 20 epochs, context gate on. |
| `configs/w6.yaml` | Supporting check | The earlier 6-channel configuration (the first, signals-only model). |
| `configs/w13.yaml` | Earlier experiment | 13-channel WESAD configuration (chest + wrist), the repository-faithful reproduction. |
| `configs/btp_13ch.yaml` | Earlier experiment | WESAD 13-channel configuration: paper-faithful 13-channel run on WESAD with PANAS labels. |
| `configs/btp_13ch_cond.yaml` | Earlier experiment | WESAD 13-channel configuration: WESAD condition labels (baseline / amusement / stress). |
| `configs/btp_13ch_global.yaml` | Earlier experiment | WESAD 13-channel configuration: PANAS labels cut across all subjects. |
| `configs/btp_13ch_paper.yaml` | Earlier experiment | WESAD 13-channel configuration: the labelling rule described in the paper. |
| `configs/btp_13ch_noTask.yaml` | Earlier experiment | WESAD 13-channel configuration: context without the Task column (leakage test). |
| `configs/btp_13ch_taskonly.yaml` | Earlier experiment | WESAD 13-channel configuration: Task as the only context (leakage test). |

## 01_wesad_preprocessing_and_audit

Reproducing the published pipeline on WESAD, checking whether its numbers can be trusted, and preparing the 7 channels the wearers can match.

| File | Role | What it does |
|---|---|---|
| `s0_check_wesad_raw.py` | Supporting check | Checks raw WESAD before any preprocessing (every subject folder, the pickle keys) instead of hiding failures in a try/except. |
| `01_prep_wesad.py` | Supporting check | A runnable copy of the repository notebook Preproc.ipynb (cells 26 to 38), which does not run as shipped (missing imports, a pandas resample code newer versions reject). The repo stays read-only. |
| `02_check_session_order.py` | Supporting check | Checks which WESAD session is which per subject: the study mixed the task order between subjects, the notebook assumes one fixed order. |
| `03_assemble_wesad.py` | Supporting check | Builds the three CSVs the training notebook expects (unlabelled for pre-training, generalized labelled, fine-tune labelled), a step the repository is missing. |
| `04_audit_wesad.py` | Supporting check | Data audit before training: does the context give away the label (Task is the session, and the label comes from one questionnaire per session), and other checks that can make an accuracy meaningless. |
| `05_train.py` | Supporting check | Self-supervised pre-training and classifier training from Stress.ipynb, without two flaws of the notebook (windows spanning two people, a random train/validation split that puts one person on both sides). |
| `06_collect_results.py` | Supporting check | Reads every runs/*/metrics.json and prints one comparison table, grouping seeds as mean and spread, so no number is typed by hand. |
| `09_package_results.py` | Supporting check | Gathers one dataset's results (runs, docs, tables) into a single folder that can be handed over, without moving anything. |
| `20_build_wesad_13ch.py` | Supporting check | Builds the 13-channel 1 Hz WESAD input that the repository's library path expects; modes literal / corrected and several label sources. |
| `21_train_subjectwise.py` | Supporting check | Re-measures the repository's 94.4 % with person-wise splits, to show how much of it is real; each change from the repo is marked. |
| `22_floor_check.py` | Supporting check | Random forest on the mean and spread of each channel: a floor that shows whether the model or the data is the bottleneck. |
| `23_context_leak_check.py` | Final pipeline | Leakage check on WESAD: each context column alone against the label. Task alone gives 1.000 on held-out people, because the task is the session and the session is the label. |
| `38_wesad_resp.py` | Final pipeline | Derives breathing rate for WESAD from the wrist pulse (BVP), and checks the method against WESAD's chest respiration belt, the only ground truth available. |
| `39_build_wesad_7ch.py` | Final pipeline | WESAD as the 7 channels that can reach the wearers: the 6 wrist channels plus the derived breathing rate. |

## 02_wearer_data_pipeline

From raw EmbracePlus files to 7 wrist channels at 1 Hz, plus the first transfer experiments on the first wearer.

| File | Role | What it does |
|---|---|---|
| `24_download_embraceplus.ipynb` | Final pipeline | Downloads the raw EmbracePlus recordings from the cloud, one CSV per person, day and sensor (resumable). Credentials removed in this copy. |
| `24b_download_remaining_users.ipynb` | Final pipeline | The same download for the remaining wearers. Credentials removed in this copy. |
| `25_build_user_6ch.py` | Final pipeline | Raw EmbracePlus files to the 6 wrist channels at 1 Hz (accelerometer x/y/z, BVP, EDA, temperature), read in chunks so a day of 64 Hz data does not exhaust memory. |
| `27_derive_pr.py` | Final pipeline | Pulse rate from the heartbeat timestamps (the gap between two beats), keeping 40 to 180 bpm and reporting how many beats were dropped. |
| `27_merge_biomarkers.py` | Supporting check | Merges the per-metric Empatica biomarker CSVs into one wide per-minute file. |
| `30_derive_context.py` | Final pipeline | All context signals for one person at 30 s: pulse rate, breathing (from heart-rhythm), activity, posture, EDA responses, time of day. Methods ported from the study team's notebook where marked. |
| `33_check_outputs.py` | Supporting check | Lists, per person, which outputs exist and whether the raw folder is safe to delete (written after raw data was deleted before the 1 Hz channels had been built). |
| `40_build_user_7ch.py` | Final pipeline | Adds the breathing channel to the 6 channels, giving the 7 channels at 1 Hz that match what the encoder was trained on. |
| `26_apply_to_users.py` | Supporting check | Scores the wearers with the signals-only WESAD model; prints each channel's range and warns of channels far outside training or almost flat. |
| `07_prep_user1.py` | Earlier experiment | Puts the first wearer (user1) into the same shape as processed WESAD (5 s bins, six statistics) for the first transfer test. |
| `08_infer_user1.py` | Earlier experiment | Applies the WESAD-trained models to user1 and compares how they behave; no labels exist, so no accuracy. |
| `s0_check_user1_context.py` | Supporting check | Looks at user1's files before designing anything: how many labels exist and which context is available. |
| `originals/Embrance_plus_raw_resume.ipynb` | Supporting check | The study team's own download notebook, unedited except that credentials are removed. |
| `originals/embrace_derived_modalities.ipynb` | Supporting check | The study team's notebook that derives pulse rate, breathing rate and the activity measures, with the citations for each method. Credentials removed. |

## 03_labels_and_context

Turning the STAI-6 answers into Low / Medium / High, and building the context columns the gate reads.

| File | Role | What it does |
|---|---|---|
| `31_build_labels.py` | Final pipeline | STAI-6 answers to labels: reverse calm / relaxed / content, add the six items, multiply by 20/6 (20 to 80), cut Low / Medium / High across everyone, attach to the windows within 15 minutes of each answer. |
| `32_user_leak_check.py` | Final pipeline | Leakage check on the wearers: how well each context column predicts the label alone, with no model. |
| `41_build_context7.py` | Earlier experiment | First 7-column context set (HOUR, IS_WEEKEND, POSTURE, TILT, ACT_COUNT, ACT_INT, INDOOR_OUTDOOR) for the labelled windows, with no signal inside it. |
| `42_build_context8.py` | Final pipeline | Labelled-window file with the eight context columns the mentor asked for, none of them a signal. The final model drops the survey-based INDOOR_OUTDOOR column, leaving 7. |
| `44_build_context7_all.py` | Final pipeline | The 7 context columns for every window a person wore the device (about 104,000), built directly from the context and biomarker files. |
| `43_check_scores.py` | Supporting check | Checks the written scores: a STAI-equivalent value on the 20 to 80 scale, and two more questions the scoring run cannot answer itself. |

## 04 / 05  model, training and evaluation (training script is in 04, the rest in 05; adastress/ package in 04)

Training the gate and classifier, scoring every window, the context ablation, and the actual-versus-detected comparison.

| File | Role | What it does |
|---|---|---|
| `34_train_users.py` | Final pipeline | Trains ONLY the gate and the classifier on the survey labels (the encoder stays frozen), with four splits (random, response, user, temporal) and answer-level results with bootstrap intervals. |
| `45_score_all_with_context.py` | Final pipeline | Scores every window of every person with the trained model; writes predictions, per-person and per-answer tables. |
| `46_context_ablation.py` | Final pipeline | Which context columns matter: drop one at a time and keep one at a time, 5 seeds each. |
| `70_actual_vs_detected.py` | Final pipeline | Puts each survey answer (actual stress) next to what the model detected, for all four splits, with and without context; writes the tables, spreadsheet and figures. |

## 07_demo_and_inference (Final_Demo/ files sit directly in 07; the rest in laptop_inference_and_dashboard/)

Showing the scores (dashboard), scoring recorded sessions at checkpoints, and running on the Jetson Orin Nano.

| File | Role | What it does |
|---|---|---|
| `dashboard.py` | Demo / device | Streamlit dashboard with six pages: how a score is made, demo-session checkpoints, one person on one day, all five people, what the context did, and how much to trust the result. |
| `60_build_demo_sessions.py` | Demo / device | Converts the short demo recordings (raw files plus per-minute CSVs) into the two inputs the scorer reads. |
| `61_demo_intervals.py` | Demo / device | The score at fixed checkpoints (every 5, 10 or 15 minutes depending on session length), cross-checked, with the figure. |
| `62_session_scores.py` | Demo / device | One command from the session table to the finished result: calls 60, 45 and 61 in order. |
| `50_inference.py` | Demo / device | Runs a trained model one window at a time like a deployed monitor: read sixty seconds, predict, wait; reports the latency of each window. |
| `51_make_sample.py` | Demo / device | Packs one person's day into a self-contained folder (signals, context, truth, model) that can be copied to another machine. |
| `52_inference_demo.py` | Demo / device | Runs a packed sample like a device and, for each window, measures how much each context column moved the score (turn one off, measure the change). |
| `54_make_demo_pack.py` | Demo / device | Builds the folder the Jetson needs (sessions, model, encoder, scaler, the laptop's reference scores). |
| `Final_Demo/55_session_checkpoints.py` | Demo / device | The inference run on the Jetson: scores all 569 windows of the five sessions, times each one, marks the 59 checkpoints, measures each context column's weight and compares with the laptop's scores (PASS / CHECK). |
| `Final_Demo/power_measure.py` | Demo / device | Reads the board's power from tegrastats while idle, while running the model, and idle again; the difference is the model's cost. |
| `Final_Demo/device_dashboard.py` | Demo / device | Dashboard for the Jetson results: window scores, latency, memory, power, and the laptop comparison. |
| `Final_Demo/run_all.sh` | Demo / device | Runs the whole device demo in order: board info, install, inference, power. |
| `Final_Demo/requirements.txt` | Demo / device | Python packages for the device demo (numpy must stay below 2). |
| `inference.py` | Earlier experiment | The reference inference script from the study team, with notification hooks; adapted into 50_inference.py. Service keys removed in this copy. |

## 08_exploration_chronos

A direction explored at the mentor's request (the SensorLLM path, without the language model). Not part of the final model.

| File | Role | What it does |
|---|---|---|
| `28_chronos_tokens.py` | Earlier experiment | Tokenises the raw-rate signals with Chronos (a direction the mentor asked for; nothing in AdaStress does this). |
| `29_token_report.py` | Earlier experiment | Reports how many distinct tokens each channel uses for each tokenisation method; showed two channels (skin temperature, pulse rate) looking nearly dead under the first method. |
| `35_tokenise_1hz.py` | Earlier experiment | Chronos tokenisation of the 1 Hz channels, turned back into numbers so the frozen encoder can read them. |
| `36_chronos_embeddings.py` | Earlier experiment | The SensorLLM path applied to our data: signal, tokens, the frozen pre-trained Chronos encoder, embeddings. |
| `37_train_on_embeddings.py` | Earlier experiment | Gate and classifier on the Chronos embeddings, stopping short of the language model, so numbers stay comparable with the rest of the project. |
| `chronos-part/make_windows_16hz.py` | Earlier experiment | 30 s windows at 16 Hz from EmbracePlus raw files and per-minute CSVs (480 samples per channel). |
| `chronos-part/chronos_embed.py` | Earlier experiment | Runs the frozen Chronos encoder on those windows, one embedding file per channel. |
| `chronos-part/add_resp_channel.py` | Earlier experiment | Adds respiration as a seventh channel to the two scripts above, keeping the originals as .orig. |
| `chronos-part/download_embrace_users.ipynb` | Earlier experiment | Download notebook for the demo recordings. Credentials removed in this copy. |

