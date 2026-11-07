# Actual stress versus model-detected stress

This folder lays the person's own answer next to what the model detected.

- **Actual stress** is the person's answer to the STAI-6 questionnaire (six items, scored 20 to 80), cut into three classes across everyone: **Low** up to 46.7, **Medium** 46.7 to 56.7 (a continuous band), **High** 56.7 and above. 88 answers from 5 people were matched to recorded windows.
- **Detected stress** is what the model predicted for the 60-second windows around that answer (windows every 30 s, 15 minutes either side), combined into one prediction per answer. The **score** is `1 x P(Medium) + 2 x P(High)`, from 0 to 2.
- **Signals only** is the model with the context switched off. **+ context** is the full ContextGate (hour, weekend, posture, activity count, activity class, body position, sleep).
- Every row was predicted by a model that did not train on that row. What "did not train on" means depends on the split, and that matters more than any other number in this folder.

## Files

| File | What it holds |
|---|---|
| `Actual_vs_Detected_Stress.xlsx` | All tables. One sheet of every answer for each split (green = match, red = miss), plus Summary, Confusion and Per person |
| `csv/answers_<split>.csv` | Every answer: actual class and STAI-6 score, detected class, score and probabilities for both models, and the survey context (activity, situation, tiredness, stress source) |
| `csv/summary_by_split.csv`, `confusion_by_split.csv`, `per_person.csv` | The tables below |
| `figures/1_confusion_by_split.png` | Actual against detected, 4 splits x 2 models |
| `figures/2_score_vs_actual.png` | Model score against the STAI-6 score, one dot per answer |
| `figures/3_per_person_mix.png` | Per person: actual class mix, detected mix without and with context |
| `05_evaluation/70_actual_vs_detected.py` (in this showcase) | Rebuilds everything here from the runs |

## 1. Headline: four splits, with and without context

Chance for balanced accuracy is 0.333. "Within one class" counts a Low called Medium, or a High called Medium, as close.

| Split | Model | Answers | Balanced acc. | 95% CI | Exact match | Within one class | Rank corr. |
|---|---|---|---|---|---|---|---|
| Random split of windows (recall) | signals only | 88 | 0.595 | 0.50 to 0.69 | 0.602 | 0.841 | 0.349 |
| Random split of windows (recall) | signals + context | 88 | 0.976 | 0.94 to 1.00 | 0.977 | 0.989 | 0.866 |
| Answers kept whole | signals only | 57 | 0.477 | 0.35 to 0.60 | 0.474 | 0.807 | -0.039 |
| Answers kept whole | signals + context | 57 | 0.372 | 0.25 to 0.50 | 0.368 | 0.667 | -0.18 |
| A person never seen | signals only | 88 | 0.15 | 0.09 to 0.22 | 0.159 | 0.659 | -0.564 |
| A person never seen | signals + context | 88 | 0.139 | 0.08 to 0.21 | 0.148 | 0.625 | -0.456 |
| Earlier train, later test | signals only | 26 | 0.558 | 0.37 to 0.74 | 0.538 | 0.923 | 0.119 |
| Earlier train, later test | signals + context | 26 | 0.5 | 0.30 to 0.68 | 0.5 | 0.846 | 0.289 |

## 2. How to read it

1. **Random split, with context (0.98).** Windows of the same answer sit on both sides of the split, and the model can find the hour of day and the weekend flag in the context. Dropping `HOUR` costs 0.118 and `IS_WEEKEND` 0.092 in the ablation, so most of the gain from context here is the clock. This measures recall, not prediction.
2. **Answers kept whole and a person never seen.** Once the model has not seen the answer, it no longer reliably beats guessing (0.48 and 0.37 against 0.33, with intervals that reach down to chance), and for a person it has not seen it is below chance. For the unseen person the rank correlation is negative (a higher actual stress gets a lower score). The five people used the scale very differently: one reported Low in all 18 answers, another never reported Low, so what the model learns from four people does not carry to the fifth.
3. **Earlier train, later test (26 answers).** Closest to real use. Signals only reach 0.56 against 0.33 for chance, with context 0.50, but the intervals (0.37 to 0.74 and 0.30 to 0.68) are wide and overlap, so this is not evidence that context helps.
4. **Five people and 88 answers are few.** Treat every number here as a picture of this data set, not a verdict on the method.

## 3. The realistic case in full: earlier train, later test (26 answers)

| Person | Answered at | STAI-6 (20-80) | Actual | Detected, signals only | Score | Detected, + context | Score  |
|---|---|---|---|---|---|---|---|
| user17 | 07-08 22:01 | 43.3 | Low | Low | 0.67 | Low | 0.14 |
| user17 | 07-09 12:03 | 56.7 | High | High | 1.46 | Medium | 1.02 |
| user17 | 07-09 17:01 | 60.0 | High | High | 1.12 | High | 1.83 |
| user17 | 07-09 20:25 | 50.0 | Medium | High | 1.22 | Low | 0.43 |
| user18 | 09-13 09:43 | 50.0 | Medium | Medium | 1.02 | Medium | 1.05 |
| user18 | 09-13 14:20 | 60.0 | High | Medium | 0.77 | High | 1.3 |
| user18 | 09-13 16:29 | 50.0 | Medium | Medium | 0.84 | High | 1.56 |
| user18 | 09-13 18:33 | 60.0 | High | Medium | 0.78 | Low | 0.61 |
| user18 | 09-14 08:59 | 53.3 | Medium | Medium | 0.82 | High | 1.52 |
| user18 | 09-14 16:48 | 50.0 | Medium | Medium | 0.81 | Medium | 0.99 |
| user18 | 09-14 19:07 | 60.0 | High | Medium | 0.79 | Low | 0.13 |
| user18 | 09-14 21:08 | 50.0 | Medium | Medium | 0.82 | Medium | 0.59 |
| user25 | 09-20 19:39 | 66.7 | High | Low | 0.78 | High | 1.16 |
| user25 | 09-20 23:25 | 66.7 | High | Medium | 0.8 | High | 1.39 |
| user25 | 09-22 09:32 | 46.7 | Low | Medium | 0.9 | Medium | 1.17 |
| user25 | 09-22 14:09 | 53.3 | Medium | Medium | 0.81 | Low | 0.31 |
| user25 | 09-22 19:55 | 66.7 | High | Low | 0.75 | Medium | 0.82 |
| user25 | 09-22 23:21 | 53.3 | Medium | Low | 0.79 | High | 1.66 |
| user25 | 09-23 11:10 | 63.3 | High | Medium | 1.03 | Medium | 1.24 |
| user26 | 09-28 22:50 | 33.3 | Low | Low | 0.75 | High | 1.25 |
| user26 | 09-29 00:53 | 30.0 | Low | Low | 0.75 | High | 1.83 |
| user26 | 09-29 12:59 | 20.0 | Low | Low | 0.73 | Low | 0.44 |
| user26 | 09-29 19:09 | 26.7 | Low | Medium | 0.81 | Low | 0.1 |
| user26 | 09-29 22:56 | 30.0 | Low | Medium | 0.83 | Low | 0.21 |
| user27 | 10-03 18:35 | 60.0 | High | High | 1.14 | High | 1.54 |
| user27 | 10-04 16:34 | 36.7 | Low | Low | 0.96 | Low | 0.63 |

### Actual (rows) against detected (columns), earlier train and later test

Signals only:

| Actual \ Detected | Low | Medium | High | Correct |
|---|---|---|---|---|
| Low | 5 | 3 | 0 | 5/8 |
| Medium | 1 | 6 | 1 | 6/8 |
| High | 2 | 5 | 3 | 3/10 |

Signals + context:

| Actual \ Detected | Low | Medium | High | Correct |
|---|---|---|---|---|
| Low | 5 | 1 | 2 | 5/8 |
| Medium | 2 | 3 | 3 | 3/8 |
| High | 2 | 3 | 5 | 5/10 |

### Per person, earlier train and later test

| Person | Answers | Actual L | Actual M | Actual H | Sig. L | Sig. M | Sig. H | Sig. correct | Ctx L | Ctx M | Ctx H | Ctx correct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| user17 | 4 | 1 | 1 | 2 | 1 | 0 | 3 | 3/4 | 2 | 1 | 1 | 2/4 |
| user18 | 8 | 0 | 5 | 3 | 0 | 8 | 0 | 5/8 | 2 | 3 | 3 | 4/8 |
| user25 | 7 | 1 | 2 | 4 | 3 | 4 | 0 | 1/7 | 1 | 3 | 3 | 2/7 |
| user26 | 5 | 5 | 0 | 0 | 3 | 2 | 0 | 3/5 | 3 | 0 | 2 | 3/5 |
| user27 | 2 | 1 | 0 | 1 | 1 | 0 | 1 | 2/2 | 1 | 0 | 1 | 2/2 |

## 4. The other splits

The confusion matrices for all four splits are in `figures/1_confusion_by_split.png` and in the Confusion sheet of the workbook. Every answer for every split is in its own sheet.

Random split (88 answers), signals + context:

| Actual \ Detected | Low | Medium | High | Correct |
|---|---|---|---|---|
| Low | 33 | 0 | 0 | 33/33 |
| Medium | 0 | 27 | 0 | 27/27 |
| High | 1 | 1 | 26 | 26/28 |

A person never seen (88 answers), signals + context:

| Actual \ Detected | Low | Medium | High | Correct |
|---|---|---|---|---|
| Low | 9 | 9 | 15 | 9/33 |
| Medium | 14 | 2 | 11 | 2/27 |
| High | 18 | 8 | 2 | 2/28 |

## 5. What is and is not shown

- The pipeline runs end to end and the model gives a score for any recording. On WESAD the accuracy is 0.935 when windows are split at random, and 0.567 for the deep model (0.733 for a random forest) on subjects it has not seen, against 0.531 for always guessing the commonest class.
- On the five wearers, the score has **not** been shown to match what people reported, except when the model has already seen the same answers.
