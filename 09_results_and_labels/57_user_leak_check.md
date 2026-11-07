# Leak check on the user context

Written by `btp/32_user_leak_check.py`. No model is trained.

For each context column: learn the most common label for each of
its values from four users, then apply that lookup to the fifth.
Whatever it scores, a model gets for free just by reading the
column.

This is the test that found `Task` predicting the WESAD label
with an accuracy of exactly 1.000 in every rotation, because the
label came from one questionnaire per session and `Task` was the
session.

| | |
|---|---|
| windows | 5119 |
| users | user17, user18, user25, user26, user27 |
| guessing baseline | 0.257 |

## Results

| column | user17 | user18 | user25 | user26 | user27 | mean | Cramer's V | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `ACT_CLASS` | 0.496 | 0.073 | 0.208 | 0.374 | 0.483 | **0.327** | 0.178 | real information (+0.070 over guessing) |
| `SLEEP` | 0.540 | 0.073 | 0.208 | 0.000 | 0.483 | **0.261** | 0.150 | carries nothing, at or below guessing |
| `IS_WEEKEND` | 0.557 | 0.000 | 0.226 | 0.000 | 0.500 | **0.257** | nan | carries nothing, at or below guessing |
| `BODY_POSITION` | 0.496 | 0.000 | 0.208 | 0.000 | 0.478 | **0.237** | 0.132 | carries nothing, at or below guessing |
| `POSTURE` | 0.406 | 0.061 | 0.160 | 0.000 | 0.422 | **0.210** | 0.119 | carries nothing, at or below guessing |
| `ACT_COUNT` | 0.281 | 0.000 | 0.235 | 0.000 | 0.447 | **0.193** | 0.111 | carries nothing, at or below guessing |
| `HOUR` | 0.244 | 0.118 | 0.072 | 0.000 | 0.447 | **0.176** | 0.253 | carries nothing, at or below guessing |
| `INDOOR_OUTDOOR` | 0.309 | 0.000 | 0.181 | 0.279 | 0.000 | **0.154** | 0.212 | carries nothing, at or below guessing |
| guessing | 0.557 | 0.000 | 0.226 | 0.000 | 0.500 | **0.257** | | |

## How to read this

Two different things can make a column score well, and they
should not be confused.

**A body signal predicting stress is the finding.** If pulse
rate or EDA predicts the label, that is what the whole project
is looking for.

**A clock or bookkeeping column predicting stress is a
problem**, because it says more about when the survey was
answered than about how the person felt. `HOUR` is the one to
watch.

## Note on the survey's own context

`responses_scored.csv` also holds what people reported about
themselves: activity, situation, environment, tiredness, and the
main source of stress in the last hour.

**Main source of stress must not become a feature.** It is close
to asking the label directly, and would be a `Task` of its own.
It is useful for explaining results, not for making them.

