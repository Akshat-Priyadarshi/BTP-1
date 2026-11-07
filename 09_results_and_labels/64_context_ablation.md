# Which context columns matter

Written by `btp/46_context_ablation.py`.

Eighteen runs on the same windows, the same frozen encoder and
the same 5 seeds. Only the context changes.

| | balanced accuracy |
|---|---:|
| all 7 columns | 0.836 |
| no context | 0.540 |
| guessing | 0.383 |

## Dropping one at a time

| column removed | balanced | change |
|---|---:|---:|
| *(none removed)* | 0.836 | — |
| `HOUR` | 0.717 | -0.118 |
| `IS_WEEKEND` | 0.743 | -0.092 |
| `POSTURE` | 0.801 | -0.035 |
| `ACT_CLASS` | 0.819 | -0.017 |
| `ACT_COUNT` | 0.825 | -0.011 |
| `BODY_POSITION` | 0.826 | -0.010 |
| `SLEEP` | 0.835 | -0.000 |

## Each column on its own

| column alone | balanced | vs no context |
|---|---:|---:|
| *(no context)* | 0.540 | — |
| `HOUR` | 0.686 | +0.146 |
| `IS_WEEKEND` | 0.602 | +0.062 |
| `POSTURE` | 0.581 | +0.041 |
| `SLEEP` | 0.565 | +0.024 |
| `ACT_CLASS` | 0.561 | +0.021 |
| `ACT_COUNT` | 0.552 | +0.012 |
| `BODY_POSITION` | 0.550 | +0.010 |

## How to read it

A column that barely matters when dropped but scores well
alone is saying the same thing as another column. One that is
flat in both tables is genuinely empty.

The split is pooled random. Windows overlap by 30 seconds and
each survey answer labels about 60 of them, so near identical
windows sit on both sides. Measured on the same data, context
is worth **+0.32** on this split and **-0.07** when survey
answers are kept whole.

So this ranks how much each column helps the model recognise
moments it has already seen. That is a fair question and worth
answering, but it is not the same as which column predicts
stress.

