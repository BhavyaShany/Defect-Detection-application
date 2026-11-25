# Evaluation report - casting-photo-v1

Held-out test images: **651** · Accuracy: **0.992** · Macro-F1: **0.991**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| def_front | 0.998 | 0.991 | 0.994 | 453 |
| ok_front | 0.980 | 0.995 | 0.987 | 198 |

Confusion matrix (rows = true, columns = predicted):

```
classes: def_front, ok_front
[[449   4]
 [  1 197]]
```

## Confidence-threshold analysis (test set)

| Threshold | Auto-decided % | Sent to REVIEW % | Accuracy on auto-decided % | Missed defects | False rejects |
|---|---|---|---|---|---|
| 0.5 | 100.0 | 0.0 | 99.23 | 4 | 1 |
| 0.6 | 99.69 | 0.31 | 99.54 | 3 | 0 |
| 0.7 | 99.39 | 0.61 | 99.54 | 3 | 0 |
| 0.75 | 99.23 | 0.77 | 99.54 | 3 | 0 |
| 0.8 | 99.08 | 0.92 | 99.53 | 3 | 0 |
| 0.85 | 98.46 | 1.54 | 99.53 | 3 | 0 |
| 0.9 | 97.54 | 2.46 | 99.53 | 3 | 0 |
| 0.95 | 94.47 | 5.53 | 99.67 | 2 | 0 |

Indicative threshold (smallest with zero missed defects on this test set): **None** - treat as a starting point; confirm on fresh data.