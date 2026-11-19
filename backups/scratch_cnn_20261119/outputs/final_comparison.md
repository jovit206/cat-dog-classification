# PawVision final model comparison

Selected using validation macro F1: **cnn128_adamw_mild_aug**. The validation-selected Cat threshold was **0.530**. The test split was not used for model or threshold selection and was scored once, after selection.

| Metric | Original baseline | Selected model | Change |
|---|---:|---:|---:|
| Test accuracy | 72.27% | 74.80% | +2.53 pp |
| Test loss | 0.5423 | 0.5261 | -0.0162 |
| Macro precision | 74.05% | 74.88% | +0.83 pp |
| Macro recall | 72.27% | 74.80% | +2.53 pp |
| Macro F1 | 71.74% | 74.78% | +3.04 pp |

## Per-class test results

| Class | Baseline precision | Selected precision | Baseline recall | Selected recall | Recall change | Baseline F1 | Selected F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dog | 67.51% | 76.27% | 85.87% | 72.00% | -13.87 pp | 75.59% | 74.07% |
| Cat | 80.59% | 73.48% | 58.67% | 77.60% | +18.93 pp | 67.90% | 75.49% |

Additional final test metrics: weighted F1 **0.7478**, ROC AUC **0.8234**, average precision (Cat positive class) **0.8044**.

The original baseline figures are the previously recorded test metrics at threshold 0.5. The selected model uses a threshold chosen on validation data; threshold differences are reported here so the comparison is transparent.
