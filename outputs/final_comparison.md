# PawVision final model comparison

Selected using validation macro F1: **efficientnetv2b0_224_finetune**. The validation-selected Cat threshold was **0.500**. The test split was not used for model or threshold selection.

| Metric | Original baseline | Scratch CNN | Selected model |
|---|---:|---:|---:|
| Test accuracy | 72.27% | 74.80% | 99.60% |
| Test loss | 0.5423 | 0.5261 | 0.0184 |
| Macro precision | 74.05% | 74.88% | 99.60% |
| Macro recall | 72.27% | 74.80% | 99.60% |
| Macro F1 | 71.74% | 74.78% | 99.60% |
| ROC AUC | — | 0.8234 | 0.9998 |

## Per-class test F1

| Metric | Original baseline | Scratch CNN | Selected model |
|---|---:|---:|---:|
| Dog | 75.59% | 74.07% | 99.60% |
| Cat | 67.90% | 75.49% | 99.60% |

The original baseline used threshold 0.5; later models use a threshold chosen on validation data only.
