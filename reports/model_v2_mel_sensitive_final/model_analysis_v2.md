# Model v2 - Melanoma-sensitive Analysis

## Objective

The objective of this version is to improve melanoma sensitivity while keeping the classifier reasonably balanced across the seven HAM10000 classes.

## Decision policy

A melanoma-sensitive threshold was selected using the validation set.

- Selected melanoma threshold: 0.36

If the predicted probability for melanoma is greater than or equal to this threshold, the final prediction is forced to melanoma.

## Test metrics without melanoma threshold

- Accuracy: 0.7246
- Recall macro: 0.6401
- F1 macro: 0.6328
- ROC-AUC OVR macro: 0.9406
- Recall melanoma: 0.7006
- Precision melanoma: 0.3611
- F1 melanoma: 0.4766

## Test metrics with melanoma-sensitive threshold

- Accuracy: 0.7046
- Recall macro: 0.6358
- F1 macro: 0.6262
- ROC-AUC OVR macro: 0.9406
- Recall melanoma: 0.7485
- Precision melanoma: 0.3324
- F1 melanoma: 0.4604

## Interpretation

This version uses EfficientNetB0 transfer learning, data augmentation, balanced sample weights, moderate melanoma weighting, label smoothing, two-phase training, and a melanoma-sensitive decision threshold.

The threshold strategy is aligned with a triage scenario: increasing the chance of flagging melanoma cases while avoiding the extreme behavior of predicting melanoma for nearly everything.
