# Sepsis Onset Prediction Under Label Noise

Predicts sepsis onset 6 hours before physician diagnosis using ICU time-series data (PhysioNet Sepsis Challenge 2019).

## Approach

- **Missingness**: Per-patient forward-fill + missingness indicators + time-since-last-measurement features
- **Leakage prevention**: 6h-ahead label shift done per patient, patient-level train/test split, train-only imputation statistics
- **Label noise**: Downweight rows near label-transition boundaries (fuzzy onset) and isolated suspicious label flips
- **Model**: XGBoost with sample-weighted training
- **Calibration**: Sigmoid (Platt) calibration via cross-validation
- **Threshold**: Percentile-based cutoff (top ~8% flagged), tuned for recall in a screening context

## Results

| Class | Precision | Recall | F1 |
|---|---|---|---|
| No sepsis | 0.98 | 0.93 | 0.96 |
| Sepsis (6h ahead) | 0.16 | 0.45 | 0.23 |

Accuracy: 92%

## Files

- `sepsis_prediction.ipynb` — full pipeline (data download, preprocessing, training, evaluation)
-  `calibration.png`
-  `confusion_matrix.png`
-  `pr_curve.png`

## How to Run

Open in Google Colab and run cells in order. The notebook downloads its own data from PhysioNet.

