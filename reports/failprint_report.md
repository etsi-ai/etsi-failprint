# failprint Report
- Timestamp: 2025-08-02T22:09:34.160637
- Total Samples: 143
- Failures: 31 (21.68%)

## Contributing Feature Segments
**Age**:
- `28.0` → 9.7% in failures (Δ +5.5%)
**Parch**:
- `0` → 87.1% in failures (Δ +12.3%)
**Fare**:
- `7.925` → 12.9% in failures (Δ +6.6%)

## Top Features Driving Failures (SHAP Analysis)
Features are ranked by their mean absolute SHAP value across all failures.
|        |   mean_abs_shap_value |
|:-------|----------------------:|
| Sex    |             0.241509  |
| Age    |             0.15351   |
| Pclass |             0.138777  |
| Fare   |             0.115429  |
| SibSp  |             0.0560138 |
| Parch  |             0.0217135 |

