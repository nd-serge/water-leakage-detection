## Project Objective

The objective of this work is to develop an artificial intelligence model capable of detecting leaks within a water distribution operator’s network using historical data collected from water meters. More specifically, the project aims to:

- analyze consumption trends collected from meters identified as relevant;
- group meters with similar consumption profiles;
- automatically detect consumption anomalies that may indicate the presence of leaks.

## Baseline Approach: Data Filtering

The data filtering approach, considered as our baseline model, consists of excluding extreme values from the consumption distribution. Observations falling between the 10th and 90th percentiles are considered normal, while values outside this range are classified as potential leaks.

**Result:** Detects consumption variations; sensitive to micro-variations.

## DBSCAN Algorithm

The DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm is a clustering and anomaly detection method based on the concept of core points and neighborhood density.

**Result:** Detects significant variations in consumption.

## Isolation Forest Algorithm

The Isolation Forest algorithm identifies anomalies by constructing multiple random decision trees based on the isolation of data points in the feature space.

**Result:** Low sensitivity to small consumption variations.

## Additional Information

- **Read the report:** [Google Drive](https://docs.google.com/presentation/d/135wj_trh2rsZ_eNXabBpfaKIyxH9gDa-/edit?usp=sharing&ouid=110200182022425202511&rtpof=true&sd=true)
- **Co-authors:** [Aurore FOHOUNDI](mailto:fohoundibehiblo@gmail.com)
