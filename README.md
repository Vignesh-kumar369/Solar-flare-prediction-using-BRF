# Scalable ML Pipeline for Solar Flare Prediction (SDO/HMI) and GOES X-Ray

### 🚀 Project Abstract

<img width="3000" height="1500" alt="priority_inversion_graph" src="https://github.com/user-attachments/assets/4d81eca2-f3de-4ec0-b945-1062cc116e91" />

Binary classification of M/X-class GOES flares within 24 hours of an SDO/HMI SHARP vector-magnetogram record, using 2014 to 2015 data (Solar Cycle 24 maximum). Models: XGBoost and Balanced Random Forest (BRF) on tabular SHARP parameters, tuned with Optuna and evaluated with TSS.

**Status:** manuscript in preparation. This repo holds the code and results as of October 2026.

This project does **not** claim to forecast flares 24 hours ahead. Its claims are:

1. Window-based flare labeling can hide an overwrite artifact ("priority inversion") that removes most of the positive class. Fixing it changed the M+X positives from 1,143 to 8,365.
2. TSS on this task is largely explained by recognizing flare-productive and repeat-flaring active regions, not by detecting temporal precursors.
3. Leakage-aware evaluation exposes both effects. A plain random split inflated test TSS to 0.9175.

---

### 🚨 Key Finding: Priority Inversion Bug

> Fixing a label-overwrite bug in the labeling loop recovered **7,222 mislabeled positive samples** (1,143 to 8,365).

A record is positive if an M or X flare from the same `NOAA_AR` starts in `[T_REC, T_REC + 24 h)`. The original loop assigned classes in the order X, M, C, B and overwrote earlier labels, so a later weak flare erased an earlier strong flare's label inside overlapping windows.

Controlled ablation (only the loop order changed):

| Class | Descending (buggy) | Ascending (fixed) | Delta |
|---|---|---|---|
| X | 16 | 1,096 | +1,080 |
| M | 1,127 | 7,269 | +6,142 |
| C | 37,117 | 32,060 | -5,057 |
| B | 4,824 | 2,659 | -2,165 |
| Non-flare | 638,374 | 638,374 | 0 |
| **Positive (M+X)** | **1,143** | **8,365** | **+7,222** |

The counts balance exactly (1,080 + 6,142 = 5,057 + 2,165), and the unchanged non-flare count shows the matching logic was held constant. A max-rank version (`np.maximum` over class ranks) makes the bug structurally impossible. The corrected positive class is about 1 in 80 records.

---

### 🛠️ Data and Pipeline

| Item | Value |
|---|---|
| Raw SHARP records (2014 to 2015) | 918,328 |
| Raw GOES flare records | 2,299 |
| Records after parsing and cleaning | 681,458 |
| Model-ready rows (after features and NaN drop) | 578,993 |

* **Data extraction:** SHARP parameters and flare records fetched with SunPy and serialized to Apache Parquet (notebook `01`).
* **Cleaning:** parsed `T_REC` (TAI), sorted by (HARPNUM, T_REC), de-duplicated. No duplicates found. The 35 s TAI to UTC offset was tested and has a negligible effect on labels.
* **Cadence gaps:** 5,055 records (0.87%) follow a gap longer than 24 minutes (57 positive). Deltas use fixed index offsets, so derivatives spanning a gap are mis-scaled. The upper bound on affected records is about 38.6k (6 h deltas) and 77.2k (12 h deltas). A sensitivity test has not been run yet.

**Features:** 16 base SHARP parameters, 2 physics ratios (`TOTUSJH/USFLUX`, `TOTUSJZ/USFLUX`), 6 h and 12 h index-based deltas, and rolling statistics, from a larger candidate set. Removing the rolling statistics raised test TSS from 0.6583 to 0.6870 and cut the train-test gap from 0.1810 to 0.1410. All final models use the **55-feature** set. Known defects: the 1e-9 epsilon creates outliers when `USFLUX` is near zero, and index-based deltas assume a perfect 720 s cadence.

**Splitting:** `GroupShuffleSplit` on HARPNUM, test size 0.10, seed 35. Train: 517,418 rows (6,486 positives). Test: 61,575 rows (1,468 positives) from 8 flare-producing active regions (11947, 11968, 11996, 12055, 12113, 12209, 12222, 12241). Seed 35 was chosen by scanning seeds 0 to 99 for test-set region diversity, which is a selection on test composition. The split is random in time, not chronological.

**Tuning:** Optuna (TPE, 50 trials), 5-fold GroupKFold, TSS objective with an in-fold threshold scan, so the CV score is optimistic. Different sampler seeds gave different best parameters (CV TSS 0.7729 vs 0.7430). Isotonic calibration failed (zero recall), so models are uncalibrated.

---

### ⚛️ Domain Context

* **Target:** M- and X-class solar flares (space weather events).
* **Metric:** True Skill Statistic (TSS = recall + specificity - 1), chosen because accuracy is meaningless at 1:80 imbalance.
* **Interpretation caution:** the model is not shown to learn flare physics. Section "Diagnostics" shows its skill is concentrated in already-flaring, magnetically extreme regions.

---

### 📊 Results

Leak-free protocol: thresholds chosen from out-of-fold predictions on the training set only, test set evaluated once.

| Model | Threshold | TSS | Recall | Precision | FP | FN |
|---|---|---|---|---|---|---|
| XGBoost | 0.47 | 0.6450 | 0.8638 | 0.0880 | 13,148 | 200 |
| Balanced Random Forest | 0.32 | 0.6499 | 0.9666 | 0.0694 | 19,040 | 49 |

* The TSS difference (0.0049) is well within noise on an 8-region test set. Neither model is shown to be better.
* XGBoost gives 30.9% fewer false positives and 10.3 points less recall than BRF.
* Precision of 6.9 to 8.8% means about 10 to 14 false alarms per hit, so these models are not operationally usable.
* An earlier phase scanned thresholds on the test set (XGBoost 0.7163, BRF 0.6814). Those figures are optimistic and kept only for transparency.

![Operational Tradeoff Curve](operational_tradeoff_curve.png)

---

### 🔬 Diagnostics

**Lead-time stratification** (threshold 0.55 from the earlier test-set scan, to be recomputed at 0.47; global FPR 0.2183):

| Lead time | n | Recall | TSS | Already flared |
|---|---|---|---|---|
| 0 to 6 h | 480 | 0.9708 | 0.7526 | 73.3% |
| 6 to 12 h | 432 | 0.8773 | 0.6590 | 71.8% |
| 12 to 18 h | 289 | 0.8997 | 0.6814 | 70.6% |
| 18 to 24 h | 266 | 1.0000 | 0.7817 | 74.8% |

* Skill does not decay with lead time. Perfect recall at 18 to 24 h is not what precursor detection would produce.
* 1,066 of 1,468 test positives (72.6%) come from regions that had already produced an M/X flare.
* On first-flare-only records (n = 402), recall across the four bins is 0.8906 / 0.5656 / 0.6588 / 1.0000. Mid-horizon recall drops once repeat flarers are removed.
* Each bin contains only 6 to 8 unique active regions, so per-bin statistics are region-clustered.
* One test region (HARPNUM 3587, 49 positives) gets zero true positives.

Working interpretation: the models behave like classifiers of flare-productive regions. A static region-level baseline would test this directly and has not been run.

---

### ⚠️ Limitations

* **Small test set:** the held-out split contains only 8 flare-producing active regions, so model comparisons are not statistically resolved. The seed (35) was chosen from seeds 0 to 99 by maximizing the number of flaring test regions (8, tied with seed 92), a criterion independent of model performance.
* **Random-in-time split:** train and test both span 2014 to 2015. A chronological train-2014/test-2015 split is not yet run.
* **Untested interpretation:** a static region-level baseline (for example a `USFLUX` threshold or "region has already flared") is not yet run, so the region-productivity interpretation remains a hypothesis.
* **Uncertainty on the model comparison:** no region-level bootstrap on the out-of-fold predictions has been done. Hyperparameters were tuned on those same folds, so out-of-fold scores are somewhat optimistic.
* **Cadence gaps:** the sensitivity test for gap-affected deltas is specified but not reported.
* **Lead-time table:** computed at threshold 0.55 (from the earlier test-set scan), not the leak-free 0.47.
* **Precision** is below 9%, and no cost-sensitive loss was tried.

---

### 📂 Tech Stack

* **Core:** Python 3.x, scikit-learn, imbalanced-learn (BRF), XGBoost, Optuna, Numba.
* **Data:** Pandas, NumPy, PyArrow (Parquet), SunPy.
* **Visualisation:** Matplotlib.

### 📝 Key Notebooks

* `01_Data_Extraction.ipynb`: data fetching and Parquet serialization.
* `02_Data_Cleaning.ipynb`: parsing, de-duplication, labeling.
* `03_Model_Training.ipynb`: features, Optuna tuning, training.
* `04_Retraining_and_Error_Analysis.ipynb`: retraining on corrected labels, TSS evaluation, diagnostics.
* `Consolidated Results from the Project.ipynb`: results summary.
