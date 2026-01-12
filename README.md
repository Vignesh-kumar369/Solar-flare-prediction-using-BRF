# Solar Flare Prediction using Machine Learning

## Project Overview
This repository contains the machine learning pipeline used to predict solar flares using SDO/HMI magnetogram data. The project focuses on handling extreme class imbalance (1:80) and identifying data quality artifacts.

## Key Features
* **Data Processing:** Cleaning and preprocessing of SDO/HMI parameters.
* **Artifact Detection:** Identification and correction of "Priority Inversion" in the dataset.
* **Model:** Random Forest Classifier (Scikit-Learn).
* **Performance:** Achieved a True Skill Statistic (TSS) of ~0.60.

## Files
* `04_Retraining_and_Error_Analysis`: The main notebook containing the training logic, validation, and TSS calculation.

## Requirements
* Python 3.x
* Scikit-Learn
* Pandas, NumPy
* Matplotlib
