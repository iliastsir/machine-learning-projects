# machine-learning-projects

# Machine Learning Projects (Supervised & Unsupervised)

This repository contains two end-to-end Machine Learning pipelines implemented in **Python** using `scikit-learn`, `pandas`, `numpy`, and `matplotlib`.

## 1. Diabetes Prediction (Supervised Learning)
A binary classification pipeline built on the **Pima Indians Diabetes Dataset** (768 samples, 8 clinical/demographic features).

* **Preprocessing:** 70/30 train-test split (`random_state=0`) and `MinMaxScaler` normalization fitted exclusively on the training set to prevent data leakage.
* **Model:** Logistic Regression (`scikit-learn`).
* **Results:**
  * **Train Accuracy:** 75.79%
  * **Test Accuracy:** 76.62% (strong generalization with no overfitting).
* **Analysis:** Extracted feature importance (top predictors: *Glucose*, *BMI*, and *Age*) and evaluated clinical risk via the Confusion Matrix, focusing on minimizing False Negatives (40 cases in the test set).

## 2. Spotify Track Recommendation System (Unsupervised Learning)
A content-based music recommendation engine built on the **Spotify Tracks Dataset**.

* **Preprocessing:** Removed missing values and duplicate tracks, resulting in **81,343 unique songs** across 113 genres. Isolated 14 numerical audio features (e.g., *danceability*, *energy*, *tempo*, *acousticness*) and normalized them to `[0, 1]` using `MinMaxScaler`.
* **Model:** **K-Means Clustering** ($k = 9$) to group tracks by acoustic profile.
* **Recommendation Engine:** Computes the **Euclidean distance** in the normalized feature space within the target song's cluster to recommend the Top-3 acoustically closest tracks.

## Tech Stack
* Python 3
* Scikit-learn, Pandas, NumPy, Matplotlib
