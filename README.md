# 23CSE301 Machine Learning — Capstone Project

**Course:** 23CSE301 Machine Learning  
**Project Tracks:** Regression, Classification, and Clustering  
**Application Name:** Erudition (Streamlit ML Platform)

---

## 1. Project Overview & Repository Structure
This repository contains an end-to-end Machine Learning pipeline across three problem tracks:
1. **Regression Track:** Predicting Concrete Compressive Strength (`data/data.csv`)
2. **Classification Track:** Alzheimer's Disease Diagnosis (`data/alzheimers_disease_data.csv`)
3. **Clustering Track:** Credit Card Customer Segmentation (`data/Credit_Card.csv`)

### Directory Structure
```text
/ (root)
├── README.md               # Project overview, results, setup instructions
├── requirements.txt        # All Python dependencies with locked versions
├── data/                   # Raw dataset files
│   ├── Credit_Card.csv
│   ├── data.csv
│   └── alzheimers_disease_data.csv
├── notebooks/              # Fully executed Jupyter notebooks
│   ├── regression.ipynb
│   ├── classification.ipynb
│   └── clustering.ipynb
├── models/                 # Serialized model artifacts (.pkl via joblib)
│   ├── regression_best_model.pkl
│   ├── classification_best_model.pkl
│   ├── kmeans_model.pkl
│   ├── agglomerative_model.pkl
│   └── clustering_scaler.pkl
├── app/                    # GUI / deployment code (Streamlit application)
│   └── app.py
└── apps/                   # Alternate deployable root for Streamlit Cloud
    └── app.py
```

---

## 2. Review 2 Deliverables & Track Summary

### Section A — Classification Track (Part B & Consolidated 10 Algorithms)
* **Algorithms Evaluated:** 
  - *Part A (Review 1):* Logistic Regression, K-Nearest Neighbors, Gaussian Naive Bayes, Decision Tree, Support Vector Machine (SVC).
  - *Part B (Review 2):* Random Forest, AdaBoost, Gradient Boosting, Bagging, MLP Classifier (Neural Network).
* **Consolidated Comparison:** All 10 models evaluated on held-out test split across **Accuracy, Precision, Recall, F1-score (weighted), and ROC-AUC**.
* **Best Model Selection & Tuning:** Random Forest identified as best-performing classifier; hyperparameter tuned (`n_estimators`, `max_depth`, `min_samples_leaf`) with documented improvement.

### Section B — Clustering Track (Credit Card Customer Segmentation)
* **Dataset:** 8,950 cardholders, 18 behavioral attributes.
* **Feature Engineering & Preprocessing:**
  - Median imputation for missing values in `MINIMUM_PAYMENTS` and `CREDIT_LIMIT`.
  - Engineered domain ratios: `BALANCE_TO_CREDIT_LIMIT` (utilization), `PURCHASE_TO_CREDIT_LIMIT`, `PAYMENT_TO_MIN_PAYMENT`.
  - Log transformation (`np.log1p`) to compress extreme right-skewness followed by `StandardScaler`.
* **Algorithms Implemented:**
  1. **K-Means Clustering:** Optimal $k=4$ validated through Elbow Method (Inertia vs. $k$) and Silhouette curve across $k \in [2, 10]$.
  2. **Agglomerative Hierarchical Clustering:** Compared linkage strategies (`ward`, `complete`, `average`) and plotted a 30-branch truncated dendrogram.
* **Mandatory Evaluation Metrics:**

| Algorithm | Silhouette Score (Higher is better) | Davies-Bouldin Index (Lower is better) | Calinski-Harabasz Index (Higher is better) |
|---|:---:|:---:|:---:|
| **K-Means ($k=4$)** | **0.2070** | **1.6993** | **2158.58** |
| **Agglomerative (Ward, $k=4$)** | 0.1429 | 1.8761 | 1631.19 |

* **Visualizations:**
  - 2D PCA projection scatter plot color-coded by cluster.
  - 2D t-SNE non-linear manifold projection.
  - Side-by-side PCA comparison of K-Means vs. Agglomerative clusters.
* **Customer Personas Identified:**
  - *Cluster 0:* Low-Activity / Dormant Cardholders (37.4%)
  - *Cluster 1:* High-Spenders / Active Transactors (18.8%)
  - *Cluster 2:* Cash Advance / Revolving Borrowers (16.4%)
  - *Cluster 3:* Disciplined Installment Shoppers (27.4%)

---

## 3. Bonus Deliverable — Interactive Web Application (Erudition)
An interactive Streamlit application is available under `app/app.py`:
- **Clustering Track:** Interactive hyperparameter controls ($k$, initialization, linkage), background model training, real-time metrics, 2D PCA plots, and live customer persona inference.
- **Regression Track:** Hyperparameter tuning across 8 regression models, real-time $R^2$, RMSE, MAE metrics, Predicted vs. Actual scatter plot, and live concrete compressive strength prediction.
- **Classification Track:** Classifier tuning (Random Forest, Decision Tree, AdaBoost, Logistic Regression, KNN), feature importances, and interactive patient cognitive impairment screener.

---

## 4. Setup and How to Run

### Environment Setup
Clone the repository and install all required dependencies:
```bash
git clone https://github.com/Key-to-Code/MLCaseStudy.git
cd MLCaseStudy
pip install -r requirements.txt
```

### Running the Notebooks
All notebooks in `notebooks/` are pre-run with all outputs and figures visible:
```bash
# Launch Jupyter Notebook or Jupyter Lab
jupyter notebook notebooks/
```

### Running the Streamlit Web Application Locally
```bash
python -m streamlit run app/app.py
```
The app will open automatically in your browser at `http://localhost:8501`.
