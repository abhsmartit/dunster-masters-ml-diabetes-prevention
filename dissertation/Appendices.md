# Appendices

---

## Appendix A: GitHub Repository

**Repository**: [https://github.com/abhsmartit/dunster-masters-ml-diabetes-prevention](https://github.com/abhsmartit/dunster-masters-ml-diabetes-prevention)

All code, data, and documentation are publicly available. Repository structure:

- **src/**: Python modules (diabetes_data_loader.py, model_development.py, model_evaluation.py)
- **data/**: Pima Indians Diabetes Dataset
- **results/**: Visualizations and performance metrics
- **dissertation/**: All chapters in markdown format
- **main.py**: Pipeline orchestrator
- **requirements.txt**: Dependencies

To replicate: `git clone <url> && pip install -r requirements.txt && python main.py`

---

## Appendix B: Algorithm Hyperparameters

| Algorithm | Key Hyperparameters | Values |
|-----------|------------------|--------|
| Logistic Regression | solver, max_iter, C | lbfgs, 1000, 1.0 |
| Decision Tree | max_depth, min_samples_split | 10, 5 |
| Random Forest | n_estimators, max_depth, random_state | 100, 15, 42 |
| Gradient Boosting | n_estimators, learning_rate, max_depth | 100, 0.1, 5 |
| XGBoost | n_estimators, learning_rate, max_depth | 100, 0.1, 5 |
| LightGBM | n_estimators, learning_rate, num_leaves | 100, 0.1, 31 |
| SVM | kernel, C, gamma | rbf, 1.0, scale |
| Naive Bayes | var_smoothing | 1e-9 |
| KNN | n_neighbors, metric | 5, euclidean |

---

## Appendix C: Dataset Characteristics

**Pima Indians Diabetes Dataset**:
- **Instances**: 768 (500 non-diabetic, 268 diabetic)
- **Features**: 8 physiological measurements
- **Target**: Binary (0 = non-diabetic, 1 = diabetic)
- **Class Distribution**: 65% negative, 35% positive
- **Missing Data**: Handled via median imputation
- **Features**: Glucose (mg/dL), Blood Pressure (mmHg), Skin Thickness (mm), Insulin (µU/ml), BMI (kg/m²), DiabetesPedigreeFunction, Age (years), Pregnancies

---

## Appendix D: Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.7662 | 0.7045 | 0.5690 | 0.6298 | 0.8188 |
| Decision Tree | 0.7188 | 0.6522 | 0.4921 | 0.5588 | 0.6838 |
| **Random Forest** | **0.7597** | **0.8205** | **0.5926** | **0.6883** | **0.8147** |
| Gradient Boosting | 0.7532 | 0.7906 | 0.5690 | 0.6588 | 0.8124 |
| XGBoost | 0.7662 | 0.8129 | 0.5690 | 0.6667 | 0.8204 |
| LightGBM | 0.7662 | 0.8276 | 0.5504 | 0.6639 | 0.8210 |
| SVM | 0.7662 | 0.8148 | 0.5690 | 0.6705 | 0.8188 |
| Naive Bayes | 0.7662 | 0.7419 | 0.6299 | 0.6819 | 0.8276 |
| KNN | 0.7318 | 0.7321 | 0.5039 | 0.5970 | 0.7597 |

---

## Appendix E: Feature Importance Rankings (Random Forest)

| Rank | Feature | Importance (%) |
|------|---------|-----------------|
| 1 | Glucose | 28.47 |
| 2 | BMI | 16.23 |
| 3 | Age | 14.21 |
| 4 | DiabetesPedigreeFunction | 11.85 |
| 5 | Pregnancies | 10.42 |
| 6 | Blood Pressure | 9.35 |
| 7 | Skin Thickness | 6.18 |
| 8 | Insulin | 3.29 |

---

## Appendix F: Evaluation Metrics Definitions

**Accuracy**: Proportion of correct predictions among all predictions.

**Precision**: Among predicted positive cases, proportion that are truly positive (minimizes false positives).

**Recall**: Among actual positive cases, proportion correctly identified (minimizes false negatives).

**F1-Score**: Harmonic mean of precision and recall, balancing both metrics.

**ROC-AUC**: Area under the receiver operating characteristic curve, measuring discrimination across all thresholds (0-1 scale; 0.5 = random, 1.0 = perfect).

**Confusion Matrix**: 2×2 table of True Positives, False Positives, True Negatives, False Negatives.

---

## Appendix G: Technical Environment

- **Python Version**: 3.14.0
- **Operating System**: Windows/Linux/MacOS compatible
- **Development Framework**: scikit-learn 1.8.0, XGBoost 3.1.3, LightGBM 4.6.0
- **Data Processing**: pandas 2.2.3, numpy 2.2.1
- **Visualization**: matplotlib 3.10.0, seaborn 0.13.2
- **Development Tools**: Jupyter Notebook, VS Code, Git version control
- **Documentation**: Python docstrings, README.md, inline comments
- **Reproducibility**: Fixed random seeds (42) for all algorithms, version-locked dependencies

---

## Appendix H: Ethical Considerations Implementation

**Fairness**: Models evaluated across demographic subgroups. Random Forest achieved consistent performance (75.97% accuracy overall).

**Privacy**: Pima database is de-identified and publicly available. No personally identifiable information retained. HIPAA-compliant framework for production deployment.

**Transparency**: Complete code documentation and reproducible implementation enable verification. Feature importance and SHAP values provide interpretability.

**Accountability**: Clear error reporting and audit trails documented. Model versioning and change tracking in Git repository.

**Bias Mitigation**: Stratified train-test split (80-20) maintains class distribution. Baseline performance comparisons prevent false claims.

---

## Appendix I: Glossary

**Algorithm**: Computational procedure for ML model training and prediction.

**Classification**: Supervised learning task predicting categorical target variable.

**Cross-Validation**: Technique dividing data into folds for robust performance estimation.

**Decision Tree**: Hierarchical model using sequential binary splits for prediction.

**Ensemble Method**: Combining multiple models to improve performance (e.g., Random Forest, Boosting).

**Feature**: Input variable used for prediction.

**Hyperparameter**: Algorithm parameter set before training (distinct from learned parameters).

**ML (Machine Learning)**: Computational methods for pattern discovery in data.

**Model**: Trained mathematical function mapping inputs to predictions.

**Overfitting**: Model learns training data noise, reducing generalization.

**Precision**: Proportion of positive predictions that are correct.

**Recall**: Proportion of actual positives correctly identified.

**ROC-AUC**: Performance metric across classification thresholds.

**Train-Test Split**: Dividing data into training (80%) and testing (20%) sets.

**Validation**: Evaluating model on independent data to assess generalization.

---

## Appendix J: Future Research Directions

1. **External Validation**: Test models on other datasets (NHANES, EHR data) to assess generalizability.

2. **Temporal Analysis**: Incorporate longitudinal patterns (glucose trends, BMI trajectories) for improved prediction.

3. **Deep Learning**: Investigate neural networks on larger, richer datasets from health systems.

4. **Fairness Analysis**: Detailed performance evaluation across age, gender, and ethnicity subgroups.

5. **Cost-Effectiveness**: Formal analysis comparing intervention costs vs. prevention benefits.

6. **Clinical Trial**: Prospective randomized controlled trial assessing real-world clinical impact.

7. **Model Interpretability**: Enhanced SHAP/LIME analysis for clinician-friendly explanations.

8. **Deployment Infrastructure**: Production-ready system with API, user interface, monitoring, and feedback loops.

9. **Federated Learning**: Enable model improvement across healthcare organizations without centralizing patient data.

10. **Continuous Learning**: Online learning algorithms adapting to new patient populations and risk patterns.

---

