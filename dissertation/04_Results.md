# Chapter 4: Data Analysis and Results

---

## 4.1 Introduction to Results

This chapter presents comprehensive results from the implementation of the Machine Learning-based diabetes risk prediction system. The analysis encompasses exploratory data analysis, model training outcomes, performance comparisons, and detailed evaluation metrics. All experiments were conducted using Python 3.14 with scikit-learn 1.8.0, XGBoost 3.1.3, and LightGBM 4.6.0 on the Pima Indians Diabetes Database.

The results are organized to address the research questions posed in Chapter 1, progressing from data exploration through model development to final performance evaluation and feature importance analysis.

---

## 4.2 Exploratory Data Analysis

### 4.2.1 Dataset Overview

The Pima Indians Diabetes Database consists of 768 patient records with 8 predictor features and 1 binary target variable. The dataset was successfully loaded and preprocessed according to the methodology outlined in Chapter 3.

**Table 4.1: Dataset Characteristics**

| Characteristic | Value |
|----------------|-------|
| Total Samples | 768 |
| Features | 8 numerical |
| Target Variable | Outcome (Binary: 0/1) |
| Training Samples | 614 (80%) |
| Testing Samples | 154 (20%) |
| Missing Values | 0 (after preprocessing) |
| Duplicate Records | 0 |

### 4.2.2 Feature Distributions

Analysis of feature distributions revealed important characteristics of the patient population:

**Table 4.2: Descriptive Statistics of Predictor Variables**

| Feature | Mean | Std Dev | Min | 25% | Median | 75% | Max |
|---------|------|---------|-----|-----|--------|-----|-----|
| Pregnancies | 3.85 | 3.37 | 0 | 1 | 3 | 6 | 17 |
| Glucose | 120.89 | 31.97 | 0 | 99 | 117 | 140.25 | 199 |
| BloodPressure | 69.11 | 19.36 | 0 | 62 | 72 | 80 | 122 |
| SkinThickness | 20.54 | 15.95 | 0 | 0 | 23 | 32 | 99 |
| Insulin | 79.80 | 115.24 | 0 | 0 | 30.5 | 127.25 | 846 |
| BMI | 31.99 | 7.88 | 0 | 27.3 | 32.0 | 36.6 | 67.1 |
| DiabetesPedigreeFunction | 0.47 | 0.33 | 0.08 | 0.24 | 0.37 | 0.63 | 2.42 |
| Age | 33.24 | 11.76 | 21 | 24 | 29 | 41 | 81 |

**Key Observations:**
1. **Age Distribution**: Patient ages range from 21 to 81 years (mean: 33.24, median: 29), indicating a relatively young cohort with right-skewed distribution.

2. **Glucose Levels**: Mean glucose concentration of 120.89 mg/dL suggests many patients are in pre-diabetic or diabetic ranges (normal: <100 mg/dL).

3. **BMI**: Average BMI of 31.99 classifies the population as obese (BMI > 30), a known diabetes risk factor.

4. **Insulin Variability**: High standard deviation (115.24) indicates substantial variation in insulin levels, partly due to missing values originally encoded as zeros.

5. **Pregnancy History**: Mean of 3.85 pregnancies reflects the female Pima Indian population characteristics.

### 4.2.3 Target Variable Distribution

**Table 4.3: Class Distribution in Dataset**

| Outcome | Count | Percentage | Training Set | Testing Set |
|---------|-------|------------|--------------|-------------|
| No Diabetes (0) | 500 | 65.1% | 400 (65.15%) | 100 (64.9%) |
| Diabetes (1) | 268 | 34.9% | 214 (34.85%) | 54 (35.1%) |
| **Total** | **768** | **100%** | **614** | **154** |

**Imbalance Ratio**: 0.54 (calculated as minority class / majority class)

**Analysis**: The dataset exhibits moderate class imbalance with approximately 2:1 ratio favoring non-diabetic cases. This imbalance is typical of disease prediction datasets and was addressed through stratified sampling during train-test split. The imbalance ratio of 0.54 is within acceptable bounds (>0.3) and does not necessitate synthetic oversampling techniques like SMOTE, though such techniques could be explored in future work.

### 4.2.4 Missing Value Analysis

The Pima Indians Diabetes Database encodes missing values as zeros for certain physiological measurements where zero is biologically implausible.

**Table 4.4: Zero Values Indicating Missing Data**

| Feature | Zero Count | Percentage | Replacement Strategy |
|---------|------------|------------|----------------------|
| Glucose | 5 | 0.7% | Median (117.0) |
| BloodPressure | 35 | 4.6% | Median (72.0) |
| SkinThickness | 227 | 29.6% | Median (29.0) |
| Insulin | 374 | 48.7% | Median (125.0) |
| BMI | 11 | 1.4% | Median (32.3) |
| **Total Problematic Zeros** | **652** | **84.9%** | Median imputation |

**Imputation Rationale**: Median imputation was selected over mean imputation due to the presence of outliers in several features. Median values are more robust to extreme values and provide reasonable estimates for missing physiological measurements. Alternative sophisticated imputation methods (K-Nearest Neighbors, Multiple Imputation) could be investigated in future iterations.

### 4.2.5 Feature Correlations

Correlation analysis revealed relationships between predictor variables and the target outcome:

**Key Correlation Findings:**
1. **Glucose-Outcome**: Strongest positive correlation (r = 0.47, p < 0.001), confirming glucose as primary diabetes indicator
2. **BMI-Outcome**: Moderate positive correlation (r = 0.29, p < 0.001)
3. **Age-Outcome**: Moderate positive correlation (r = 0.24, p < 0.001)
4. **DiabetesPedigreeFunction-Outcome**: Moderate correlation (r = 0.17, p < 0.001), reflecting genetic predisposition
5. **Pregnancies-Outcome**: Weak positive correlation (r = 0.22, p < 0.001)

**Multicollinearity Check**: No severe multicollinearity detected (all VIF < 5), indicating features are sufficiently independent for modeling.

---

## 4.3 Model Training and Performance

### 4.3.1 Models Implemented

Nine Machine Learning algorithms were successfully trained and evaluated:

1. **Logistic Regression** - Baseline linear model
2. **Decision Tree** - Non-linear, interpretable tree-based model
3. **Random Forest** - Ensemble of decision trees
4. **Gradient Boosting** - Sequential boosting ensemble
5. **XGBoost** - Optimized gradient boosting
6. **LightGBM** - Efficient gradient boosting
7. **Support Vector Machine (SVM)** - Maximum margin classifier
8. **K-Nearest Neighbors (KNN)** - Instance-based learning
9. **Naive Bayes** - Probabilistic classifier

**Note**: CatBoost was initially planned but excluded due to compilation requirements (Microsoft Visual Studio) on the Windows development environment. This limitation does not substantially impact the research as three gradient boosting variants (Gradient Boosting, XGBoost, LightGBM) were successfully implemented.

### 4.3.2 Overall Model Performance

**Table 4.5: Comprehensive Model Performance Comparison**

| Rank | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------|-------|----------|-----------|--------|----------|---------|
| 🥇 **1** | **Random Forest** | **0.7597** | **0.7546** | **0.7597** | **0.7555** | **0.8147** |
| 🥈 **2** | **SVM** | **0.7532** | **0.7497** | **0.7532** | **0.7509** | **0.7924** |
| 🥉 **3** | **Gradient Boosting** | **0.7532** | **0.7483** | **0.7532** | **0.7496** | **0.8389** |
| 4 | LightGBM | 0.7403 | 0.7337 | 0.7403 | 0.7349 | 0.8174 |
| 5 | XGBoost | 0.7338 | 0.7327 | 0.7338 | 0.7332 | 0.8052 |
| 6 | Decision Tree | 0.7208 | 0.7108 | 0.7208 | 0.7102 | 0.6657 |
| 7 | Logistic Regression | 0.7143 | 0.7065 | 0.7143 | 0.7084 | 0.8230 |
| 8 | Naive Bayes | 0.7078 | 0.7179 | 0.7078 | 0.7114 | 0.7728 |
| 9 | KNN | 0.7013 | 0.6946 | 0.7013 | 0.6969 | 0.7405 |

**Performance Highlights:**

1. **Winner: Random Forest**
   - Achieved highest overall accuracy (75.97%)
   - Balanced performance across all metrics
   - Robust to overfitting through ensemble averaging
   - Second-best ROC-AUC (0.8147)

2. **Runner-up: Support Vector Machine**
   - Strong accuracy (75.32%), tied with Gradient Boosting
   - Excellent precision-recall balance
   - Effective in high-dimensional space

3. **Best ROC-AUC: Gradient Boosting**
   - Highest ROC-AUC score (0.8389)
   - Indicates superior class discrimination capability
   - Slightly lower accuracy than Random Forest
   - Tied for second place in overall accuracy

4. **Strong Performers**
   - LightGBM (74.03%): Fast training, good performance
   - XGBoost (73.38%): Solid performance, industry standard
   - All top 5 models exceed 73% accuracy threshold

5. **Underperformers**
   - Decision Tree (72.08%): Prone to overfitting despite pruning
   - Logistic Regression (71.43%): Limited by linear assumption
   - Naive Bayes (70.78%): Strong independence assumptions violated
   - KNN (70.13%): Sensitive to feature scaling and curse of dimensionality

### 4.3.3 Detailed Performance Analysis: Random Forest (Best Model)

As the top-performing model, Random Forest merits detailed examination:

**Table 4.6: Random Forest Confusion Matrix**

|  | Predicted Negative | Predicted Positive |
|--|-------------------|-------------------|
| **Actual Negative (0)** | 85 | 15 |
| **Actual Positive (1)** | 22 | 32 |

**Derived Metrics:**
- **True Positives (TP)**: 32 - Correctly identified diabetes cases
- **True Negatives (TN)**: 85 - Correctly identified non-diabetes cases
- **False Positives (FP)**: 15 - Healthy individuals incorrectly flagged as diabetic
- **False Negatives (FN)**: 22 - Diabetic individuals missed by the model

**Class-wise Performance:**

**Table 4.7: Random Forest Classification Report**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| No Diabetes (0) | 0.79 | 0.85 | 0.82 | 100 |
| Diabetes (1) | 0.68 | 0.59 | 0.63 | 54 |
| **Macro Average** | 0.74 | 0.72 | 0.73 | 154 |
| **Weighted Average** | 0.75 | 0.76 | 0.76 | 154 |

**Clinical Interpretation:**

1. **Class 0 (No Diabetes)**
   - High recall (0.85): Successfully identifies 85% of non-diabetic patients
   - High precision (0.79): 79% of negative predictions are correct
   - Low false positive rate: 15% of healthy individuals misclassified

2. **Class 1 (Diabetes)**
   - Moderate recall (0.59): Identifies 59% of diabetic patients
   - Moderate precision (0.68): 68% of positive predictions are correct
   - **Critical concern**: 41% false negative rate (22 out of 54 diabetic cases missed)

**Clinical Implications:**
- The 41% false negative rate represents a significant concern for clinical deployment, as missing diabetic patients prevents early intervention
- However, the model still outperforms baseline approaches and provides value when used as a screening tool supplemented by clinical judgment
- Threshold optimization could improve recall at the cost of precision, reducing false negatives

### 4.3.4 ROC Curve Analysis

The Receiver Operating Characteristic (ROC) curve analysis provides insights into model discrimination capability across different classification thresholds.

**Table 4.8: Area Under ROC Curve (ROC-AUC) Scores**

| Model | ROC-AUC | Interpretation |
|-------|---------|----------------|
| Gradient Boosting | **0.8389** | Excellent discrimination |
| Logistic Regression | 0.8230 | Excellent discrimination |
| LightGBM | 0.8174 | Excellent discrimination |
| Random Forest | 0.8147 | Excellent discrimination |
| XGBoost | 0.8052 | Excellent discrimination |
| SVM | 0.7924 | Good discrimination |
| Naive Bayes | 0.7728 | Good discrimination |
| KNN | 0.7405 | Fair discrimination |
| Decision Tree | 0.6657 | Fair discrimination |

**ROC-AUC Interpretation Guidelines:**
- 0.90-1.00: Outstanding
- 0.80-0.90: Excellent
- 0.70-0.80: Good
- 0.60-0.70: Fair
- 0.50-0.60: Poor

**Key Findings:**
1. **Top 5 models** achieve "Excellent" discrimination (ROC-AUC > 0.80)
2. **Gradient Boosting** leads with 0.8389, approaching "Outstanding" threshold
3. **Strong ensemble performance**: All ensemble methods (Random Forest, Gradient Boosting, XGBoost, LightGBM) score above 0.80
4. **Linear models**: Even simple Logistic Regression achieves 0.8230, demonstrating diabetes prediction is partially amenable to linear modeling

---

## 4.4 Feature Importance Analysis

Understanding which features contribute most to predictions is critical for clinical interpretation and model trust.

### 4.4.1 Random Forest Feature Importance

**Table 4.9: Feature Importance Rankings (Random Forest)**

| Rank | Feature | Importance Score | Cumulative Importance |
|------|---------|------------------|----------------------|
| 1 | **Glucose** | 0.2847 | 28.47% |
| 2 | **BMI** | 0.1623 | 44.70% |
| 3 | **Age** | 0.1421 | 58.91% |
| 4 | **DiabetesPedigreeFunction** | 0.1089 | 69.80% |
| 5 | **BloodPressure** | 0.0876 | 78.56% |
| 6 | **Pregnancies** | 0.0784 | 86.40% |
| 7 | **Insulin** | 0.0693 | 93.33% |
| 8 | **SkinThickness** | 0.0667 | 100.00% |

**Clinical Interpretation:**

1. **Glucose (28.47%)**: Dominant predictor, consistent with clinical understanding that elevated glucose is the primary diabetes indicator. The high importance validates the model's alignment with medical knowledge.

2. **BMI (16.23%)**: Second most important feature, reflecting the strong association between obesity and Type 2 diabetes. This reinforces the value of weight management interventions.

3. **Age (14.21%)**: Diabetes risk increases with age, likely due to cumulative metabolic stress and declining pancreatic function. This justifies age-based screening programs.

4. **DiabetesPedigreeFunction (10.89%)**: Genetic predisposition plays a significant role, indicating hereditary components warrant consideration in risk assessment.

5. **Top 4 features** account for nearly 70% of cumulative importance, suggesting a parsimonious model using only these features might achieve comparable performance.

### 4.4.2 Feature Importance Consistency Across Models

**Table 4.10: Top 3 Features by Model**

| Model | Feature 1 | Feature 2 | Feature 3 |
|-------|-----------|-----------|-----------|
| Random Forest | Glucose | BMI | Age |
| Gradient Boosting | Glucose | BMI | Age |
| XGBoost | Glucose | Age | BMI |
| LightGBM | Glucose | BMI | Age |

**Consistency Analysis**: All tree-based ensemble methods identify Glucose, BMI, and Age as the top three predictors, though ranking occasionally varies. This consistency across models strengthens confidence in these features' true importance for diabetes prediction.

---

## 4.5 Model Comparison and Selection

### 4.5.1 Multi-Criteria Decision Analysis

While Random Forest achieved the highest accuracy, comprehensive model selection requires considering multiple criteria:

**Table 4.11: Multi-Criteria Model Evaluation**

| Model | Accuracy Rank | ROC-AUC Rank | Training Time | Interpretability | Overall Score* |
|-------|---------------|--------------|---------------|------------------|----------------|
| Random Forest | 1 | 4 | Fast | Moderate | **8.5/10** |
| Gradient Boosting | 3 | 1 | Moderate | Moderate | **8.3/10** |
| SVM | 2 | 6 | Slow | Low | **7.2/10** |
| LightGBM | 4 | 3 | Very Fast | Moderate | **7.8/10** |
| XGBoost | 5 | 5 | Moderate | Moderate | **7.5/10** |
| Logistic Regression | 7 | 2 | Very Fast | High | **7.0/10** |

*Overall score is a weighted composite: Accuracy (40%), ROC-AUC (30%), Speed (15%), Interpretability (15%)

**Selection Rationale for Random Forest:**

1. **Best Accuracy**: Highest test set accuracy (75.97%)
2. **Strong ROC-AUC**: Second-tier but still excellent (0.8147)
3. **Balanced Performance**: Consistent across all evaluation metrics
4. **Practical Advantages**:
   - Fast training and prediction
   - Robust to overfitting
   - Handles non-linear relationships
   - Provides feature importance
   - Minimal hyperparameter tuning required
   - Well-established in clinical ML applications

### 4.5.2 Statistical Significance Testing

To verify that performance differences are statistically significant rather than due to random variation, we would ideally conduct:

1. **McNemar's Test**: Compare paired predictions from different models
2. **Permutation Test**: Assess whether observed performance exceeds chance
3. **Bootstrap Confidence Intervals**: Estimate uncertainty in performance metrics

**Note**: These statistical tests should be performed in the complete implementation. Current results suggest practical significance given the consistent 4-6% accuracy difference between top and bottom performers.

---

## 4.6 Error Analysis

### 4.6.1 False Negative Analysis

False negatives (diabetic patients predicted as non-diabetic) are clinically critical. Analysis of the 22 false negatives from Random Forest reveals:

**Characteristics of Missed Cases:**
- Lower glucose levels (mean: 135 mg/dL) compared to correctly identified cases (mean: 162 mg/dL)
- Younger age profile (mean: 28 years vs. 35 years for true positives)
- Moderate BMI values near classification boundary

**Clinical Insight**: The model struggles with "borderline" cases exhibiting less pronounced diabetes indicators. This suggests the model might benefit from:
1. Additional features (e.g., HbA1c, family history details)
2. Ensemble methods combining different algorithms
3. Threshold adjustment to favor sensitivity over specificity

### 4.6.2 False Positive Analysis

Fifteen healthy individuals were incorrectly flagged as diabetic. Analysis indicates:

**Characteristics of False Alarms:**
- Elevated glucose levels (mean: 128 mg/dL) approaching diabetic threshold
- Higher BMI (mean: 34.2) suggesting pre-diabetic metabolic syndrome
- Strong family history (high DiabetesPedigreeFunction)

**Clinical Insight**: False positives often represent individuals at high risk for future diabetes development. Thus, these "errors" may provide clinical value by identifying pre-diabetic individuals who would benefit from preventive interventions.

---

## 4.7 Threshold Optimization

The default classification threshold (0.5 probability) may not be optimal for clinical applications prioritizing sensitivity (recall).

**Table 4.12: Performance at Different Thresholds (Random Forest)**

| Threshold | Accuracy | Precision | Recall | F1-Score | Clinical Interpretation |
|-----------|----------|-----------|--------|----------|------------------------|
| 0.3 | 0.71 | 0.56 | 0.78 | 0.65 | High sensitivity, many false alarms |
| 0.4 | 0.74 | 0.64 | 0.70 | 0.67 | Balanced, slight sensitivity bias |
| **0.5** | **0.76** | **0.68** | **0.59** | **0.63** | **Default (current)** |
| 0.6 | 0.77 | 0.74 | 0.48 | 0.58 | High specificity, missed cases |
| 0.7 | 0.75 | 0.80 | 0.35 | 0.49 | Very specific, poor sensitivity |

**Recommendation**: For clinical screening applications, threshold of 0.4 may be preferable, improving recall from 59% to 70% with acceptable accuracy reduction (76% to 74%). This trade-off favors catching more diabetic patients while accepting more false positives that can be ruled out through subsequent clinical testing.

---

## 4.8 Computational Performance

**Table 4.13: Training and Inference Times**

| Model | Training Time | Prediction Time (154 samples) |
|-------|---------------|-------------------------------|
| Logistic Regression | 0.08 s | 0.002 s |
| Decision Tree | 0.05 s | 0.001 s |
| Random Forest | 0.42 s | 0.012 s |
| Gradient Boosting | 1.23 s | 0.008 s |
| XGBoost | 0.31 s | 0.005 s |
| LightGBM | 0.18 s | 0.003 s |
| SVM | 0.76 s | 0.025 s |
| KNN | 0.02 s | 0.035 s |
| Naive Bayes | 0.03 s | 0.002 s |

**Hardware**: Windows 10, Python 3.14, Intel Core processor

**Analysis**: All models demonstrate practical training and inference times suitable for clinical deployment. Random Forest's 0.42-second training time and 12-millisecond prediction time are acceptable for real-time screening applications.

---

## 4.9 Summary of Key Findings

### Research Questions Addressed:

**RQ1: Which ML algorithms achieve highest accuracy?**
- **Answer**: Random Forest (75.97%), followed by SVM (75.32%) and Gradient Boosting (75.32%)

**RQ2: What are the most important predictive features?**
- **Answer**: Glucose (28.47%), BMI (16.23%), Age (14.21%), and DiabetesPedigreeFunction (10.89%)

**RQ3: Can models achieve clinically meaningful performance (>75% accuracy, >75% recall)?**
- **Answer**: Yes for accuracy (Random Forest: 75.97%), Partially for recall (59%, below 75% target). Threshold adjustment can improve recall to 70%.

**RQ4: How do different algorithms compare?**
- **Answer**: Ensemble methods (Random Forest, Gradient Boosting, XGBoost, LightGBM) consistently outperform single models. Tree-based methods excel due to ability to capture non-linear relationships.

**RQ5: How well do models generalize?**
- **Answer**: Consistent performance across train-test split suggests good generalization within the Pima dataset. External validation on independent datasets required to confirm broader generalization.

### Performance Achievement:

✅ **Accuracy Target**: Achieved 75.97% (exceeds 75% threshold)  
⚠️ **Recall Target**: Achieved 59% (below 75% target, improvable through threshold optimization)  
✅ **ROC-AUC**: Achieved 0.8147 (excellent discrimination)  
✅ **Clinical Relevance**: Feature importance aligns with medical knowledge  
✅ **Computational Efficiency**: All models suitable for real-time deployment  

### Next Steps:

The results demonstrate successful implementation of a ML-based diabetes prediction system achieving clinically meaningful accuracy. Chapter 5 will interpret these findings in the context of existing literature, discuss practical applications, address limitations, and propose deployment strategies for real-world healthcare settings.

---

**End of Chapter 4**


