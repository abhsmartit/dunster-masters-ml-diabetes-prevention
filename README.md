# Machine Learning-Based Diabetes Risk Prediction System

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Design, Development, and Evaluation for Healthcare Decision Support**

## 🔗 GitHub Repository

**Public Repository**: [https://github.com/abhsmartit/dunster-masters-ml-diabetes-prevention](https://github.com/abhsmartit/dunster-masters-ml-diabetes-prevention)

## Project Title
**Machine Learning-Based Diabetes Risk Prediction System: Design, Development, and Evaluation for Early Intervention and Healthcare Optimization**

## Project Overview
This project develops a comprehensive machine learning system for predicting diabetes risk in patients using the Pima Indians Diabetes Database. By analyzing patient health metrics including glucose levels, BMI, blood pressure, and other clinical indicators, the system enables early detection and intervention, supporting healthcare providers in making data-driven decisions and optimizing patient outcomes.

### Domain: Healthcare Analytics
### Application: Clinical Decision Support & Preventive Care
### Dataset: Pima Indians Diabetes Database (768 instances, 8 features)

## Project Structure
```
dunster-masters/
│
├── data/
│   ├── raw/              # Original, immutable data
│   └── processed/        # Cleaned and transformed data
│
├── notebooks/            # Jupyter notebooks for exploration and analysis
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experimentation.ipynb
│
├── src/                  # Source code for the project
│   ├── __init__.py
│   ├── data_processing.py      # Data loading and preprocessing
│   ├── feature_engineering.py  # Feature creation and selection
│   ├── model_development.py    # ML model implementations
│   ├── model_evaluation.py     # Evaluation metrics and visualization
│   └── utils.py               # Utility functions
│
├── models/               # Trained models (saved as .pkl or .joblib)
│
├── results/              # Generated figures, reports, and outputs
│
├── docs/                 # Documentation and dissertation materials
│
├── main.py              # Main script to run the ML pipeline
├── requirements.txt     # Python dependencies
└── README.md           # This file

```

## Key Requirements Addressed

### 1. Problem / Domain Selection
- **Domain:** Healthcare Analytics
- **Application Area:** Diabetes Risk Prediction
- **Problem Statement:** Diabetes affects over 537 million adults globally. Early prediction enables preventive interventions, reduces complications, and optimizes healthcare resources. This project develops ML models to accurately predict diabetes risk based on patient health indicators.

### 2. Data Understanding & Preparation
- Data collection and preprocessing pipeline
- Feature selection and transformation methods
- Data quality assessment and validation

### 3. Model Development
- Multiple ML algorithms implementation:
  - Linear Models (Logistic Regression, Linear Regression)
  - Tree-based Models (Decision Trees, Random Forest, XGBoost, LightGBM, CatBoost)
  - Support Vector Machines
  - Neural Networks (if applicable)
- Model training, testing, and validation framework
- Hyperparameter optimization

### 4. Performance Evaluation
- Comprehensive metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC, etc.)
- Cross-validation strategies
- Model comparison and interpretation
- Visualization of results

### 5. Application & Improvement Strategy
- Practical deployment considerations
- Model improvement recommendations
- Scalability and maintenance strategies

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone or download this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

4. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Quick Start: Diabetes Prediction

#### Step 1: Download Dataset
```bash
# Run the data download notebook
jupyter notebook notebooks/00_diabetes_data_download.ipynb

# Or use the specialized loader
python src/diabetes_data_loader.py
```

#### Step 2: Run the Full Pipeline
```bash
python main.py --data data/raw/diabetes.csv --target Outcome --task classification
```

### Running Individual Components
```python
# Diabetes data loading and preprocessing
from src.diabetes_data_loader import DiabetesDataLoader
loader = DiabetesDataLoader()
X_train, X_test, y_train, y_test = loader.prepare_for_modeling()

# Model training
from src.model_development import ModelDeveloper
developer = ModelDeveloper(task_type='classification')
models = developer.train_models(X_train, y_train)

# Model evaluation
from src.model_evaluation import ModelEvaluator
evaluator = ModelEvaluator(task_type='classification')
results = evaluator.evaluate_models(models, X_test, y_test)
```

### Jupyter Notebooks (Recommended for Analysis)
```bash
# 1. Data download and initial setup
jupyter notebook notebooks/00_diabetes_data_download.ipynb

# 2. Exploratory data analysis
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb

# 3. Feature engineering
jupyter notebook notebooks/02_feature_engineering.ipynb

# 4. Model experimentation
jupyter notebook notebooks/03_model_experimentation.ipynb
results = evaluator.evaluate_models(models, X_test, y_test)
```

## Expected Learning Outcomes
✓ Understanding of AI and Machine Learning concepts  
✓ Ability to design and evaluate ML models  
✓ Practical exposure to data-driven decision-making  
✓ Application of AI solutions to real-world problems  

## Project Timeline
- Phase 1: Data Collection & Understanding (Weeks 1-2)
- Phase 2: Data Preprocessing & Feature Engineering (Weeks 3-4)
- Phase 3: Model Development & Training (Weeks 5-8)
- Phase 4: Model Evaluation & Optimization (Weeks 9-10)
- Phase 5: Documentation & Dissertation Writing (Weeks 11-16)

## Final Deliverable
A structured 15,000–20,000 word dissertation including:
- Machine learning models and algorithms
- Evaluation metrics and result analysis
- Model architectures and datasets (appendices)
- Performance summaries and recommendations

## Author
Mohammed Azhar

## License
This project is part of a Masters dissertation.

## Contact
For questions or collaboration, please contact through the university portal.
