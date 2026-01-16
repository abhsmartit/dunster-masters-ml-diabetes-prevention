# Diabetes Dataset Information

## How to Download the Pima Indians Diabetes Dataset

### Option 1: Direct Download from UCI Repository

1. **Visit UCI ML Repository:**
   - URL: https://archive.ics.uci.edu/dataset/34/diabetes
   - Alternative: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

2. **Download the dataset:**
   - File name: `diabetes.csv` or `pima-indians-diabetes.csv`
   - Place it in this directory: `data/raw/`

3. **Expected file:**
   - `diabetes.csv` with 768 rows and 9 columns

### Option 2: Kaggle Dataset

```bash
# Install Kaggle CLI (if not installed)
pip install kaggle

# Configure Kaggle credentials
# Download from: https://www.kaggle.com/settings -> Create New API Token
# Place kaggle.json in: C:\Users\YourUsername\.kaggle\

# Download the dataset
kaggle datasets download -d uciml/pima-indians-diabetes-database
```

### Option 3: Python Script (Automated Download)

```python
# Run this script to download the dataset
import pandas as pd
from sklearn.datasets import fetch_openml

# Download from OpenML
diabetes = fetch_openml(name='diabetes', version=1, as_frame=True)
df = diabetes.frame

# Save to CSV
df.to_csv('data/raw/diabetes.csv', index=False)
print("✓ Dataset downloaded successfully!")
```

## Dataset Structure

### Columns:
1. **Pregnancies** - Number of times pregnant (int)
2. **Glucose** - Plasma glucose concentration (int)
3. **BloodPressure** - Diastolic blood pressure (mm Hg) (int)
4. **SkinThickness** - Triceps skin fold thickness (mm) (int)
5. **Insulin** - 2-Hour serum insulin (mu U/ml) (int)
6. **BMI** - Body mass index (float)
7. **DiabetesPedigreeFunction** - Diabetes pedigree function (float)
8. **Age** - Age (years) (int)
9. **Outcome** - Class variable (0 or 1) (int) - **TARGET VARIABLE**

### Sample Data:
```
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
6,148,72,35,0,33.6,0.627,50,1
1,85,66,29,0,26.6,0.351,31,0
8,183,64,0,0,23.3,0.672,32,1
```

## Data Quality Notes

### Missing Values (Encoded as Zeros):
- **Glucose**: 5 zeros (biologically impossible)
- **BloodPressure**: 35 zeros
- **SkinThickness**: 227 zeros
- **Insulin**: 374 zeros
- **BMI**: 11 zeros

**Action Required:** Replace zeros with median/mean values during preprocessing

### Class Distribution:
- **Class 0 (No Diabetes):** 500 instances (65.1%)
- **Class 1 (Diabetes):** 268 instances (34.9%)

**Note:** Slight class imbalance - consider using stratified sampling or SMOTE

## Citation

If using this dataset in your dissertation, cite:

```
Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). 
Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. 
In Proceedings of the Symposium on Computer Applications and Medical Care (pp. 261--265). 
IEEE Computer Society Press.
```

## Alternative Datasets (Optional)

### 1. Diabetes Health Indicators Dataset (Kaggle)
- **Size:** 253,680 instances
- **Features:** 21 variables
- **Source:** CDC Behavioral Risk Factor Surveillance System
- **Use:** Model validation, transfer learning

### 2. Early Stage Diabetes Risk Prediction Dataset
- **Size:** 520 instances
- **Features:** 17 attributes
- **Source:** UCI ML Repository
- **Use:** Comparative analysis

## Quick Start

Once you have the dataset, run:

```bash
# Verify dataset
python src/data_processing.py

# Or use Jupyter notebook
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

## Dataset Status

- [ ] Dataset downloaded
- [ ] Data quality verified
- [ ] Missing values identified
- [ ] Exploratory analysis completed
- [ ] Ready for modeling

**Last Updated:** January 16, 2026
