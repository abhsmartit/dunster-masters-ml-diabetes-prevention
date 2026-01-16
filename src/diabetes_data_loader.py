"""
Diabetes Dataset Loader
Specialized module for loading and preparing the Pima Indians Diabetes dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class DiabetesDataLoader:
    """
    Specialized data loader for Pima Indians Diabetes Dataset
    """
    
    def __init__(self, data_path='data/raw/diabetes.csv'):
        """
        Initialize the diabetes data loader
        
        Parameters:
        -----------
        data_path : str
            Path to the diabetes.csv file
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Define column names for Pima dataset
        self.column_names = [
            'Pregnancies',
            'Glucose',
            'BloodPressure',
            'SkinThickness',
            'Insulin',
            'BMI',
            'DiabetesPedigreeFunction',
            'Age',
            'Outcome'
        ]
        
        # Columns that should not have zero values (biologically impossible)
        self.zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 
                                  'Insulin', 'BMI']
    
    def _normalize_columns(self):
        """
        Normalize column names to standard format
        """
        if self.df is None:
            return
        
        # Mapping from abbreviated to full names
        column_mapping = {
            'preg': 'Pregnancies',
            'plas': 'Glucose',
            'pres': 'BloodPressure',
            'skin': 'SkinThickness',
            'insu': 'Insulin',
            'mass': 'BMI',
            'pedi': 'DiabetesPedigreeFunction',
            'age': 'Age',
            'class': 'Outcome'
        }
        
        # Rename columns
        self.df.rename(columns=column_mapping, inplace=True)
        
        # Convert Outcome to numeric if it's text
        if 'Outcome' in self.df.columns and self.df['Outcome'].dtype == 'object':
            outcome_mapping = {
                'tested_positive': 1,
                'tested_negative': 0,
                'positive': 1,
                'negative': 0,
                'yes': 1,
                'no': 0
            }
            self.df['Outcome'] = self.df['Outcome'].map(outcome_mapping)
            print("✓ Converted Outcome to numeric (0: No Diabetes, 1: Diabetes)")
    
    def download_dataset(self):
        """
        Download dataset from sklearn if local file not found
        """
        try:
            from sklearn.datasets import fetch_openml
            
            print("📥 Downloading diabetes dataset from OpenML...")
            diabetes = fetch_openml(name='diabetes', version=1, as_frame=True)
            self.df = diabetes.frame
            
            # Rename columns to match standard names
            if 'class' in self.df.columns:
                self.df.rename(columns={'class': 'Outcome'}, inplace=True)
            
            # Save to file
            self.df.to_csv(self.data_path, index=False)
            print(f"✓ Dataset saved to {self.data_path}")
            
            return self.df
            
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            print("\nPlease download manually from:")
            print("https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database")
            return None
    
    def load_data(self, auto_download=True):
        """
        Load the diabetes dataset
        
        Parameters:
        -----------
        auto_download : bool
            If True, attempt to download if file not found
            
        Returns:
        --------
        pd.DataFrame : Loaded dataset
        """
        try:
            # Try to load from file
            self.df = pd.read_csv(self.data_path)
            print(f"✓ Dataset loaded successfully from {self.data_path}")
            print(f"  Shape: {self.df.shape}")
            
        except FileNotFoundError:
            print(f"⚠️  File not found: {self.data_path}")
            
            if auto_download:
                self.df = self.download_dataset()
            else:
                print("\nPlease download the dataset and place it in data/raw/diabetes.csv")
                print("See data/raw/DATASET_README.md for instructions")
                return None
        
        # Normalize column names if needed
        self._normalize_columns()
        
        # Validate dataset
        self._validate_dataset()
        
        return self.df
    
    def _validate_dataset(self):
        """
        Validate the loaded dataset has expected structure
        """
        if self.df is None:
            return
        
        # Check for target column
        if 'Outcome' not in self.df.columns:
            # Try to find it with different names
            possible_names = ['class', 'target', 'label', 'diabetes']
            for name in possible_names:
                if name in self.df.columns:
                    self.df.rename(columns={name: 'Outcome'}, inplace=True)
                    print(f"✓ Renamed '{name}' to 'Outcome'")
                    break
        
        # Check expected columns
        expected_cols = ['Glucose', 'BMI', 'Age', 'Outcome']
        missing_cols = [col for col in expected_cols if col not in self.df.columns]
        
        if missing_cols:
            print(f"⚠️  Warning: Missing expected columns: {missing_cols}")
        else:
            print("✓ Dataset structure validated")
    
    def get_data_info(self):
        """
        Get detailed information about the dataset
        
        Returns:
        --------
        dict : Dataset information
        """
        if self.df is None:
            print("Please load data first using load_data()")
            return None
        
        info = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'zero_values': {},
            'class_distribution': self.df['Outcome'].value_counts().to_dict() if 'Outcome' in self.df.columns else None,
            'statistical_summary': self.df.describe().to_dict()
        }
        
        # Check for zero values that shouldn't be zero
        for col in self.zero_not_accepted:
            if col in self.df.columns:
                zero_count = (self.df[col] == 0).sum()
                info['zero_values'][col] = zero_count
        
        # Print summary
        print("\n" + "="*60)
        print("DIABETES DATASET INFORMATION")
        print("="*60)
        print(f"\n📊 Shape: {info['shape']}")
        print(f"🔢 Features: {info['shape'][1] - 1} (+ 1 target)")
        
        if info['class_distribution']:
            print(f"\n🎯 Class Distribution:")
            for cls, count in info['class_distribution'].items():
                pct = (count / len(self.df)) * 100
                label = 'Diabetes' if cls == 1 else 'No Diabetes'
                print(f"   Class {cls} ({label}): {count} ({pct:.1f}%)")
        
        total_zeros = sum(info['zero_values'].values())
        if total_zeros > 0:
            print(f"\n⚠️  Zero Values (Potential Missing Data):")
            for col, count in info['zero_values'].items():
                if count > 0:
                    pct = (count / len(self.df)) * 100
                    print(f"   {col}: {count} ({pct:.1f}%)")
            print(f"\n   Total problematic zeros: {total_zeros}")
        
        return info
    
    def handle_zero_values(self, strategy='median'):
        """
        Replace zero values with median/mean for biological impossibilities
        
        Parameters:
        -----------
        strategy : str
            'median' or 'mean' for replacement
            
        Returns:
        --------
        pd.DataFrame : Dataset with zeros replaced
        """
        if self.df is None:
            print("Please load data first")
            return None
        
        print(f"\n🔧 Handling zero values using {strategy} strategy...")
        
        for col in self.zero_not_accepted:
            if col in self.df.columns:
                # Count zeros before
                zeros_before = (self.df[col] == 0).sum()
                
                if zeros_before > 0:
                    # Calculate replacement value from non-zero values
                    non_zero_values = self.df[col][self.df[col] != 0]
                    
                    if strategy == 'median':
                        replacement = non_zero_values.median()
                    else:  # mean
                        replacement = non_zero_values.mean()
                    
                    # Replace zeros
                    self.df[col] = self.df[col].replace(0, replacement)
                    
                    print(f"   ✓ {col}: {zeros_before} zeros replaced with {replacement:.2f}")
        
        print("✓ Zero values handled successfully")
        return self.df
    
    def prepare_for_modeling(self, test_size=0.2, random_state=42, 
                            handle_zeros=True, scale_features=True):
        """
        Prepare data for machine learning
        
        Parameters:
        -----------
        test_size : float
            Proportion of data for testing
        random_state : int
            Random seed for reproducibility
        handle_zeros : bool
            Whether to handle zero values
        scale_features : bool
            Whether to scale features
            
        Returns:
        --------
        tuple : X_train, X_test, y_train, y_test
        """
        if self.df is None:
            print("Please load data first")
            return None
        
        print("\n⚙️  Preparing data for modeling...")
        
        # Handle zero values if requested
        if handle_zeros:
            self.handle_zero_values()
        
        # Separate features and target
        X = self.df.drop('Outcome', axis=1)
        y = self.df['Outcome']
        
        # Split data with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"✓ Data split: {len(self.X_train)} training, {len(self.X_test)} testing")
        
        # Scale features if requested
        if scale_features:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            
            self.X_train = pd.DataFrame(
                scaler.fit_transform(self.X_train),
                columns=self.X_train.columns,
                index=self.X_train.index
            )
            self.X_test = pd.DataFrame(
                scaler.transform(self.X_test),
                columns=self.X_test.columns,
                index=self.X_test.index
            )
            print("✓ Features scaled using StandardScaler")
        
        # Print class distribution
        print(f"\nClass Distribution:")
        print(f"  Training - Diabetes: {(self.y_train == 1).sum()} ({(self.y_train == 1).sum() / len(self.y_train) * 100:.1f}%)")
        print(f"  Training - No Diabetes: {(self.y_train == 0).sum()} ({(self.y_train == 0).sum() / len(self.y_train) * 100:.1f}%)")
        print(f"  Testing - Diabetes: {(self.y_test == 1).sum()} ({(self.y_test == 1).sum() / len(self.y_test) * 100:.1f}%)")
        print(f"  Testing - No Diabetes: {(self.y_test == 0).sum()} ({(self.y_test == 0).sum() / len(self.y_test) * 100:.1f}%)")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def save_processed_data(self, output_path='data/processed/diabetes_processed.csv'):
        """
        Save processed dataset
        
        Parameters:
        -----------
        output_path : str
            Path to save processed data
        """
        if self.df is None:
            print("No data to save")
            return
        
        self.df.to_csv(output_path, index=False)
        print(f"✓ Processed data saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("DIABETES DATA LOADER")
    print("="*60)
    
    # Initialize loader
    loader = DiabetesDataLoader()
    
    # Load data
    df = loader.load_data(auto_download=True)
    
    if df is not None:
        # Get dataset information
        info = loader.get_data_info()
        
        # Prepare for modeling
        X_train, X_test, y_train, y_test = loader.prepare_for_modeling(
            test_size=0.2,
            handle_zeros=True,
            scale_features=True
        )
        
        print("\n" + "="*60)
        print("✓ Data loader ready for use!")
        print("="*60)
        print("\nUsage in your scripts:")
        print(">>> from src.diabetes_data_loader import DiabetesDataLoader")
        print(">>> loader = DiabetesDataLoader()")
        print(">>> X_train, X_test, y_train, y_test = loader.prepare_for_modeling()")
