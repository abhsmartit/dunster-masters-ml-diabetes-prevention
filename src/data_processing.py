"""
Data Processing Module
Handles data loading, preprocessing, cleaning, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    """
    Main class for data processing operations
    """
    
    def __init__(self, data_path=None, random_state=42):
        """
        Initialize DataProcessor
        
        Parameters:
        -----------
        data_path : str
            Path to the dataset file
        random_state : int
            Random state for reproducibility
        """
        self.data_path = data_path
        self.random_state = random_state
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, data_path=None):
        """
        Load data from CSV, Excel, or other formats
        
        Parameters:
        -----------
        data_path : str
            Path to dataset (optional, uses self.data_path if not provided)
            
        Returns:
        --------
        pd.DataFrame : Loaded dataset
        """
        if data_path:
            self.data_path = data_path
            
        if self.data_path is None:
            raise ValueError("Please provide a data path")
        
        # Determine file type and load accordingly
        if self.data_path.endswith('.csv'):
            self.df = pd.read_csv(self.data_path)
        elif self.data_path.endswith(('.xlsx', '.xls')):
            self.df = pd.read_excel(self.data_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel files.")
        
        print(f"✓ Data loaded successfully!")
        print(f"  Shape: {self.df.shape}")
        print(f"  Columns: {list(self.df.columns)}")
        
        return self.df
    
    def explore_data(self):
        """
        Perform basic exploratory data analysis
        
        Returns:
        --------
        dict : Dictionary containing data statistics
        """
        if self.df is None:
            raise ValueError("Please load data first using load_data()")
        
        info = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'missing_percentage': (self.df.isnull().sum() / len(self.df) * 100).to_dict(),
            'duplicates': self.df.duplicated().sum(),
            'numerical_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object', 'category']).columns),
        }
        
        print("\n" + "="*60)
        print("DATA EXPLORATION SUMMARY")
        print("="*60)
        print(f"\n📊 Dataset Shape: {info['shape']}")
        print(f"🔢 Numerical Columns: {len(info['numerical_columns'])}")
        print(f"📝 Categorical Columns: {len(info['categorical_columns'])}")
        print(f"❌ Total Missing Values: {sum(info['missing_values'].values())}")
        print(f"🔄 Duplicate Rows: {info['duplicates']}")
        
        if sum(info['missing_values'].values()) > 0:
            print("\n⚠️  Columns with Missing Values:")
            for col, count in info['missing_values'].items():
                if count > 0:
                    pct = info['missing_percentage'][col]
                    print(f"   - {col}: {count} ({pct:.2f}%)")
        
        return info
    
    def handle_missing_values(self, strategy='mean', threshold=0.5):
        """
        Handle missing values in the dataset
        
        Parameters:
        -----------
        strategy : str
            Strategy for imputation ('mean', 'median', 'mode', 'drop')
        threshold : float
            If proportion of missing > threshold, drop the column
            
        Returns:
        --------
        pd.DataFrame : Dataset with handled missing values
        """
        if self.df is None:
            raise ValueError("Please load data first")
        
        print("\n🔧 Handling Missing Values...")
        
        # Drop columns with too many missing values
        missing_ratio = self.df.isnull().sum() / len(self.df)
        cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
        
        if cols_to_drop:
            print(f"   Dropping {len(cols_to_drop)} columns with >{threshold*100}% missing values")
            self.df = self.df.drop(columns=cols_to_drop)
        
        # Handle numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0 and strategy != 'drop':
            imputer_num = SimpleImputer(strategy=strategy if strategy in ['mean', 'median'] else 'mean')
            self.df[numerical_cols] = imputer_num.fit_transform(self.df[numerical_cols])
        
        # Handle categorical columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0 and strategy != 'drop':
            imputer_cat = SimpleImputer(strategy='most_frequent')
            self.df[categorical_cols] = imputer_cat.fit_transform(self.df[categorical_cols])
        
        # If strategy is 'drop', drop remaining rows with missing values
        if strategy == 'drop':
            initial_rows = len(self.df)
            self.df = self.df.dropna()
            print(f"   Dropped {initial_rows - len(self.df)} rows with missing values")
        
        print(f"✓ Missing values handled. Remaining missing: {self.df.isnull().sum().sum()}")
        
        return self.df
    
    def encode_categorical_features(self, columns=None):
        """
        Encode categorical features using Label Encoding
        
        Parameters:
        -----------
        columns : list
            List of columns to encode (if None, all categorical columns)
            
        Returns:
        --------
        pd.DataFrame : Dataset with encoded features
        """
        if self.df is None:
            raise ValueError("Please load data first")
        
        if columns is None:
            columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print(f"\n🔤 Encoding {len(columns)} categorical features...")
        
        for col in columns:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
                print(f"   ✓ {col}: {len(le.classes_)} unique values")
        
        return self.df
    
    def scale_features(self, columns=None, method='standard'):
        """
        Scale numerical features
        
        Parameters:
        -----------
        columns : list
            List of columns to scale (if None, all numerical columns)
        method : str
            Scaling method ('standard' or 'minmax')
            
        Returns:
        --------
        pd.DataFrame : Dataset with scaled features
        """
        if self.df is None:
            raise ValueError("Please load data first")
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"\n📏 Scaling features using {method} scaling...")
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Method must be 'standard' or 'minmax'")
        
        self.df[columns] = scaler.fit_transform(self.df[columns])
        self.scaler = scaler
        
        print(f"✓ Scaled {len(columns)} features")
        
        return self.df
    
    def prepare_data(self, target_column, test_size=0.2, scale=True):
        """
        Prepare data for machine learning (split and scale)
        
        Parameters:
        -----------
        target_column : str
            Name of the target variable column
        test_size : float
            Proportion of data for testing
        scale : bool
            Whether to scale features
            
        Returns:
        --------
        tuple : X_train, X_test, y_train, y_test
        """
        if self.df is None:
            raise ValueError("Please load data first")
        
        if target_column not in self.df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        print("\n⚙️  Preparing data for ML...")
        
        # Separate features and target
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y if y.nunique() < 30 else None
        )
        
        print(f"✓ Data split: {len(self.X_train)} training, {len(self.X_test)} testing samples")
        
        # Scale features if requested
        if scale:
            print("✓ Scaling features...")
            self.scaler = StandardScaler()
            self.X_train = pd.DataFrame(
                self.scaler.fit_transform(self.X_train),
                columns=self.X_train.columns,
                index=self.X_train.index
            )
            self.X_test = pd.DataFrame(
                self.scaler.transform(self.X_test),
                columns=self.X_test.columns,
                index=self.X_test.index
            )
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_feature_info(self):
        """
        Get information about features in the dataset
        
        Returns:
        --------
        pd.DataFrame : Feature information
        """
        if self.df is None:
            raise ValueError("Please load data first")
        
        feature_info = pd.DataFrame({
            'Column': self.df.columns,
            'Type': self.df.dtypes,
            'Null_Count': self.df.isnull().sum(),
            'Null_Percentage': (self.df.isnull().sum() / len(self.df) * 100).round(2),
            'Unique_Values': [self.df[col].nunique() for col in self.df.columns],
            'Sample_Value': [self.df[col].iloc[0] if len(self.df) > 0 else None for col in self.df.columns]
        })
        
        return feature_info
    
    def save_processed_data(self, output_path):
        """
        Save processed data to file
        
        Parameters:
        -----------
        output_path : str
            Path to save the processed data
        """
        if self.df is None:
            raise ValueError("No data to save")
        
        if output_path.endswith('.csv'):
            self.df.to_csv(output_path, index=False)
        elif output_path.endswith('.xlsx'):
            self.df.to_excel(output_path, index=False)
        else:
            raise ValueError("Output path must end with .csv or .xlsx")
        
        print(f"✓ Processed data saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    print("Data Processing Module")
    print("=" * 60)
    print("\nThis module provides comprehensive data processing capabilities:")
    print("- Data loading (CSV, Excel)")
    print("- Missing value handling")
    print("- Categorical encoding")
    print("- Feature scaling")
    print("- Train-test splitting")
    print("\nImport and use in your scripts:")
    print(">>> from src.data_processing import DataProcessor")
    print(">>> processor = DataProcessor('data/raw/dataset.csv')")
    print(">>> processor.load_data()")
