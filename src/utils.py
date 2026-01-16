"""
Utility Functions
Helper functions used across the project
"""

import os
import json
import pickle
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


def create_directory(path):
    """
    Create directory if it doesn't exist
    
    Parameters:
    -----------
    path : str
        Directory path to create
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"✓ Created directory: {path}")
    return path


def save_model(model, filepath, method='joblib'):
    """
    Save trained model to file
    
    Parameters:
    -----------
    model : object
        Trained model object
    filepath : str
        Path to save the model
    method : str
        Saving method ('joblib' or 'pickle')
    """
    create_directory(os.path.dirname(filepath))
    
    if method == 'joblib':
        joblib.dump(model, filepath)
    elif method == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
    else:
        raise ValueError("Method must be 'joblib' or 'pickle'")
    
    print(f"✓ Model saved to {filepath}")


def load_model(filepath, method='joblib'):
    """
    Load trained model from file
    
    Parameters:
    -----------
    filepath : str
        Path to the saved model
    method : str
        Loading method ('joblib' or 'pickle')
        
    Returns:
    --------
    object : Loaded model
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    if method == 'joblib':
        model = joblib.load(filepath)
    elif method == 'pickle':
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
    else:
        raise ValueError("Method must be 'joblib' or 'pickle'")
    
    print(f"✓ Model loaded from {filepath}")
    return model


def save_results(results, filepath):
    """
    Save results dictionary to JSON file
    
    Parameters:
    -----------
    results : dict
        Results dictionary to save
    filepath : str
        Path to save the results
    """
    create_directory(os.path.dirname(filepath))
    
    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj
    
    results_clean = convert_types(results)
    
    with open(filepath, 'w') as f:
        json.dump(results_clean, f, indent=4)
    
    print(f"✓ Results saved to {filepath}")


def load_results(filepath):
    """
    Load results from JSON file
    
    Parameters:
    -----------
    filepath : str
        Path to the results file
        
    Returns:
    --------
    dict : Loaded results
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Results file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    print(f"✓ Results loaded from {filepath}")
    return results


def get_timestamp():
    """
    Get current timestamp as string
    
    Returns:
    --------
    str : Timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def print_section_header(title):
    """
    Print a formatted section header
    
    Parameters:
    -----------
    title : str
        Section title
    """
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def calculate_class_distribution(y):
    """
    Calculate and display class distribution
    
    Parameters:
    -----------
    y : array-like
        Target variable
        
    Returns:
    --------
    dict : Class distribution
    """
    unique, counts = np.unique(y, return_counts=True)
    distribution = dict(zip(unique, counts))
    
    print("\n📊 Class Distribution:")
    for label, count in distribution.items():
        percentage = (count / len(y)) * 100
        print(f"   Class {label}: {count} ({percentage:.2f}%)")
    
    return distribution


def check_data_balance(y, threshold=0.3):
    """
    Check if dataset is balanced
    
    Parameters:
    -----------
    y : array-like
        Target variable
    threshold : float
        Imbalance threshold (default: 0.3 = 30%)
        
    Returns:
    --------
    bool : True if balanced, False if imbalanced
    """
    distribution = calculate_class_distribution(y)
    counts = list(distribution.values())
    
    min_count = min(counts)
    max_count = max(counts)
    
    imbalance_ratio = min_count / max_count
    
    if imbalance_ratio < threshold:
        print(f"\n⚠️  Dataset is IMBALANCED (ratio: {imbalance_ratio:.2f})")
        print("   Consider using techniques like:")
        print("   - SMOTE (Synthetic Minority Over-sampling)")
        print("   - Class weights in model training")
        print("   - Stratified sampling")
        return False
    else:
        print(f"\n✓ Dataset is relatively BALANCED (ratio: {imbalance_ratio:.2f})")
        return True


def set_plot_style():
    """
    Set consistent plot style for the project
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 10


def save_figure(fig, filepath, dpi=300):
    """
    Save matplotlib figure
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save
    filepath : str
        Path to save the figure
    dpi : int
        Resolution (dots per inch)
    """
    create_directory(os.path.dirname(filepath))
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"✓ Figure saved to {filepath}")


def memory_usage(df):
    """
    Calculate memory usage of a DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze
        
    Returns:
    --------
    str : Memory usage string
    """
    mem = df.memory_usage(deep=True).sum() / 1024**2
    return f"{mem:.2f} MB"


def reduce_memory_usage(df):
    """
    Reduce memory usage of DataFrame by optimizing dtypes
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to optimize
        
    Returns:
    --------
    pd.DataFrame : Optimized DataFrame
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage before optimization: {start_mem:.2f} MB")
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage after optimization: {end_mem:.2f} MB")
    print(f"Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%")
    
    return df


if __name__ == "__main__":
    print("Utility Functions Module")
    print("=" * 60)
    print("\nAvailable utility functions:")
    print("- create_directory() - Create directories")
    print("- save_model() / load_model() - Model persistence")
    print("- save_results() / load_results() - Results management")
    print("- calculate_class_distribution() - Analyze target distribution")
    print("- check_data_balance() - Check dataset balance")
    print("- set_plot_style() - Consistent visualization style")
    print("- reduce_memory_usage() - Optimize DataFrame memory")
