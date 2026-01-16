"""
Model Development Module
Implements various ML algorithms for classification and regression tasks
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

# CatBoost is optional (requires Visual Studio on Windows)
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️  CatBoost not available (requires Visual Studio). Continuing without it.")

from sklearn.model_selection import cross_val_score, GridSearchCV
import warnings
warnings.filterwarnings('ignore')


class ModelDeveloper:
    """
    Main class for ML model development and training
    """
    
    def __init__(self, task_type='classification', random_state=42):
        """
        Initialize ModelDeveloper
        
        Parameters:
        -----------
        task_type : str
            Type of ML task ('classification' or 'regression')
        random_state : int
            Random state for reproducibility
        """
        self.task_type = task_type
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.best_model = None
        self.best_model_name = None
        
    def get_default_models(self):
        """
        Get dictionary of default models based on task type
        
        Returns:
        --------
        dict : Dictionary of model name: model object
        """
        if self.task_type == 'classification':
            models = {
                'Logistic Regression': LogisticRegression(
                    random_state=self.random_state, max_iter=1000
                ),
                'Decision Tree': DecisionTreeClassifier(
                    random_state=self.random_state
                ),
                'Random Forest': RandomForestClassifier(
                    random_state=self.random_state, n_estimators=100
                ),
                'Gradient Boosting': GradientBoostingClassifier(
                    random_state=self.random_state
                ),
                'XGBoost': XGBClassifier(
                    random_state=self.random_state, eval_metric='logloss'
                ),
                'LightGBM': LGBMClassifier(
                    random_state=self.random_state, verbose=-1
                ),
                'SVM': SVC(
                    random_state=self.random_state, probability=True
                ),
                'KNN': KNeighborsClassifier(),
                'Naive Bayes': GaussianNB()
            }
            # Add CatBoost if available
            if CATBOOST_AVAILABLE:
                models['CatBoost'] = CatBoostClassifier(
                    random_state=self.random_state, verbose=0
                )
        elif self.task_type == 'regression':
            models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(
                    random_state=self.random_state
                ),
                'Lasso Regression': Lasso(
                    random_state=self.random_state
                ),
                'Decision Tree': DecisionTreeRegressor(
                    random_state=self.random_state
                ),
                'Random Forest': RandomForestRegressor(
                    random_state=self.random_state, n_estimators=100
                ),
                'Gradient Boosting': GradientBoostingRegressor(
                    random_state=self.random_state
                ),
                'XGBoost': XGBRegressor(
                    random_state=self.random_state
                ),
                'LightGBM': LGBMRegressor(
                    random_state=self.random_state, verbose=-1
                ),
                'SVR': SVR(),
                'KNN': KNeighborsRegressor()
            }
            # Add CatBoost if available
            if CATBOOST_AVAILABLE:
                models['CatBoost'] = CatBoostRegressor(
                    random_state=self.random_state, verbose=0
                )
        else:
            raise ValueError("task_type must be 'classification' or 'regression'")
        
        self.models = models
        return models
    
    def train_single_model(self, model, X_train, y_train, model_name="Model"):
        """
        Train a single model
        
        Parameters:
        -----------
        model : object
            Scikit-learn compatible model
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        model_name : str
            Name of the model for tracking
            
        Returns:
        --------
        object : Trained model
        """
        print(f"   Training {model_name}...", end=" ")
        try:
            model.fit(X_train, y_train)
            print("✓")
            return model
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            return None
    
    def train_models(self, X_train, y_train, models=None):
        """
        Train multiple models
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        models : dict
            Dictionary of models (if None, uses default models)
            
        Returns:
        --------
        dict : Dictionary of trained models
        """
        if models is None:
            models = self.get_default_models()
        
        print("\n🚀 Training Models...")
        print("=" * 60)
        
        for name, model in models.items():
            trained_model = self.train_single_model(model, X_train, y_train, name)
            if trained_model is not None:
                self.trained_models[name] = trained_model
        
        print(f"\n✓ Successfully trained {len(self.trained_models)} models")
        
        return self.trained_models
    
    def cross_validate_model(self, model, X, y, cv=5, scoring=None):
        """
        Perform cross-validation on a model
        
        Parameters:
        -----------
        model : object
            Model to cross-validate
        X : array-like
            Features
        y : array-like
            Target
        cv : int
            Number of cross-validation folds
        scoring : str
            Scoring metric
            
        Returns:
        --------
        dict : Cross-validation results
        """
        if scoring is None:
            scoring = 'accuracy' if self.task_type == 'classification' else 'neg_mean_squared_error'
        
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        return {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std(),
            'scoring': scoring
        }
    
    def cross_validate_all_models(self, X, y, cv=5):
        """
        Cross-validate all trained models
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target
        cv : int
            Number of cross-validation folds
            
        Returns:
        --------
        dict : Cross-validation results for all models
        """
        print("\n🔄 Cross-Validating Models...")
        print("=" * 60)
        
        cv_results = {}
        
        for name, model in self.trained_models.items():
            print(f"   {name}...", end=" ")
            try:
                cv_result = self.cross_validate_model(model, X, y, cv=cv)
                cv_results[name] = cv_result
                print(f"Mean Score: {cv_result['mean']:.4f} (±{cv_result['std']:.4f})")
            except Exception as e:
                print(f"✗ Error: {str(e)}")
        
        return cv_results
    
    def hyperparameter_tuning(self, model, param_grid, X_train, y_train, cv=5, scoring=None):
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Parameters:
        -----------
        model : object
            Model to tune
        param_grid : dict
            Parameter grid for GridSearchCV
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        cv : int
            Number of cross-validation folds
        scoring : str
            Scoring metric
            
        Returns:
        --------
        object : Best model after tuning
        """
        if scoring is None:
            scoring = 'accuracy' if self.task_type == 'classification' else 'neg_mean_squared_error'
        
        print(f"\n🔧 Hyperparameter Tuning...")
        
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring=scoring,
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\n✓ Best Parameters: {grid_search.best_params_}")
        print(f"✓ Best Score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def get_feature_importance(self, model_name, feature_names=None, top_n=20):
        """
        Get feature importance from tree-based models
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        feature_names : list
            List of feature names
        top_n : int
            Number of top features to return
            
        Returns:
        --------
        pd.DataFrame : Feature importance DataFrame
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.trained_models[model_name]
        
        if not hasattr(model, 'feature_importances_'):
            print(f"⚠️  Model '{model_name}' does not have feature_importances_")
            return None
        
        importances = model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(top_n)
        
        return importance_df
    
    def get_model_params(self, model_name):
        """
        Get parameters of a trained model
        
        Parameters:
        -----------
        model_name : str
            Name of the model
            
        Returns:
        --------
        dict : Model parameters
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found")
        
        return self.trained_models[model_name].get_params()
    
    def predict(self, model_name, X):
        """
        Make predictions using a trained model
        
        Parameters:
        -----------
        model_name : str
            Name of the model to use
        X : array-like
            Features for prediction
            
        Returns:
        --------
        array : Predictions
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found")
        
        return self.trained_models[model_name].predict(X)
    
    def predict_proba(self, model_name, X):
        """
        Get prediction probabilities (for classification)
        
        Parameters:
        -----------
        model_name : str
            Name of the model to use
        X : array-like
            Features for prediction
            
        Returns:
        --------
        array : Prediction probabilities
        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba only available for classification tasks")
        
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.trained_models[model_name]
        
        if not hasattr(model, 'predict_proba'):
            raise ValueError(f"Model '{model_name}' does not support predict_proba")
        
        return model.predict_proba(X)


# Predefined hyperparameter grids for common models
CLASSIFICATION_PARAM_GRIDS = {
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3]
    },
    'Logistic Regression': {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    }
}

REGRESSION_PARAM_GRIDS = {
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3]
    },
    'Ridge': {
        'alpha': [0.01, 0.1, 1, 10, 100]
    },
    'Lasso': {
        'alpha': [0.01, 0.1, 1, 10, 100]
    }
}


if __name__ == "__main__":
    print("Model Development Module")
    print("=" * 60)
    print(f"\nClassification Models: {len(ModelDeveloper('classification').get_default_models())}")
    print(f"Regression Models: {len(ModelDeveloper('regression').get_default_models())}")
    print("\nImport and use in your scripts:")
    print(">>> from src.model_development import ModelDeveloper")
    print(">>> developer = ModelDeveloper(task_type='classification')")
    print(">>> models = developer.train_models(X_train, y_train)")
