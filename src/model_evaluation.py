"""
Model Evaluation Module
Comprehensive evaluation metrics, visualizations, and model comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Main class for model evaluation and visualization
    """
    
    def __init__(self, task_type='classification'):
        """
        Initialize ModelEvaluator
        
        Parameters:
        -----------
        task_type : str
            Type of ML task ('classification' or 'regression')
        """
        self.task_type = task_type
        self.results = {}
        
    def evaluate_classification(self, y_true, y_pred, y_pred_proba=None, model_name="Model"):
        """
        Evaluate classification model
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        y_pred_proba : array-like
            Prediction probabilities (optional)
        model_name : str
            Name of the model
            
        Returns:
        --------
        dict : Evaluation metrics
        """
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        # Add ROC-AUC if probabilities are provided
        if y_pred_proba is not None:
            try:
                # For binary classification
                if y_pred_proba.shape[1] == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                # For multi-class classification
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
            except Exception as e:
                metrics['roc_auc'] = None
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Classification report
        metrics['classification_report'] = classification_report(y_true, y_pred, zero_division=0)
        
        return metrics
    
    def evaluate_regression(self, y_true, y_pred, model_name="Model"):
        """
        Evaluate regression model
        
        Parameters:
        -----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
        model_name : str
            Name of the model
            
        Returns:
        --------
        dict : Evaluation metrics
        """
        metrics = {
            'model_name': model_name,
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred),
        }
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            metrics['mape'] = mape
        else:
            metrics['mape'] = None
        
        return metrics
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """
        Evaluate a single model
        
        Parameters:
        -----------
        model : object
            Trained model
        X_test : array-like
            Test features
        y_test : array-like
            Test target
        model_name : str
            Name of the model
            
        Returns:
        --------
        dict : Evaluation metrics
        """
        y_pred = model.predict(X_test)
        
        if self.task_type == 'classification':
            # Get probabilities if available
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
            
            metrics = self.evaluate_classification(y_test, y_pred, y_pred_proba, model_name)
        else:
            metrics = self.evaluate_regression(y_test, y_pred, model_name)
        
        self.results[model_name] = metrics
        return metrics
    
    def evaluate_models(self, models, X_test, y_test):
        """
        Evaluate multiple models
        
        Parameters:
        -----------
        models : dict
            Dictionary of trained models
        X_test : array-like
            Test features
        y_test : array-like
            Test target
            
        Returns:
        --------
        pd.DataFrame : Comparison of model results
        """
        print("\n📊 Evaluating Models...")
        print("=" * 60)
        
        results_list = []
        
        for name, model in models.items():
            print(f"   Evaluating {name}...", end=" ")
            try:
                metrics = self.evaluate_model(model, X_test, y_test, name)
                results_list.append(metrics)
                print("✓")
            except Exception as e:
                print(f"✗ Error: {str(e)}")
        
        # Create comparison DataFrame
        if self.task_type == 'classification':
            comparison_df = pd.DataFrame([
                {
                    'Model': r['model_name'],
                    'Accuracy': r['accuracy'],
                    'Precision': r['precision'],
                    'Recall': r['recall'],
                    'F1-Score': r['f1_score'],
                    'ROC-AUC': r.get('roc_auc', None)
                }
                for r in results_list
            ]).sort_values('Accuracy', ascending=False)
        else:
            comparison_df = pd.DataFrame([
                {
                    'Model': r['model_name'],
                    'MSE': r['mse'],
                    'RMSE': r['rmse'],
                    'MAE': r['mae'],
                    'R²': r['r2_score'],
                    'MAPE': r.get('mape', None)
                }
                for r in results_list
            ]).sort_values('R²', ascending=False)
        
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name="Model", figsize=(8, 6)):
        """
        Plot confusion matrix
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        model_name : str
            Name of the model
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure : Confusion matrix plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Confusion Matrix - {model_name}')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curve(self, y_true, y_pred_proba, model_name="Model", figsize=(8, 6)):
        """
        Plot ROC curve for binary classification
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred_proba : array-like
            Prediction probabilities
        model_name : str
            Name of the model
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure : ROC curve plot
        """
        # For binary classification, use probabilities of positive class
        if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] == 2:
            y_scores = y_pred_proba[:, 1]
        else:
            y_scores = y_pred_proba
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {model_name}')
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, model_name="Model", figsize=(8, 6)):
        """
        Plot Precision-Recall curve
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred_proba : array-like
            Prediction probabilities
        model_name : str
            Name of the model
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure : PR curve plot
        """
        # For binary classification
        if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] == 2:
            y_scores = y_pred_proba[:, 1]
        else:
            y_scores = y_pred_proba
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {avg_precision:.2f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve - {model_name}')
        ax.legend(loc="lower left")
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, model, feature_names, model_name="Model", 
                               top_n=20, figsize=(10, 8)):
        """
        Plot feature importance
        
        Parameters:
        -----------
        model : object
            Trained model with feature_importances_
        feature_names : list
            List of feature names
        model_name : str
            Name of the model
        top_n : int
            Number of top features to display
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure : Feature importance plot
        """
        if not hasattr(model, 'feature_importances_'):
            print(f"⚠️  Model does not have feature_importances_")
            return None
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Use actual number of features (min of top_n and available features)
        n_features = min(top_n, len(indices))
        indices = indices[:n_features]
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(range(n_features), importances[indices])
        ax.set_yticks(range(n_features))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {n_features} Feature Importances - {model_name}')
        ax.invert_yaxis()
        
        plt.tight_layout()
        return fig
    
    def plot_predictions_vs_actual(self, y_true, y_pred, model_name="Model", figsize=(8, 8)):
        """
        Plot predicted vs actual values for regression
        
        Parameters:
        -----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
        model_name : str
            Name of the model
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure : Predictions vs actual plot
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(y_true, y_pred, alpha=0.5)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'Predicted vs Actual - {model_name}')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_residuals(self, y_true, y_pred, model_name="Model", figsize=(10, 4)):
        """
        Plot residuals for regression
        
        Parameters:
        -----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
        model_name : str
            Name of the model
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure : Residuals plot
        """
        residuals = y_true - y_pred
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Residuals scatter plot
        ax1.scatter(y_pred, residuals, alpha=0.5)
        ax1.axhline(y=0, color='r', linestyle='--', lw=2)
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title(f'Residuals Plot - {model_name}')
        ax1.grid(alpha=0.3)
        
        # Residuals histogram
        ax2.hist(residuals, bins=30, edgecolor='black')
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Residuals Distribution')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_model_comparison(self, comparison_df, metric='Accuracy', figsize=(10, 6)):
        """
        Plot model comparison
        
        Parameters:
        -----------
        comparison_df : pd.DataFrame
            DataFrame with model comparison results
        metric : str
            Metric to plot
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure : Model comparison plot
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        models = comparison_df['Model']
        values = comparison_df[metric]
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
        bars = ax.barh(models, values, color=colors)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax.text(value, i, f' {value:.4f}', va='center')
        
        ax.set_xlabel(metric)
        ax.set_title(f'Model Comparison - {metric}')
        ax.invert_yaxis()
        
        plt.tight_layout()
        return fig
    
    def generate_evaluation_report(self, model_name):
        """
        Generate a comprehensive evaluation report for a model
        
        Parameters:
        -----------
        model_name : str
            Name of the model
            
        Returns:
        --------
        str : Formatted evaluation report
        """
        if model_name not in self.results:
            raise ValueError(f"No results found for model '{model_name}'")
        
        metrics = self.results[model_name]
        
        report = f"\n{'='*60}\n"
        report += f"EVALUATION REPORT: {model_name}\n"
        report += f"{'='*60}\n\n"
        
        if self.task_type == 'classification':
            report += f"Accuracy:  {metrics['accuracy']:.4f}\n"
            report += f"Precision: {metrics['precision']:.4f}\n"
            report += f"Recall:    {metrics['recall']:.4f}\n"
            report += f"F1-Score:  {metrics['f1_score']:.4f}\n"
            if metrics.get('roc_auc'):
                report += f"ROC-AUC:   {metrics['roc_auc']:.4f}\n"
            report += f"\n{'-'*60}\n"
            report += "CLASSIFICATION REPORT:\n"
            report += f"{'-'*60}\n"
            report += metrics['classification_report']
        else:
            report += f"Mean Squared Error (MSE):  {metrics['mse']:.4f}\n"
            report += f"Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}\n"
            report += f"Mean Absolute Error (MAE): {metrics['mae']:.4f}\n"
            report += f"R² Score: {metrics['r2_score']:.4f}\n"
            if metrics.get('mape'):
                report += f"MAPE: {metrics['mape']:.2f}%\n"
        
        report += f"\n{'='*60}\n"
        
        return report


if __name__ == "__main__":
    print("Model Evaluation Module")
    print("=" * 60)
    print("\nAvailable evaluation capabilities:")
    print("- Classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC)")
    print("- Regression metrics (MSE, RMSE, MAE, R², MAPE)")
    print("- Confusion matrix visualization")
    print("- ROC and Precision-Recall curves")
    print("- Feature importance plots")
    print("- Model comparison charts")
    print("\nImport and use in your scripts:")
    print(">>> from src.model_evaluation import ModelEvaluator")
    print(">>> evaluator = ModelEvaluator(task_type='classification')")
    print(">>> results = evaluator.evaluate_models(models, X_test, y_test)")
