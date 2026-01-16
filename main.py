"""
Main Application Script
Orchestrates the complete ML pipeline from data loading to model evaluation
"""

import os
import sys
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_processing import DataProcessor
from src.model_development import ModelDeveloper
from src.model_evaluation import ModelEvaluator
from src.utils import (
    create_directory, save_model, save_results, 
    print_section_header, check_data_balance
)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Masters AI/ML Project - ML Pipeline'
    )
    parser.add_argument(
        '--data', type=str, required=True,
        help='Path to the dataset file (CSV or Excel)'
    )
    parser.add_argument(
        '--target', type=str, required=True,
        help='Name of the target variable column'
    )
    parser.add_argument(
        '--task', type=str, choices=['classification', 'regression'],
        default='classification',
        help='Type of ML task (default: classification)'
    )
    parser.add_argument(
        '--test-size', type=float, default=0.2,
        help='Proportion of data for testing (default: 0.2)'
    )
    parser.add_argument(
        '--random-state', type=int, default=42,
        help='Random state for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--no-scale', action='store_true',
        help='Skip feature scaling'
    )
    parser.add_argument(
        '--output-dir', type=str, default='results',
        help='Directory for output files (default: results)'
    )
    
    return parser.parse_args()


def main():
    """Main execution function"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    create_directory(output_dir)
    create_directory('models')
    
    print("\n" + "="*80)
    print(" MASTERS IN AI & ML PROJECT - ML PIPELINE")
    print("="*80)
    print(f"\n📅 Timestamp: {timestamp}")
    print(f"📁 Data: {args.data}")
    print(f"🎯 Target: {args.target}")
    print(f"🔧 Task: {args.task}")
    print(f"📊 Test Size: {args.test_size}")
    print(f"🎲 Random State: {args.random_state}")
    
    # ========================================================================
    # STEP 1: DATA LOADING AND PREPROCESSING
    # ========================================================================
    print_section_header("STEP 1: DATA LOADING AND PREPROCESSING")
    
    # Check if this is diabetes dataset and use specialized loader
    if 'diabetes.csv' in args.data.lower():
        from src.diabetes_data_loader import DiabetesDataLoader
        diabetes_loader = DiabetesDataLoader(args.data)
        df = diabetes_loader.load_data(auto_download=True)
        # The diabetes loader already handles column normalization and target conversion
    else:
        processor = DataProcessor(args.data, random_state=args.random_state)
        df = processor.load_data()
    
    # Continue with standard processing
    processor = DataProcessor(args.data, random_state=args.random_state)
    processor.df = df  # Use the already loaded data
    
    # Explore data
    data_info = processor.explore_data()
    
    # Handle missing values
    if df.isnull().sum().sum() > 0:
        df = processor.handle_missing_values(strategy='mean')
    
    # Encode categorical features
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if args.target in categorical_cols:
        categorical_cols.remove(args.target)
    
    if categorical_cols:
        df = processor.encode_categorical_features(categorical_cols)
    
    # Prepare data
    X_train, X_test, y_train, y_test = processor.prepare_data(
        target_column=args.target,
        test_size=args.test_size,
        scale=not args.no_scale
    )
    
    # Check data balance
    check_data_balance(y_train)
    
    # Save processed data
    processed_data_path = os.path.join('data', 'processed', f'processed_{timestamp}.csv')
    processor.save_processed_data(processed_data_path)
    
    # ========================================================================
    # STEP 2: MODEL DEVELOPMENT
    # ========================================================================
    print_section_header("STEP 2: MODEL DEVELOPMENT")
    
    developer = ModelDeveloper(
        task_type=args.task,
        random_state=args.random_state
    )
    
    # Get default models
    models = developer.get_default_models()
    print(f"\n📋 Models to train: {list(models.keys())}")
    
    # Train all models
    trained_models = developer.train_models(X_train, y_train)
    
    # ========================================================================
    # STEP 3: MODEL EVALUATION
    # ========================================================================
    print_section_header("STEP 3: MODEL EVALUATION")
    
    evaluator = ModelEvaluator(task_type=args.task)
    
    # Evaluate all models
    comparison_df = evaluator.evaluate_models(trained_models, X_test, y_test)
    
    # Save comparison results
    comparison_path = os.path.join(output_dir, 'model_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n✓ Model comparison saved to: {comparison_path}")
    
    # ========================================================================
    # STEP 4: VISUALIZATIONS
    # ========================================================================
    print_section_header("STEP 4: GENERATING VISUALIZATIONS")
    
    # Model comparison plot
    if args.task == 'classification':
        metric = 'Accuracy'
    else:
        metric = 'R²'
    
    fig = evaluator.plot_model_comparison(comparison_df, metric=metric)
    fig_path = os.path.join(output_dir, 'model_comparison.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {fig_path}")
    
    # Get best model
    best_model_name = comparison_df.iloc[0]['Model']
    best_model = trained_models[best_model_name]
    
    # Best model visualizations
    y_pred = best_model.predict(X_test)
    
    if args.task == 'classification':
        # Confusion Matrix
        fig = evaluator.plot_confusion_matrix(y_test, y_pred, best_model_name)
        fig_path = os.path.join(output_dir, 'confusion_matrix.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {fig_path}")
        
        # ROC Curve (if applicable)
        if hasattr(best_model, 'predict_proba'):
            y_pred_proba = best_model.predict_proba(X_test)
            try:
                fig = evaluator.plot_roc_curve(y_test, y_pred_proba, best_model_name)
                fig_path = os.path.join(output_dir, 'roc_curve.png')
                fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                print(f"✓ Saved: {fig_path}")
            except:
                print("⚠️  ROC curve not applicable for this problem")
    
    else:  # Regression
        # Predictions vs Actual
        fig = evaluator.plot_predictions_vs_actual(y_test, y_pred, best_model_name)
        fig_path = os.path.join(output_dir, 'predictions_vs_actual.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {fig_path}")
        
        # Residuals Plot
        fig = evaluator.plot_residuals(y_test, y_pred, best_model_name)
        fig_path = os.path.join(output_dir, 'residuals.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {fig_path}")
    
    # Feature Importance (if available)
    if hasattr(best_model, 'feature_importances_'):
        fig = evaluator.plot_feature_importance(
            best_model, 
            X_train.columns.tolist(),
            best_model_name,
            top_n=20
        )
        fig_path = os.path.join(output_dir, 'feature_importance.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {fig_path}")
    
    # ========================================================================
    # STEP 5: SAVE BEST MODEL
    # ========================================================================
    print_section_header("STEP 5: SAVING BEST MODEL")
    
    model_filename = f'best_model_{best_model_name.replace(" ", "_")}_{timestamp}.joblib'
    model_path = os.path.join('models', model_filename)
    save_model(best_model, model_path)
    
    # ========================================================================
    # STEP 6: GENERATE REPORT
    # ========================================================================
    print_section_header("STEP 6: GENERATING FINAL REPORT")
    
    report = evaluator.generate_evaluation_report(best_model_name)
    print(report)
    
    # Save report
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n✓ Report saved to: {report_path}")
    
    # Save comprehensive results
    results = {
        'timestamp': timestamp,
        'data_path': args.data,
        'target_column': args.target,
        'task_type': args.task,
        'test_size': args.test_size,
        'random_state': args.random_state,
        'dataset_shape': {
            'total_samples': len(df),
            'n_features': X_train.shape[1],
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        },
        'best_model': best_model_name,
        'best_model_path': model_path,
        'model_comparison': comparison_df.to_dict('records'),
        'output_directory': output_dir
    }
    
    results_path = os.path.join(output_dir, 'results.json')
    save_results(results, results_path)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print(" PIPELINE EXECUTION COMPLETE")
    print("="*80)
    print(f"\n🏆 Best Model: {best_model_name}")
    
    if args.task == 'classification':
        print(f"📊 Test Accuracy: {comparison_df.iloc[0]['Accuracy']:.4f}")
        print(f"📈 F1-Score: {comparison_df.iloc[0]['F1-Score']:.4f}")
    else:
        print(f"📊 R² Score: {comparison_df.iloc[0]['R²']:.4f}")
        print(f"📈 RMSE: {comparison_df.iloc[0]['RMSE']:.4f}")
    
    print(f"\n📁 Output Directory: {output_dir}")
    print(f"💾 Model Saved: {model_path}")
    print(f"📋 Results Saved: {results_path}")
    
    print("\n" + "="*80)
    print(" Thank you for using the ML Pipeline!")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
