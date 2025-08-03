import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from src.data.load_data import load_data
from src.models.train_model import prepare_data_for_training
from src.models.evaluate_model import print_model_comparison, feature_importance_analysis
from src.utils.utils import load_object, save_object
from src.config.paths import SAVED_MODELS_DIR, METRICS_DIR, PLOTS_DIR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

def load_trained_models():
    """
    Load all trained models and comparison results
    
    Returns:
        tuple: (trained_models, comparison_results, encoders)
    """
    # Load model comparison results
    comparison_results = load_object(str(SAVED_MODELS_DIR / 'model_comparison_results.pkl'))
    
    # Load all trained models
    trained_models = {}
    for _, row in comparison_results.iterrows():
        model_name = row['model_name']
        model_filename = f"{model_name.lower().replace(' ', '_')}.pkl"
        model_path = SAVED_MODELS_DIR / model_filename
        
        if model_path.exists():
            trained_models[model_name] = load_object(str(model_path))
            print(f"Loaded: {model_name}")
        else:
            print(f"Warning: Model file not found: {model_filename}")
    
    # Load preprocessing encoders
    encoders = load_object(str(SAVED_MODELS_DIR / 'preprocessing_encoders.pkl'))
    
    return trained_models, comparison_results, encoders

def evaluate_on_test_data():
    """
    Evaluate trained models on the test dataset (if target is available)
    """
    print("="*80)
    print("EVALUATING MODELS ON TEST DATA")
    print("="*80)
    
    # Load data
    print("\n1. Loading data...")
    train_df, test_df = load_data()
    
    # Check if test data has target column
    if 'Item_Outlet_Sales' not in test_df.columns:
        print("Test data doesn't have target column. Cannot evaluate on test data.")
        return None
    
    # Load trained models
    print("\n2. Loading trained models...")
    trained_models, comparison_results, encoders = load_trained_models()
    
    # Prepare test data
    print("\n3. Preparing test data...")
    X_test, y_test, _, _ = prepare_data_for_training(
        test_df, test_df=None, target_column='Item_Outlet_Sales'
    )
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Test target shape: {y_test.shape}")
    
    # Evaluate each model
    print("\n4. Evaluating models on test data...")
    test_results = []
    
    for model_name, model in trained_models.items():
        print(f"\nEvaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_r2 = r2_score(y_test, y_pred)
        test_mae = mean_absolute_error(y_test, y_pred)
        
        test_results.append({
            'model_name': model_name,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'test_mae': test_mae
        })
        
        print(f"  Test RMSE: {test_rmse:.2f}")
        print(f"  Test R²: {test_r2:.4f}")
        print(f"  Test MAE: {test_mae:.2f}")
    
    # Create comparison dataframe
    test_results_df = pd.DataFrame(test_results)
    test_results_df = test_results_df.sort_values('test_rmse')
    
    # Save test results
    test_results_path = METRICS_DIR / 'test_evaluation_results.pkl'
    test_csv_path = METRICS_DIR / 'test_evaluation_results.csv'
    save_object(test_results_df, str(test_results_path))
    test_results_df.to_csv(test_csv_path, index=False)
    
    print(f"\nTest evaluation results saved to: {test_csv_path}")
    
    return test_results_df

def evaluate_on_validation_split():
    """
    Evaluate models on a validation split of the training data
    """
    print("="*80)
    print("EVALUATING MODELS ON VALIDATION SPLIT")
    print("="*80)
    
    # Load data
    print("\n1. Loading data...")
    train_df, test_df = load_data()
    
    # Load trained models
    print("\n2. Loading trained models...")
    trained_models, comparison_results, encoders = load_trained_models()
    
    # Prepare training data
    print("\n3. Preparing training data...")
    X_train, y_train, _, _ = prepare_data_for_training(
        train_df, test_df=None, target_column='Item_Outlet_Sales'
    )
    
    # Create validation split
    print("\n4. Creating validation split...")
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"Training split: {X_train_split.shape}")
    print(f"Validation split: {X_val.shape}")
    
    # Evaluate each model
    print("\n5. Evaluating models on validation data...")
    val_results = []
    
    for model_name, model in trained_models.items():
        print(f"\nEvaluating {model_name}...")
        
        # Retrain model on training split
        model.fit(X_train_split, y_train_split)
        
        # Make predictions
        y_pred = model.predict(X_val)
        
        # Calculate metrics
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        val_r2 = r2_score(y_val, y_pred)
        val_mae = mean_absolute_error(y_val, y_pred)
        
        val_results.append({
            'model_name': model_name,
            'val_rmse': val_rmse,
            'val_r2': val_r2,
            'val_mae': val_mae
        })
        
        print(f"  Validation RMSE: {val_rmse:.2f}")
        print(f"  Validation R²: {val_r2:.4f}")
        print(f"  Validation MAE: {val_mae:.2f}")
    
    # Create comparison dataframe
    val_results_df = pd.DataFrame(val_results)
    val_results_df = val_results_df.sort_values('val_rmse')
    
    # Save validation results
    val_results_path = METRICS_DIR / 'validation_evaluation_results.pkl'
    val_csv_path = METRICS_DIR / 'validation_evaluation_results.csv'
    save_object(val_results_df, str(val_results_path))
    val_results_df.to_csv(val_csv_path, index=False)
    
    print(f"\nValidation evaluation results saved to: {val_csv_path}")
    
    return val_results_df

def create_evaluation_plots(test_results_df=None, val_results_df=None):
    """
    Create visualization plots for model evaluation results
    
    Args:
        test_results_df: Test evaluation results dataframe
        val_results_df: Validation evaluation results dataframe
    """
    print("\n6. Creating evaluation plots...")
    
    # Determine how many plots we need
    has_test = test_results_df is not None
    has_val = val_results_df is not None
    
    if has_test and has_val:
        # Both test and validation results available
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: RMSE Comparison
        axes[0, 0].bar(test_results_df['model_name'], test_results_df['test_rmse'])
        axes[0, 0].set_title('Test RMSE Comparison')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        axes[0, 1].bar(val_results_df['model_name'], val_results_df['val_rmse'])
        axes[0, 1].set_title('Validation RMSE Comparison')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 2: R² Comparison
        axes[1, 0].bar(test_results_df['model_name'], test_results_df['test_r2'])
        axes[1, 0].set_title('Test R² Comparison')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        axes[1, 1].bar(val_results_df['model_name'], val_results_df['val_r2'])
        axes[1, 1].set_title('Validation R² Comparison')
        axes[1, 1].set_ylabel('R²')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
    elif has_val:
        # Only validation results available
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # RMSE Comparison
        axes[0].bar(val_results_df['model_name'], val_results_df['val_rmse'])
        axes[0].set_title('Validation RMSE Comparison')
        axes[0].set_ylabel('RMSE')
        axes[0].tick_params(axis='x', rotation=45)
        
        # R² Comparison
        axes[1].bar(val_results_df['model_name'], val_results_df['val_r2'])
        axes[1].set_title('Validation R² Comparison')
        axes[1].set_ylabel('R²')
        axes[1].tick_params(axis='x', rotation=45)
        
    elif has_test:
        # Only test results available
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # RMSE Comparison
        axes[0].bar(test_results_df['model_name'], test_results_df['test_rmse'])
        axes[0].set_title('Test RMSE Comparison')
        axes[0].set_ylabel('RMSE')
        axes[0].tick_params(axis='x', rotation=45)
        
        # R² Comparison
        axes[1].bar(test_results_df['model_name'], test_results_df['test_r2'])
        axes[1].set_title('Test R² Comparison')
        axes[1].set_ylabel('R²')
        axes[1].tick_params(axis='x', rotation=45)
    
    else:
        print("No evaluation results available for plotting")
        return
    
    plt.tight_layout()
    plot_path = PLOTS_DIR / 'model_evaluation_comparison.png'
    plt.savefig(str(plot_path), bbox_inches='tight', dpi=300)
    plt.show()
    
    print(f"Evaluation plots saved to: {plot_path}")

def print_evaluation_summary(test_results_df=None, val_results_df=None):
    """
    Print a summary of evaluation results
    
    Args:
        test_results_df: Test evaluation results dataframe
        val_results_df: Validation evaluation results dataframe
    """
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    if test_results_df is not None:
        print("\nTEST DATA EVALUATION:")
        print(test_results_df[['model_name', 'test_rmse', 'test_r2', 'test_mae']].to_string(index=False, float_format='%.4f'))
        
        best_test = test_results_df.iloc[0]
        print(f"\nBest model on test data: {best_test['model_name']}")
        print(f"Test RMSE: {best_test['test_rmse']:.2f}")
        print(f"Test R²: {best_test['test_r2']:.4f}")
    
    if val_results_df is not None:
        print("\nVALIDATION DATA EVALUATION:")
        print(val_results_df[['model_name', 'val_rmse', 'val_r2', 'val_mae']].to_string(index=False, float_format='%.4f'))
        
        best_val = val_results_df.iloc[0]
        print(f"\nBest model on validation data: {best_val['model_name']}")
        print(f"Validation RMSE: {best_val['val_rmse']:.2f}")
        print(f"Validation R²: {best_val['val_r2']:.4f}")

def evaluate_models():
    """
    Main evaluation function
    """
    print("="*80)
    print("BIGMART SALES PREDICTION - MODEL EVALUATION")
    print("="*80)
    
    # Evaluate on validation split (since test data might not have target)
    print("Starting model evaluation...")
    
    val_results_df = evaluate_on_validation_split()
    
    # Try to evaluate on test data if target is available
    test_results_df = evaluate_on_test_data()
    
    # Create evaluation plots
    create_evaluation_plots(test_results_df, val_results_df)
    
    # Print summary
    print_evaluation_summary(test_results_df, val_results_df)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETED!")
    print("="*80)
    print("Next steps:")
    print("1. Run 'python scripts/predict.py' to make predictions")
    print("="*80)

if __name__ == "__main__":
    evaluate_models()
