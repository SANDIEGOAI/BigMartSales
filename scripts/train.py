import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.load_data import load_data
from src.models.train_model import prepare_data_for_training
from src.models.evaluate_model import compare_models, save_models_and_results, print_model_comparison, feature_importance_analysis
from src.utils.utils import save_object
from sklearn.model_selection import train_test_split
import numpy as np

def train_models():
    """
    Main function to train multiple models
    """
    print("="*80)
    print("BIGMART SALES PREDICTION - MODEL TRAINING")
    print("="*80)
    
    print("\n1. Loading data...")
    train_df, test_df = load_data()
    
    print("\n2. Preparing data for training...")
    # Use the prepare_data_for_training function
    X_train, y_train, X_test, encoders = prepare_data_for_training(
        train_df, test_df, target_column='Item_Outlet_Sales'
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Target shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    print("\n3. Splitting data for validation...")
    # Split training data for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"Training split: {X_train_split.shape}")
    print(f"Validation split: {X_val.shape}")
    
    print("\n4. Training and comparing models...")
    # Train and compare multiple models
    comparison_df, trained_models = compare_models(
        X_train_split, y_train_split, X_val, y_val
    )
    
    print("\n5. Model comparison results...")
    # Print detailed comparison
    print_model_comparison(comparison_df)
    
    print("\n6. Feature importance analysis...")
    # Analyze feature importance for the best model
    best_model_name = comparison_df.iloc[0]['model_name']
    best_model = trained_models[best_model_name]
    feature_names = X_train.columns.tolist()
    
    feature_importance_analysis(best_model, feature_names, best_model_name)
    
    print("\n7. Saving models and results...")
    # Save all models and results
    save_models_and_results(trained_models, comparison_df, encoders)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("Next steps:")
    print("1. Run 'python scripts/evaluate.py' to evaluate models")
    print("2. Run 'python scripts/predict.py' to make predictions")
    print("="*80)
    
    return trained_models, comparison_df, encoders

if __name__ == "__main__":
    trained_models, comparison_df, encoders = train_models()
