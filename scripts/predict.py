import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from src.data.load_data import load_data
from src.models.train_model import prepare_data_for_training
from src.models.predict import load_best_model, make_predictions
from src.utils.utils import save_object
from src.config.paths import PROCESSED_DATA_DIR

def predict_on_test_data():
    """
    Make predictions on the test dataset
    """
    print("="*80)
    print("MAKING PREDICTIONS ON TEST DATA")
    print("="*80)
    
    # Load data
    print("\n1. Loading test data...")
    train_df, test_df = load_data()
    
    # Load best model
    print("\n2. Loading best model...")
    best_model, encoders, comparison_results = load_best_model()
    
    # Make predictions
    print("\n3. Making predictions...")
    predictions = make_predictions(test_df, best_model, encoders)
    
    # Create submission dataframe
    print("\n4. Creating submission file...")
    submission_df = pd.DataFrame({
        'Item_Identifier': test_df['Item_Identifier'],
        'Outlet_Identifier': test_df['Outlet_Identifier'],
        'Item_Outlet_Sales': predictions
    })
    
    # Save predictions
    submission_path = PROCESSED_DATA_DIR / 'predictions.csv'
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(submission_path, index=False)
    
    print(f"\nPredictions saved to: {submission_path}")
    print(f"Number of predictions: {len(predictions)}")
    print(f"Prediction range: {predictions.min():.2f} - {predictions.max():.2f}")
    
    return submission_df, predictions

def predict_on_custom_data(custom_data_path):
    """
    Make predictions on custom data
    
    Args:
        custom_data_path: Path to custom data file
    """
    print("="*80)
    print("MAKING PREDICTIONS ON CUSTOM DATA")
    print("="*80)
    
    # Load custom data
    print(f"\n1. Loading custom data from: {custom_data_path}")
    custom_data = pd.read_csv(custom_data_path)
    
    # Load best model
    print("\n2. Loading best model...")
    best_model, encoders, comparison_results = load_best_model()
    
    # Make predictions
    print("\n3. Making predictions...")
    predictions = make_predictions(custom_data, best_model, encoders)
    
    # Add predictions to dataframe
    custom_data['Predicted_Sales'] = predictions
    
    # Save results
    output_path = custom_data_path.replace('.csv', '_with_predictions.csv')
    custom_data.to_csv(output_path, index=False)
    
    print(f"\nResults saved to: {output_path}")
    print(f"Number of predictions: {len(predictions)}")
    print(f"Prediction range: {predictions.min():.2f} - {predictions.max():.2f}")
    
    return custom_data, predictions

def make_predictions_pipeline():
    """
    Main prediction pipeline
    """
    print("="*80)
    print("BIGMART SALES PREDICTION - PREDICTION PIPELINE")
    print("="*80)
    
    # Make predictions on test data
    submission_df, predictions = predict_on_test_data()
    
    print("\n" + "="*80)
    print("PREDICTION COMPLETED!")
    print("="*80)
    print("Files created:")
    print(f"1. {PROCESSED_DATA_DIR / 'predictions.csv'} - Submission file")
    print("="*80)
    
    return submission_df, predictions

if __name__ == "__main__":
    # Make predictions on test data
    submission_df, predictions = make_predictions_pipeline()
    
    # Example: Make predictions on custom data (uncomment if needed)
    # custom_data, custom_predictions = predict_on_custom_data('path/to/custom_data.csv')
