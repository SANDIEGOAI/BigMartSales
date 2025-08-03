import os
import numpy as np
from ..utils.utils import load_object
from ..models.train_model import prepare_data_for_training
from ..config.paths import SAVED_MODELS_DIR

def load_best_model():
    """
    Load the best performing model and preprocessing artifacts
    
    Returns:
        tuple: (best_model, encoders, comparison_results)
    """
    # Load model comparison results
    comparison_results = load_object(str(SAVED_MODELS_DIR / 'model_comparison_results.pkl'))
    best_model_name = comparison_results.iloc[0]['model_name']
    
    # Load the best model
    model_filename = f"{best_model_name.lower().replace(' ', '_')}.pkl"
    best_model = load_object(str(SAVED_MODELS_DIR / model_filename))
    
    # Load preprocessing encoders
    encoders = load_object(str(SAVED_MODELS_DIR / 'preprocessing_encoders.pkl'))
    
    print(f"Loaded best model: {best_model_name}")
    print(f"Model performance - RMSE: {comparison_results.iloc[0]['val_rmse']:.2f}, R²: {comparison_results.iloc[0]['val_r2']:.4f}")
    
    return best_model, encoders, comparison_results

def prepare_data_for_prediction(data, encoders, target_column='Item_Outlet_Sales'):
    """
    Prepare data for prediction (same as training but without outlier removal)
    
    Args:
        data: Input dataframe
        encoders: Preprocessing encoders
        target_column: Name of target column (if present)
        
    Returns:
        pd.DataFrame: Processed features
    """
    from ..data.preprocess import handle_missing_values
    from ..features.build_features import clean_categorical_variables, create_features, apply_encoding_to_test_data
    
    # Handle missing values (same as training)
    data_processed = handle_missing_values(data)
    
    # Clean categorical variables (same as training)
    data_processed = clean_categorical_variables(data_processed)
    
    # Create features (same as training)
    data_processed = create_features(data_processed)
    
    # Remove target column if present
    if target_column in data_processed.columns:
        X_processed = data_processed.drop(columns=[target_column])
    else:
        X_processed = data_processed
    
    # Apply encoding using the trained encoders (same as training)
    X_encoded = apply_encoding_to_test_data(X_processed, encoders)
    
    return X_encoded

def make_predictions(data, model, encoders, target_column='Item_Outlet_Sales'):
    """
    Make predictions on new data using the trained model
    
    Args:
        data: Input dataframe
        model: Trained model
        encoders: Preprocessing encoders
        target_column: Name of target column (if present)
        
    Returns:
        np.array: Predictions
    """
    # Prepare data for prediction (without outlier removal)
    X_processed = prepare_data_for_prediction(data, encoders, target_column)
    
    # Make predictions
    predictions = model.predict(X_processed)
    
    return predictions

def predict_single_item(item_data, model, encoders):
    """
    Make prediction for a single item
    
    Args:
        item_data: Dictionary or DataFrame with single item data
        model: Trained model
        encoders: Preprocessing encoders
        
    Returns:
        float: Predicted sales value
    """
    # Convert to DataFrame if it's a dictionary
    if isinstance(item_data, dict):
        import pandas as pd
        item_data = pd.DataFrame([item_data])
    
    # Make prediction
    prediction = make_predictions(item_data, model, encoders)
    
    return prediction[0]  # Return single value

def get_prediction_confidence(model, X_data, method='std'):
    """
    Get prediction confidence/uncertainty (for models that support it)
    
    Args:
        model: Trained model
        X_data: Input features
        method: Method for confidence ('std' for ensemble models)
        
    Returns:
        np.array: Confidence scores
    """
    if hasattr(model, 'estimators_') and method == 'std':
        # For ensemble models like Random Forest
        predictions = []
        for estimator in model.estimators_:
            predictions.append(estimator.predict(X_data))
        
        predictions = np.array(predictions)
        confidence = np.std(predictions, axis=0)
        return confidence
    
    else:
        # Default confidence (1.0 for all predictions)
        return np.ones(len(X_data))
