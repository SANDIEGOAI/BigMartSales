import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from ..utils.utils import save_object, load_object
from ..config.paths import SAVED_MODELS_DIR, METRICS_DIR, PLOTS_DIR
import warnings
warnings.filterwarnings('ignore')

def get_models():
    """
    Define and return a dictionary of models to train and evaluate
    
    Returns:
        dict: Dictionary of model names and their instances
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0, random_state=42),
        'Lasso Regression': Lasso(alpha=0.1, random_state=42),
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
    }
    return models

def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, model_name):
    """
    Train a single model and evaluate its performance
    
    Args:
        model: The model instance to train
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        model_name: Name of the model for reporting
        
    Returns:
        dict: Dictionary containing model performance metrics
    """
    print(f"Training {model_name}...")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    results = {
        'model_name': model_name,
        'model': model,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'train_mae': train_mae,
        'val_mae': val_mae,
        'cv_rmse': cv_rmse,
        'cv_std': cv_scores.std()
    }
    
    print(f"{model_name} - Validation RMSE: {val_rmse:.2f}, R²: {val_r2:.4f}")
    
    return results

def compare_models(X_train, y_train, X_val, y_val):
    """
    Train and compare multiple models
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        
    Returns:
        pd.DataFrame: Comparison results
        dict: Dictionary of trained models
    """
    models = get_models()
    results_list = []
    trained_models = {}
    
    for name, model in models.items():
        results = train_and_evaluate_model(model, X_train, y_train, X_val, y_val, name)
        results_list.append(results)
        trained_models[name] = results['model']
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(results_list)
    comparison_df = comparison_df.drop('model', axis=1)  # Remove model objects from dataframe
    
    # Sort by validation RMSE (best first)
    comparison_df = comparison_df.sort_values('val_rmse')
    
    return comparison_df, trained_models

def save_models_and_results(trained_models, comparison_df, encoders, save_dir=None):
    """
    Save trained models, results, and preprocessing artifacts
    
    Args:
        trained_models: Dictionary of trained models
        comparison_df: Model comparison results
        encoders: Preprocessing encoders
        save_dir: Directory to save files
    """
    if save_dir is None:
        save_dir = SAVED_MODELS_DIR
    
    # Create directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each model
    for name, model in trained_models.items():
        model_filename = f"{name.lower().replace(' ', '_')}.pkl"
        model_path = save_dir / model_filename
        save_object(model, str(model_path))
        print(f"Saved model: {model_filename}")
    
    # Save comparison results
    comparison_path = save_dir / 'model_comparison_results.pkl'
    save_object(comparison_df, str(comparison_path))
    
    # Save encoders
    encoders_path = save_dir / 'preprocessing_encoders.pkl'
    save_object(encoders, str(encoders_path))
    
    # Save results as CSV for easy viewing
    comparison_csv_path = save_dir / 'model_comparison_results.csv'
    comparison_df.to_csv(comparison_csv_path, index=False)
    
    print(f"Models and results saved to {save_dir}")

def print_model_comparison(comparison_df):
    """
    Print a formatted comparison of model results
    
    Args:
        comparison_df: Model comparison results dataframe
    """
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    
    # Display key metrics
    display_cols = ['model_name', 'val_rmse', 'val_r2', 'val_mae', 'cv_rmse']
    print(comparison_df[display_cols].to_string(index=False, float_format='%.4f'))
    
    print("\n" + "="*80)
    print("BEST MODEL")
    print("="*80)
    best_model = comparison_df.iloc[0]
    print(f"Model: {best_model['model_name']}")
    print(f"Validation RMSE: {best_model['val_rmse']:.2f}")
    print(f"Validation R²: {best_model['val_r2']:.4f}")
    print(f"Validation MAE: {best_model['val_mae']:.2f}")
    print(f"Cross-validation RMSE: {best_model['cv_rmse']:.2f}")

def feature_importance_analysis(model, feature_names, model_name):
    """
    Analyze and return feature importance for models that support it
    
    Args:
        model: Trained model
        feature_names: List of feature names
        model_name: Name of the model
        
    Returns:
        pd.DataFrame: Feature importance dataframe
    """
    importance_df = None
    
    if hasattr(model, 'feature_importances_'):
        # For tree-based models (Random Forest, XGBoost)
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
    elif hasattr(model, 'coef_'):
        # For linear models
        coefficients = np.abs(model.coef_)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients
        }).sort_values('coefficient', ascending=False)
    
    if importance_df is not None:
        print(f"\nTop 10 features for {model_name}:")
        print(importance_df.head(10).to_string(index=False))
    
    return importance_df
