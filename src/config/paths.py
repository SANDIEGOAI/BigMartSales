import os
from pathlib import Path

# Get the project root directory (assuming this file is in src/config/)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
EXTERNAL_DATA_DIR = PROJECT_ROOT / "data" / "external"

# Data file paths
TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = PROJECT_ROOT / "sample_submission.csv"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

# Results directories
RESULTS_DIR = PROJECT_ROOT / "reports"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "figures"

# Logs directory
LOGS_DIR = PROJECT_ROOT / "logs"

# Notebooks directory
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
EXPLORATORY_NOTEBOOKS_DIR = NOTEBOOKS_DIR / "exploratory"
MODELING_NOTEBOOKS_DIR = NOTEBOOKS_DIR / "modeling"
PRODUCTION_NOTEBOOKS_DIR = NOTEBOOKS_DIR / "production"

# Reports directory
REPORTS_DIR = PROJECT_ROOT / "reports"

# Create directories if they don't exist
def create_directories():
    """Create all necessary directories for the project"""
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR,
        SAVED_MODELS_DIR,
        CHECKPOINTS_DIR,
        PREDICTIONS_DIR,
        METRICS_DIR,
        PLOTS_DIR,
        LOGS_DIR,
        EXPLORATORY_NOTEBOOKS_DIR,
        MODELING_NOTEBOOKS_DIR,
        PRODUCTION_NOTEBOOKS_DIR,
        REPORTS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

# File naming conventions
def get_model_path(model_name: str, version: str = "v1") -> Path:
    """Generate model file path with naming convention"""
    return SAVED_MODELS_DIR / f"{model_name}_{version}.pkl"

def get_prediction_path(model_name: str, version: str = "v1") -> Path:
    """Generate prediction file path with naming convention"""
    return PREDICTIONS_DIR / f"predictions_{model_name}_{version}.csv"

def get_metrics_path(model_name: str, version: str = "v1") -> Path:
    """Generate metrics file path with naming convention"""
    return METRICS_DIR / f"metrics_{model_name}_{version}.json"

def get_plot_path(plot_name: str, format: str = "png") -> Path:
    """Generate plot file path with naming convention"""
    return PLOTS_DIR / f"{plot_name}.{format}"