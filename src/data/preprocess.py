import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from ..utils.utils import detect_outliers
import warnings
warnings.filterwarnings('ignore')

def handle_missing_values(df):
    """
    Handle missing values in the dataset
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with missing values handled
    """
    df_clean = df.copy()
    
    # Handle Item_Weight missing values - impute with median
    if 'Item_Weight' in df_clean.columns:
        df_clean['Item_Weight'].fillna(df_clean['Item_Weight'].median(), inplace=True)
    
    # Handle Outlet_Size missing values - impute with mode
    if 'Outlet_Size' in df_clean.columns:
        df_clean['Outlet_Size'].fillna(df_clean['Outlet_Size'].mode()[0], inplace=True)
    
    return df_clean

def handle_outliers(df):
    """
    Handle outliers in the dataset
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    df_clean = df.copy()

    # Handle Item_Outlet_Sales outliers - remove outliers
    if 'Item_Outlet_Sales' in df_clean.columns:
        upper_limit, lower_limit = detect_outliers(df_clean, 'Item_Outlet_Sales')
        df_clean = df_clean[(df_clean['Item_Outlet_Sales'] >= lower_limit) & 
                           (df_clean['Item_Outlet_Sales'] <= upper_limit)]
    
    # Handle Item_Visibility outliers - remove outliers
    if 'Item_Visibility' in df_clean.columns:
        upper_limit, lower_limit = detect_outliers(df_clean, 'Item_Visibility')
        df_clean = df_clean[(df_clean['Item_Visibility'] >= lower_limit) & 
                           (df_clean['Item_Visibility'] <= upper_limit)]

    return df_clean

def get_preprocessing_pipeline():
    """
    Create a scikit-learn preprocessing pipeline
    
    Returns:
        sklearn.pipeline.Pipeline: Preprocessing pipeline
    """
    # Define numerical and categorical features
    numerical_features = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Age']
    categorical_features = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 
                           'Outlet_Location_Type', 'Outlet_Type', 'Item_Type_Combined',
                           'Item_MRP_Bins', 'Item_Visibility_Bins', 'Item_Identifier_Categories']
    
    # Create preprocessing steps
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', LabelEncoder())
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    return preprocessor


