import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def separate_features(df):
    numerical_features = df.select_dtypes(
        include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(
        include=['object']).columns.tolist()

    if 'Outlet_Establishment_Year' in numerical_features:
        numerical_features.remove('Outlet_Establishment_Year')

    if 'Item_Outlet_Sales' in numerical_features:
        numerical_features.remove('Item_Outlet_Sales')
        target = 'Item_Outlet_Sales'
    else:
        target = None

    return numerical_features, categorical_features, target


def create_features(df):
    """
    Create new features for better model performance

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Dataframe with new features
    """
    df_features = df.copy()

    # Create Item_Type_Combined (combine similar item types)
    if 'Item_Type' in df_features.columns:
        item_type_mapping = {
            'Dairy': 'Perishable',
            'Meat': 'Perishable',
            'Fruits and Vegetables': 'Perishable',
            'Frozen Foods': 'Perishable',
            'Baking Goods': 'Non-Perishable',
            'Snack Foods': 'Non-Perishable',
            'Beverages': 'Non-Perishable',
            'Hard Drinks': 'Non-Perishable',
            'Canned': 'Non-Perishable',
            'Breads': 'Perishable',
            'Starchy Foods': 'Perishable',
            'Breakfast': 'Non-Perishable',
            'Health and Hygiene': 'Non-Perishable',
            'Household': 'Non-Perishable',
            'Others': 'Non-Perishable',
            'Seafood': 'Perishable'
        }
        df_features['Item_Type_Combined'] = df_features['Item_Type'].map(
            item_type_mapping)

    # Create Outlet_Age (years since establishment)
    if 'Outlet_Establishment_Year' in df_features.columns:
        current_year = 2025  # You might want to make this dynamic
        df_features['Outlet_Age'] = current_year - \
            df_features['Outlet_Establishment_Year']

        del df_features['Outlet_Establishment_Year']

    # Create Item_MRP_Bins (price categories)
    if 'Item_MRP' in df_features.columns:
        df_features['Item_MRP_Bins'] = pd.cut(
            df_features['Item_MRP'],
            bins=[0, 50, 100, 150, 200, 300],
            labels=['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High'],
            include_lowest=True
        ).astype(str)

    # Create Item_Visibility_Bins
    if 'Item_Visibility' in df_features.columns:
        df_features['Item_Visibility_Bins'] = pd.cut(
            df_features['Item_Visibility'],
            bins=[0, 0.05, 0.1, 0.15, 0.2, 1],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
            include_lowest=True
        ).astype(str)

    if 'Outlet_Size' in df_features.columns:
        df_features['Outlet_Size'] = df_features['Outlet_Size'].map({'Small': 1,
                                                                     'Medium': 2,
                                                                     'High': 3
                                                                     }).astype(int)

    if 'Outlet_Location_Type' in df_features.columns:
        df_features['Outlet_Location_Type'] = df_features['Outlet_Location_Type'].str[-1:].astype(
            int)

    if 'Item_Identifier' in df_features.columns:
        df_features['Item_Identifier_Categories'] = df_features['Item_Identifier'].str[0:2]
        del df_features['Item_Identifier']

    return df_features


def clean_categorical_variables(df):
    """
    Clean and standardize categorical variables

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Dataframe with cleaned categorical variables
    """
    df_clean = df.copy()

    # Standardize Item_Fat_Content
    if 'Item_Fat_Content' in df_clean.columns:
        fat_content_mapping = {
            'low fat': 'Low Fat',
            'LF': 'Low Fat',
            'reg': 'Regular'
        }
        df_clean['Item_Fat_Content'] = df_clean['Item_Fat_Content'].replace(
            fat_content_mapping)

    return df_clean


def encode_categorical_variables(df, categorical_columns=None):
    """
    Encode categorical variables using Label Encoding

    Args:
        df (pd.DataFrame): Input dataframe
        categorical_columns (list): List of categorical columns to encode

    Returns:
        pd.DataFrame: Dataframe with encoded categorical variables
        dict: Dictionary of label encoders for each column
    """
    df_encoded = df.copy()
    encoders = {}

    if categorical_columns is None:
        categorical_columns = df.select_dtypes(
            include=['object']).columns.tolist()

    for col in categorical_columns:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le

    return df_encoded, encoders

def apply_encoding_to_test_data(df, encoders):
    """
    Apply encoding to test data, handling unseen categories
    
    Args:
        df (pd.DataFrame): Test dataframe
        encoders (dict): Dictionary of label encoders
        
    Returns:
        pd.DataFrame: Encoded test dataframe
    """
    df_encoded = df.copy()
    
    for col, encoder in encoders.items():
        if col in df_encoded.columns:
            # Convert to string and handle unseen categories
            col_data = df_encoded[col].astype(str)
            
            # Find unseen categories
            seen_categories = set(encoder.classes_)
            unseen_mask = ~col_data.isin(seen_categories)
            
            if unseen_mask.any():
                print(f"Warning: Found unseen categories in column {col}: {col_data[unseen_mask].unique()}")
                # Replace unseen categories with the most frequent category
                most_frequent = encoder.classes_[0]  # First category as fallback
                col_data[unseen_mask] = most_frequent
            
            # Apply encoding
            df_encoded[col] = encoder.transform(col_data)
    
    return df_encoded
