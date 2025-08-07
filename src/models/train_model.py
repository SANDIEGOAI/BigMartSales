from ..data.preprocess import handle_missing_values, handle_outliers
from ..features.build_features import clean_categorical_variables, create_features, encode_categorical_variables, apply_encoding_to_test_data


def prepare_data_for_training(train_df, test_df=None, target_column='Item_Outlet_Sales'):
    """
    Complete data preprocessing pipeline for training

    Args:
        train_df (pd.DataFrame): Training dataframe
        test_df (pd.DataFrame): Test dataframe (optional)
        target_column (str): Name of target column

    Returns:
        tuple: Processed training and test dataframes, and preprocessing info
    """
    # Handle missing values
    train_processed = handle_missing_values(train_df)
    if test_df is not None:
        test_processed = handle_missing_values(test_df)

    # Handle outliers (only for training data)
    train_processed = handle_outliers(train_processed)

    # Clean categorical variables
    train_processed = clean_categorical_variables(train_processed)
    if test_df is not None:
        test_processed = clean_categorical_variables(test_processed)

    # Create features
    train_processed = create_features(train_processed)
    if test_df is not None:
        test_processed = create_features(test_processed)

    # Separate features and target
    if target_column in train_processed.columns:
        X_train = train_processed.drop(columns=[target_column])
        y_train = train_processed[target_column]
    else:
        X_train = train_processed
        y_train = None

    if test_df is not None:
        X_test = test_processed
    else:
        X_test = None

    # Encode categorical variables
    categorical_columns = X_train.select_dtypes(
        include=['object']).columns.tolist()
    X_train_encoded, encoders = encode_categorical_variables(
        X_train, categorical_columns)

    if X_test is not None:
        # Apply same encoding to test data with handling for unseen categories
        X_test = apply_encoding_to_test_data(X_test, encoders)

    # Check for NaN values after encoding
    if X_train_encoded.isnull().any().any():
        print("Warning: NaN values found in training data after encoding")
        print("Columns with NaN values:")
        print(X_train_encoded.isnull().sum()[X_train_encoded.isnull().sum() > 0])
        
        # Fill NaN values with 0 for categorical columns
        for col in categorical_columns:
            if col in X_train_encoded.columns and X_train_encoded[col].isnull().any():
                X_train_encoded[col].fillna(0, inplace=True)
                if X_test is not None and col in X_test.columns:
                    X_test[col].fillna(0, inplace=True)

    return X_train_encoded, y_train, X_test, encoders
