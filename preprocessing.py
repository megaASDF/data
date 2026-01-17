import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os


def preprocess_data(df, scaler=None, fit_scaler=True):
    """
    Preprocess the data for training or prediction.
    
    Args:
        df: Input DataFrame
        scaler: StandardScaler object (None to create new one)
        fit_scaler: Whether to fit the scaler (True for training, False for prediction)
        
    Returns:
        Preprocessed DataFrame ready for model, fitted scaler
    """
    # Make a copy to avoid modifying original
    df_processed = df.copy()
    
    # Identify categorical and numerical columns
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    
    # Remove Customer_number if it exists (it's just an ID)
    if 'Customer_number' in df_processed.columns:
        df_processed = df_processed.drop(columns=['Customer_number'])
        if 'Customer_number' in categorical_cols:
            categorical_cols.remove('Customer_number')
    
    # Remove target column if it exists (for training data)
    if 'Avg_Trans_Amount' in df_processed.columns:
        df_processed = df_processed.drop(columns=['Avg_Trans_Amount'])
    
    # Encode categorical variables
    for col in categorical_cols:
        le = LabelEncoder()
        # Handle missing values
        df_processed[col] = df_processed[col].fillna('missing')
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    
    # Fill missing numerical values with median
    numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    for col in numerical_cols:
        df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    # Scale numerical features
    if scaler is None:
        scaler = StandardScaler()
    
    if fit_scaler:
        df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
    else:
        df_processed[numerical_cols] = scaler.transform(df_processed[numerical_cols])
    
    return df_processed, scaler
