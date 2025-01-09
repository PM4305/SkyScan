from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def create_preprocessor():
    """
    Create a preprocessing pipeline for both numeric and categorical features.
    
    Returns:
        ColumnTransformer: Preprocessor pipeline
    """
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])
    
    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, ['duration', 'days_left']),
            ('cat', categorical_transformer, ['airline', 'source_city', 'destination_city', 
                                           'departure_time', 'stops', 'arrival_time', 'class'])
        ])

def create_feature_pipeline():
    """
    Create a complete feature engineering pipeline including preprocessing and feature selection.
    
    Returns:
        Pipeline: Complete feature engineering pipeline
    """
    return Pipeline(steps=[
        ('preprocessor', create_preprocessor()),
        ('feature_selection', SelectFromModel(
            RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        ))
    ])

def remove_price_outliers(df, lower_bound=0, upper_bound=200000):
    """
    Remove price outliers from the dataset.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        lower_bound (float): Lower bound for prices
        upper_bound (float): Upper bound for prices
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    return df[(df.price > lower_bound) & (df.price <= upper_bound)].copy()

def prepare_features(df):
    """
    Prepare features for modeling by separating inputs and targets.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        tuple: (X, y) where X contains features and y contains target variable
    """
    input_cols = ['airline', 'source_city', 'departure_time', 'stops', 
                  'arrival_time', 'destination_city', 'class', 'duration', 
                  'days_left']
    target_col = 'price'
    
    X = df[input_cols].copy()
    y = df[target_col].copy()
    
    return X, y