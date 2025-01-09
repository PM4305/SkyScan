"""
SkyScan - Flight Price Prediction Package

This package provides tools and models for predicting flight prices
based on various features like airline, route, time, and class.
"""

from .data_preprocessing import (
    load_data,
    preprocess_data,
    flight_time,
    clean_time_data,
    categorize_stops,
    clean_stops_data,
    convert_duration,
    clean_duration_data,
    clean_price_data
)

from .feature_engineering import (
    create_preprocessor,
    create_feature_pipeline,
    remove_price_outliers,
    prepare_features
)

from .model_training import (
    create_model_pipeline,
    create_param_grid,
    train_model,
    evaluate_model,
    save_model,
    load_model
)

# Package metadata
__version__ = '1.0.0'
__author__ = 'PRAKHAR MADNANI'
__email__ = 'prakhar.madnani@gmail.com'
__description__ = 'A machine learning package for predicting flight prices'

# Define what should be available on "from skyscan import *"
__all__ = [
    # Data preprocessing
    'load_data',
    'preprocess_data',
    'flight_time',
    'clean_time_data',
    'categorize_stops',
    'clean_stops_data',
    'convert_duration',
    'clean_duration_data',
    'clean_price_data',
    
    # Feature engineering
    'create_preprocessor',
    'create_feature_pipeline',
    'remove_price_outliers',
    'prepare_features',
    
    # Model training
    'create_model_pipeline',
    'create_param_grid',
    'train_model',
    'evaluate_model',
    'save_model',
    'load_model'
]

