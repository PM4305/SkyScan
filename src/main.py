#!/usr/bin/env python3

import os
import argparse
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

from data_preprocessing import load_data, preprocess_data
from feature_engineering import (create_feature_pipeline, remove_price_outliers,
                               prepare_features)
from model_training import train_model, evaluate_model, save_model, load_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_arg_parser():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description='SkyScan: Flight Price Prediction Model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument(
        '--economy-data',
        type=str,
        default='data/economy.csv',
        help='Path to economy class data CSV (default: data/economy.csv)'
    )
    train_parser.add_argument(
        '--business-data',
        type=str,
        default='data/business.csv',
        help='Path to business class data CSV (default: data/business.csv)'
    )
    train_parser.add_argument(
        '--model-output',
        type=str,
        default='models/flight_price_model.joblib',
        help='Path to save trained model (default: models/flight_price_model.joblib)'
    )
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained model file'
    )
    predict_parser.add_argument(
        '--airline',
        type=str,
        required=True,
        help='Airline name'
    )
    predict_parser.add_argument(
        '--class',
        type=str,
        choices=['Economy', 'Business'],
        required=True,
        help='Flight class'
    )
    predict_parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Source city'
    )
    predict_parser.add_argument(
        '--destination',
        type=str,
        required=True,
        help='Destination city'
    )
    predict_parser.add_argument(
        '--departure-time',
        type=str,
        choices=['Early Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late Night'],
        required=True,
        help='Departure time category'
    )
    predict_parser.add_argument(
        '--stops',
        type=str,
        choices=['zero', 'one', 'two_or_more'],
        required=True,
        help='Number of stops'
    )
    predict_parser.add_argument(
        '--arrival-time',
        type=str,
        choices=['Early Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late Night'],
        required=True,
        help='Arrival time category'
    )
    predict_parser.add_argument(
        '--duration',
        type=float,
        required=True,
        help='Flight duration in hours'
    )
    predict_parser.add_argument(
        '--days-left',
        type=int,
        required=True,
        help='Number of days until departure'
    )
    
    return parser

def train(args):
    """Train the model using provided data."""
    try:
        logger.info("Starting model training process...")
        
        # Create necessary directories
        os.makedirs(os.path.dirname(args.model_output), exist_ok=True)
        
        # Load and preprocess data
        logger.info("Loading data...")
        raw_data = load_data(args.economy_data, args.business_data)
        
        logger.info("Preprocessing data...")
        processed_data = preprocess_data(raw_data)
        
        # Remove outliers
        logger.info("Removing outliers...")
        clean_data = remove_price_outliers(processed_data)
        
        # Prepare features
        logger.info("Preparing features...")
        X, y = prepare_features(clean_data)
        
        # Create feature pipeline
        feature_pipeline = create_feature_pipeline()
        
        # Train model
        logger.info("Training model...")
        model, X_train, X_test, y_train, y_test = train_model(X, y, feature_pipeline)
        
        # Evaluate model
        logger.info("Evaluating model...")
        mse, r2 = evaluate_model(model, X_test, y_test)
        logger.info(f"Model Performance:")
        logger.info(f"Mean Squared Error: {mse:.2f}")
        logger.info(f"R-squared Score: {r2:.3f}")
        
        # Save model
        logger.info(f"Saving model to {args.model_output}...")
        save_model(model, args.model_output)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

def predict(args):
    """Make predictions using trained model."""
    try:
        logger.info("Loading model for prediction...")
        model = load_model(args.model_path)
        
        # Prepare input data
        input_data = pd.DataFrame({
            'airline': [args.airline],
            'class': [getattr(args, 'class')],
            'source_city': [args.source],
            'destination_city': [args.destination],
            'departure_time': [args.departure_time],
            'stops': [args.stops],
            'arrival_time': [args.arrival_time],
            'duration': [args.duration],
            'days_left': [args.days_left]
        })
        
        # Make prediction
        logger.info("Making prediction...")
        predicted_price = model.predict(input_data)
        
        logger.info("\nPrediction Results:")
        logger.info("-" * 50)
        logger.info(f"Predicted Price: ₹{predicted_price[0]:,.2f}")
        logger.info("-" * 50)
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

def main():
    """Main execution function."""
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    if args.command == 'train':
        train(args)
    elif args.command == 'predict':
        predict(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")