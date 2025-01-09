from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def create_model_pipeline(feature_pipeline):
    """
    Create a complete model pipeline including feature engineering and model.
    
    Args:
        feature_pipeline: Feature engineering pipeline
    
    Returns:
        Pipeline: Complete model pipeline
    """
    return Pipeline(steps=[
        ('features', feature_pipeline),
        ('regressor', LinearRegression())
    ])

def create_param_grid():
    """
    Create parameter grid for GridSearchCV.
    
    Returns:
        list: Parameter grid
    """
    return [
        {
            'features__feature_selection__estimator': [RandomForestRegressor()],
            'features__feature_selection__estimator__n_estimators': [100, 200],
            'features__feature_selection__estimator__max_depth': [3, 5, 7],
            'regressor': [LinearRegression()]
        },
        {
            'features__feature_selection__estimator': [RandomForestRegressor()],
            'features__feature_selection__estimator__n_estimators': [100],
            'features__feature_selection__estimator__max_depth': [5],
            'regressor': [DecisionTreeRegressor()],
            'regressor__max_depth': [3, 5, 7]
        },
        {
            'features__feature_selection__estimator': [RandomForestRegressor()],
            'features__feature_selection__estimator__n_estimators': [100],
            'features__feature_selection__estimator__max_depth': [5],
            'regressor': [RandomForestRegressor()],
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [3, 5, 7]
        },
        {
            'features__feature_selection__estimator': [RandomForestRegressor()],
            'features__feature_selection__estimator__n_estimators': [100],
            'features__feature_selection__estimator__max_depth': [5],
            'regressor': [KNeighborsRegressor()],
            'regressor__n_neighbors': [3, 5, 7]
        }
    ]

def train_model(X, y, feature_pipeline):
    """
    Train model using GridSearchCV.
    
    Args:
        X: Features
        y: Target variable
        feature_pipeline: Feature engineering pipeline
    
    Returns:
        tuple: (best_model, X_train, X_test, y_train, y_test)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Create and train model
    pipeline = create_model_pipeline(feature_pipeline)
    param_grid = create_param_grid()
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring='neg_mean_squared_error'
    )
    
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
    
    Returns:
        tuple: (mse, r2)
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, r2

def save_model(model, filepath):
    """
    Save trained model to file.
    
    Args:
        model: Trained model
        filepath (str): Path to save model
    """
    joblib.dump(model, filepath)

def load_model(filepath):
    """
    Load trained model from file.
    
    Args:
        filepath (str): Path to model file
    
    Returns:
        Trained model
    """
    return joblib.load(filepath)