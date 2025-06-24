import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any
import mlflow
from model_registry import log_model, log_experiment_metadata, transition_model_stage

# Set random seed for reproducibility
np.random.seed(42)

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('reports/figures', exist_ok=True)

def load_and_validate_data(filepath: str = 'training_data.csv') -> pd.DataFrame:
    """Load and validate the training data."""
    print("Loading and validating data...")
    
    # Load the data
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
        
    df = pd.read_csv(filepath)
    
    # Basic validation
    if len(df) == 0:
        raise ValueError("Empty dataset. Please check your data source.")
    
    print(f"Initial data points: {len(df)}")
    
    # Check for required columns
    required_columns = {'timestamp', 'aqi'}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Convert timestamp to datetime and set as index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Handle duplicate timestamps by taking the mean of duplicates
    duplicate_count = df['timestamp'].duplicated().sum()
    
    
    df = df.sort_values('timestamp')
    print(f"Final data points after processing: {len(df)}")
    
    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features."""
    if df.empty:
        raise ValueError("Cannot create features: Empty DataFrame")
        
    df = df.copy()
    
    # Ensure we have a datetime index for time-based operations
    df = df.set_index('timestamp').sort_index()
    
    # Time-based features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['is_weekend'] = df.index.dayofweek // 5 == 1
    
    # Only create lag features if we have enough data
    if len(df) > 24:  # At least 24 hours of data
        # Lag features for AQI
        for lag in [1, 2, 3, 6, 12, 24]:  # 1h, 2h, 3h, 6h, 12h, 24h
            df[f'aqi_lag_{lag}'] = df['aqi'].shift(lag)
        
        # Rolling statistics for AQI
        for window in [3, 6, 12, 24]:
            df[f'aqi_rolling_mean_{window}'] = df['aqi'].rolling(window=window, min_periods=1).mean()
            df[f'aqi_rolling_std_{window}'] = df['aqi'].rolling(window=window, min_periods=1).std()
    
    
    
    if df.empty:
        raise ValueError("No data left after creating features")
        
    return df.reset_index()

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list]:
    """Preprocess data and split into features and target."""
    if df.empty:
        raise ValueError("Empty DataFrame in preprocess_data")
        
    # Drop columns that are not useful for modeling
    columns_to_drop = ['station', 'timezone', 'unix_timestamp', 'dominant_pollutant', 'timestamp']
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    
    # Keep only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df[numeric_cols]
    
    # Separate features and target
    if 'aqi' not in df.columns:
        raise ValueError("Target column 'aqi' not found in the data")
        
    X = df.drop(columns=['aqi'], errors='ignore')
    y = df['aqi']
    
    # Get feature names for later use
    feature_names = X.columns.tolist()
    
    if not feature_names:
        raise ValueError("No features available for modeling")
        
    return X, y, feature_names

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[RandomForestRegressor, Dict[str, Any], Dict[str, float]]:
    """Train a Random Forest model with hyperparameter tuning."""
    print("Training Random Forest model...")
    
    # Set up cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    # Initialize the model
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1,
        return_train_score=True
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {-grid_search.best_score_:.4f}")
    
    # Get feature importances
    feature_importances = dict(zip(X_train.columns, grid_search.best_estimator_.feature_importances_))
    
    return grid_search.best_estimator_, grid_search.cv_results_, feature_importances

def evaluate_model(model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluate the model and return metrics."""
    print("Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'mape': mean_absolute_percentage_error(y_test, y_pred) * 100,  # as percentage
        'r2': r2_score(y_test, y_pred)
    }
    
    # Print metrics
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    return metrics

def plot_feature_importance(model: RandomForestRegressor, feature_names: list, top_n: int = 20):
    """Plot feature importance."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig('reports/figures/feature_importance.png')
    plt.close()

def plot_actual_vs_predicted(y_true: pd.Series, y_pred: np.ndarray):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted Values')
    plt.tight_layout()
    plt.savefig('reports/figures/actual_vs_predicted.png')
    plt.close()

def save_model(model: RandomForestRegressor, filename: str = 'models/air_quality_model.pkl'):
    """Save the trained model and metadata."""
    # Create a dictionary with model and metadata
    model_data = {
        'model': model,
        'feature_names': model.feature_names_in_.tolist() if hasattr(model, 'feature_names_in_') else None,
        'model_type': 'RandomForestRegressor',
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    joblib.dump(model_data, filename)
    print(f"Model saved to {filename}")

def main():
    try:
        # Set experiment name
        experiment_name = "air_quality_prediction"
        
        # Log experiment metadata
        log_experiment_metadata(
            experiment_name=experiment_name,
            tags={
                "project": "air_quality_mlops",
                "team": "data_science",
                "model_type": "random_forest"
            },
            data_source="training_data.csv",
            model_version="1.0.0"
        )
        
        # Load and validate data
        print("\n=== Loading Data ===")
        df = load_and_validate_data('training_data.csv')
        
        # Create features
        print("\n=== Creating Features ===")
        df = create_features(df)
        
        # Preprocess data
        print("\n=== Preprocessing Data ===")
        X, y, feature_names = preprocess_data(df)
        
        # Split data
        train_size = int(0.8 * len(X))
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        # Train model
        print("\n=== Training Model ===")
        model, cv_results, feature_importances = train_random_forest(X_train, y_train)
        
        # Evaluate model
        print("\n=== Model Evaluation ===")
        metrics = evaluate_model(model, X_test, y_test)
        
        # Log model to registry
        print("\n=== Logging Model to Registry ===")
        run_id = log_model(
            model=model,
            model_name="air_quality_predictor",
            params=model.get_params(),
            metrics=metrics,
            feature_importance=feature_importances
        )
        
        # Optionally transition model to production
        if metrics.get('r2', 0) > 0.7:  # Only promote if R2 > 0.7
            print("\n=== Promoting Model to Production ===")
            try:
                # Get the latest version of the model
                client = mlflow.tracking.MlflowClient()
                model_versions = client.search_model_versions(f"name='air_quality_predictor'")
                latest_version = max([int(mv.version) for mv in model_versions]) if model_versions else 1
                
                transition_model_stage(
                    model_name="air_quality_predictor",
                    version=latest_version,  # Use the latest version
                    stage="Production"
                )
            except Exception as e:
                print(f"Warning: Could not promote model to production: {str(e)}")
                print("The model was still trained and logged, but not promoted to production.")
        
        # Generate and save plots
        print("\n=== Generating Plots ===")
        plot_feature_importance(model, feature_names)
        plot_actual_vs_predicted(y_test, model.predict(X_test))
        
        print("\n=== Training Completed Successfully! ===")
        print("\nModel evaluation metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
            
    except Exception as e:
        print(f"\n=== ERROR ===")
        print(f"Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    main()