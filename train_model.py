import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('reports/figures', exist_ok=True)

def load_and_preprocess_data(filepath='training_data.csv'):
    """Load and preprocess the training data."""
    print("Loading and preprocessing data...")
    
    # Load the data
    df = pd.read_csv(filepath)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Drop columns that are not useful for modeling
    columns_to_drop = ['station', 'timestamp', 'timezone', 'unix_timestamp']
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=columns_to_drop)
    
    # Handle categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].nunique() > 10:  # If too many categories, drop
            df = df.drop(columns=[col])
        else:
            # For columns with few categories, use one-hot encoding
            df = pd.get_dummies(df, columns=[col], drop_first=True)
    
    # Fill remaining missing values with column means
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())
    print(df.head())
    return df

def prepare_features_target(df, target_col='aqi'):
    """Prepare features and target variable."""
    # Ensure target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the data. Available columns: {df.columns.tolist()}")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return X, y

def train_random_forest(X_train, y_train):
    """Train a Random Forest model with hyperparameter tuning."""
    print("Training Random Forest model...")
    
    # Base model
    rf = RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    )
    
    # Hyperparameter grid for tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    }
    
    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics."""
    print("Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Plot feature importance
    plot_feature_importance(model, X_test.columns)
    
    # Plot actual vs predicted
    plot_actual_vs_predicted(y_test, y_pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def plot_feature_importance(model, feature_names):
    """Plot feature importance."""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importance")
    plt.bar(range(min(20, len(feature_names))), 
            importance[indices][:20],
            align="center")
    plt.xticks(range(min(20, len(feature_names))), 
              [feature_names[i] for i in indices][:20], 
              rotation=90)
    plt.tight_layout()
    plt.savefig('reports/figures/feature_importance.png')
    plt.close()

def plot_actual_vs_predicted(y_true, y_pred):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], 
             [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted AQI Values')
    plt.tight_layout()
    plt.savefig('reports/figures/actual_vs_predicted.png')
    plt.close()

def save_model(model, filename='models/air_quality_model.pkl'):
    """Save the trained model."""
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def main():
    # Load and preprocess data
    df = load_and_preprocess_data('training_data.csv')
    
    # Prepare features and target
    X, y = prepare_features_target(df, target_col='aqi')
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # Train the model
    model = train_random_forest(X_train, y_train)
    
    # Evaluate the model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save the model
    save_model(model)
    
    print("\nTraining completed successfully!")
    print("Model evaluation metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()