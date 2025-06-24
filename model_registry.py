import os
import mlflow
import mlflow.sklearn
import joblib
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# Configure MLflow
tracking_uri = "file://" + str(Path.cwd() / "mlruns")
mlflow.set_tracking_uri(tracking_uri)

def log_model(
    model: Any,
    model_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    feature_importance: Dict[str, float],
    model_dir: str = "models"
) -> str:
    """
    Log a trained model to MLflow tracking server.
    
    Args:
        model: Trained model object
        model_name: Name to give to the model
        params: Dictionary of model parameters
        metrics: Dictionary of evaluation metrics
        feature_importance: Dictionary of feature importances
        model_dir: Directory to save the model files
        
    Returns:
        str: Run ID of the logged model
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # End any active runs
    if mlflow.active_run() is not None:
        mlflow.end_run()
    
    # Start MLflow run
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params(params)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log feature importance as a metric (first 10 most important features)
        top_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10])
        for feature, importance in top_features.items():
            # Clean the feature name to be MLflow compatible
            clean_feature = ''.join(c if c.isalnum() else '_' for c in str(feature))
            # Ensure the metric name is not too long and doesn't start with a number
            metric_name = f"feature_importance_{clean_feature}"[:250]  # MLflow has a 250 char limit
            if metric_name[0].isdigit():
                metric_name = f"f_{metric_name}"  # Add prefix if starts with digit
            mlflow.log_metric(metric_name, float(importance))  # Ensure importance is float
        
        # Log the model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=model_name,
            registered_model_name=model_name
        )
        
        # Save model locally as well
        model_path = os.path.join(model_dir, f"{model_name}_{run.info.run_id}.pkl")
        joblib.dump(model, model_path)
        
        # Log the model path as an artifact
        mlflow.log_artifact(model_path)
        
        print(f"Model logged with run_id: {run.info.run_id}")
        
        return run.info.run_id

def load_model(model_name: str, stage: str = "Production") -> Any:
    """
    Load a model from MLflow model registry.
    
    Args:
        model_name: Name of the model in the registry
        stage: Stage of the model (Production/Staging/None)
        
    Returns:
        Loaded model object
    """
    try:
        if stage:
            model_uri = f"models:/{model_name}/{stage}"
        else:
            model_uri = f"models:/{model_name}/latest"
            
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Loaded model {model_name} from {stage} stage")
        return model
    except Exception as e:
        print(f"Error loading model {model_name} from registry: {str(e)}")
        return None

def transition_model_stage(
    model_name: str,
    version: int,
    stage: str,
    archive_existing_versions: bool = True
) -> bool:
    """
    Transition a model version to a specific stage.
    
    Args:
        model_name: Name of the model
        version: Version number of the model
        stage: Target stage (Staging, Production, Archived, None)
        archive_existing_versions: Whether to archive existing models in the target stage
        
    Returns:
        bool: True if transition was successful, False otherwise
    """
    try:
        client = mlflow.tracking.MlflowClient()
        
        if archive_existing_versions and stage != "None":
            # Archive existing models in the target stage
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=True
            )
        else:
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            
        print(f"Transitioned model {model_name} version {version} to {stage} stage")
        return True
    except Exception as e:
        print(f"Error transitioning model {model_name} version {version} to {stage} stage: {str(e)}")
        return False

def log_experiment_metadata(
    experiment_name: str,
    tags: Optional[Dict[str, str]] = None,
    **kwargs
) -> str:
    """
    Log additional metadata to the current MLflow run.
    
    Args:
        experiment_name: Name of the experiment
        tags: Dictionary of tags to log
        **kwargs: Additional key-value pairs to log as parameters
        
    Returns:
        str: Experiment ID
    """
    # Set the experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    
    mlflow.set_experiment(experiment_name=experiment_name)
    
    # Log additional parameters
    if kwargs:
        mlflow.log_params(kwargs)
    
    # Log tags
    if tags:
        mlflow.set_tags(tags)
    
    return experiment_id
