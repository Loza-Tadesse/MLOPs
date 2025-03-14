import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import json
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlProject import logger
from mlProject.entity.config_entity import DeepModelEvaluationConfig
from mlProject.utils.common import save_json
from pathlib import Path
import mlflow
from urllib.parse import urlparse


class CryptoPriceNet(nn.Module):
    """Neural Network for Cryptocurrency Price Prediction"""
    
    def __init__(self, input_size, hidden_layers, dropout_rate=0.2):
        super(CryptoPriceNet, self).__init__()
        
        layers_list = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_layers:
            layers_list.append(nn.Linear(prev_size, hidden_size))
            layers_list.append(nn.ReLU())
            layers_list.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers_list.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers_list)
    
    def forward(self, x):
        return self.network(x)


class DeepModelEvaluation:
    def __init__(self, config: DeepModelEvaluationConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def log_into_mlflow(self):
        try:
            # Load test data
            test_data = pd.read_csv(self.config.test_data_path)
            
            # Load model configuration
            with open(self.config.model_config_path, 'r') as f:
                model_config = json.load(f)
            
            # Load scaler
            scaler = joblib.load(self.config.scaler_path)
            
            # Initialize model architecture
            model = CryptoPriceNet(
                input_size=model_config['input_size'],
                hidden_layers=model_config['hidden_layers'],
                dropout_rate=model_config['dropout_rate']
            ).to(self.device)
            
            # Load trained weights
            model.load_state_dict(torch.load(self.config.model_path, map_location=self.device))
            model.eval()
            
            # Prepare test data
            test_x = test_data.drop([self.config.target_column], axis=1)
            test_y = test_data[self.config.target_column]
            
            # Scale features
            test_x_scaled = scaler.transform(test_x)
            test_x_tensor = torch.FloatTensor(test_x_scaled).to(self.device)
            
            # Make predictions
            with torch.no_grad():
                predicted_qualities = model(test_x_tensor).cpu().numpy().flatten()
            
            # Calculate metrics
            (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)
            
            # Save metrics locally
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)
            
            # MLflow logging with unique experiment management
            mlflow.set_registry_uri(self.config.mlflow_uri)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            # Set or create unique experiment for deep learning models
            experiment_name = "Deep_Learning_CryptoPredict"
            try:
                mlflow.create_experiment(experiment_name)
            except mlflow.exceptions.MlflowException:
                # Experiment already exists, which is fine
                pass
            
            mlflow.set_experiment(experiment_name)
            
            # Create unique run name with timestamp
            import time
            run_name = f"deep_model_eval_{int(time.time())}"
            
            with mlflow.start_run(run_name=run_name):
                # Log parameters safely
                params_to_log = self.config.all_params.copy() if hasattr(self.config, 'all_params') else {}
                params_to_log.update({
                    "model_type": "PyTorch_Neural_Network",
                    "architecture": str(model_config['hidden_layers']),
                    "device": str(self.device),
                    "input_size": model_config['input_size'],
                    "dropout_rate": model_config['dropout_rate']
                })
                
                for key, value in params_to_log.items():
                    try:
                        mlflow.log_param(key, value)
                    except mlflow.exceptions.MlflowException as param_e:
                        logger.warning(f"Could not log parameter {key}: {param_e}")
                
                # Log metrics
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)
                
                # Register model if not using file store
                if tracking_url_type_store != "file":
                    # Model registry does not work with file store
                    mlflow.pytorch.log_model(model, "model", registered_model_name="DeepCryptoPriceModel")
                else:
                    mlflow.pytorch.log_model(model, "model")
            
            logger.info(f"Deep model evaluation completed. RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
            
        except Exception as e:
            logger.exception(f"Error during deep model evaluation: {str(e)}")
            raise e