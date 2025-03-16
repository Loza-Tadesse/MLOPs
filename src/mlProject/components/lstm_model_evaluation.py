"""LSTM Model Evaluation for Cryptocurrency Price Prediction"""
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


class CryptoLSTM(nn.Module):
    """LSTM Network for Cryptocurrency Price Prediction"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout_rate=0.2):
        super(CryptoLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out


class LSTMModelEvaluation:
    def __init__(self, config: DeepModelEvaluationConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    
    def create_sequences(self, data, sequence_length):
        """Create sequences for LSTM evaluation"""
        sequences = []
        for i in range(len(data) - sequence_length):
            sequences.append(data[i:i + sequence_length])
        return np.array(sequences)

    def log_into_mlflow(self):
        try:
            # Load test data
            test_data = pd.read_csv(self.config.test_data_path)
            
            # Load model configuration
            with open(self.config.model_config_path, 'r') as f:
                model_config = json.load(f)
            
            # Load scaler
            scaler = joblib.load(self.config.scaler_path)
            
            # Check if LSTM or standard model
            is_lstm = model_config.get('model_type') == 'LSTM'
            sequence_length = model_config.get('sequence_length', 10)
            
            if is_lstm:
                # Initialize LSTM model
                model = CryptoLSTM(
                    input_size=model_config['input_size'],
                    hidden_size=model_config.get('hidden_size', 128),
                    num_layers=model_config.get('num_layers', 2),
                    dropout_rate=model_config['dropout_rate']
                ).to(self.device)
                
                logger.info("Evaluating LSTM model")
            else:
                # Standard neural network
                from mlProject.pipeline.unified_prediction import CryptoPriceNet
                model = CryptoPriceNet(
                    input_size=model_config['input_size'],
                    hidden_layers=model_config['hidden_layers'],
                    dropout_rate=model_config['dropout_rate']
                ).to(self.device)
                
                logger.info("Evaluating standard neural network")
            
            # Load trained weights
            model.load_state_dict(torch.load(self.config.model_path, map_location=self.device))
            model.eval()
            
            # Prepare test data
            test_x = test_data.drop([self.config.target_column], axis=1)
            test_y = test_data[self.config.target_column].values
            
            # Scale features
            test_x_scaled = scaler.transform(test_x)
            
            if is_lstm:
                # Create sequences for LSTM
                test_sequences = self.create_sequences(test_x_scaled, sequence_length)
                test_y_adjusted = test_y[sequence_length:]  # Adjust targets to match sequences
                
                test_x_tensor = torch.FloatTensor(test_sequences).to(self.device)
            else:
                test_x_tensor = torch.FloatTensor(test_x_scaled).to(self.device)
                test_y_adjusted = test_y
            
            # Make predictions
            with torch.no_grad():
                predicted_qualities = model(test_x_tensor).cpu().numpy().flatten()
            
            # Calculate metrics
            (rmse, mae, r2) = self.eval_metrics(test_y_adjusted, predicted_qualities)
            
            # Save metrics locally
            scores = {"rmse": rmse, "mae": mae, "r2": r2, "model_type": model_config.get('model_type', 'Unknown')}
            save_json(path=Path(self.config.metric_file_name), data=scores)
            
            # MLflow logging
            mlflow.set_registry_uri(self.config.mlflow_uri)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            experiment_name = "LSTM_CryptoPredict" if is_lstm else "Deep_Learning_CryptoPredict"
            try:
                mlflow.create_experiment(experiment_name)
            except mlflow.exceptions.MlflowException:
                pass
            
            mlflow.set_experiment(experiment_name)
            
            import time
            run_name = f"{'lstm' if is_lstm else 'nn'}_eval_{int(time.time())}"
            
            with mlflow.start_run(run_name=run_name):
                # Log parameters
                params_to_log = {
                    "model_type": model_config.get('model_type', 'Unknown'),
                    "device": str(self.device),
                    "input_size": model_config['input_size'],
                    "dropout_rate": model_config['dropout_rate']
                }
                
                if is_lstm:
                    params_to_log.update({
                        "hidden_size": model_config.get('hidden_size', 128),
                        "num_layers": model_config.get('num_layers', 2),
                        "sequence_length": sequence_length
                    })
                else:
                    params_to_log["architecture"] = str(model_config['hidden_layers'])
                
                for key, value in params_to_log.items():
                    try:
                        mlflow.log_param(key, value)
                    except mlflow.exceptions.MlflowException as param_e:
                        logger.warning(f"Could not log parameter {key}: {param_e}")
                
                # Log metrics
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)
                
                # Register model
                if tracking_url_type_store != "file":
                    model_name = "LSTM_CryptoPriceModel" if is_lstm else "DeepCryptoPriceModel"
                    mlflow.pytorch.log_model(model, "model", registered_model_name=model_name)
                else:
                    mlflow.pytorch.log_model(model, "model")
            
            logger.info(f"{'LSTM' if is_lstm else 'NN'} model evaluation completed. RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
            
        except Exception as e:
            logger.exception(f"Error during model evaluation: {str(e)}")
            raise e
