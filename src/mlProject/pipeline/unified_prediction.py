import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from pathlib import Path
from mlProject import logger
from mlProject.components.feature_validator import FeatureValidator
from mlProject.components.feature_engineering import FeatureEngineering


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


class UnifiedPredictionPipeline:
    def __init__(self, model_type='traditional', enable_validation=True):
        """
        Initialize prediction pipeline
        
        Args:
            model_type (str): Either 'traditional' for Random Forest or 'deep'/'deep_learning' for PyTorch model
            enable_validation (bool): Whether to enable feature validation
        """
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.enable_validation = enable_validation
        self.is_lstm = False
        self.sequence_length = 10
        
        # Initialize validator if enabled
        if self.enable_validation:
            self.validator = FeatureValidator()
            logger.info("Feature validation enabled")
        else:
            self.validator = None
            logger.info("Feature validation disabled")
        
        if model_type == 'traditional':
            self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
            self.scaler = None
            logger.info("✅ Loaded Random Forest model from artifacts/model_trainer/model.joblib")
        elif model_type in ['deep', 'deep_learning']:
            try:
                self._load_deep_model()
                logger.info(f"✅ Loaded deep learning model (LSTM={self.is_lstm})")
            except Exception as e:
                logger.warning(f"⚠️  Deep learning model not found, falling back to Random Forest: {e}")
                logger.warning("💡 To train deep learning model, run: python train_improved_dnn.py")
                # Fall back to Random Forest
                self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
                self.scaler = None
                self.model_type = 'traditional'  # Switch to traditional mode
                logger.info("✅ Using Random Forest model as fallback")
        else:
            raise ValueError("model_type must be either 'traditional', 'deep', or 'deep_learning'")
    
    def _load_deep_model(self):
        """Load PyTorch deep learning model (DNN, LSTM, or standard NN)"""
        try:
            # Load model configuration
            config_path = Path('artifacts/deep_model_trainer/model_config.json')
            with open(config_path, 'r') as f:
                model_config = json.load(f)
            
            model_type = model_config.get('model_type', 'NN')
            
            # Check if LSTM model
            if model_type == 'LSTM':
                # Import LSTM model
                from mlProject.components.lstm_model_trainer import CryptoLSTM
                
                self.model = CryptoLSTM(
                    input_size=model_config['input_size'],
                    hidden_size=model_config.get('hidden_size', 128),
                    num_layers=model_config.get('num_layers', 2),
                    dropout_rate=model_config['dropout_rate']
                ).to(self.device)
                
                self.sequence_length = model_config.get('sequence_length', 10)
                self.is_lstm = True
                logger.info(f"Loaded LSTM model (sequence_length={self.sequence_length})")
            
            elif model_type == 'DNN':
                # Import improved DNN model
                from mlProject.components.dnn_model import ImprovedDNN
                
                self.model = ImprovedDNN(
                    input_size=model_config['input_size'],
                    dropout_rate=model_config.get('dropout_rate', 0.3)
                ).to(self.device)
                
                self.is_lstm = False
                logger.info(f"Loaded Improved DNN model (input_size={model_config['input_size']})")
            
            else:
                # Standard neural network
                self.model = CryptoPriceNet(
                    input_size=model_config['input_size'],
                    hidden_layers=model_config['hidden_layers'],
                    dropout_rate=model_config['dropout_rate']
                ).to(self.device)
                
                self.is_lstm = False
                logger.info("Loaded standard neural network")
            
            # Load trained weights
            model_path = Path('artifacts/deep_model_trainer/best_deep_model.pth')
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            
            # Load scaler
            scaler_path = Path('artifacts/deep_model_trainer/scaler.joblib')
            self.scaler = joblib.load(scaler_path)
            
        except Exception as e:
            raise Exception(f"Failed to load deep learning model: {str(e)}")

    def predict(self, data):
        """
        Make predictions using the loaded model
        
        Args:
            data (np.array): Input features for prediction
            
        Returns:
            np.array: Predictions
        """
        # Validate features if enabled
        if self.enable_validation and self.validator:
            try:
                validation_results = self.validator.validate_all(
                    features=data,
                    feature_names=FeatureEngineering.get_feature_names(),
                    check_drift=True,
                    drift_threshold=3.0
                )
                
                if not validation_results['valid']:
                    logger.error(f"Feature validation failed: {validation_results['errors']}")
                    # Continue with prediction but log warnings
                
                if validation_results['warnings']:
                    for warning in validation_results['warnings']:
                        logger.warning(f"Feature validation warning: {warning}")
                
                # Log drift details if detected
                if validation_results.get('drift_report'):
                    drift_report = validation_results['drift_report']
                    if drift_report.get('drifted_features'):
                        logger.warning(
                            f"Feature drift detected in {len(drift_report['drifted_features'])} features"
                        )
                        
            except Exception as e:
                logger.error(f"Feature validation error: {e}")
                # Continue with prediction
        
        if self.model_type == 'traditional':
            return self._predict_traditional(data)
        elif self.model_type in ['deep', 'deep_learning']:
            return self._predict_deep(data)
    
    def _predict_traditional(self, data):
        """Predict using traditional Random Forest model"""
        prediction = self.model.predict(data)
        logger.info(f"Random Forest raw prediction: {prediction}")
        return prediction
    
    def _predict_deep(self, data):
        """Predict using deep learning model"""
        # Scale the data
        data_scaled = self.scaler.transform(data)
        logger.info(f"Deep learning model type: {'LSTM' if self.is_lstm else 'DNN/NN'}")
        
        if self.is_lstm:
            # LSTM requires sequences
            # If single data point provided, create sequence by repeating
            # In production, should collect actual historical sequence
            if data_scaled.shape[0] == 1:
                # Repeat the single point to create a sequence
                # This is a simplification - ideally use real historical data
                sequence = np.repeat(data_scaled, self.sequence_length, axis=0)
                sequence = sequence.reshape(1, self.sequence_length, -1)
            else:
                # If multiple points provided, use them as sequence
                if data_scaled.shape[0] < self.sequence_length:
                    # Pad with first point if not enough
                    padding = np.repeat(
                        data_scaled[0:1], 
                        self.sequence_length - data_scaled.shape[0], 
                        axis=0
                    )
                    sequence = np.vstack([padding, data_scaled])
                else:
                    # Use last sequence_length points
                    sequence = data_scaled[-self.sequence_length:]
                
                sequence = sequence.reshape(1, self.sequence_length, -1)
            
            # Convert to tensor
            data_tensor = torch.FloatTensor(sequence).to(self.device)
        else:
            # Standard NN - use data as-is
            data_tensor = torch.FloatTensor(data_scaled).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(data_tensor).cpu().numpy().flatten()
        
        logger.info(f"Deep learning raw prediction: {prediction}")
        return prediction


# Backward compatibility - keep original class
class PredictionPipeline:
    def __init__(self):
        self.unified_pipeline = UnifiedPredictionPipeline(model_type='traditional')

    def predict(self, data):
        return self.unified_pipeline.predict(data)