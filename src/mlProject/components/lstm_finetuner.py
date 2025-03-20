import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from mlProject import logger
from mlProject.entity.config_entity import DeepModelTrainerConfig
from mlProject.components.lstm_model_trainer import CryptoLSTM, TimeSeriesDataset
from torch.utils.data import DataLoader
import joblib


class LSTMFineTuner:
    """Fine-tune LSTM model with real-world prediction errors"""
    
    def __init__(self, config: DeepModelTrainerConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        self.model_config = None
        
    def load_model_and_scaler(self):
        """Load the trained LSTM model and scaler"""
        try:
            # Load model configuration
            config_path = os.path.join(os.path.dirname(self.config.model_path), 'model_config.json')
            with open(config_path, 'r') as f:
                self.model_config = json.load(f)
            
            # Load scaler
            scaler_path = os.path.join(os.path.dirname(self.config.model_path), 'scaler.joblib')
            self.scaler = joblib.load(scaler_path)
            
            # Initialize and load model
            self.model = CryptoLSTM(
                input_size=self.model_config['input_size'],
                hidden_size=self.model_config['hidden_size'],
                num_layers=self.model_config['num_layers'],
                dropout_rate=self.model_config['dropout_rate']
            ).to(self.device)
            
            self.model.load_state_dict(torch.load(self.config.model_path, map_location=self.device))
            logger.info("Successfully loaded LSTM model and scaler for fine-tuning")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def prepare_finetuning_data(self, performance_file: str, min_samples: int = 10):
        """
        Prepare fine-tuning data from verified predictions
        
        Args:
            performance_file: Path to performance tracking file
            min_samples: Minimum number of verified predictions needed
        
        Returns:
            X, y arrays for fine-tuning, or None if insufficient data
        """
        try:
            if not os.path.exists(performance_file):
                logger.warning(f"Performance file not found: {performance_file}")
                return None, None
            
            # Read verified predictions
            verified_predictions = []
            with open(performance_file, 'r') as f:
                for line in f:
                    pred = json.loads(line)
                    if pred.get('actual') is not None and pred.get('error_percent') is not None:
                        verified_predictions.append(pred)
            
            if len(verified_predictions) < min_samples:
                logger.info(f"Insufficient verified predictions for fine-tuning: {len(verified_predictions)} < {min_samples}")
                return None, None
            
            logger.info(f"Found {len(verified_predictions)} verified predictions for fine-tuning")
            
            # Extract features and actual prices
            # Note: We would need to reconstruct the full feature vectors
            # For now, we'll use a simplified approach with available data
            features = []
            targets = []
            
            for pred in verified_predictions[-50:]:  # Use last 50 verified predictions
                # This is a simplified version - in production, you'd want to store
                # the full feature vector with each prediction
                current_price = pred.get('current_price', pred['predicted'])
                actual_price = pred['actual']
                
                # Create a simple feature vector (would be replaced with actual features)
                # In production, store full features in performance file
                feature_vec = np.array([current_price] * self.model_config['input_size'])
                features.append(feature_vec)
                targets.append(actual_price)
            
            return np.array(features), np.array(targets)
            
        except Exception as e:
            logger.error(f"Error preparing fine-tuning data: {e}")
            return None, None
    
    def finetune(self, X, y, epochs: int = 5, learning_rate: float = 0.0001):
        """
        Fine-tune the LSTM model with new data
        
        Args:
            X: Feature data
            y: Target data
            epochs: Number of fine-tuning epochs
            learning_rate: Learning rate for fine-tuning (smaller than initial training)
        """
        try:
            if X is None or y is None or len(X) == 0:
                logger.warning("No data available for fine-tuning")
                return False
            
            self.model.train()
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Create sequences
            sequence_length = self.model_config['sequence_length']
            X_sequences = []
            y_sequences = []
            
            for i in range(len(X_scaled) - sequence_length):
                X_sequences.append(X_scaled[i:i + sequence_length])
                y_sequences.append(y[i + sequence_length])
            
            if len(X_sequences) == 0:
                logger.warning("Not enough data to create sequences for fine-tuning")
                return False
            
            X_sequences = np.array(X_sequences)
            y_sequences = np.array(y_sequences).reshape(-1, 1)
            
            # Create dataset and dataloader
            dataset = TimeSeriesDataset(
                torch.FloatTensor(X_sequences),
                torch.FloatTensor(y_sequences)
            )
            dataloader = DataLoader(dataset, batch_size=min(8, len(dataset)), shuffle=True)
            
            # Fine-tuning setup
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
            
            logger.info(f"Starting fine-tuning with {len(X_sequences)} sequences for {epochs} epochs")
            
            initial_loss = None
            for epoch in range(epochs):
                epoch_loss = 0.0
                for batch_X, batch_y in dataloader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    
                    # Gradient clipping to prevent instability
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(dataloader)
                if initial_loss is None:
                    initial_loss = avg_loss
                
                if (epoch + 1) % 2 == 0:
                    logger.info(f"Fine-tuning Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.4f}")
            
            # Save fine-tuned model
            torch.save(self.model.state_dict(), self.config.model_path)
            
            improvement = ((initial_loss - avg_loss) / initial_loss * 100) if initial_loss > 0 else 0
            logger.info(f"Fine-tuning completed! Loss improvement: {improvement:.2f}%")
            logger.info(f"Fine-tuned model saved to {self.config.model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during fine-tuning: {e}")
            return False
    
    def finetune_from_performance_file(self, performance_file: str, min_samples: int = 10):
        """
        Complete fine-tuning workflow from performance file
        
        Args:
            performance_file: Path to performance tracking file
            min_samples: Minimum verified predictions needed
        
        Returns:
            bool: True if fine-tuning was successful
        """
        try:
            # Load model
            self.load_model_and_scaler()
            
            # Prepare data
            X, y = self.prepare_finetuning_data(performance_file, min_samples)
            
            if X is None or y is None:
                return False
            
            # Fine-tune
            success = self.finetune(X, y, epochs=5, learning_rate=0.0001)
            
            return success
            
        except Exception as e:
            logger.error(f"Error in fine-tuning workflow: {e}")
            return False


class IncrementalLSTMTrainer:
    """Incremental online learning for LSTM - updates model with each new verified prediction"""
    
    def __init__(self, model_path: str, scaler_path: str):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        
    def load_model(self):
        """Load model for incremental training"""
        try:
            # Load model config
            config_path = os.path.join(os.path.dirname(self.model_path), 'model_config.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Load scaler
            self.scaler = joblib.load(self.scaler_path)
            
            # Load model
            self.model = CryptoLSTM(
                input_size=config['input_size'],
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                dropout_rate=config['dropout_rate']
            ).to(self.device)
            
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            
            # Initialize optimizer for incremental learning
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00001, weight_decay=1e-5)
            
            logger.info("Model loaded for incremental training")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def update_with_single_sample(self, features: np.ndarray, actual_price: float):
        """
        Update model with a single verified prediction
        
        Args:
            features: Feature vector that was used for prediction
            actual_price: Actual price that occurred
        """
        try:
            if self.model is None:
                self.load_model()
            
            self.model.train()
            
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Convert to tensors
            X = torch.FloatTensor(features_scaled).unsqueeze(0).to(self.device)  # Add sequence dimension
            y = torch.FloatTensor([[actual_price]]).to(self.device)
            
            # Single gradient step
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.criterion(output, y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            self.optimizer.step()
            
            # Periodically save model (every N updates)
            torch.save(self.model.state_dict(), self.model_path)
            
            logger.info(f"Incremental update: loss={loss.item():.4f}")
            
        except Exception as e:
            logger.error(f"Error in incremental update: {e}")
