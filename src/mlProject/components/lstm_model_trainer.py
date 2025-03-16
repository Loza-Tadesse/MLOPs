"""LSTM Model Trainer for Cryptocurrency Price Prediction"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os
from mlProject import logger
from mlProject.entity.config_entity import DeepModelTrainerConfig


class CryptoLSTM(nn.Module):
    """LSTM Network for Cryptocurrency Price Prediction with Time Series"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout_rate=0.2):
        super(CryptoLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last hidden state
        out = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out


class TimeSeriesDataset(Dataset):
    """Dataset for creating sequences from time series data"""
    
    def __init__(self, features, targets, sequence_length=10):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.features) - self.sequence_length
    
    def __getitem__(self, idx):
        # Get sequence of features
        feature_sequence = self.features[idx:idx + self.sequence_length]
        
        # Get target (next time step after sequence)
        target = self.targets[idx + self.sequence_length]
        
        return (
            torch.FloatTensor(feature_sequence),
            torch.FloatTensor([target])
        )


class LSTMModelTrainer:
    def __init__(self, config: DeepModelTrainerConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
    def create_sequences(self, data, sequence_length=10):
        """Create sequences for LSTM from time series data"""
        sequences = []
        for i in range(len(data) - sequence_length):
            sequences.append(data[i:i + sequence_length])
        return np.array(sequences)
    
    def train(self, use_lstm=True, sequence_length=10):
        """Train LSTM or standard neural network model"""
        try:
            # Load data
            logger.info("Loading training and test data...")
            train_data = pd.read_csv(self.config.train_data_path)
            test_data = pd.read_csv(self.config.test_data_path)
            
            # Prepare features and target
            train_x = train_data.drop([self.config.target_column], axis=1)
            test_x = test_data.drop([self.config.target_column], axis=1)
            train_y = train_data[self.config.target_column].values
            test_y = test_data[self.config.target_column].values
            
            input_size = train_x.shape[1]
            logger.info(f"Input features: {input_size}")
            logger.info(f"Training samples: {len(train_x)}, Test samples: {len(test_x)}")
            
            # Scale features
            logger.info("Scaling features...")
            scaler = StandardScaler()
            train_x_scaled = scaler.fit_transform(train_x)
            test_x_scaled = scaler.transform(test_x)
            
            # Save scaler
            scaler_path = os.path.join(self.config.root_dir, 'scaler.joblib')
            joblib.dump(scaler, scaler_path)
            logger.info(f"Scaler saved to {scaler_path}")
            
            if use_lstm:
                # Create sequences for LSTM
                logger.info(f"Creating sequences with length {sequence_length}...")
                
                # Create datasets with sequences
                train_dataset = TimeSeriesDataset(
                    train_x_scaled, train_y, sequence_length
                )
                test_dataset = TimeSeriesDataset(
                    test_x_scaled, test_y, sequence_length
                )
                
                # Data loaders
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=self.config.batch_size,
                    shuffle=True
                )
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=False
                )
                
                # Initialize LSTM model
                model = CryptoLSTM(
                    input_size=input_size,
                    hidden_size=128,
                    num_layers=2,
                    dropout_rate=self.config.dropout_rate
                ).to(self.device)
                
                logger.info(f"LSTM Model Architecture:\n{model}")
                
            else:
                # Standard neural network (for comparison)
                from mlProject.pipeline.unified_prediction import CryptoPriceNet
                
                train_dataset = torch.utils.data.TensorDataset(
                    torch.FloatTensor(train_x_scaled),
                    torch.FloatTensor(train_y).unsqueeze(1)
                )
                test_dataset = torch.utils.data.TensorDataset(
                    torch.FloatTensor(test_x_scaled),
                    torch.FloatTensor(test_y).unsqueeze(1)
                )
                
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=True
                )
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=False
                )
                
                model = CryptoPriceNet(
                    input_size=input_size,
                    hidden_layers=self.config.hidden_layers,
                    dropout_rate=self.config.dropout_rate
                ).to(self.device)
                
                logger.info(f"Standard NN Model Architecture:\n{model}")
            
            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )
            
            # Training loop
            best_loss = float('inf')
            patience_counter = 0
            train_losses = []
            test_losses = []
            
            logger.info("Starting training...")
            for epoch in range(self.config.epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                
                for batch_features, batch_targets in train_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_targets)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                train_losses.append(train_loss)
                
                # Validation phase
                model.eval()
                test_loss = 0.0
                
                with torch.no_grad():
                    for batch_features, batch_targets in test_loader:
                        batch_features = batch_features.to(self.device)
                        batch_targets = batch_targets.to(self.device)
                        
                        outputs = model(batch_features)
                        loss = criterion(outputs, batch_targets)
                        test_loss += loss.item()
                
                test_loss /= len(test_loader)
                test_losses.append(test_loss)
                
                # Learning rate scheduling
                scheduler.step(test_loss)
                
                # Log progress
                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch [{epoch+1}/{self.config.epochs}] "
                        f"Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}"
                    )
                
                # Early stopping
                if test_loss < best_loss:
                    best_loss = test_loss
                    patience_counter = 0
                    
                    # Save best model
                    model_path = os.path.join(self.config.root_dir, self.config.model_name)
                    torch.save(model.state_dict(), model_path)
                    logger.info(f"Best model saved with test loss: {best_loss:.6f}")
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Save model configuration
            model_config = {
                'model_type': 'LSTM' if use_lstm else 'StandardNN',
                'input_size': input_size,
                'hidden_size': 128 if use_lstm else None,
                'num_layers': 2 if use_lstm else None,
                'hidden_layers': self.config.hidden_layers if not use_lstm else None,
                'dropout_rate': self.config.dropout_rate,
                'sequence_length': sequence_length if use_lstm else None,
                'feature_names': list(train_x.columns)
            }
            
            config_path = os.path.join(self.config.root_dir, 'model_config.json')
            with open(config_path, 'w') as f:
                json.dump(model_config, f, indent=4)
            logger.info(f"Model config saved to {config_path}")
            
            # Save training history
            history = {
                'train_losses': train_losses,
                'test_losses': test_losses,
                'best_loss': best_loss,
                'epochs_trained': len(train_losses)
            }
            
            history_path = os.path.join(self.config.root_dir, 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=4)
            logger.info(f"Training history saved to {history_path}")
            
            logger.info("Training completed successfully!")
            logger.info(f"Best test loss: {best_loss:.6f}")
            
        except Exception as e:
            logger.exception(f"Error during LSTM training: {str(e)}")
            raise e
