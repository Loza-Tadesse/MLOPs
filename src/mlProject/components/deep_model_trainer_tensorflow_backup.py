import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from mlProject import logger
from mlProject.entity.config_entity import DeepModelTrainerConfig
import joblib
import json


class DeepModelTrainer:
    def __init__(self, config: DeepModelTrainerConfig):
        self.config = config
        self.scaler = StandardScaler()
        
    def build_model(self, input_shape):
        """Build neural network architecture based on config"""
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Dense(
            self.config.hidden_layers[0], 
            activation=self.config.activation,
            input_shape=(input_shape,)
        ))
        model.add(layers.Dropout(self.config.dropout_rate))
        
        # Hidden layers
        for units in self.config.hidden_layers[1:]:
            model.add(layers.Dense(units, activation=self.config.activation))
            model.add(layers.Dropout(self.config.dropout_rate))
        
        # Output layer (regression)
        model.add(layers.Dense(1, activation='linear'))
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=self.config.loss,
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train(self):
        """Train deep learning model"""
        try:
            # Load data
            train_data = pd.read_csv(self.config.train_data_path)
            test_data = pd.read_csv(self.config.test_data_path)
            
            # Prepare features and targets
            train_x = train_data.drop([self.config.target_column], axis=1)
            test_x = test_data.drop([self.config.target_column], axis=1)
            train_y = train_data[self.config.target_column]
            test_y = test_data[self.config.target_column]
            
            # Scale features
            train_x_scaled = self.scaler.fit_transform(train_x)
            test_x_scaled = self.scaler.transform(test_x)
            
            # Build model
            model = self.build_model(train_x_scaled.shape[1])
            
            logger.info(f"Model architecture: {model.summary()}")
            
            # Callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7
                )
            ]
            
            # Train model
            history = model.fit(
                train_x_scaled, train_y,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                validation_data=(test_x_scaled, test_y),
                callbacks=callbacks,
                verbose=1
            )
            
            # Save model and scaler
            model.save(os.path.join(self.config.root_dir, "deep_model.h5"))
            joblib.dump(self.scaler, os.path.join(self.config.root_dir, "scaler.joblib"))
            
            # Save training history
            history_dict = {
                'loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history['val_loss']],
                'mae': [float(x) for x in history.history['mae']],
                'val_mae': [float(x) for x in history.history['val_mae']]
            }
            
            with open(os.path.join(self.config.root_dir, "training_history.json"), 'w') as f:
                json.dump(history_dict, f, indent=4)
            
            logger.info("Deep learning model training completed successfully!")
            
        except Exception as e:
            logger.exception(f"Error during deep model training: {str(e)}")
            raise e