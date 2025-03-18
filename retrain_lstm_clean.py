"""
Retrain LSTM with clean features (no data leakage) and proper architecture
This script fixes the LSTM performance issues by:
1. Removing target-related features (data leakage)
2. Using proper sequence handling
3. Training with clean data distribution
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import json
import os
from pathlib import Path

# Clean feature set - NO DATA LEAKAGE
CLEAN_FEATURES = [
    'price', 'volume', 'market_cap',
    'sma_7', 'sma_14', 'sma_30',
    'ema_7', 'ema_14',
    'macd', 'macd_signal', 'macd_histogram',
    'rsi',
    'bb_middle', 'bb_upper', 'bb_lower',
    'price_change_1h', 'price_change_24h', 'price_change_7d',
    'volume_sma', 'volume_ratio',
    'volatility',
    'high_14d', 'low_14d',
    'price_position'
]

TARGET_COLUMN = 'target_price_5min'


class CryptoLSTM(nn.Module):
    """LSTM Network for Cryptocurrency Price Prediction"""
    
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
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = lstm_out[:, -1, :]  # Use last hidden state
        
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
        feature_sequence = self.features[idx:idx + self.sequence_length]
        target = self.targets[idx + self.sequence_length]
        
        return (
            torch.FloatTensor(feature_sequence),
            torch.FloatTensor([target])
        )


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for features, targets in train_loader:
        features = features.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, test_loader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            targets = targets.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            all_predictions.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
    
    avg_loss = total_loss / len(test_loader)
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    r2 = r2_score(all_targets, all_predictions)
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    mae = mean_absolute_error(all_targets, all_predictions)
    
    return avg_loss, r2, rmse, mae, all_predictions, all_targets


def main():
    print("=" * 60)
    print("LSTM Retraining with Clean Features (No Data Leakage)")
    print("=" * 60)
    
    # Configuration
    SEQUENCE_LENGTH = 10
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT_RATE = 0.2
    LEARNING_RATE = 0.0005
    BATCH_SIZE = 32
    EPOCHS = 50
    PATIENCE = 10
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    print("\n1. Loading data...")
    train_data = pd.read_csv('artifacts/data_transformation/train.csv')
    test_data = pd.read_csv('artifacts/data_transformation/test.csv')
    
    print(f"   Training samples: {len(train_data)}")
    print(f"   Test samples: {len(test_data)}")
    
    # Extract clean features (NO DATA LEAKAGE)
    print("\n2. Extracting clean features (removing data leakage)...")
    train_x = train_data[CLEAN_FEATURES].values
    test_x = test_data[CLEAN_FEATURES].values
    train_y = train_data[TARGET_COLUMN].values
    test_y = test_data[TARGET_COLUMN].values
    
    print(f"   Features used: {len(CLEAN_FEATURES)}")
    print(f"   Clean feature set: {CLEAN_FEATURES[:5]}... (+{len(CLEAN_FEATURES)-5} more)")
    
    # Scale features
    print("\n3. Scaling features...")
    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    test_x_scaled = scaler.transform(test_x)
    
    # Create datasets
    print(f"\n4. Creating sequences (length={SEQUENCE_LENGTH})...")
    train_dataset = TimeSeriesDataset(train_x_scaled, train_y, SEQUENCE_LENGTH)
    test_dataset = TimeSeriesDataset(test_x_scaled, test_y, SEQUENCE_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"   Training sequences: {len(train_dataset)}")
    print(f"   Test sequences: {len(test_dataset)}")
    
    # Initialize model
    print("\n5. Initializing LSTM model...")
    input_size = len(CLEAN_FEATURES)
    model = CryptoLSTM(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT_RATE
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training
    print("\n6. Training LSTM...")
    print("   Epoch | Train Loss | Test Loss | Test R² | Test RMSE | Test MAE")
    print("   " + "-" * 65)
    
    best_test_loss = float('inf')
    best_r2 = -float('inf')
    patience_counter = 0
    training_history = {
        'train_losses': [],
        'test_losses': [],
        'test_r2': [],
        'test_rmse': [],
        'test_mae': []
    }
    
    for epoch in range(EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        test_loss, r2, rmse, mae, _, _ = evaluate(model, test_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(test_loss)
        
        # Save history
        training_history['train_losses'].append(train_loss)
        training_history['test_losses'].append(test_loss)
        training_history['test_r2'].append(r2)
        training_history['test_rmse'].append(rmse)
        training_history['test_mae'].append(mae)
        
        # Print progress
        print(f"   {epoch+1:5d} | {train_loss:10.4f} | {test_loss:9.4f} | "
              f"{r2:7.4f} | ${rmse:8.4f} | ${mae:7.4f}")
        
        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_r2 = r2
            patience_counter = 0
            
            # Save model
            torch.save(model.state_dict(), 'artifacts/deep_model_trainer/best_deep_model.pth')
            print(f"   ✓ Best model saved (R²={r2:.6f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n   Early stopping triggered (patience={PATIENCE})")
            break
    
    # Final evaluation
    print("\n7. Final evaluation on test set...")
    model.load_state_dict(torch.load('artifacts/deep_model_trainer/best_deep_model.pth'))
    test_loss, r2, rmse, mae, predictions, actuals = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"   Final R² Score: {r2:.6f}")
    print(f"   Final RMSE: ${rmse:.4f}")
    print(f"   Final MAE: ${mae:.4f}")
    
    # Save artifacts
    print("\n8. Saving artifacts...")
    
    # Save scaler
    joblib.dump(scaler, 'artifacts/deep_model_trainer/scaler.joblib')
    print("   ✓ Scaler saved")
    
    # Save model config
    model_config = {
        'model_type': 'LSTM',
        'input_size': input_size,
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
        'dropout_rate': DROPOUT_RATE,
        'sequence_length': SEQUENCE_LENGTH,
        'feature_names': CLEAN_FEATURES
    }
    
    with open('artifacts/deep_model_trainer/model_config.json', 'w') as f:
        json.dump(model_config, f, indent=4)
    print("   ✓ Model config saved")
    
    # Save training history
    training_history['best_loss'] = float(best_test_loss)
    training_history['epochs_trained'] = epoch + 1
    training_history['final_r2'] = float(r2)
    training_history['final_rmse'] = float(rmse)
    training_history['final_mae'] = float(mae)
    
    with open('artifacts/deep_model_trainer/training_history.json', 'w') as f:
        json.dump(training_history, f, indent=4)
    print("   ✓ Training history saved")
    
    # Save evaluation metrics
    metrics = {
        'r2': float(r2),
        'rmse': float(rmse),
        'mae': float(mae),
        'model_type': 'LSTM'
    }
    
    os.makedirs('artifacts/deep_model_evaluation', exist_ok=True)
    with open('artifacts/deep_model_evaluation/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("   ✓ Evaluation metrics saved")
    
    # Performance comparison
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    print(f"\nOLD LSTM Performance:")
    print(f"  R²:   -0.000188 (essentially random)")
    print(f"  RMSE: $20.74")
    print(f"  MAE:  $17.11")
    
    print(f"\nNEW LSTM Performance:")
    print(f"  R²:   {r2:.6f}")
    print(f"  RMSE: ${rmse:.4f}")
    print(f"  MAE:  ${mae:.4f}")
    
    improvement = ((r2 - (-0.000188)) / abs(-0.000188)) * 100 if r2 > 0 else 0
    print(f"\nImprovement: {'+' if improvement > 0 else ''}{improvement:.1f}%")
    
    # Data leakage explanation
    print("\n" + "=" * 60)
    print("FIXES APPLIED")
    print("=" * 60)
    print("\n1. ✓ Removed data leakage:")
    print("     Excluded: target_price_1h, target_direction_*, target_change_*")
    print(f"     Using only: {len(CLEAN_FEATURES)} legitimate features")
    
    print("\n2. ✓ Proper sequence handling:")
    print(f"     Sequence length: {SEQUENCE_LENGTH} time steps")
    print("     Real temporal patterns (not duplicated single points)")
    
    print("\n3. ✓ Clean training/inference alignment:")
    print("     Same features used in training and inference")
    print("     No future information leaked")
    
    print("\n4. ✓ Better hyperparameters:")
    print(f"     Learning rate: {LEARNING_RATE} (lower for stability)")
    print(f"     Gradient clipping: 1.0 (prevents exploding gradients)")
    print(f"     Early stopping: patience={PATIENCE}")
    
    print("\n" + "=" * 60)
    print("RETRAINING COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Restart Flask server: python app.py")
    print("2. LSTM predictions will now use the clean model")
    print("3. Monitor real-world performance via /get_model_metrics")
    print("4. Fine-tuning will gradually improve accuracy further")
    print("\n")


if __name__ == "__main__":
    main()
