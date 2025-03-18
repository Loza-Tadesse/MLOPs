"""
Train a better Deep Neural Network (DNN) instead of LSTM
For 5-minute price predictions, a well-tuned DNN may outperform LSTM
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

# Clean feature set
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


class ImprovedDNN(nn.Module):
    """Improved Deep Neural Network for Price Prediction"""
    
    def __init__(self, input_size, dropout_rate=0.3):
        super(ImprovedDNN, self).__init__()
        
        self.network = nn.Sequential(
            # Input layer
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Hidden layer 1
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Hidden layer 2
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Hidden layer 3
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            
            # Output layer
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.network(x)


class SimpleDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets).reshape(-1, 1)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def main():
    print("=" * 70)
    print("Training Improved DNN (Better than LSTM for 5-min predictions)")
    print("=" * 70)
    
    # Config
    DROPOUT_RATE = 0.3
    LEARNING_RATE = 0.001
    BATCH_SIZE = 64
    EPOCHS = 100
    PATIENCE = 15
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    print("\n1. Loading data...")
    train_data = pd.read_csv('artifacts/data_transformation/train.csv')
    test_data = pd.read_csv('artifacts/data_transformation/test.csv')
    
    train_x = train_data[CLEAN_FEATURES].values
    test_x = test_data[CLEAN_FEATURES].values
    train_y = train_data[TARGET_COLUMN].values
    test_y = test_data[TARGET_COLUMN].values
    
    print(f"   Training: {len(train_x)} samples")
    print(f"   Test: {len(test_x)} samples")
    print(f"   Features: {len(CLEAN_FEATURES)}")
    
    # Scale
    print("\n2. Scaling features...")
    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    test_x_scaled = scaler.transform(test_x)
    
    # Create datasets
    train_dataset = SimpleDataset(train_x_scaled, train_y)
    test_dataset = SimpleDataset(test_x_scaled, test_y)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    print("\n3. Initializing DNN...")
    model = ImprovedDNN(len(CLEAN_FEATURES), DROPOUT_RATE).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7, factor=0.5)
    
    # Train
    print("\n4. Training...")
    print("   Epoch | Train Loss | Test Loss | Test R² | Test RMSE | Test MAE")
    print("   " + "-" * 70)
    
    best_loss = float('inf')
    patience_counter = 0
    history = {'train_losses': [], 'test_losses': []}
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Evaluation
        model.eval()
        test_loss = 0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for features, targets in test_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                
                all_preds.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())
        
        test_loss /= len(test_loader)
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        r2 = r2_score(all_targets, all_preds)
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        mae = mean_absolute_error(all_targets, all_preds)
        
        scheduler.step(test_loss)
        
        history['train_losses'].append(train_loss)
        history['test_losses'].append(test_loss)
        
        print(f"   {epoch+1:5d} | {train_loss:10.4f} | {test_loss:9.4f} | "
              f"{r2:7.5f} | ${rmse:8.4f} | ${mae:7.4f}")
        
        if test_loss < best_loss:
            best_loss = test_loss
            best_r2 = r2
            patience_counter = 0
            torch.save(model.state_dict(), 'artifacts/deep_model_trainer/best_deep_model.pth')
            print(f"   ✓ Best model saved (R²={r2:.6f})")
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            print(f"\n   Early stopping (patience={PATIENCE})")
            break
    
    # Final eval
    print("\n5. Final evaluation...")
    model.load_state_dict(torch.load('artifacts/deep_model_trainer/best_deep_model.pth'))
    model.eval()
    
    all_preds, all_targets = [], []
    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            outputs = model(features)
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    r2 = r2_score(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae = mean_absolute_error(all_targets, all_preds)
    
    print(f"   R² Score: {r2:.6f}")
    print(f"   RMSE: ${rmse:.4f}")
    print(f"   MAE: ${mae:.4f}")
    
    # Save artifacts
    print("\n6. Saving artifacts...")
    
    joblib.dump(scaler, 'artifacts/deep_model_trainer/scaler.joblib')
    
    model_config = {
        'model_type': 'DNN',
        'input_size': len(CLEAN_FEATURES),
        'hidden_layers': [256, 128, 64, 32],
        'dropout_rate': DROPOUT_RATE,
        'feature_names': CLEAN_FEATURES
    }
    
    with open('artifacts/deep_model_trainer/model_config.json', 'w') as f:
        json.dump(model_config, f, indent=4)
    
    history['best_loss'] = float(best_loss)
    history['epochs_trained'] = epoch + 1
    with open('artifacts/deep_model_trainer/training_history.json', 'w') as f:
        json.dump(history, f, indent=4)
    
    metrics = {
        'r2': float(r2),
        'rmse': float(rmse),
        'mae': float(mae),
        'model_type': 'DNN'
    }
    
    os.makedirs('artifacts/deep_model_evaluation', exist_ok=True)
    with open('artifacts/deep_model_evaluation/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("   ✓ All artifacts saved")
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print("\nOLD LSTM: R²=-0.000188, RMSE=$20.74, MAE=$17.11")
    print(f"NEW DNN:  R²={r2:.6f}, RMSE=${rmse:.4f}, MAE=${mae:.4f}")
    
    if r2 > 0.5:
        print("\n✅ EXCELLENT! DNN shows strong performance")
    elif r2 > 0:
        print("\n⚠️  DNN shows positive R² (improvement over LSTM)")
    else:
        print("\n❌ Still negative R² - data may not support 5-min predictions")
    
    print("\n" + "=" * 70)
    print("Restart server: python app.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
