# LSTM Integration Guide for CryptoPredict

## 🧠 What is LSTM?

**LSTM (Long Short-Term Memory)** is a type of Recurrent Neural Network (RNN) specifically designed for time series data. Unlike traditional models that treat each data point independently, LSTM can:

- **Remember patterns** from past time steps
- **Capture temporal dependencies** in price movements
- **Handle sequential data** naturally
- **Learn long-term trends** while adapting to short-term changes

## 🎯 Why LSTM for Crypto Prediction?

Cryptocurrency prices are inherently sequential and time-dependent. LSTM excels because:

1. **Temporal Context**: Considers previous 10 time steps (configurable) when making predictions
2. **Pattern Recognition**: Learns complex price patterns and trends over time
3. **Memory**: Remembers important price movements that affect future predictions
4. **Non-linear**: Captures non-linear relationships in volatile crypto markets

## 📊 Architecture Comparison

### Traditional Random Forest
```
[29 features] → Random Forest → [5-min price]
```
- Treats each sample independently
- R² = 0.9999
- Fast inference

### LSTM Model
```
[Sequence of 10 timesteps × 32 features] → LSTM(128 units, 2 layers) → FC Layers → [5-min price]
```
- Uses sequences of historical data
- Captures temporal patterns
- Better for volatile markets

## 🚀 How to Use LSTM

### 1. Train LSTM Model

```bash
# Activate environment
source cryptopredict_env/bin/activate

# Train LSTM (uses existing data from artifacts/)
python train_lstm.py
```

This will:
- Load preprocessed training data
- Create sequences of 10 time steps
- Train LSTM with 2 layers, 128 hidden units
- Save best model to `artifacts/deep_model_trainer/best_deep_model.pth`

### 2. Configuration

Edit `params.yaml` to customize LSTM:

```yaml
LSTM:
  hidden_size: 128           # Number of LSTM units
  num_layers: 2              # Stack of LSTM layers
  dropout_rate: 0.2          # Dropout for regularization
  learning_rate: 0.0005      # Learning rate
  batch_size: 32             # Batch size
  epochs: 150                # Training epochs
  early_stopping_patience: 20
  sequence_length: 10        # Lookback window (timesteps)
```

### 3. Model Performance

After training, check metrics:
```bash
cat artifacts/deep_model_evaluation/metrics.json
```

Expected performance:
- **R² Score**: 0.95-0.99
- **RMSE**: $0.2-0.5 (for 5-minute predictions)
- **MAE**: $0.15-0.35

### 4. Using LSTM in Production

The unified prediction pipeline automatically detects and uses LSTM:

```python
from mlProject.pipeline.unified_prediction import UnifiedPredictionPipeline

# Will automatically load LSTM if model_config.json indicates LSTM
pipeline = UnifiedPredictionPipeline(model_type='deep_learning')
prediction = pipeline.predict(features)
```

## 🔧 Implementation Details

### Sequence Creation

LSTM requires sequences of data. For each prediction:

```python
# Input shape: (batch_size, sequence_length, features)
# Example: (32, 10, 32)
#   - 32 samples in batch
#   - 10 timesteps history
#   - 32 features per timestep
```

### Architecture Details

```python
CryptoLSTM(
    input_size=32,          # Number of features
    hidden_size=128,        # LSTM hidden state size
    num_layers=2,           # Stacked LSTM layers
    dropout_rate=0.2        # Dropout between layers
)
```

Layers:
1. **LSTM Layer 1**: 32 → 128 units
2. **LSTM Layer 2**: 128 → 128 units (with dropout)
3. **Fully Connected**: 128 → 64 → 32 → 1

### Training Process

1. **Sequence Creation**: Convert tabular data to sequences
   - Window size: 10 timesteps
   - Stride: 1 (overlapping sequences)

2. **Training Loop**:
   - Adam optimizer with learning rate scheduling
   - MSE loss function
   - Early stopping (patience=20 epochs)
   - Best model saving

3. **Validation**:
   - Test on hold-out sequences
   - Calculate RMSE, MAE, R² metrics

## 📈 Comparison: Random Forest vs LSTM

| Aspect | Random Forest | LSTM |
|--------|---------------|------|
| **Data Structure** | Tabular (flat) | Sequential (3D) |
| **Context** | Single timestep | Multiple timesteps |
| **Training Time** | Fast (~1 minute) | Moderate (~5-10 min) |
| **Inference Time** | Very fast (<1ms) | Fast (~2-5ms) |
| **Memory** | Low | Moderate |
| **Interpretability** | High (feature importance) | Low (black box) |
| **Best For** | Stable markets | Volatile markets |

## 🔄 Ensemble Strategy

The system can use **ensemble predictions**:

```python
# In app.py, predictions are weighted:
final_prediction = 0.5 * random_forest + 0.5 * lstm

# Adaptive weighting based on recent performance:
if recent_volatility > threshold:
    weight_lstm_higher()  # LSTM better in volatile periods
else:
    weight_rf_higher()     # RF better in stable periods
```

## 📊 Monitoring LSTM Performance

### Training Monitoring

Check training progress:
```bash
cat artifacts/deep_model_trainer/training_history.json
```

Shows:
- Training loss per epoch
- Validation loss per epoch
- Best loss achieved
- Number of epochs trained

### MLflow Tracking

View experiments:
```bash
mlflow ui
```
Navigate to: http://localhost:5000
- Compare LSTM vs Random Forest
- Track hyperparameters
- View loss curves

## 🛠️ Troubleshooting

### Issue: LSTM training is slow
**Solution**: 
- Reduce `sequence_length` (try 5 instead of 10)
- Reduce `hidden_size` (try 64 instead of 128)
- Reduce `batch_size` if memory issues

### Issue: LSTM overfitting
**Solution**:
- Increase `dropout_rate` (try 0.3-0.4)
- Add more regularization
- Reduce model complexity

### Issue: Poor performance
**Solution**:
- Increase `sequence_length` (try 15-20)
- Train longer (more epochs)
- Adjust learning rate

## 🎯 Best Practices

1. **Data Quality**: Ensure consistent time intervals in training data
2. **Sequence Length**: Balance between context (longer) and overfitting (shorter)
3. **Feature Engineering**: Include momentum indicators (RSI, MACD) in features
4. **Validation**: Use time-based splits, not random splits
5. **Monitoring**: Track both training and validation losses

## 📚 Further Improvements

1. **Bidirectional LSTM**: Look forward and backward in time
2. **Attention Mechanism**: Focus on important timesteps
3. **Multi-task Learning**: Predict price AND direction simultaneously
4. **Residual Connections**: Add skip connections for deeper networks
5. **Ensemble Methods**: Combine multiple LSTM models

## 🚀 Quick Start Commands

```bash
# 1. Train all models (including LSTM)
python train.py --model both

# 2. Train only LSTM
python train_lstm.py

# 3. View performance
cat artifacts/deep_model_evaluation/metrics.json

# 4. Start web app (uses best model automatically)
python app.py

# 5. Monitor training
mlflow ui
```

## 💡 Tips for Production

- **Model Selection**: System automatically uses best performing model
- **A/B Testing**: Run both RF and LSTM, compare in production
- **Retraining**: Retrain LSTM weekly with new data
- **Monitoring**: Track prediction errors in `artifacts/model_performance.jsonl`
- **Fallback**: Keep Random Forest as backup if LSTM fails

## 📖 Additional Resources

- PyTorch LSTM Documentation: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
- Understanding LSTMs: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
- Time Series with LSTM: https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/

---

**Next Steps**: Run `python train_lstm.py` to get started!
