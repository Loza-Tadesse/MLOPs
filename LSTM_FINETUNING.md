# LSTM Fine-Tuning System

## Overview
The CryptoPredict system now includes automatic and manual LSTM fine-tuning capabilities to continuously improve model performance using real-world prediction verification data.

## Features

### 1. Automatic Fine-Tuning
- **Trigger Threshold**: Automatically fine-tunes LSTM after every 20 verified predictions
- **Background Processing**: Runs in a separate thread to avoid blocking predictions
- **Non-intrusive**: Doesn't interrupt the prediction service

### 2. Manual Fine-Tuning
- **API Endpoint**: `/finetune_lstm` (POST)
- **On-Demand**: Trigger fine-tuning whenever needed
- **Synchronous**: Waits for completion and returns status

### 3. Fine-Tuning Status Monitoring
- **API Endpoint**: `/get_finetuning_status` (GET)
- **Metrics**: Shows verified predictions, pending count, and threshold

## How It Works

### Automatic Fine-Tuning Workflow

1. **Prediction Verification**
   - System makes 5-minute price predictions
   - After 3 minutes, actual prices are compared with predictions
   - Verification data is stored in `artifacts/model_performance.jsonl`

2. **Threshold Check**
   - After each verification, system checks if 20+ new verified predictions exist
   - When threshold reached, automatic fine-tuning is triggered

3. **Background Fine-Tuning**
   - LSTM model is loaded with current weights
   - Last 50 verified predictions are used for fine-tuning
   - Model trains for 5 epochs with low learning rate (0.0001)
   - Fine-tuned model is saved, replacing the previous version

4. **Continuous Learning**
   - Process repeats automatically as more predictions are verified
   - Model gradually adapts to changing market conditions

### Manual Fine-Tuning

#### Trigger Manual Fine-Tuning
```bash
curl -X POST http://localhost:8080/finetune_lstm?min_samples=10
```

**Response (Success)**:
```json
{
  "status": "success",
  "message": "LSTM fine-tuning completed successfully",
  "verified_predictions_used": 25
}
```

**Response (Insufficient Data)**:
```json
{
  "status": "error",
  "message": "Insufficient verified predictions: 5 < 10",
  "verified_predictions": 5,
  "required": 10
}
```

#### Check Fine-Tuning Status
```bash
curl http://localhost:8080/get_finetuning_status
```

**Response**:
```json
{
  "auto_finetune_enabled": true,
  "finetune_threshold": 20,
  "last_finetune_count": 40,
  "verified_predictions": 55,
  "pending_for_finetune": 5
}
```

## Technical Details

### Fine-Tuning Parameters
- **Learning Rate**: 0.0001 (10x smaller than initial training)
- **Epochs**: 5
- **Batch Size**: 8 (or dataset size if smaller)
- **Gradient Clipping**: max_norm=1.0
- **Weight Decay**: 1e-5

### Data Requirements
- **Minimum Samples**: 10 verified predictions (configurable)
- **Sequence Creation**: Uses last 50 verified predictions
- **Feature Scaling**: Uses same scaler from initial training

### Safety Mechanisms
1. **Gradient Clipping**: Prevents unstable updates
2. **Low Learning Rate**: Ensures gradual, stable improvements
3. **Minimum Data Check**: Won't fine-tune with insufficient data
4. **Error Handling**: Logs errors without crashing the service

## Configuration

### Adjust Auto Fine-Tune Threshold
In `app.py`, change the `FINETUNE_THRESHOLD` constant:
```python
FINETUNE_THRESHOLD = 30  # Fine-tune after 30 verified predictions
```

### Modify Fine-Tuning Parameters
In `src/mlProject/components/lstm_finetuner.py`, adjust:
```python
def finetune(self, X, y, epochs: int = 10, learning_rate: float = 0.00005):
    # Custom parameters
```

## Monitoring

### Log Messages
Fine-tuning activities are logged:
```
[INFO] 🔄 Triggering LSTM fine-tuning (25 verified predictions)
[INFO] Starting fine-tuning with 15 sequences for 5 epochs
[INFO] Fine-tuning Epoch [2/5] Loss: 325.4567
[INFO] Fine-tuning completed! Loss improvement: 15.23%
[INFO] ✅ LSTM fine-tuning completed successfully
```

### Performance Tracking
Check `artifacts/model_performance.jsonl` for verification data:
```json
{"crypto_id": "solana", "predicted": 165.23, "actual": 165.45, "error_percent": 0.133, "timestamp": "..."}
```

## Benefits

1. **Adaptive Learning**: Model continuously learns from real market data
2. **Performance Improvement**: Gradual reduction in prediction errors
3. **Market Adaptation**: Adjusts to changing crypto market conditions
4. **Zero Downtime**: Fine-tuning happens in background without service interruption
5. **Transparent**: All fine-tuning events are logged and monitorable

## Best Practices

1. **Monitor Logs**: Regularly check logs for fine-tuning success/failures
2. **Track Metrics**: Use `/get_model_metrics` to monitor R² improvements
3. **Manual Triggers**: Use manual fine-tuning after major market events
4. **Data Quality**: Ensure prediction verification is working correctly
5. **Backup Models**: Periodically backup `artifacts/deep_model_trainer/best_deep_model.pth`

## Troubleshooting

### Fine-Tuning Not Triggering
- Check verified predictions count: `/get_finetuning_status`
- Ensure predictions are being verified (wait 3+ minutes)
- Check logs for errors in verification process

### Poor Fine-Tuning Results
- Increase minimum samples requirement
- Adjust learning rate (decrease for more stability)
- Check data quality in `model_performance.jsonl`

### Memory Issues
- Reduce batch size in `lstm_finetuner.py`
- Limit number of sequences used for fine-tuning
- Clear old performance data periodically

## Future Enhancements

- [ ] A/B testing between fine-tuned and original models
- [ ] Fine-tuning scheduler (e.g., daily at specific times)
- [ ] Multi-crypto fine-tuning with separate models
- [ ] Federated learning across multiple instances
- [ ] Automated hyperparameter tuning for fine-tuning
