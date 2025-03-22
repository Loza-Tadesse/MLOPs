# Dual Model UI Implementation

## Overview
This document describes the implementation of separate Random Forest and LSTM prediction displays in the CryptoPredict UI.

## Changes Made

### 1. Backend API Updates (`app.py`)

#### Modified `_generate_ensemble_predictions()` Function
- **Previous**: Returned a list with a single ensemble prediction
- **New**: Returns a dictionary containing:
  - `random_forest`: Individual Random Forest prediction with confidence
  - `lstm`: Individual LSTM prediction with confidence  
  - `ensemble`: Combined prediction using 70/30 weighting
  - `timestamp`: Prediction time
  - `interval`: Time ahead (5 minutes)

**Return Structure**:
```python
{
    'random_forest': {
        'price': float,
        'change_percent': float,
        'confidence': float  # 0.95 typical
    },
    'lstm': {
        'price': float,
        'change_percent': float,
        'confidence': float  # 0.75 typical
    },
    'ensemble': {
        'price': float,
        'change_percent': float,
        'confidence': float,
        'weight_rf': 0.7,
        'weight_lstm': 0.3
    },
    'timestamp': ISO string,
    'interval': '+5min'
}
```

#### Enhanced `get_model_metrics()` Endpoint
- **Previous**: Only returned deep learning metrics
- **New**: Returns comprehensive metrics for both models

**Return Structure**:
```json
{
    "random_forest": {
        "r2_score": 0.999945,
        "rmse": 0.1545,
        "mae": 0.1122,
        "model_type": "Random Forest"
    },
    "lstm": {
        "r2_score": 0.838891,
        "rmse": 8.1663,
        "mae": 6.3329,
        "model_type": "LSTM"
    },
    "comparison": {
        "better_model": "Random Forest",
        "r2_difference": 0.161054,
        "ensemble_weights": {
            "random_forest": 0.7,
            "lstm": 0.3
        }
    },
    "training_samples": 1443,
    "real_time_performance": {
        "last_error": float,
        "avg_error_10": float,
        "predictions_tracked": int,
        "error_history": [...]
    }
}
```

### 2. Frontend UI Updates (`templates/index.html`)

#### Added Model Predictions Comparison Section
Three side-by-side cards displaying:

1. **Random Forest Prediction**
   - Green-themed card with tree icon
   - Price prediction
   - Percentage change badge
   - Confidence score

2. **LSTM Neural Net Prediction**
   - Blue-themed card with network icon
   - Price prediction
   - Percentage change badge
   - Confidence score

3. **Ensemble Prediction**
   - Purple-themed card with layers icon
   - Weighted combination (70/30)
   - Price prediction
   - Percentage change badge
   - Confidence score

#### Enhanced Model Performance Comparison Section
Two side-by-side metric cards:

1. **Random Forest Metrics**
   - R² Score (typically ~99.99%)
   - RMSE (~$0.15)
   - MAE (~$0.11)

2. **LSTM Neural Net Metrics**
   - R² Score (typically ~83.89%)
   - RMSE (~$8.17)
   - MAE (~$6.33)

Plus:
- Model comparison summary showing which model performs better
- Training samples count
- Real-time performance metrics

#### Updated JavaScript Functions

**`updateChart()` Function**:
- Modified to parse new prediction structure
- Updates all three prediction display cards
- Shows individual model confidence levels
- Uses ensemble prediction for chart visualization

**`loadModelMetrics()` Function**:
- Loads metrics for both Random Forest and LSTM
- Updates separate metric displays
- Shows model comparison
- Handles real-time performance tracking

## Visual Design

### Color Scheme
- **Random Forest**: Green (`bg-success`, `border-success`)
- **LSTM**: Blue (`bg-info`, `border-info`)
- **Ensemble**: Purple (`bg-primary`, `border-primary`)

### Icons
- Random Forest: `fa-tree`
- LSTM: `fa-network-wired`
- Ensemble: `fa-layer-group`

## Model Performance Comparison

### Random Forest
- **R² Score**: 0.999945 (99.99%)
- **RMSE**: $0.1545
- **MAE**: $0.1122
- **Best For**: Stable, highly accurate predictions
- **Ensemble Weight**: 70%

### LSTM Neural Network
- **R² Score**: 0.838891 (83.89%)
- **RMSE**: $8.1663
- **MAE**: $6.3329
- **Best For**: Capturing temporal patterns and trends
- **Ensemble Weight**: 30%

## Technical Implementation Details

### Ensemble Strategy
The ensemble combines both models using weighted averaging:
```python
ensemble_price = 0.7 * rf_price + 0.3 * lstm_price
```

Weights are based on R² performance:
- Random Forest has superior R² (0.999945), so it gets 70% weight
- LSTM captures temporal dynamics, so it contributes 30%

### Confidence Scores
- **Random Forest**: 0.95 (high confidence due to excellent training metrics)
- **LSTM**: 0.75 (moderate confidence, lower than RF but captures time series patterns)
- **Ensemble**: 0.85 (balanced confidence when both models agree)

### Prediction Bounds
Both predictions are clipped to prevent extreme values:
- Maximum change: ±1.5% (adaptive based on 24h volatility)
- Base change: ±0.5% with volatility adjustment

## API Endpoints Modified

### `/get_price_predictions/<crypto_id>`
- Returns separate predictions for RF, LSTM, and ensemble
- Includes confidence scores and change percentages
- 5-minute ahead predictions

### `/get_model_metrics`
- Returns comprehensive metrics for both models
- Includes model comparison
- Real-time performance tracking

## User Experience Improvements

1. **Transparency**: Users can see both models' predictions and performance
2. **Comparison**: Side-by-side display makes it easy to compare models
3. **Confidence**: Confidence scores help users gauge prediction reliability
4. **Performance**: Clear metrics show each model's strengths
5. **Ensemble Logic**: Users understand how the 70/30 weighting works

## Future Enhancements

Potential improvements:
1. Add model selection (let users choose which prediction to follow)
2. Historical prediction accuracy charts per model
3. Dynamic weight adjustment based on recent performance
4. Model-specific prediction intervals/uncertainty bands
5. Comparative backtesting results

## Running the Application

```bash
cd /Users/philipt/VinoPredict
source cryptopredict_env/bin/activate
python3 app.py
```

Then visit: http://localhost:8080

## Screenshot Guide

The UI now shows:
1. **Top Section**: Three prediction cards (RF, LSTM, Ensemble)
2. **Middle Section**: Two performance metric cards (RF metrics, LSTM metrics)
3. **Bottom Section**: Real-time performance tracking
4. **Chart**: Price chart with ensemble prediction line

## Testing Checklist

- [x] Backend returns separate predictions
- [x] UI displays all three predictions
- [x] Metrics load for both models
- [x] Confidence scores display correctly
- [x] Change percentages show with correct badges
- [x] Model comparison summary appears
- [x] Real-time performance tracking works
- [x] Chart uses ensemble prediction
- [x] Responsive design works on mobile

## Notes

- Uses Python 3.13 (f-strings require Python 3.6+)
- Flask server runs on port 8080
- CoinGecko API data cached for 5 minutes
- Predictions update every 5 minutes with auto-refresh
