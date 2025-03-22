# Past Predictions Display Feature

## Overview
The UI now displays past predictions for both Random Forest and LSTM models on the price chart, allowing you to visually compare how each model performed over time alongside the current 5-minute predictions.

## Features Added

### 1. Backend API Endpoint
**Endpoint**: `/get_past_predictions/<crypto_id>`
- **Parameters**: 
  - `max`: Maximum number of past predictions to retrieve (default: 20)
- **Returns**: JSON with separate arrays for RF, LSTM, ensemble predictions, and actual prices
- **Data Structure**:
  ```json
  {
    "random_forest": [
      {"timestamp": "...", "price": 193.5, "actual": 193.2, "error": 0.155}
    ],
    "lstm": [
      {"timestamp": "...", "price": 194.1, "actual": 193.2, "error": 0.466}
    ],
    "ensemble": [...],
    "actuals": [...]
  }
  ```

### 2. Enhanced Performance Tracking
The `model_performance.jsonl` file now stores:
- `rf_predicted`: Random Forest prediction
- `lstm_predicted`: LSTM prediction
- `predicted`: Ensemble prediction (95% RF + 5% LSTM)
- `actual`: Verified actual price (after 3+ minutes)
- `rf_error`: Random Forest error percentage
- `lstm_error`: LSTM error percentage
- `error_percent`: Ensemble error percentage
- `prediction_time`: Timestamp when prediction was made for next 5 minutes

### 3. Chart Visualization

#### Four Chart Datasets:
1. **Historical Price** (Purple/Blue)
   - Real market data from last 4 hours (5-minute intervals)
   - Solid line with fill

2. **Ensemble Current Prediction** (Red)
   - Current 5-minute ahead prediction
   - Dashed line with triangle markers
   - 95% Random Forest + 5% LSTM weighted ensemble

3. **RF Past Predictions** (Green)
   - Historical Random Forest predictions overlaid on chart
   - Dashed line with square markers
   - Shows how RF model predicted prices at different times

4. **LSTM Past Predictions** (Blue)
   - Historical LSTM predictions overlaid on chart
   - Dashed line with circle markers
   - Shows how LSTM model predicted prices at different times

### 4. Toggle Control
The "Show Next 5-Min Prediction" checkbox now controls:
- Current ensemble prediction
- All past RF predictions
- All past LSTM predictions

When unchecked, all prediction datasets are hidden, showing only historical prices.

## Chart Legend
- 📊 Historical Data (solid line)
- 🔮 Ensemble Prediction (red dashed, triangle markers)
- 🌲 RF Past Predictions (green dashed, square markers)
- 🧠 LSTM Past Predictions (blue dashed, circle markers)

## How It Works

### Prediction Flow:
1. User requests prediction for cryptocurrency (e.g., Solana)
2. System generates three predictions:
   - Random Forest: R²=0.999945 (excellent)
   - LSTM: R²=-0.000188 (poor, being improved via fine-tuning)
   - Ensemble: 95% RF + 5% LSTM
3. All three predictions stored in performance file with timestamp
4. After 3+ minutes, system verifies prediction against actual price
5. Individual model errors calculated and stored

### Chart Update Flow:
1. Frontend fetches last 20 past predictions via `/get_past_predictions/solana`
2. Fetches 5-minute historical data via `/get_5min_data/solana`
3. Fetches current 5-minute prediction via `/get_price_predictions/solana`
4. Maps past predictions to chart timestamps
5. Displays all datasets with proper time alignment

## Benefits

### Visual Model Comparison
- **See which model is more accurate**: Compare green (RF) vs blue (LSTM) markers against actual prices
- **Track model performance over time**: Observe if predictions cluster around actual prices
- **Identify prediction patterns**: See if models consistently over/under-predict

### Debugging & Validation
- **Verify fine-tuning effectiveness**: After LSTM fine-tuning, check if blue markers get closer to actual prices
- **Spot systematic errors**: If all predictions are off by similar amount, indicates potential issue
- **Real-time model health monitoring**: Quickly identify if a model starts degrading

### User Confidence
- **Transparency**: Users see historical accuracy, not just current predictions
- **Trust building**: Showing past predictions demonstrates confidence in system
- **Decision support**: Users can weight predictions based on historical accuracy

## Example Use Cases

### 1. Model Performance Analysis
```
Scenario: After 20 predictions, user wants to see which model is better
Action: Toggle predictions on, observe green (RF) vs blue (LSTM) clusters
Result: RF predictions (green squares) cluster tightly around actual prices
        LSTM predictions (blue circles) show more variance
Conclusion: RF model is more reliable for this timeframe
```

### 2. Fine-Tuning Validation
```
Scenario: LSTM was fine-tuned after 20 verified predictions
Action: Compare blue markers before and after fine-tuning threshold
Result: Blue markers after fine-tuning show reduced error spread
Conclusion: Fine-tuning is improving LSTM performance
```

### 3. Real-Time Decision Making
```
Scenario: User wants to make trading decision at 2:30 PM
Action: Check chart at 2:30 PM
Observation:
  - Current price: $193.20
  - RF predicts (green): $193.93 (+0.38%)
  - LSTM predicts (blue): $194.15 (+0.49%)
  - Ensemble: $193.95 (+0.39%)
  - Past RF predictions: 90% within 0.5% error
  - Past LSTM predictions: 60% within 0.5% error
Decision: Trust ensemble but weight toward RF (which it does at 95%)
```

## Configuration

### Adjust Number of Past Predictions Displayed
Edit the fetch call in `index.html`:
```javascript
const pastPredResponse = await fetch(`/get_past_predictions/${cryptoId}?max=20`);
```
Change `max=20` to display more or fewer past predictions (e.g., `max=50` for last 50 predictions).

### Adjust Chart Time Window
The chart displays 4 hours of 5-minute data (48 points). To show more history, modify:
```javascript
params = {'vs_currency': 'usd', 'days': '0.2'}  // 0.2 days = ~4.8 hours
```
Increase to `'days': '0.5'` for ~12 hours, or `'days': '1'` for 24 hours.

## Technical Details

### Chart Dataset Configuration
```javascript
Dataset 0 (Historical): Solid line, fill, main crypto color
Dataset 1 (Ensemble): Red, dashed [5,5], triangle points
Dataset 2 (RF Past): Green, dashed [3,3], square points
Dataset 3 (LSTM Past): Blue, dashed [3,3], circle points
```

### Time Alignment
Past predictions are mapped to chart labels by matching timestamps:
```javascript
const predTime = new Date(pred.timestamp);
const predLabel = predTime.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
const labelIndex = allLabels.indexOf(predLabel);
if (labelIndex >= 0) {
    rfPastData[labelIndex] = pred.price;
}
```

### Performance Considerations
- Only last 20 predictions loaded by default (configurable)
- Past predictions cached in browser until next chart update
- Chart updates every time user refreshes predictions manually
- Auto-refresh disabled by default to avoid excessive API calls

## Future Enhancements

### Potential Additions:
1. **Prediction Accuracy Bands**: Show confidence intervals around predictions
2. **Error Metrics on Hover**: Display exact error percentage when hovering over past predictions
3. **Model Comparison View**: Toggle individual models on/off independently
4. **Prediction Heatmap**: Color-code past predictions by accuracy (green=accurate, red=inaccurate)
5. **Trend Analysis**: Show moving average of prediction errors
6. **Export Functionality**: Download past predictions and actuals for external analysis

## Troubleshooting

### Past predictions not showing on chart
**Issue**: Green/blue markers not visible
**Solutions**:
1. Ensure "Show Next 5-Min Prediction" checkbox is checked
2. Verify predictions exist: Check `artifacts/model_performance.jsonl`
3. Check console logs: Look for "Processing X RF past predictions"
4. Ensure timestamps align: 5-minute intervals should match

### Past predictions showing but not aligned
**Issue**: Markers appear at wrong time positions
**Solution**: 
- Check timezone consistency
- Verify timestamp format in performance file
- Ensure chart labels match prediction timestamp format

### Performance file growing too large
**Issue**: `model_performance.jsonl` becomes very large
**Solution**:
- Implement log rotation (keep only last 1000 predictions)
- Archive old predictions to separate file
- Clear file periodically: `> artifacts/model_performance.jsonl`

## API Documentation

### GET `/get_past_predictions/<crypto_id>`
Retrieve past predictions for specified cryptocurrency.

**Parameters:**
- `crypto_id` (path): Cryptocurrency identifier (e.g., "solana")
- `max` (query, optional): Maximum predictions to return (default: 20)

**Response:**
```json
{
  "random_forest": [
    {
      "timestamp": "2025-10-24T02:06:36.997146",
      "price": 193.93,
      "actual": 193.35,
      "error": 0.3
    }
  ],
  "lstm": [...],
  "ensemble": [...],
  "actuals": [...]
}
```

**Example:**
```bash
curl http://localhost:8080/get_past_predictions/solana?max=10
```

## Maintenance

### Regular Tasks:
1. **Monitor performance file size**: Check `artifacts/model_performance.jsonl` size weekly
2. **Verify prediction accuracy**: Review chart weekly to ensure models performing well
3. **Fine-tune LSTM**: System auto-fine-tunes every 20 verified predictions
4. **Clear old data**: Archive predictions older than 30 days

### Backup:
```bash
# Backup performance history
cp artifacts/model_performance.jsonl backups/model_performance_$(date +%Y%m%d).jsonl

# Compress old backups
gzip backups/model_performance_*.jsonl
```

## Summary
This feature provides complete transparency into model predictions, allowing users to:
- See how each model (RF and LSTM) predicted prices historically
- Compare current predictions with past performance
- Validate the effectiveness of the ensemble approach
- Monitor LSTM fine-tuning improvements over time
- Make informed decisions based on historical model accuracy

The visualization makes it clear that Random Forest (green) is currently the superior model, which is why the ensemble weights it at 95%. As LSTM fine-tuning continues with more verified data, users will be able to observe the blue (LSTM) predictions improving on the chart.
