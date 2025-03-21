# Testing Guide: Past Predictions Display Feature

## Quick Start

### 1. Access the Application
Open your browser and navigate to:
```
http://localhost:8080
```

### 2. View the Enhanced Chart
1. Scroll down to the "Price Chart with AI Predictions" section
2. You'll see the chart with legend showing:
   - 📊 Historical Data (solid line)
   - 🔮 Ensemble Prediction (red dashed line)
   - 🌲 RF Past Predictions (green dashed line)
   - 🧠 LSTM Past Predictions (blue dashed line)

### 3. Generate New Predictions
1. Click the "Refresh Predictions" button
2. Wait for the chart to update
3. Observe the current ensemble prediction (red triangle marker)
4. Wait 5+ minutes for the prediction to be verified

### 4. Test Past Predictions Display
After a few predictions have been made and verified (3+ minutes after each):
1. Click "Refresh Predictions" again
2. You should now see:
   - Green square markers: Where Random Forest predicted prices would be
   - Blue circle markers: Where LSTM predicted prices would be
   - Red triangle: Current 5-minute ahead prediction

### 5. Toggle Predictions On/Off
1. Uncheck "Show Next 5-Min Prediction" checkbox
2. All prediction markers disappear, showing only historical data
3. Check the box again to show predictions

## Testing Scenarios

### Scenario 1: First-Time User (No Past Predictions)
**Expected Behavior:**
- Chart shows historical price data (purple/blue solid line)
- Current ensemble prediction shows (red triangle)
- No green or blue markers yet (no past predictions)
- After 5+ minutes, make another prediction
- Previous prediction will now appear as green/blue markers

**Verification:**
```bash
# Check how many predictions have been made
wc -l artifacts/model_performance.jsonl
```

### Scenario 2: After Multiple Predictions (5+ predictions)
**Expected Behavior:**
- Historical price line (solid)
- Current prediction (red triangle at future time)
- 5 green squares (RF past predictions)
- 5 blue circles (LSTM past predictions)
- All markers align with their predicted times

**Verification:**
```bash
# View last 5 predictions
tail -5 artifacts/model_performance.jsonl | jq '.rf_predicted, .lstm_predicted, .predicted'
```

### Scenario 3: After Verification (3+ minutes later)
**Expected Behavior:**
- Past predictions show accuracy
- Green/blue markers should be close to actual historical price
- Console logs show verification messages

**Verification:**
```bash
# Check verified predictions
grep -c '"actual":[0-9]' artifacts/model_performance.jsonl
```

### Scenario 4: Model Comparison
**Expected Behavior:**
- Green markers (RF) cluster tightly around actual prices
- Blue markers (LSTM) show more variance
- Visual confirmation that RF is more accurate

**Verification:**
- Hover over markers to see exact values
- Compare distances from solid line (actual prices)

## API Testing

### Test Backend Endpoints

#### 1. Get Past Predictions
```bash
# Get last 5 predictions
curl -s http://localhost:8080/get_past_predictions/solana?max=5 | python3 -m json.tool

# Expected output:
{
  "random_forest": [
    {"timestamp": "...", "price": 192.31, "actual": null, "error": null},
    ...
  ],
  "lstm": [...],
  "ensemble": [...],
  "actuals": [...]
}
```

#### 2. Get Current Prediction
```bash
# Get new prediction
curl -s http://localhost:8080/get_price_predictions/solana | python3 -m json.tool

# Check predictions section:
{
  "predictions": {
    "random_forest": {"price": 192.31, "change_percent": -0.045, ...},
    "lstm": {"price": 192.40, "change_percent": 0.000, ...},
    "ensemble": {"price": 192.31, "change_percent": -0.045, ...}
  }
}
```

#### 3. Check Performance File
```bash
# View last prediction with all fields
tail -1 artifacts/model_performance.jsonl | python3 -m json.tool

# Should include:
# - rf_predicted
# - lstm_predicted
# - predicted (ensemble)
# - actual (null until verified)
# - rf_error (null until verified)
# - lstm_error (null until verified)
# - prediction_time
```

## Browser Console Testing

### 1. Open Browser Console
- Chrome/Edge: F12 or Cmd+Option+I (Mac)
- Firefox: F12 or Cmd+Option+K (Mac)
- Safari: Cmd+Option+C (Mac)

### 2. Monitor Chart Updates
Look for these console log messages:
```
🔄 Updating chart for solana...
📜 Past predictions loaded: {random_forest: Array(5), lstm: Array(5), ...}
📊 Historical data response: {prices: Array(48), volumes: Array(48), ...}
🌲 Processing 5 RF past predictions
🧠 Processing 5 LSTM past predictions
📈 Dataset 0 (historical): 48 points
🤖 Dataset 1 (ensemble current): 2 points
🌲 Dataset 2 (RF past): 5 points
🧠 Dataset 3 (LSTM past): 5 points
✅ Chart updated successfully
```

### 3. Inspect Chart Datasets
```javascript
// In browser console
console.log(priceChart.data.datasets);

// Expected output:
// [
//   {label: "Solana Historical Price", data: [...], borderColor: "#9945ff", ...},
//   {label: "Solana Ensemble Prediction (5min)", data: [...], borderColor: "#ff6b6b", ...},
//   {label: "Solana RF Past Predictions", data: [...], borderColor: "#27ae60", ...},
//   {label: "Solana LSTM Past Predictions", data: [...], borderColor: "#3498db", ...}
// ]
```

### 4. Check Dataset Visibility
```javascript
// Check which datasets are visible
priceChart.data.datasets.forEach((ds, i) => {
  console.log(`Dataset ${i} (${ds.label}): hidden=${ds.hidden}`);
});
```

## Performance Verification

### 1. Check Prediction Accuracy

#### Via UI:
1. Scroll to "Model Performance Comparison" section
2. Look at "Real-Time Performance" section:
   - **Last Prediction Error**: Should be < 0.5% for good performance
   - **Average Error (Last 10)**: Should be < 0.5%
   - **Predictions Tracked**: Number of verified predictions

#### Via CLI:
```bash
# Calculate average error for last 10 verified predictions
tail -100 artifacts/model_performance.jsonl | \
  grep '"actual":[0-9]' | tail -10 | \
  python3 -c "
import sys, json
errors = [json.loads(line)['error_percent'] for line in sys.stdin]
print(f'Average error: {sum(errors)/len(errors):.3f}%')
print(f'RF errors: {[json.loads(line)[\"rf_error\"] for line in errors]}')
print(f'LSTM errors: {[json.loads(line)[\"lstm_error\"] for line in errors]}')
"
```

### 2. Model Comparison
```bash
# Compare RF vs LSTM accuracy
python3 << 'EOF'
import json

with open('artifacts/model_performance.jsonl', 'r') as f:
    lines = [json.loads(line) for line in f if json.loads(line).get('actual')]

rf_errors = [p['rf_error'] for p in lines if p.get('rf_error')]
lstm_errors = [p['lstm_error'] for p in lines if p.get('lstm_error')]

print(f"RF Average Error: {sum(rf_errors)/len(rf_errors):.3f}%")
print(f"LSTM Average Error: {sum(lstm_errors)/len(lstm_errors):.3f}%")
print(f"RF is better by: {(sum(lstm_errors)/len(lstm_errors) - sum(rf_errors)/len(rf_errors)):.3f}%")
EOF
```

## Common Issues and Solutions

### Issue 1: No Past Predictions Showing
**Symptoms:**
- Chart only shows historical data and current prediction
- No green/blue markers visible

**Solution:**
```bash
# Check if predictions have rf_predicted and lstm_predicted fields
tail -5 artifacts/model_performance.jsonl | jq '.rf_predicted, .lstm_predicted'

# If null, make new predictions:
curl http://localhost:8080/get_price_predictions/solana
```

### Issue 2: Predictions Not Aligning with Time
**Symptoms:**
- Green/blue markers appear at wrong times on chart
- Markers don't overlap with historical data

**Diagnosis:**
```bash
# Check timestamp format in performance file
tail -1 artifacts/model_performance.jsonl | jq '.timestamp, .prediction_time'

# Check browser console for time parsing errors
# Look for: "Processing X RF past predictions" but seeing 0 points on chart
```

**Solution:**
- Ensure prediction_time is 5 minutes ahead of timestamp
- Verify timezone consistency
- Check that chart labels match time format (HH:MM)

### Issue 3: Chart Toggle Not Working
**Symptoms:**
- Clicking "Show Next 5-Min Prediction" doesn't hide/show markers

**Solution:**
```javascript
// In browser console, manually toggle:
priceChart.data.datasets.forEach((ds, i) => {
  if (i > 0) ds.hidden = !ds.hidden;
});
priceChart.update();
```

### Issue 4: Server Not Running
**Symptoms:**
- Page won't load
- API calls fail with connection errors

**Solution:**
```bash
# Check if server is running
lsof -i :8080

# If not running, start it:
cd /Users/philipt/VinoPredict
source cryptopredict_env/bin/activate
python3 app.py
```

## Automated Testing Script

Create this test script to verify everything works:

```bash
#!/bin/bash
# test_past_predictions.sh

echo "=== Testing Past Predictions Feature ==="

# 1. Check server
echo -e "\n1. Checking server status..."
if curl -s http://localhost:8080/ > /dev/null; then
    echo "✅ Server is running"
else
    echo "❌ Server is not running"
    exit 1
fi

# 2. Make prediction
echo -e "\n2. Making new prediction..."
PRED=$(curl -s http://localhost:8080/get_price_predictions/solana)
CURRENT=$(echo $PRED | jq -r '.current_price')
RF=$(echo $PRED | jq -r '.predictions.random_forest.price')
LSTM=$(echo $PRED | jq -r '.predictions.lstm.price')
echo "Current: \$$CURRENT"
echo "RF: \$$RF"
echo "LSTM: \$$LSTM"

# 3. Check storage
echo -e "\n3. Checking prediction storage..."
LAST=$(tail -1 artifacts/model_performance.jsonl)
if echo $LAST | jq -e '.rf_predicted' > /dev/null; then
    echo "✅ RF prediction stored"
else
    echo "❌ RF prediction not stored"
fi
if echo $LAST | jq -e '.lstm_predicted' > /dev/null; then
    echo "✅ LSTM prediction stored"
else
    echo "❌ LSTM prediction not stored"
fi

# 4. Test API endpoint
echo -e "\n4. Testing past predictions endpoint..."
PAST=$(curl -s "http://localhost:8080/get_past_predictions/solana?max=3")
RF_COUNT=$(echo $PAST | jq '.random_forest | length')
LSTM_COUNT=$(echo $PAST | jq '.lstm | length')
echo "RF past predictions: $RF_COUNT"
echo "LSTM past predictions: $LSTM_COUNT"

# 5. Check verified predictions
echo -e "\n5. Checking verified predictions..."
VERIFIED=$(grep -c '"actual":[0-9]' artifacts/model_performance.jsonl)
echo "Verified predictions: $VERIFIED"

echo -e "\n=== Test Complete ==="
```

Run it:
```bash
chmod +x test_past_predictions.sh
./test_past_predictions.sh
```

## Visual Verification Checklist

✅ **Chart displays correctly**
- [ ] Historical price line visible (solid, colored)
- [ ] Current ensemble prediction visible (red dashed, triangle)
- [ ] RF past predictions visible (green dashed, squares)
- [ ] LSTM past predictions visible (blue dashed, circles)
- [ ] All datasets have proper legends

✅ **Toggle works**
- [ ] Unchecking hides all prediction datasets
- [ ] Checking shows all prediction datasets
- [ ] Historical data remains visible

✅ **Time alignment**
- [ ] Past predictions align with correct times on x-axis
- [ ] Current prediction is 5 minutes ahead
- [ ] No overlapping markers

✅ **Model comparison**
- [ ] RF markers (green) closer to actual prices
- [ ] LSTM markers (blue) show more variance
- [ ] Visual difference reflects R² scores

✅ **Data accuracy**
- [ ] Hover shows correct price values
- [ ] Tooltips display model names
- [ ] Values match API responses

## Next Steps After Testing

1. **Monitor for 24 hours**: Let system accumulate 20+ predictions
2. **Observe verification**: Check that predictions get verified after 3+ minutes
3. **Watch fine-tuning**: After 20 verified, LSTM should auto fine-tune
4. **Compare before/after**: LSTM blue markers should improve post-fine-tuning
5. **Export data**: Save performance file for analysis

## Documentation References

- **Feature Overview**: See `PAST_PREDICTIONS_DISPLAY.md`
- **Fine-tuning Guide**: See `LSTM_FINETUNING.md`
- **Data Flow**: See `DATA_FLOW_DOCUMENTATION.md`

## Support

If you encounter issues not covered here:
1. Check browser console for JavaScript errors
2. Check Flask logs for Python errors
3. Verify all dependencies are installed
4. Ensure model artifacts exist in `artifacts/` directory
5. Review performance file format

Happy testing! 🚀
