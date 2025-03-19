from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import json
import os
import logging
import time
import requests
from mlProject.pipeline.unified_prediction import UnifiedPredictionPipeline
from mlProject.components.feature_engineering import FeatureEngineering
from mlProject import logger
from mlProject.config.configuration import ConfigurationManager
from mlProject.components.lstm_finetuner import LSTMFineTuner

app = Flask(__name__)  # initializing a flask app

# Constants
PERFORMANCE_FILE = 'artifacts/model_performance.jsonl'
API_CACHE = {}
CACHE_DURATION = 300  # 5 minute cache
RATE_LIMIT_DELAY = 1.2  # 1.2 seconds between API calls
MIN_VERIFICATION_DELAY = 180  # 3 minutes
FINETUNE_THRESHOLD = 20  # Fine-tune after 20 verified predictions
last_api_call = 0
last_finetune_count = 0

# Initialize feature engineering component (shared across all requests)
feature_engineer = FeatureEngineering()

def get_cached_api_data(url, params, cache_key):
    """Get data from API with enhanced caching and rate limiting"""
    global last_api_call
    current_time = time.time()
    
    # Check if we have cached data that's still valid
    if cache_key in API_CACHE:
        cached_time, cached_data = API_CACHE[cache_key]
        if current_time - cached_time < CACHE_DURATION:
            logger.info(f"Using cached data for {cache_key}")
            return cached_data
    
    # Rate limiting: ensure minimum delay between API calls
    time_since_last_call = current_time - last_api_call
    if time_since_last_call < RATE_LIMIT_DELAY:
        sleep_time = RATE_LIMIT_DELAY - time_since_last_call
        logger.info(f"Rate limiting: sleeping {sleep_time:.2f} seconds")
        time.sleep(sleep_time)
    
    # Make API call with enhanced error handling
    try:
        logger.info(f"Making API call to {url} with params {params}")
        response = requests.get(url, params=params, timeout=15)
        last_api_call = time.time()
        
        if response.status_code == 429:
            logger.warning("Rate limited by API. Using cached data if available.")
            if cache_key in API_CACHE:
                _, cached_data = API_CACHE[cache_key]
                return cached_data
            else:
                # Wait longer and try once more
                time.sleep(5)
                response = requests.get(url, params=params, timeout=15)
                last_api_call = time.time()
        
        response.raise_for_status()
        data = response.json()
        
        # Validate response has expected structure
        if not data:
            logger.warning(f"Empty response from API for {cache_key}")
            if cache_key in API_CACHE:
                _, cached_data = API_CACHE[cache_key]
                return cached_data
            return {}
        
        # Cache the successful response
        API_CACHE[cache_key] = (current_time, data)
        logger.info(f"Successfully cached data for {cache_key}")
        return data
        
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error for {cache_key}: {e}")
        if cache_key in API_CACHE:
            logger.info(f"Using expired cache for {cache_key}")
            _, cached_data = API_CACHE[cache_key]
            return cached_data
        return {}
    except Exception as e:
        logger.error(f"API error for {cache_key}: {str(e)}")
        if cache_key in API_CACHE:
            logger.info(f"Using expired cache for {cache_key}")
            _, cached_data = API_CACHE[cache_key]
            return cached_data
        return {}

# Configure logging to file only (not stdout) to avoid mixing with JSON responses
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        # Remove StreamHandler to prevent console output
    ]
)


@app.route('/', methods=['GET'])  # route to display the home page
def home_page():
    return render_template("index.html")


@app.route('/get_cryptocurrencies')
def get_configured_cryptocurrencies():
    """Get the list of configured cryptocurrencies"""
    try:
        config = ConfigurationManager()
        cryptocurrencies = config.get_cryptocurrencies()
        return jsonify({"cryptocurrencies": cryptocurrencies})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/get_crypto_data/<crypto_id>')
def get_live_crypto_data(crypto_id):
    """Get live crypto data for display"""
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            'ids': crypto_id,
            'vs_currencies': 'usd',
            'include_24hr_change': 'true',
            'include_market_cap': 'true',
            'include_24hr_vol': 'true'
        }
        cache_key = f"crypto_data_{crypto_id}"
        data = get_cached_api_data(url, params, cache_key)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/get_5min_data/<crypto_id>', methods=['GET'])
def get_5min_data(crypto_id):
    """Get 5-minute granular price data for the last 4 hours"""
    try:
        logger.info(f"Fetching 5-minute granular data for {crypto_id}")
        
        # CoinGecko API: days=1 with interval=5minute gives 5-min data
        url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
        # Use 0.2 days (~4.8 hours) to get approximately 4 hours of 5-minute data
        params = {'vs_currency': 'usd', 'days': '0.2'}
        
        cache_key = f"5min_data_{crypto_id}"
        hist_data = get_cached_api_data(url, params, cache_key)
        
        if 'prices' not in hist_data or not hist_data['prices']:
            return jsonify({"error": "No price data available", "prices": [], "volumes": []})
        
        # Process last 4 hours (48 x 5-minute intervals)
        processed_prices = []
        processed_volumes = []
        
        for price_data in hist_data.get('prices', [])[-48:]:
            if len(price_data) >= 2 and price_data[1] is not None:
                processed_prices.append([int(price_data[0]), float(price_data[1])])
        
        for vol_data in hist_data.get('total_volumes', [])[-48:]:
            if len(vol_data) >= 2 and vol_data[1] is not None:
                processed_volumes.append([int(vol_data[0]), float(vol_data[1])])
        
        return jsonify({
            'prices': processed_prices,
            'volumes': processed_volumes,
            'crypto_id': crypto_id,
            'interval': '5min',
            'hours': 4
        })
        
    except Exception as e:
        logger.exception(f"Error fetching 5-minute data: {str(e)}")
        return jsonify({"error": str(e), "prices": [], "volumes": []})


@app.route('/get_historical_data/<crypto_id>', methods=['GET'])
def get_historical_data(crypto_id):
    """Get historical price data for cryptocurrency with enhanced error handling"""
    try:
        days = request.args.get('days', default=7, type=int)
        logger.info(f"Fetching {days} days of historical data for {crypto_id}")
        
        url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
        params = {'vs_currency': 'usd', 'days': days}
        
        cache_key = f"historical_data_{crypto_id}_{days}"
        hist_data = get_cached_api_data(url, params, cache_key)
        
        # Validate response data
        if 'prices' not in hist_data or not hist_data['prices']:
            return jsonify({"error": "No price data available", "prices": [], "volumes": []})
        
        # Process data
        processed_prices = [[int(p[0]), float(p[1])] for p in hist_data.get('prices', []) if len(p) >= 2 and p[1] is not None]
        processed_volumes = [[int(v[0]), float(v[1])] for v in hist_data.get('total_volumes', []) if len(v) >= 2 and v[1] is not None]
        
        return jsonify({
            'prices': processed_prices,
            'volumes': processed_volumes,
            'crypto_id': crypto_id,
            'days': days
        })
        
    except Exception as e:
        logger.exception(f"Error fetching historical data: {str(e)}")
        return jsonify({"error": str(e), "prices": [], "volumes": []})


def _extract_price_data(hist_data, crypto_id, current_price):
    """Extract and process price data from historical data"""
    if not hist_data or 'prices' not in hist_data or not hist_data.get('prices'):
        logger.warning(f"No historical price data available for {crypto_id}, using current price as fallback")
        return [current_price * (1 + (0.02 * (0.5 - (i % 10) / 10))) for i in range(24)]
    
    try:
        price_data = hist_data.get('prices', [])
        prices = [item[1] for item in price_data[-24:] if len(item) > 1 and item[1] is not None] if price_data else []
        while len(prices) < 24:
            prices.insert(0, current_price)
        return prices
    except (IndexError, TypeError, KeyError) as e:
        logger.error(f"Error processing historical data for {crypto_id}: {e}")
        return [current_price] * 24


def _calculate_technical_indicators(prices, current_price):
    """
    DEPRECATED: Use FeatureEngineering.calculate_technical_indicators() instead
    
    Calculate technical indicators from price history - focusing on MACD, RSI, Volume
    This function is kept for backward compatibility only.
    """
    indicators = {}
    
    # Moving averages
    indicators['sma_7'] = sum(prices[-7:]) / 7 if len(prices) >= 7 else current_price
    indicators['sma_14'] = sum(prices[-14:]) / 14 if len(prices) >= 14 else current_price
    indicators['ema_7'] = prices[-1] if prices else current_price
    
    # MACD (Moving Average Convergence Divergence) - KEY INDICATOR
    if len(prices) >= 26:
        # Calculate EMA-12 and EMA-26
        ema_12 = prices[-1]  # Simplified for real-time
        ema_26 = sum(prices[-26:]) / 26
        for i in range(-26, 0):
            alpha_12 = 2 / (12 + 1)
            alpha_26 = 2 / (26 + 1)
            ema_12 = prices[i] * alpha_12 + ema_12 * (1 - alpha_12)
            ema_26 = prices[i] * alpha_26 + ema_26 * (1 - alpha_26)
        
        indicators['macd'] = ema_12 - ema_26
        # MACD signal line (9-period EMA of MACD)
        indicators['macd_signal'] = indicators['macd'] * 0.9  # Simplified
        indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
    else:
        indicators['macd'] = 0.0
        indicators['macd_signal'] = 0.0
        indicators['macd_histogram'] = 0.0
    
    # RSI (Relative Strength Index) - KEY INDICATOR
    if len(prices) >= 14:
        gains, losses = [], []
        for i in range(1, min(15, len(prices))):
            change = prices[-i] - prices[-i-1]
            gains.append(max(0, change))
            losses.append(abs(min(0, change)))
        
        avg_gain = sum(gains) / len(gains) if gains else 0.01
        avg_loss = sum(losses) / len(losses) if losses else 0.01
        rs = avg_gain / avg_loss if avg_loss > 0 else 1
        indicators['rsi'] = 100 - (100 / (1 + rs))
    else:
        indicators['rsi'] = 50
    
    # Bollinger Bands
    if len(prices) >= 20:
        sma_20 = sum(prices[-20:]) / 20
        variance = sum([(price - sma_20) ** 2 for price in prices[-20:]]) / 20
        std_dev = variance ** 0.5
        indicators['bb_upper'] = sma_20 + (2 * std_dev)
        indicators['bb_lower'] = sma_20 - (2 * std_dev)
    else:
        indicators['bb_upper'] = current_price * 1.05
        indicators['bb_lower'] = current_price * 0.95
    
    # Volatility
    if len(prices) >= 2:
        price_changes = [abs(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        indicators['volatility'] = sum(price_changes) / len(price_changes)
    else:
        indicators['volatility'] = 0.02
    
    # Price position
    if len(prices) >= 14:
        high_14d = max(prices[-14:])
        low_14d = min(prices[-14:])
        indicators['price_position'] = (current_price - low_14d) / (high_14d - low_14d) if high_14d != low_14d else 0.5
    else:
        indicators['price_position'] = 0.5
    
    return indicators


def _prepare_prediction_input(current_price, current_volume, current_market_cap, indicators, market_data, crypto_id, prices):
    """
    DEPRECATED: Use FeatureEngineering.prepare_features() instead
    
    Prepare input features for model prediction - 29 features (target columns excluded during training)
    This function is kept for backward compatibility only.
    """
    price_change_1h = (current_price - prices[-2]) / prices[-2] if len(prices) >= 2 else 0.0
    price_change_24h = (market_data[crypto_id].get('usd_24h_change', 0.0) or 0.0) / 100.0
    price_change_7d = (current_price - prices[-7]) / prices[-7] if len(prices) >= 7 else 0.0
    
    volume_sma = current_volume
    volume_ratio = 1.0
    
    # 32 input features (excluding target_price_5min which is dropped during model training)
    # Training process: CSV has 33 cols -> drops target_price_5min -> 32 features for model
    input_features = [
        current_price,                # 0: price
        current_volume,               # 1: volume
        current_market_cap,           # 2: market_cap
        indicators['sma_7'],          # 3: sma_7
        indicators['sma_14'],         # 4: sma_14
        indicators['sma_14'],         # 5: sma_30 (approximation)
        indicators['ema_7'],          # 6: ema_7
        indicators['ema_7'],          # 7: ema_14 (approximation)
        indicators['macd'],           # 8: macd
        indicators['macd_signal'],    # 9: macd_signal
        indicators['macd_histogram'], # 10: macd_histogram
        indicators['rsi'],            # 11: rsi
        indicators['sma_14'],         # 12: bb_middle
        indicators['bb_upper'],       # 13: bb_upper
        indicators['bb_lower'],       # 14: bb_lower
        price_change_1h,              # 15: price_change_1h
        price_change_24h,             # 16: price_change_24h
        price_change_7d,              # 17: price_change_7d
        volume_sma,                   # 18: volume_sma
        volume_ratio,                 # 19: volume_ratio
        indicators['volatility'],     # 20: volatility
        max(prices[-14:]) if len(prices) >= 14 else current_price,  # 21: high_14d
        min(prices[-14:]) if len(prices) >= 14 else current_price,  # 22: low_14d
        indicators['price_position'], # 23: price_position
        # Note: target_price_1h (index 24) is excluded - it's the target, not a feature
        current_price,                # 24: target_price_24h (secondary target, kept as feature)
        0,                            # 25: target_direction_1h
        0,                            # 26: target_direction_24h
        0.0,                          # 27: target_change_1h
        0.0                           # 28: target_change_24h
    ]
    return np.array(input_features).reshape(1, -1)


def _check_and_trigger_finetuning(performance_lines):
    """
    Check if LSTM fine-tuning should be triggered based on verified predictions
    
    Args:
        performance_lines: Lines from performance file
    """
    global last_finetune_count
    
    try:
        # Count verified predictions
        verified_count = sum(1 for line in performance_lines 
                           if json.loads(line).get('actual') is not None)
        
        # Trigger fine-tuning if we have enough new verified predictions
        if verified_count - last_finetune_count >= FINETUNE_THRESHOLD:
            logger.info(f"🔄 Triggering LSTM fine-tuning ({verified_count} verified predictions)")
            
            # Run fine-tuning in background (non-blocking)
            import threading
            thread = threading.Thread(target=_run_lstm_finetuning, args=(PERFORMANCE_FILE,))
            thread.daemon = True
            thread.start()
            
            last_finetune_count = verified_count
            
    except Exception as e:
        logger.error(f"Error checking fine-tuning trigger: {e}")


def _run_lstm_finetuning(performance_file: str):
    """Run LSTM fine-tuning in background thread"""
    try:
        config = ConfigurationManager()
        deep_trainer_config = config.get_deep_model_trainer_config()
        
        finetuner = LSTMFineTuner(deep_trainer_config)
        success = finetuner.finetune_from_performance_file(
            performance_file=performance_file,
            min_samples=10
        )
        
        if success:
            logger.info("✅ LSTM fine-tuning completed successfully")
        else:
            logger.warning("⚠️ LSTM fine-tuning skipped (insufficient data or error)")
            
    except Exception as e:
        logger.error(f"Error in background fine-tuning: {e}")


def _generate_ensemble_predictions(input_data, current_price, volatility, current_time):
    """Generate 5-minute predictions - returns separate RF, LSTM, and ensemble predictions"""
    traditional_obj = UnifiedPredictionPipeline(model_type='traditional')
    deep_obj = UnifiedPredictionPipeline(model_type='deep_learning')
    
    # Get predictions with multiple variations for robustness
    trad_predictions = []
    deep_predictions = []
    
    rng = np.random.default_rng(int(current_time.timestamp()))
    for _ in range(3):  # 3 ensemble members
        noise = rng.random() * 0.0005 - 0.00025  # ±0.025% noise
        varied_input = input_data.copy()
        varied_input[0][0] = current_price * (1 + noise)
        
        try:
            trad_pred = traditional_obj.predict(varied_input)
            if len(trad_pred) > 0 and not np.isnan(trad_pred[0]):
                trad_predictions.append(float(trad_pred[0]))
        except Exception:
            pass
        
        try:
            deep_pred = deep_obj.predict(varied_input)
            if len(deep_pred) > 0 and not np.isnan(deep_pred[0]):
                deep_predictions.append(float(deep_pred[0]))
        except Exception:
            pass
    
    # Calculate individual model predictions
    rf_price = np.median(trad_predictions) if trad_predictions else current_price
    lstm_price = np.median(deep_predictions) if deep_predictions else current_price
    
    # Weighted ensemble: Random Forest R²=0.9999 vs LSTM R²=-0.0002 (poor performance)
    # Use 95% Random Forest since LSTM is currently unreliable
    if trad_predictions and deep_predictions:
        ensemble_price = 0.95 * rf_price + 0.05 * lstm_price  # 95% RF, 5% LSTM
    elif trad_predictions:
        ensemble_price = rf_price
    elif deep_predictions:
        ensemble_price = lstm_price
    else:
        ensemble_price = current_price
    
    # Apply reasonable bounds based on volatility
    price_change_24h = float(input_data[0][16])  # 24h change
    adaptive_max = 0.005 * (1 + abs(price_change_24h) * 0.3)
    adaptive_max = min(adaptive_max, 0.015)  # Cap at 1.5%
    
    # Clip predictions
    rf_price = np.clip(rf_price, current_price * (1 - adaptive_max), current_price * (1 + adaptive_max))
    lstm_price = np.clip(lstm_price, current_price * (1 - adaptive_max), current_price * (1 + adaptive_max))
    ensemble_price = np.clip(ensemble_price, current_price * (1 - adaptive_max), current_price * (1 + adaptive_max))
    
    pred_time = current_time + pd.Timedelta(minutes=5)
    
    # Return detailed predictions for both models
    return {
        'random_forest': {
            'price': float(rf_price),
            'change_percent': ((rf_price - current_price) / current_price) * 100,
            'confidence': 0.95 if trad_predictions else 0.70
        },
        'lstm': {
            'price': float(lstm_price),
            'change_percent': ((lstm_price - current_price) / current_price) * 100,
            'confidence': 0.75 if deep_predictions else 0.60
        },
        'ensemble': {
            'price': float(ensemble_price),
            'change_percent': ((ensemble_price - current_price) / current_price) * 100,
            'confidence': 0.95 if (trad_predictions and deep_predictions) else 0.75,
            'weight_rf': 0.95,
            'weight_lstm': 0.05
        },
        'timestamp': pred_time.isoformat(),
        'interval': '+5min'
    }


@app.route('/get_price_predictions/<crypto_id>')
def get_price_predictions(crypto_id):
    """Generate price predictions for the next hour in 15-minute intervals"""
    try:
        config = ConfigurationManager()
        if crypto_id not in config.get_cryptocurrencies():
            return jsonify({"error": f"Cryptocurrency {crypto_id} not supported"})
        
        market_url = "https://api.coingecko.com/api/v3/simple/price"
        market_params = {
            'ids': crypto_id, 'vs_currencies': 'usd',
            'include_24hr_change': 'true', 'include_market_cap': 'true', 'include_24hr_vol': 'true'
        }
        cache_key = f"crypto_data_{crypto_id}"
        market_data = get_cached_api_data(market_url, market_params, cache_key)
        
        if not market_data or crypto_id not in market_data:
            logger.error(f"crypto_id '{crypto_id}' not found in market_data: {market_data}")
            return jsonify({"error": f"No market data available for {crypto_id}"})
        
        current_price = market_data[crypto_id]['usd']
        current_volume = market_data[crypto_id].get('usd_24h_vol', 1000000)
        current_market_cap = market_data[crypto_id].get('usd_market_cap', current_price * 1000000)
        
        hist_url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
        hist_params = {'vs_currency': 'usd', 'days': '2'}
        hist_cache_key = f"historical_data_{crypto_id}_2"
        hist_data = get_cached_api_data(hist_url, hist_params, hist_cache_key)
        
        prices = _extract_price_data(hist_data, crypto_id, current_price)
        
        # Use shared FeatureEngineering component
        price_change_24h = market_data[crypto_id].get('usd_24h_change', 0.0)
        volumes = [current_volume] * len(prices)  # Simplified volume history
        input_data = feature_engineer.prepare_features(
            current_price=current_price,
            current_volume=current_volume,
            current_market_cap=current_market_cap,
            prices=prices,
            volumes=volumes,
            price_change_24h=price_change_24h
        )
        
        # Calculate volatility for ensemble predictions
        volatility = feature_engineer.calculate_volatility(prices)
        
        # Generate predictions
        current_time = pd.Timestamp.now()
        try:
            predictions = _generate_ensemble_predictions(input_data, current_price, volatility, current_time)
        except Exception as pred_error:
            logger.error(f"Prediction error: {str(pred_error)}")
            # Fallback: trend-based predictions
            trend = market_data[crypto_id].get('usd_24h_change', 0) or 0
            predictions = {
                'random_forest': {'price': current_price * (1 + trend/100/12), 'change_percent': trend/12, 'confidence': 0.5},
                'lstm': {'price': current_price * (1 + trend/100/12), 'change_percent': trend/12, 'confidence': 0.5},
                'ensemble': {'price': current_price * (1 + trend/100/12), 'change_percent': trend/12, 'confidence': 0.5},
                'timestamp': (current_time + pd.Timedelta(minutes=5)).isoformat(),
                'interval': '+5min'
            }
        
        os.makedirs(os.path.dirname(PERFORMANCE_FILE), exist_ok=True)
        
        # Verify previous prediction and trigger fine-tuning if needed
        try:
            if os.path.exists(PERFORMANCE_FILE):
                with open(PERFORMANCE_FILE, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last_pred = json.loads(lines[-1])
                        last_time = pd.Timestamp(last_pred['timestamp'])
                        time_elapsed = (current_time - last_time).total_seconds()
                        
                        if (last_pred['crypto_id'] == crypto_id and time_elapsed >= MIN_VERIFICATION_DELAY):
                            error_percent = abs(current_price - last_pred['predicted']) / current_price * 100
                            
                            # Calculate individual model errors
                            rf_error = None
                            lstm_error = None
                            if 'rf_predicted' in last_pred:
                                rf_error = round(abs(current_price - last_pred['rf_predicted']) / current_price * 100, 3)
                            if 'lstm_predicted' in last_pred:
                                lstm_error = round(abs(current_price - last_pred['lstm_predicted']) / current_price * 100, 3)
                            
                            last_pred.update({
                                'actual': current_price,
                                'error_percent': round(error_percent, 3),
                                'rf_error': rf_error,
                                'lstm_error': lstm_error,
                                'verified_at': current_time.isoformat()
                            })
                            lines[-1] = json.dumps(last_pred) + '\n'
                            with open(PERFORMANCE_FILE, 'w') as fw:
                                fw.writelines(lines)
                            logger.info(f"✅ Verified - Ensemble: {error_percent:.3f}%, RF: {rf_error}%, LSTM: {lstm_error}%")
                            
                            # Check if we should trigger LSTM fine-tuning
                            _check_and_trigger_finetuning(lines)
        except Exception as verify_error:
            logger.warning(f"Could not verify last prediction: {verify_error}")
        
        # Store new prediction (will be verified next time)
        if predictions and isinstance(predictions, dict) and 'ensemble' in predictions:
            new_prediction = {
                'crypto_id': crypto_id,
                'predicted': predictions['ensemble']['price'],
                'rf_predicted': predictions['random_forest']['price'],
                'lstm_predicted': predictions['lstm']['price'],
                'actual': None,
                'error_percent': None,
                'rf_error': None,
                'lstm_error': None,
                'timestamp': current_time.isoformat(),
                'current_price': current_price,
                'prediction_time': predictions['timestamp']
            }
            with open(PERFORMANCE_FILE, 'a') as f:
                f.write(json.dumps(new_prediction) + '\n')
        
        return jsonify({
            'crypto_id': crypto_id,
            'current_price': current_price,
            'predictions': predictions,
            'timestamp': current_time.isoformat()
        })
        
    except Exception as e:
        logger.exception(f"Error generating predictions: {str(e)}")
        return jsonify({"error": str(e)})


@app.route('/get_model_metrics')
def get_model_metrics():
    """Get model accuracy metrics for both Random Forest and LSTM"""
    try:
        metrics = {
            'random_forest': {},
            'lstm': {},
            'comparison': {}
        }
        
        # Load Random Forest metrics
        rf_metrics_path = 'artifacts/model_evaluation/metrics.json'
        if os.path.exists(rf_metrics_path):
            with open(rf_metrics_path, 'r') as f:
                rf_data = json.load(f)
                metrics['random_forest'] = {
                    'r2_score': round(rf_data.get('r2', 0), 6),
                    'rmse': round(rf_data.get('rmse', 0), 4),
                    'mae': round(rf_data.get('mae', 0), 4),
                    'model_type': 'Random Forest'
                }
        
        # Load LSTM/Deep learning model metrics
        lstm_metrics_path = 'artifacts/deep_model_evaluation/metrics.json'
        if os.path.exists(lstm_metrics_path):
            with open(lstm_metrics_path, 'r') as f:
                lstm_data = json.load(f)
                metrics['lstm'] = {
                    'r2_score': round(lstm_data.get('r2', 0), 6),
                    'rmse': round(lstm_data.get('rmse', 0), 4),
                    'mae': round(lstm_data.get('mae', 0), 4),
                    'model_type': lstm_data.get('model_type', 'LSTM')
                }
        
        # Training data size
        train_path = 'artifacts/data_transformation/train.csv'
        if os.path.exists(train_path):
            train_df = pd.read_csv(train_path)
            metrics['training_samples'] = len(train_df)
        
        # Comparison and ensemble info
        if metrics['random_forest'] and metrics['lstm']:
            metrics['comparison'] = {
                'better_model': 'Random Forest' if metrics['random_forest']['r2_score'] > metrics['lstm']['r2_score'] else 'LSTM',
                'r2_difference': round(abs(metrics['random_forest']['r2_score'] - metrics['lstm']['r2_score']), 6),
                'ensemble_weights': {
                    'random_forest': 0.95,
                    'lstm': 0.05
                },
                'note': 'LSTM currently has poor performance (negative R²). Using 95% Random Forest weighting.'
            }
        
        # Real-time performance tracking
        if os.path.exists(PERFORMANCE_FILE):
            with open(PERFORMANCE_FILE, 'r') as f:
                errors = [json.loads(line)['error_percent'] 
                         for line in f 
                         if json.loads(line).get('error_percent') is not None]
                
                errors = errors[-10:]  # Last 10 verified predictions
                metrics['real_time_performance'] = {
                    'last_error': round(errors[-1], 3) if errors else 0,
                    'avg_error_10': round(sum(errors) / len(errors), 3) if errors else 0,
                    'predictions_tracked': len(errors),
                    'error_history': [round(e, 2) for e in errors] if errors else []
                }
        
        return jsonify(metrics)
    except Exception as e:
        logger.exception(f"Error getting model metrics: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/finetune_model', methods=['POST'])
def finetune_model():
    """Fine-tune model with actual price data from last prediction"""
    try:
        data = request.get_json()
        crypto_id = data.get('crypto_id')
        predicted_price = float(data.get('predicted_price'))
        actual_price = float(data.get('actual_price'))
        
        error = abs(actual_price - predicted_price) / actual_price * 100
        logger.info(f"Fine-tuning {crypto_id}: error={error:.2f}%")
        
        performance_data = {
            'crypto_id': crypto_id,
            'predicted': predicted_price,
            'actual': actual_price,
            'error_percent': error,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(PERFORMANCE_FILE), exist_ok=True)
        with open(PERFORMANCE_FILE, 'a') as f:
            f.write(json.dumps(performance_data) + '\n')
        
        return jsonify({
            'status': 'success',
            'error_percent': error,
            'message': f'Model performance logged. Error: {error:.2f}%'
        })
    except Exception as e:
        logger.exception(f"Error in fine-tuning: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/finetune_lstm', methods=['POST'])
def finetune_lstm_manual():
    """Manually trigger LSTM fine-tuning"""
    try:
        min_samples = request.args.get('min_samples', default=10, type=int)
        
        if not os.path.exists(PERFORMANCE_FILE):
            return jsonify({
                'status': 'error',
                'message': 'No performance data available for fine-tuning'
            }), 400
        
        # Count verified predictions
        with open(PERFORMANCE_FILE, 'r') as f:
            verified_count = sum(1 for line in f if json.loads(line).get('actual') is not None)
        
        if verified_count < min_samples:
            return jsonify({
                'status': 'error',
                'message': f'Insufficient verified predictions: {verified_count} < {min_samples}',
                'verified_predictions': verified_count,
                'required': min_samples
            }), 400
        
        logger.info(f"🔄 Manual LSTM fine-tuning triggered with {verified_count} verified predictions")
        
        # Run fine-tuning synchronously for manual trigger
        config = ConfigurationManager()
        deep_trainer_config = config.get_deep_model_trainer_config()
        
        finetuner = LSTMFineTuner(deep_trainer_config)
        success = finetuner.finetune_from_performance_file(
            performance_file=PERFORMANCE_FILE,
            min_samples=min_samples
        )
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'LSTM fine-tuning completed successfully',
                'verified_predictions_used': verified_count
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'LSTM fine-tuning failed',
                'verified_predictions': verified_count
            }), 500
            
    except Exception as e:
        logger.exception(f"Error in manual LSTM fine-tuning: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/get_finetuning_status')
def get_finetuning_status():
    """Get LSTM fine-tuning status and statistics"""
    try:
        status = {
            'auto_finetune_enabled': True,
            'finetune_threshold': FINETUNE_THRESHOLD,
            'last_finetune_count': last_finetune_count,
            'verified_predictions': 0,
            'pending_for_finetune': 0
        }
        
        if os.path.exists(PERFORMANCE_FILE):
            with open(PERFORMANCE_FILE, 'r') as f:
                lines = f.readlines()
                verified = sum(1 for line in lines if json.loads(line).get('actual') is not None)
                status['verified_predictions'] = verified
                status['pending_for_finetune'] = max(0, FINETUNE_THRESHOLD - (verified - last_finetune_count))
        
        return jsonify(status)
    except Exception as e:
        logger.exception(f"Error getting fine-tuning status: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/get_past_predictions/<crypto_id>')
def get_past_predictions(crypto_id):
    """Get past predictions for both RF and LSTM models for charting"""
    try:
        max_predictions = request.args.get('max', default=20, type=int)
        
        past_predictions = {
            'random_forest': [],
            'lstm': [],
            'ensemble': [],
            'actuals': []
        }
        
        if not os.path.exists(PERFORMANCE_FILE):
            return jsonify(past_predictions)
        
        # Read all predictions for this crypto
        with open(PERFORMANCE_FILE, 'r') as f:
            lines = f.readlines()
        
        # Filter for this crypto and get recent predictions (including the latest unverified one)
        crypto_predictions = []
        for line in lines:
            try:
                pred = json.loads(line)
                if pred.get('crypto_id') == crypto_id:
                    crypto_predictions.append(pred)
            except json.JSONDecodeError:
                continue
        
        # Get last N predictions (this now includes the most recent prediction)
        recent_predictions = crypto_predictions[-max_predictions:]
        
        for pred in recent_predictions:
            timestamp = pred.get('prediction_time') or pred.get('timestamp')
            
            # Random Forest predictions
            if 'rf_predicted' in pred:
                past_predictions['random_forest'].append({
                    'timestamp': timestamp,
                    'price': pred['rf_predicted'],
                    'actual': pred.get('actual'),
                    'error': pred.get('rf_error')
                })
            
            # LSTM predictions
            if 'lstm_predicted' in pred:
                past_predictions['lstm'].append({
                    'timestamp': timestamp,
                    'price': pred['lstm_predicted'],
                    'actual': pred.get('actual'),
                    'error': pred.get('lstm_error')
                })
            
            # Ensemble predictions
            if 'predicted' in pred:
                past_predictions['ensemble'].append({
                    'timestamp': timestamp,
                    'price': pred['predicted'],
                    'actual': pred.get('actual'),
                    'error': pred.get('error_percent')
                })
            
            # Actual prices (for verified predictions)
            if pred.get('actual') is not None:
                past_predictions['actuals'].append({
                    'timestamp': pred.get('verified_at') or timestamp,
                    'price': pred['actual']
                })
        
        return jsonify(past_predictions)
        
    except Exception as e:
        logger.exception(f"Error getting past predictions: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/get_historical_predictions/<crypto_id>')
def get_historical_predictions(crypto_id):
    """Generate predictions for each historical data point"""
    try:
        # Get 5-minute historical data
        hist_url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
        hist_params = {'vs_currency': 'usd', 'days': '0.2'}  # Last ~5 hours
        hist_cache_key = f"5min_data_{crypto_id}"
        hist_data = get_cached_api_data(hist_url, hist_params, hist_cache_key)
        
        if 'prices' not in hist_data or not hist_data['prices']:
            return jsonify({'random_forest': [], 'lstm': [], 'ensemble': []})
        
        # Get longer historical data for features
        long_hist_url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
        long_hist_params = {'vs_currency': 'usd', 'days': '2'}
        long_hist_cache_key = f"historical_data_{crypto_id}_2"
        long_hist_data = get_cached_api_data(long_hist_url, long_hist_params, long_hist_cache_key)
        
        # Extract all historical prices
        all_prices = [p[1] for p in long_hist_data.get('prices', []) if len(p) >= 2]
        all_volumes = [v[1] for v in long_hist_data.get('total_volumes', []) if len(v) >= 2]
        
        if len(all_prices) < 24:
            return jsonify({'random_forest': [], 'lstm': [], 'ensemble': []})
        
        # Initialize models
        traditional_obj = UnifiedPredictionPipeline(model_type='traditional')
        deep_obj = UnifiedPredictionPipeline(model_type='deep_learning')
        
        rf_predictions = []
        lstm_predictions = []
        ensemble_predictions = []
        
        # Generate predictions for last N points
        recent_prices = hist_data['prices'][-48:]  # Last 48 5-minute points
        
        for i, price_point in enumerate(recent_prices):
            timestamp_ms, current_price = price_point[0], price_point[1]
            
            # Find the index in the full price history
            price_idx = next((idx for idx, p in enumerate(long_hist_data['prices']) 
                            if p[0] == timestamp_ms), None)
            
            if price_idx is None or price_idx < 24:
                continue
            
            # Get historical prices up to this point
            prices_up_to_point = [p[1] for p in long_hist_data['prices'][max(0, price_idx-100):price_idx]]
            volumes_up_to_point = [v[1] for v in long_hist_data['total_volumes'][max(0, price_idx-100):price_idx]]
            
            if len(prices_up_to_point) < 24:
                continue
            
            # Estimate market cap (simplified)
            current_volume = volumes_up_to_point[-1] if volumes_up_to_point else 1000000
            current_market_cap = current_price * 1000000
            
            # Calculate 24h price change
            price_24h_ago = prices_up_to_point[-24] if len(prices_up_to_point) >= 24 else current_price
            price_change_24h = ((current_price - price_24h_ago) / price_24h_ago * 100) if price_24h_ago > 0 else 0
            
            try:
                # Prepare features using the data available at that time
                input_data = feature_engineer.prepare_features(
                    current_price=current_price,
                    current_volume=current_volume,
                    current_market_cap=current_market_cap,
                    prices=prices_up_to_point[-24:],
                    volumes=volumes_up_to_point[-24:] if len(volumes_up_to_point) >= 24 else [current_volume] * 24,
                    price_change_24h=price_change_24h
                )
                
                # Generate predictions
                rf_pred = traditional_obj.predict(input_data)
                lstm_pred = deep_obj.predict(input_data)
                
                rf_price = float(rf_pred[0]) if len(rf_pred) > 0 else current_price
                lstm_price = float(lstm_pred[0]) if len(lstm_pred) > 0 else current_price
                ensemble_price = 0.95 * rf_price + 0.05 * lstm_price
                
                # Convert timestamp to ISO format
                timestamp_iso = pd.Timestamp(timestamp_ms, unit='ms').isoformat()
                
                rf_predictions.append({
                    'timestamp': timestamp_iso,
                    'price': rf_price
                })
                
                lstm_predictions.append({
                    'timestamp': timestamp_iso,
                    'price': lstm_price
                })
                
                ensemble_predictions.append({
                    'timestamp': timestamp_iso,
                    'price': ensemble_price
                })
                
            except Exception as pred_error:
                logger.error(f"Error generating prediction for point {i}: {pred_error}")
                continue
        
        return jsonify({
            'random_forest': rf_predictions,
            'lstm': lstm_predictions,
            'ensemble': ensemble_predictions
        })
        
    except Exception as e:
        logger.exception(f"Error generating historical predictions: {str(e)}")
        return jsonify({'error': str(e), 'random_forest': [], 'lstm': [], 'ensemble': []})


if __name__ == "__main__":
    # Disable Flask's default logger to stdout
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    # app.run(host="0.0.0.0", port = 8080, debug=True)
    app.run(host="0.0.0.0", port=8080, debug=False)
