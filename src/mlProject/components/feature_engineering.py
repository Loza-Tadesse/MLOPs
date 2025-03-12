"""
Feature Engineering Component
Centralizes all feature engineering logic for both training and prediction pipelines.
Ensures consistency and prevents feature drift.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional
from mlProject import logger


class FeatureEngineering:
    """
    Unified feature engineering component for cryptocurrency price prediction.
    Calculates technical indicators and prepares features for model input.
    """
    
    # Feature names in exact order expected by models
    # Note: target_price_5min is the target (dropped during training)
    # Features are ordered to match training data column order (32 features total)
    FEATURE_NAMES = [
        'price', 'volume', 'market_cap', 'sma_7', 'sma_14', 'sma_30',
        'ema_7', 'ema_14', 'macd', 'macd_signal', 'macd_histogram',
        'rsi', 'bb_middle', 'bb_upper', 'bb_lower', 'price_change_1h',
        'price_change_24h', 'price_change_7d', 'volume_sma', 'volume_ratio',
        'volatility', 'high_14d', 'low_14d', 'price_position',
        'target_price_1h', 'target_price_24h', 'target_direction_5min',
        'target_direction_1h', 'target_direction_24h', 'target_change_5min',
        'target_change_1h', 'target_change_24h'
    ]
    
    def __init__(self):
        """Initialize feature engineering component"""
        logger.info("Initializing FeatureEngineering component")
        
    @staticmethod
    def calculate_sma(prices: List[float], period: int) -> float:
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return prices[-1] if prices else 0.0
        return sum(prices[-period:]) / period
    
    @staticmethod
    def calculate_ema(prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        if not prices:
            return 0.0
        if len(prices) < period:
            return prices[-1]
        
        alpha = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = price * alpha + ema * (1 - alpha)
        return ema
    
    @staticmethod
    def calculate_macd(prices: List[float]) -> Dict[str, float]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        Returns: dict with macd, macd_signal, and macd_histogram
        """
        if len(prices) < 26:
            return {'macd': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0}
        
        # Calculate EMA-12 and EMA-26
        ema_12 = FeatureEngineering.calculate_ema(prices, 12)
        ema_26 = FeatureEngineering.calculate_ema(prices, 26)
        
        macd = ema_12 - ema_26
        
        # MACD signal line (9-period EMA of MACD)
        # Simplified: use 90% of MACD value
        macd_signal = macd * 0.9
        macd_histogram = macd - macd_signal
        
        return {
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_histogram': macd_histogram
        }
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0  # Neutral
        
        gains, losses = [], []
        for i in range(1, min(period + 1, len(prices))):
            change = prices[-i] - prices[-i-1]
            gains.append(max(0, change))
            losses.append(abs(min(0, change)))
        
        avg_gain = sum(gains) / len(gains) if gains else 0.01
        avg_loss = sum(losses) / len(losses) if losses else 0.01
        rs = avg_gain / avg_loss if avg_loss > 0 else 1
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_bollinger_bands(prices: List[float], period: int = 20) -> Dict[str, float]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            current_price = prices[-1] if prices else 0.0
            return {
                'bb_middle': current_price,
                'bb_upper': current_price * 1.05,
                'bb_lower': current_price * 0.95
            }
        
        sma = sum(prices[-period:]) / period
        variance = sum([(price - sma) ** 2 for price in prices[-period:]]) / period
        std_dev = variance ** 0.5
        
        return {
            'bb_middle': sma,
            'bb_upper': sma + (2 * std_dev),
            'bb_lower': sma - (2 * std_dev)
        }
    
    @staticmethod
    def calculate_volatility(prices: List[float]) -> float:
        """Calculate price volatility"""
        if len(prices) < 2:
            return 0.02
        
        price_changes = [
            abs(prices[i] - prices[i-1]) / prices[i-1] 
            for i in range(1, len(prices)) if prices[i-1] != 0
        ]
        return sum(price_changes) / len(price_changes) if price_changes else 0.02
    
    @staticmethod
    def calculate_price_position(prices: List[float], period: int = 14) -> float:
        """Calculate price position within recent range"""
        if len(prices) < period:
            return 0.5
        
        recent_prices = prices[-period:]
        high = max(recent_prices)
        low = min(recent_prices)
        current = prices[-1]
        
        if high == low:
            return 0.5
        return (current - low) / (high - low)
    
    def calculate_technical_indicators(
        self, 
        prices: List[float], 
        current_price: float,
        volumes: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        Calculate all technical indicators from price history
        
        Args:
            prices: Historical price data (at least 24 hours recommended)
            current_price: Most recent price
            volumes: Optional volume data
            
        Returns:
            Dictionary of technical indicators
        """
        if not prices:
            prices = [current_price]
        
        indicators = {}
        
        # Moving averages
        indicators['sma_7'] = self.calculate_sma(prices, 7)
        indicators['sma_14'] = self.calculate_sma(prices, 14)
        indicators['sma_30'] = self.calculate_sma(prices, 30)
        indicators['ema_7'] = self.calculate_ema(prices, 7)
        indicators['ema_14'] = self.calculate_ema(prices, 14)
        
        # MACD
        macd_values = self.calculate_macd(prices)
        indicators.update(macd_values)
        
        # RSI
        indicators['rsi'] = self.calculate_rsi(prices, 14)
        
        # Bollinger Bands
        bb_values = self.calculate_bollinger_bands(prices, 20)
        indicators.update(bb_values)
        
        # Volatility
        indicators['volatility'] = self.calculate_volatility(prices)
        
        # Price range indicators
        if len(prices) >= 14:
            indicators['high_14d'] = max(prices[-14:])
            indicators['low_14d'] = min(prices[-14:])
        else:
            indicators['high_14d'] = current_price
            indicators['low_14d'] = current_price
        
        indicators['price_position'] = self.calculate_price_position(prices, 14)
        
        # Volume indicators (if available)
        if volumes and len(volumes) >= 7:
            indicators['volume_sma'] = sum(volumes[-7:]) / 7
            indicators['volume_ratio'] = volumes[-1] / indicators['volume_sma'] if indicators['volume_sma'] > 0 else 1.0
        else:
            indicators['volume_sma'] = volumes[-1] if volumes else 0.0
            indicators['volume_ratio'] = 1.0
        
        logger.info(f"Calculated {len(indicators)} technical indicators")
        return indicators
    
    def prepare_features(
        self,
        current_price: float,
        current_volume: float,
        current_market_cap: float,
        prices: List[float],
        volumes: Optional[List[float]] = None,
        price_change_24h: Optional[float] = None
    ) -> np.ndarray:
        """
        Prepare complete feature vector for model prediction
        
        Args:
            current_price: Current cryptocurrency price
            current_volume: Current 24h trading volume
            current_market_cap: Current market capitalization
            prices: Historical price data
            volumes: Optional historical volume data
            price_change_24h: Optional 24h price change percentage
            
        Returns:
            numpy array with 29 features in correct order
        """
        # Calculate technical indicators
        indicators = self.calculate_technical_indicators(prices, current_price, volumes)
        
        # Calculate price changes
        price_change_1h = (current_price - prices[-2]) / prices[-2] if len(prices) >= 2 else 0.0
        price_change_24h_calc = price_change_24h / 100.0 if price_change_24h else 0.0
        price_change_7d = (current_price - prices[-7]) / prices[-7] if len(prices) >= 7 else 0.0
        
        # Build feature vector in exact order expected by model
        features = [
            current_price,                      # 0: price
            current_volume,                     # 1: volume
            current_market_cap,                 # 2: market_cap
            indicators['sma_7'],                # 3: sma_7
            indicators['sma_14'],               # 4: sma_14
            indicators['sma_30'],               # 5: sma_30
            indicators['ema_7'],                # 6: ema_7
            indicators['ema_14'],               # 7: ema_14
            indicators['macd'],                 # 8: macd
            indicators['macd_signal'],          # 9: macd_signal
            indicators['macd_histogram'],       # 10: macd_histogram
            indicators['rsi'],                  # 11: rsi
            indicators['bb_middle'],            # 12: bb_middle
            indicators['bb_upper'],             # 13: bb_upper
            indicators['bb_lower'],             # 14: bb_lower
            price_change_1h,                    # 15: price_change_1h
            price_change_24h_calc,              # 16: price_change_24h
            price_change_7d,                    # 17: price_change_7d
            indicators['volume_sma'],           # 18: volume_sma
            indicators['volume_ratio'],         # 19: volume_ratio
            indicators['volatility'],           # 20: volatility
            indicators['high_14d'],             # 21: high_14d
            indicators['low_14d'],              # 22: low_14d
            indicators['price_position'],       # 23: price_position
            current_price,                      # 24: target_price_1h (placeholder)
            current_price,                      # 25: target_price_24h (placeholder)
            0,                                  # 26: target_direction_5min (placeholder)
            0,                                  # 27: target_direction_1h (placeholder)
            0,                                  # 28: target_direction_24h (placeholder)
            0.0,                                # 29: target_change_5min (placeholder)
            0.0,                                # 30: target_change_1h (placeholder)
            0.0                                 # 31: target_change_24h (placeholder)
        ]
        
        feature_array = np.array(features, dtype=np.float64).reshape(1, -1)
        
        # Validate feature count
        expected_count = len(self.FEATURE_NAMES)
        actual_count = feature_array.shape[1]
        if actual_count != expected_count:
            raise ValueError(
                f"Feature count mismatch: expected {expected_count}, got {actual_count}"
            )
        
        logger.info(f"Prepared {actual_count} features for prediction")
        return feature_array
    
    def prepare_features_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features from a DataFrame (for training pipeline)
        
        Args:
            df: DataFrame with raw cryptocurrency data
            
        Returns:
            DataFrame with engineered features
        """
        logger.info(f"Engineering features for {len(df)} samples")
        
        # Validate required columns
        required_cols = ['price', 'volume', 'market_cap']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # This method would be used during training to apply feature engineering
        # to the entire dataset. For now, return as-is since training data
        # already has features calculated
        return df
    
    @staticmethod
    def get_feature_names() -> List[str]:
        """Get list of feature names in correct order"""
        return FeatureEngineering.FEATURE_NAMES.copy()
    
    @staticmethod
    def validate_features(features: np.ndarray) -> bool:
        """
        Validate feature array shape and values
        
        Args:
            features: Feature array to validate
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        expected_count = len(FeatureEngineering.FEATURE_NAMES)
        
        if features.ndim != 2:
            raise ValueError(f"Features must be 2D array, got {features.ndim}D")
        
        if features.shape[1] != expected_count:
            raise ValueError(
                f"Expected {expected_count} features, got {features.shape[1]}"
            )
        
        if np.any(np.isnan(features)):
            raise ValueError("Features contain NaN values")
        
        if np.any(np.isinf(features)):
            raise ValueError("Features contain infinite values")
        
        return True
