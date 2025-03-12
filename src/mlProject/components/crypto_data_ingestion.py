import requests
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta
from mlProject import logger
from mlProject.entity.config_entity import DataIngestionConfig
import zipfile


class CryptoDataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.base_url = "https://api.coingecko.com/api/v3"
        
    def get_cryptocompare_data(self, crypto_symbol, limit=2000):
        """Fetch hourly data from CryptoCompare (Free: 2000 hours)"""
        try:
            url = "https://min-api.cryptocompare.com/data/v2/histohour"
            params = {'fsym': crypto_symbol.upper(), 'tsym': 'USD', 'limit': limit}
            
            logger.info(f"Fetching {limit} hours from CryptoCompare for {crypto_symbol}")
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            if data['Response'] == 'Success':
                df = pd.DataFrame(data['Data']['Data'])
                df['datetime'] = pd.to_datetime(df['time'], unit='s')
                df = df.rename(columns={
                    'close': 'price',
                    'volumefrom': 'volume',
                    'volumeto': 'market_cap'
                })
                df = df[['datetime', 'price', 'volume', 'market_cap']].set_index('datetime')
                logger.info(f"CryptoCompare: Fetched {len(df)} hourly data points")
                return df
            return None
        except Exception as e:
            logger.warning(f"CryptoCompare API failed: {e}")
            return None
    
    def get_binance_data(self, symbol, interval='1h', limit=1000):
        """Fetch data from Binance (Free: 1000 candles per request)"""
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {'symbol': f'{symbol}USDT', 'interval': interval, 'limit': limit}
            
            logger.info(f"Fetching {limit} candles from Binance for {symbol}")
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['price'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            df['market_cap'] = df['quote_volume'].astype(float)
            df = df[['datetime', 'price', 'volume', 'market_cap']].set_index('datetime')
            logger.info(f"Binance: Fetched {len(df)} data points")
            return df
        except Exception as e:
            logger.warning(f"Binance API failed: {e}")
            return None
        
    def get_crypto_data(self, crypto_id="bitcoin", vs_currency="usd", days=365):
        """
        Fetch cryptocurrency data from CoinGecko API
        
        Args:
            crypto_id (str): Cryptocurrency ID (bitcoin, ethereum, etc.)
            vs_currency (str): Currency to compare against
            days (int): Number of days of historical data
        """
        try:
            # Historical price data
            price_url = f"{self.base_url}/coins/{crypto_id}/market_chart"
            params = {
                'vs_currency': vs_currency,
                'days': days,
                'interval': 'hourly' if days <= 90 else 'daily'
            }
            
            logger.info(f"Fetching {crypto_id} data for {days} days...")
            response = requests.get(price_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to DataFrame
            prices = data['prices']
            volumes = data['total_volumes']
            market_caps = data['market_caps']
            
            df = pd.DataFrame({
                'timestamp': [item[0] for item in prices],
                'price': [item[1] for item in prices],
                'volume': [item[1] for item in volumes],
                'market_cap': [item[1] for item in market_caps]
            })
            
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.drop('timestamp', axis=1)
            df = df.set_index('datetime')
            
            return df
            
        except Exception as e:
            logger.exception(f"Error fetching crypto data: {str(e)}")
            raise e
    
    def add_technical_indicators(self, df):
        """Add technical indicators to the dataset"""
        try:
            # Simple Moving Averages
            df['sma_7'] = df['price'].rolling(window=7).mean()
            df['sma_14'] = df['price'].rolling(window=14).mean()
            df['sma_30'] = df['price'].rolling(window=30).mean()
            
            # Exponential Moving Averages  
            df['ema_7'] = df['price'].ewm(span=7).mean()
            df['ema_14'] = df['price'].ewm(span=14).mean()
            
            # MACD (Moving Average Convergence Divergence)
            ema_12 = df['price'].ewm(span=12).mean()
            ema_26 = df['price'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # RSI (Relative Strength Index)
            delta = df['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df['price'].rolling(window=20).mean()
            bb_std = df['price'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # Price change percentages
            df['price_change_1h'] = df['price'].pct_change(1)
            df['price_change_24h'] = df['price'].pct_change(24)
            df['price_change_7d'] = df['price'].pct_change(168)  # 7 days * 24 hours
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=14).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Volatility
            df['volatility'] = df['price'].rolling(window=24).std()
            
            # Price position relative to recent high/low
            df['high_14d'] = df['price'].rolling(window=336).max()  # 14 days
            df['low_14d'] = df['price'].rolling(window=336).min()
            df['price_position'] = (df['price'] - df['low_14d']) / (df['high_14d'] - df['low_14d'])
            
            return df
            
        except Exception as e:
            logger.exception(f"Error adding technical indicators: {str(e)}")
            raise e
    
    def create_prediction_targets(self, df):
        """Create prediction targets (future prices)"""
        try:
            # Next hour price (for calculating 5-min target)
            df['target_price_1h'] = df['price'].shift(-1)
            
            # 5-minute target: scale 1-hour change to 5-minute timeframe
            # Formula: price + (1h_change * 5/60)
            df['target_price_5min'] = df['price'] + (df['target_price_1h'] - df['price']) * (5.0 / 60.0)
            
            # Next day price (24 hours) - kept for reference
            df['target_price_24h'] = df['price'].shift(-24)
            
            # Price direction (classification targets)
            df['target_direction_5min'] = np.where(df['target_price_5min'] > df['price'], 1, 0)
            df['target_direction_1h'] = np.where(df['target_price_1h'] > df['price'], 1, 0)
            df['target_direction_24h'] = np.where(df['target_price_24h'] > df['price'], 1, 0)
            
            # Price change percentage targets
            df['target_change_5min'] = (df['target_price_5min'] - df['price']) / df['price'] * 100
            df['target_change_1h'] = (df['target_price_1h'] - df['price']) / df['price'] * 100
            df['target_change_24h'] = (df['target_price_24h'] - df['price']) / df['price'] * 100
            
            return df
            
        except Exception as e:
            logger.exception(f"Error creating prediction targets: {str(e)}")
            raise e
    
    def download_file(self, cryptocurrencies=None):
        """Main method to fetch and aggregate crypto data from multiple APIs"""
        try:
            os.makedirs(os.path.dirname(self.config.local_data_file), exist_ok=True)
            
            if cryptocurrencies is None:
                cryptos = ['solana']
            else:
                cryptos = cryptocurrencies
            
            # Crypto symbol mapping for different APIs
            symbol_map = {
                'solana': {'cryptocompare': 'SOL', 'binance': 'SOL', 'coingecko': 'solana'},
                'bitcoin': {'cryptocompare': 'BTC', 'binance': 'BTC', 'coingecko': 'bitcoin'},
                'ethereum': {'cryptocompare': 'ETH', 'binance': 'ETH', 'coingecko': 'ethereum'}
            }
            
            all_data = []
            
            for crypto in cryptos:
                logger.info(f"=" * 70)
                logger.info(f"Aggregating data for {crypto} from multiple sources...")
                
                dfs = []
                
                # Source 1: CryptoCompare (2000 hours = ~83 days of hourly data)
                if crypto in symbol_map:
                    cc_df = self.get_cryptocompare_data(symbol_map[crypto]['cryptocompare'], limit=2000)
                    if cc_df is not None and len(cc_df) > 0:
                        dfs.append(cc_df)
                    time.sleep(1)
                
                # Source 2: Binance (1000 hours of data)
                if crypto in symbol_map:
                    binance_df = self.get_binance_data(symbol_map[crypto]['binance'], interval='1h', limit=1000)
                    if binance_df is not None and len(binance_df) > 0:
                        dfs.append(binance_df)
                    time.sleep(1)
                
                # Source 3: CoinGecko (limited but still useful)
                try:
                    cg_df = self.get_crypto_data(crypto_id=crypto, days=365)
                    if len(cg_df) > 0:
                        dfs.append(cg_df)
                except Exception as e:
                    logger.warning(f"CoinGecko failed for {crypto}: {e}")
                
                # Aggregate all sources
                if dfs:
                    # Concatenate and remove duplicates by timestamp
                    combined_df = pd.concat(dfs)
                    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
                    combined_df = combined_df.sort_index()
                    
                    logger.info(f"Total data points after aggregation: {len(combined_df)}")
                    
                    # Add technical indicators
                    combined_df = self.add_technical_indicators(combined_df)
                    
                    # Create prediction targets
                    combined_df = self.create_prediction_targets(combined_df)
                    
                    # Add crypto identifier
                    combined_df['crypto_symbol'] = crypto
                    
                    # Clean data (remove NaN values)
                    combined_df = combined_df.dropna()
                    
                    logger.info(f"Final clean data points: {len(combined_df)}")
                    all_data.append(combined_df)
                else:
                    logger.error(f"No data sources available for {crypto}")
                
                time.sleep(1)
            
            # Combine all crypto data
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Save to CSV
            output_path = self.config.local_data_file.replace('.zip', '.csv')
            combined_df.to_csv(output_path, index=False)
            
            logger.info(f"Crypto data saved to {output_path}")
            logger.info(f"Dataset shape: {combined_df.shape}")
            logger.info(f"Columns: {list(combined_df.columns)}")
            
            return output_path
            
        except Exception as e:
            logger.exception(f"Error in download_file: {str(e)}")
            raise e
    
    def extract_zip_file(self):
        """Extract zip file (not needed for crypto data, but keeping for compatibility)"""
        try:
            # For crypto data, we don't need zip extraction
            # But we'll create the unzip directory anyway
            os.makedirs(self.config.unzip_dir, exist_ok=True)
            
            # Check if CSV file exists (no copying needed since paths are the same)
            csv_file = self.config.local_data_file.replace('.zip', '.csv')
            if os.path.exists(csv_file):
                logger.info(f"Crypto data already available at {csv_file}")
            else:
                logger.error(f"Crypto data file not found at {csv_file}")
            
        except Exception as e:
            logger.exception(f"Error in extract_zip_file: {str(e)}")
            raise e