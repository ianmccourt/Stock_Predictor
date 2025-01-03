# Import from original StockPredictor
from dateutil.relativedelta import relativedelta
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import ta
import logging
import torch
from torch.cuda import is_available as cuda_available
from google.colab import drive
from tqdm.notebook import tqdm
import gc
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import warnings
pd.options.mode.chained_assignment = None  # Suppress the warning globally
import json
import os
from sklearn.model_selection import TimeSeriesSplit
import pickle
from scipy.stats import norm

class EnhancedStockPredictor:
    def __init__(self, base_params: Dict = None):
        # Detect available hardware
        self.device = self._get_device()
        print(f"Using device: {self.device}")

        # Add model caching
        self.model_cache = {}
        self.model_timestamp = {}
        self.cache_duration = timedelta(hours=24)

        # Modify base parameters for GPU
        stock_specific_params = {
            'n_estimators': 500,
            'learning_rate': 0.003,
            'max_depth': 12,
            'num_leaves': 128,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 50,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'random_state': 42,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'metric': 'rmse',
            'boost_from_average': True,
            'num_threads': -1
        }

        # Add GPU-specific parameters only if GPU is available
        if self.device == 'gpu':
            stock_specific_params.update({
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0
            })

        self.base_params = base_params or stock_specific_params
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.performance_tracker = {}
        self.warm_up_size = 1000
        self.timeframes = ['1d', '1wk', '1mo']  # Updated timeframes list
        self.models_by_timeframe = {tf: {} for tf in self.timeframes}

        # Add cache path
        self.cache_path = '/content/drive/MyDrive/stock_predictor/model_cache'
        os.makedirs(self.cache_path, exist_ok=True)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Mount Google Drive for model storage
        drive.mount('/content/drive')
        self.model_path = '/content/drive/MyDrive/stock_predictor'
        os.makedirs(self.model_path, exist_ok=True)

        # Add more stock symbols for training
        self.training_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',  # Tech
            'JPM', 'BAC', 'GS',                        # Finance
            'JNJ', 'PFE', 'UNH',                       # Healthcare
            'XOM', 'CVX', 'NEE',                       # Energy
            'PG', 'KO', 'WMT'                          # Consumer
        ]

        # Add minimum required points for each timeframe
        self.min_required_points = {
            '1d': 50,
            '1wk': 26,
            '1mo': 24
        }

        # Add default lookback periods
        self.default_lookback = {
            '1d': 365,
            '1wk': 520,  # 10 years of weekly data
            '1mo': 120   # 10 years of monthly data
        }

    def _get_device(self) -> str:
        """Detect and return the best available device"""
        try:
            # Check for GPU
            if cuda_available():
                return 'gpu'
            return 'cpu'
        except Exception as e:
            self.logger.warning(f"Error detecting device: {str(e)}")
            return 'cpu'

    def prepare_market_data(self, df: pd.DataFrame, timeframe: str = '1d') -> pd.DataFrame:
        """Convert market data into features with advanced indicators"""
        try:
            # Create a deep copy to avoid warnings
            X = df.copy(deep=True)

            # Handle MultiIndex columns first
            if isinstance(X.columns, pd.MultiIndex):
                X.columns = [col[0] for col in X.columns]
            X.columns = X.columns.str.lower()

            # Verify required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in X.columns for col in required_columns):
                self.logger.error(f"Missing required columns. Available: {X.columns.tolist()}")
                return None

            try:
                # Initialize feature columns list
                feature_cols = []

                # Calculate base features
                high_low_diff = X['high'] - X['low']
                if (high_low_diff <= 0).any():
                    high_low_diff = high_low_diff.replace(0, np.nan).ffill().bfill()
                X.loc[:, 'feature_normalized_price'] = (X['close'] - X['low']) / high_low_diff
                feature_cols.append('feature_normalized_price')

                # Calculate window-based features
                timeframe_windows = {
                    '1d': [5, 10, 20, 50],
                    '1wk': [2, 4, 8, 13, 26],
                    '1mo': [2, 3, 6, 12]
                }
                windows = timeframe_windows.get(timeframe, [5, 10, 20, 50])

                for window in windows:
                    if len(X) > window:
                        # Price-based features with consistent naming
                        feature_names = [
                            f'feature_reg_coef_{window}p',
                            f'feature_volatility_{window}p',
                            f'feature_volume_volatility_{window}p',
                            f'feature_sma_{window}p',
                            f'feature_ema_{window}p'
                        ]

                        X.loc[:, feature_names[0]] = self._calculate_regression_coefficient(X['close'], window)
                        X.loc[:, feature_names[1]] = X['close'].pct_change().rolling(window).std()
                        X.loc[:, feature_names[2]] = X['volume'].pct_change().rolling(window).std()
                        X.loc[:, feature_names[3]] = X['close'].rolling(window=window).mean()
                        X.loc[:, feature_names[4]] = ta.trend.ema_indicator(X['close'], window=window)

                        feature_cols.extend(feature_names)

                # Add technical indicators with consistent naming
                tech_indicators = {
                    'feature_rsi': lambda: ta.momentum.rsi(X['close']),
                    'feature_macd': lambda: ta.trend.macd_diff(X['close']),
                    'feature_adx': lambda: ta.trend.adx(X['high'], X['low'], X['close']),
                    'feature_cci': lambda: ta.trend.cci(X['high'], X['low'], X['close'])
                }

                for name, func in tech_indicators.items():
                    X.loc[:, name] = func()
                    feature_cols.append(name)

                # Handle missing values
                X[feature_cols] = X[feature_cols].ffill().bfill()

                # Store feature columns for validation
                self.feature_cols = feature_cols

                # Return only the feature columns and close price
                return X[feature_cols + ['close']]

            except Exception as e:
                self.logger.error(f"Error in feature calculation: {str(e)}")
                print(f"DataFrame info:")
                print(X.info())
                return None

        except Exception as e:
            self.logger.error(f"Error in prepare_market_data: {str(e)}")
            return None

    def _verify_data(self, df: pd.DataFrame, timeframe: str) -> bool:
        """Verify that the downloaded data is valid"""
        try:
            # Handle MultiIndex columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                # Create a mapping of old column names to new ones
                col_map = {
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                }

                # Create new DataFrame with correct column names
                df = pd.DataFrame({
                    col_map[col[0]]: df[col]
                    for col in df.columns
                    if col[0] in col_map
                })

            required_columns = ['open', 'high', 'low', 'close', 'volume']

            # Check if all required columns exist
            if not all(col in df.columns for col in required_columns):
                self.logger.error(f"Missing required columns. Available: {df.columns.tolist()}")
                return False

            # Check for sufficient data points
            if len(df) < self.min_required_points.get(timeframe, 50):
                self.logger.error(f"Insufficient data points for {timeframe}")
                return False

            # Check for missing values
            if df[required_columns].isnull().any().any():
                self.logger.error("Found missing values in required columns")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error in data verification: {str(e)}")
            return False

    def _create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced technical indicators"""
        try:
            X = df.copy()

            # Handle MultiIndex columns if present
            if isinstance(X.columns, pd.MultiIndex):
                col_map = {
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                }
                X = pd.DataFrame({
                    col_map[col[0]]: X[col]
                    for col in X.columns
                    if col[0] in col_map
                })

            try:
                # Volume Profile
                X['volume_price_trend'] = X['close'] * X['volume']
                X['volume_price_ratio'] = X['volume_price_trend'] / X['volume_price_trend'].rolling(20).mean()

                # Advanced Momentum
                adx_indicator = ta.trend.ADXIndicator(X['high'], X['low'], X['close'])
                X['adx'] = adx_indicator.adx()

                cci_indicator = ta.trend.CCIIndicator(X['high'], X['low'], X['close'])
                X['cci'] = cci_indicator.cci()

                stoch_indicator = ta.momentum.StochasticOscillator(X['high'], X['low'], X['close'])
                X['stoch_k'] = stoch_indicator.stoch()

            except Exception as e:
                print(f"Warning: Error calculating advanced indicators: {str(e)}")
                # Set default values
                X['adx'] = 25
                X['cci'] = 0
                X['stoch_k'] = 50

            # Market Structure
            X['support_level'] = X['low'].rolling(20).min()
            X['resistance_level'] = X['high'].rolling(20).max()
            X['price_position'] = (X['close'] - X['support_level']) / (X['resistance_level'] - X['support_level'])

            return X

        except Exception as e:
            print(f"Error in _create_advanced_features: {str(e)}")
            print(f"DataFrame info:")
            print(df.info())
            return df

    def _get_cached_model(self, timeframe: str) -> Optional[List[lgb.Booster]]:
        """Get cached model if it exists and is not expired"""
        cache_key = f"{timeframe}_{datetime.now().strftime('%Y%m%d')}"

        if cache_key in self.model_cache:
            if datetime.now() - self.model_timestamp[cache_key] < self.cache_duration:
                self.logger.info(f"Using cached model for {timeframe}")
                return self.model_cache[cache_key]

        return None

    def _save_model_to_cache(self, models: List[lgb.Booster], timeframe: str):
        """Save model to cache with timestamp"""
        cache_key = f"{timeframe}_{datetime.now().strftime('%Y%m%d')}"
        self.model_cache[cache_key] = models
        self.model_timestamp[cache_key] = datetime.now()

        # Save to disk
        cache_file = os.path.join(self.cache_path, f"{cache_key}.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'models': models,
                'timestamp': datetime.now(),
                'performance': self.performance_tracker.get(timeframe, {})
            }, f)

    def _create_model(self, params: Optional[Dict] = None) -> lgb.LGBMRegressor:
        """Create and return a LightGBM model with optimized parameters"""
        default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'n_estimators': 400,
            'learning_rate': 0.01,
            'num_leaves': 31,
            'feature_fraction': 0.8,  # Remove colsample_bytree
            'bagging_fraction': 0.8,  # Remove subsample
            'bagging_freq': 5,
            'min_child_samples': 20,
            'min_data_in_leaf': 20,
            'max_depth': -1,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'n_jobs': -1,
            'random_state': 42,
            'verbose': -1,
            'force_col_wise': True  # Add this to remove threading overhead warning
        }

        if params:
            default_params.update(params)

        return lgb.LGBMRegressor(**default_params)

    def _train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: pd.DataFrame, y_val: pd.Series) -> lgb.LGBMRegressor:
        """Train the LightGBM model with early stopping"""
        model = self._create_model()

        # Set up early stopping
        eval_set = [(X_val, y_val)]
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100)
        ]

        # Train the model
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=callbacks
        )

        return model

    def _predict_timeframe(self, X: pd.DataFrame, timeframe: str) -> Dict:
        """Make ensemble prediction with improved confidence calculation"""
        try:
            if timeframe not in self.models:
                self.logger.error(f"No models found for timeframe {timeframe}")
                return None

            models = self.models[timeframe]
            if not isinstance(models, list):
                self.logger.error(f"Models for {timeframe} is not a list")
                return None

            feature_cols = [col for col in X.columns
                           if col.startswith(('feature_', 'reg_coef_', 'normalized_price'))]

            # Make predictions with all models in ensemble
            predictions = []
            for model in models:
                try:
                    pred = model.predict(X[feature_cols])
                    predictions.append(pred)
                except Exception as e:
                    self.logger.warning(f"Error in model prediction: {str(e)}")
                    continue

            if not predictions:
                self.logger.error("No valid predictions generated")
                return None

            # Calculate ensemble prediction
            predictions_array = np.array(predictions)
            ensemble_pred = np.mean(predictions_array, axis=0)
            pred_std = np.std(predictions_array, axis=0)

            # Calculate prediction intervals
            confidence_level = 0.95
            z_score = norm.ppf((1 + confidence_level) / 2)

            confidence_interval = {
                'lower': ensemble_pred - z_score * pred_std,
                'upper': ensemble_pred + z_score * pred_std
            }

            # Get current market regime
            market_regime = self._detect_market_regime(self.latest_data)

            # Calculate confidence score
            confidence = self._calculate_ensemble_confidence(
                predictions_array[:, -1],  # Use last prediction point
                timeframe,
                market_regime
            )

            current_price = float(X['close'].iloc[-1])
            predicted_price = current_price * (1 + ensemble_pred[-1])

            return {
                'price': predicted_price,
                'raw_prediction': float(ensemble_pred[-1]),
                'current_price': current_price,
                'confidence': confidence,
                'confidence_interval': {
                    'lower': float(confidence_interval['lower'][-1]),
                    'upper': float(confidence_interval['upper'][-1])
                },
                'ensemble_std': float(pred_std[-1]),
                'market_regime': market_regime
            }

        except Exception as e:
            self.logger.error(f"Error in ensemble prediction: {str(e)}")
            return None

    def _adjust_params_by_timeframe(self, timeframe: str) -> Dict:
        """Adjust model parameters based on timeframe"""
        base_params = self.base_params.copy()

        if timeframe == '1d':
            # Daily models need more complexity and slower learning
            params = {
                'learning_rate': 0.001,
                'num_leaves': 256,
                'n_estimators': 1000,
                'max_depth': 15,
                'min_child_samples': 100,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'reg_alpha': 0.2,
                'reg_lambda': 1.2
            }
        elif timeframe == '1wk':
            # Weekly models need moderate complexity
            params = {
                'learning_rate': 0.002,
                'num_leaves': 128,
                'n_estimators': 600,
                'max_depth': 12,
                'min_child_samples': 70,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.15,
                'reg_lambda': 1.1
            }
        else:  # 1mo
            # Monthly models need less complexity to avoid overfitting
            params = {
                'learning_rate': 0.003,
                'num_leaves': 64,
                'n_estimators': 400,
                'max_depth': 8,
                'min_child_samples': 50,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0
            }

        base_params.update(params)
        return base_params

    def train_on_multiple_stocks(self, timeframe: str) -> Optional[List[lgb.Booster]]:
        """Train models on multiple stocks with optimized parameters"""
        try:
            # Check cache first
            cached_models = self._get_cached_model(timeframe)
            if cached_models is not None:
                return cached_models

            # Get adjusted parameters
            params = self._adjust_params_by_timeframe(timeframe)

            # Store original params
            original_params = self.base_params.copy()

            # Update base params temporarily
            self.base_params.update(params)

            try:
                # Prepare combined data
                all_data = []
                for symbol in tqdm(self.training_symbols, desc="Processing stocks"):
                    try:
                        # Get historical data
                        data = yf.download(
                            symbol,
                            start=(datetime.now() - relativedelta(years=10)),
                            end=datetime.now(),
                            interval=timeframe
                        )

                        if len(data) < self.min_required_points.get(timeframe, 50):
                            continue

                        # Prepare features
                        prepared_data = self.prepare_market_data(data, timeframe)
                        if prepared_data is not None:
                            prepared_data['symbol'] = symbol
                            all_data.append(prepared_data)

                    except Exception as e:
                        self.logger.warning(f"Error processing {symbol}: {str(e)}")
                        continue

                if not all_data:
                    self.logger.error("No valid data collected for training")
                    return None

                # Combine all stock data
                combined_data = pd.concat(all_data, axis=0)
                combined_data = combined_data.sort_index()

                # Train models
                models = self._train_model(combined_data, timeframe)

                return models

            finally:
                # Restore original params
                self.base_params = original_params

        except Exception as e:
            self.logger.error(f"Error in training multiple stocks: {str(e)}")
            return None

    def predict_with_confidence(self, symbol: str, lookback_days: int = 365,
                          prediction_horizon: int = 5) -> Dict:
        """Make predictions with realistic confidence scores"""
        try:
            if not self.models:
                print("Training models first...")
                self.train_on_multiple_stocks()

            if not self.models:
                raise ValueError("Failed to train models")

            predictions = {}

            for timeframe in self.timeframes:
                try:
                    print(f"\nProcessing {timeframe} timeframe...")

                    # Get data
                    df = yf.download(
                        symbol,
                        start=(datetime.now() - timedelta(days=lookback_days)),
                        end=datetime.now(),
                        interval=timeframe
                    )

                    print(f"Downloaded {len(df)} periods of data")

                    if len(df) < 100:
                        print(f"Insufficient data for {timeframe} timeframe")
                        continue

                    # Store latest data for market regime detection
                    self.latest_data = df.copy()

                    # Prepare features
                    X = self.prepare_market_data(df)
                    if X is None:
                        print(f"Failed to prepare market data for {timeframe}")
                        continue

                    # Get feature columns in the same order as training
                    feature_cols = [
                        'normalized_price',
                        'reg_coef_3d',
                        'reg_coef_5d',
                        'reg_coef_10d',
                        'reg_coef_20d'
                    ] + [col for col in X.columns if col.startswith('feature_')]

                    print(f"Generated {len(feature_cols)} features")

                    if not feature_cols:
                        print("No features generated")
                        continue

                    # Scale features if scaler exists
                    if timeframe in self.scalers:
                        try:
                            X[feature_cols] = self.scalers[timeframe].transform(X[feature_cols])
                            print("Features scaled successfully")
                        except Exception as e:
                            print(f"Error scaling features: {str(e)}")
                            print("Available features:", X.columns.tolist())
                            print("Expected features:", feature_cols)
                            continue
                    else:
                        print(f"No scaler found for {timeframe}, using unscaled features")

                    # Make prediction
                    if timeframe in self.models:
                        model = self.models[timeframe]
                        print("Model found, making prediction...")

                        try:
                            # Get latest data point as numpy array
                            latest_features = X[feature_cols].iloc[-1:].values
                            print("Feature shape:", latest_features.shape)

                            # Make prediction
                            pred = model.predict(latest_features)
                            print(f"Raw prediction: {float(pred[0]):.6f}")

                            # Calculate predicted price
                            current_price = float(df['Close'].iloc[-1])
                            predicted_price = current_price * (1 + float(pred[0]))
                            print(f"Current price: ${current_price:.2f}")
                            print(f"Predicted price: ${predicted_price:.2f}")

                            # Get confidence
                            metrics = self.performance_tracker.get(timeframe, {})
                            direction_accuracy = float(metrics.get('direction_accuracy', [0.5])[-1])
                            confidence = min(direction_accuracy * 100, 85.0)  # Cap at 85%
                            print(f"Confidence: {confidence:.2f}%")

                            predictions[timeframe] = {
                                'price': predicted_price,
                                'confidence': confidence
                            }

                            print(f"Successfully generated prediction for {timeframe}")
                        except Exception as e:
                            print(f"Error making prediction: {str(e)}")
                            continue
                    else:
                        print(f"No model found for {timeframe}")
                        continue

                except Exception as e:
                    print(f"Error processing {timeframe}: {str(e)}")
                    continue

            if not predictions:
                raise ValueError(f"Could not generate any predictions for {symbol}")

            # Combine predictions
            print("\nCombining predictions from all timeframes...")
            combined_pred = self._combine_timeframe_predictions(predictions)

            if combined_pred is None:
                raise ValueError("Failed to combine predictions")

            return {
                'symbol': symbol,
                'timeframe_predictions': predictions,
                'combined_prediction': combined_pred
            }

        except Exception as e:
            print(f"Error in predict_with_confidence for {symbol}: {str(e)}")
            return None

    def _optimize_for_device(self, X: pd.DataFrame) -> pd.DataFrame:
        """Optimize data structure for the current device"""
        if self.device == 'gpu':
            float_cols = X.select_dtypes(include=['float64']).columns
            X[float_cols] = X[float_cols].astype('float32')
        return X

    def clear_memory(self):
        """Clear memory when needed"""
        import gc
        gc.collect()

        if hasattr(self, 'models'):
            self.models.clear()
        if hasattr(self, 'scalers'):
            self.scalers.clear()

        if self.device == 'gpu':
            torch.cuda.empty_cache()

    def _combine_timeframe_predictions(self, predictions: Dict) -> Dict:
        """Combine predictions from multiple timeframes with realistic confidence scores"""
        try:
            if not predictions:
                return None

            # Initialize weights based on timeframe reliability
            weights = {
                '1h': 0.4,  # Lower weight for shorter timeframe
                '1d': 0.6   # Higher weight for daily predictions
            }

            weighted_price = 0
            weighted_confidence = 0
            total_weight = 0

            for timeframe, pred in predictions.items():
                # Handle Series/DataFrame values
                if isinstance(pred['price'], (pd.Series, pd.DataFrame)):
                    price = float(pred['price'].iloc[0])
                else:
                    price = float(pred['price'])

                # Calculate base confidence from metrics
                metrics = self.performance_tracker.get(timeframe, {})
                direction_accuracy = metrics.get('direction_accuracy', [])
                rmse = metrics.get('rmse', [])

                if direction_accuracy and rmse:
                    # Use latest metrics
                    latest_accuracy = direction_accuracy[-1]
                    latest_rmse = rmse[-1]

                    # Calculate confidence components
                    accuracy_score = min(latest_accuracy * 100, 75.0)  # Cap at 75%
                    rmse_score = max(0, (1 - latest_rmse) * 25)  # Up to 25% based on RMSE

                    # Combine scores
                    confidence = accuracy_score + rmse_score
                else:
                    confidence = 50.0  # Default confidence

                # Apply timeframe weight
                weight = weights.get(timeframe, 0.5)
                weighted_price += price * weight
                weighted_confidence += confidence * weight
                total_weight += weight

            # Normalize
            if total_weight > 0:
                final_price = weighted_price / total_weight
                final_confidence = weighted_confidence / total_weight

                # Apply market regime adjustment
                if hasattr(self, 'latest_data'):
                    regime_score = self._detect_market_regime(self.latest_data)
                    final_confidence *= (0.5 + regime_score/2)

                # Ensure confidence stays within reasonable bounds
                final_confidence = min(max(final_confidence, 35.0), 85.0)
            else:
                final_price = sum(pred['price'] for pred in predictions.values()) / len(predictions)
                final_confidence = 50.0

            return {
                'price': final_price,
                'confidence': final_confidence,
                'weights': weights
            }

        except Exception as e:
            print(f"Error in combining predictions: {str(e)}")
            return None

    def _detect_market_regime(self, data: pd.DataFrame) -> float:
        """Detect the current market regime (bullish, bearish, or neutral)"""
        try:
            # Ensure we're working with a copy to avoid modifying original data
            df = data.copy()

            # If we have a MultiIndex, use the price level index
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index(level=0)  # Reset the first level of MultiIndex

            # Calculate short and long term trends
            short_ma = df['Close'].rolling(window=20).mean()
            long_ma = df['Close'].rolling(window=50).mean()

            # Calculate RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            # Get latest values
            current_short_ma = short_ma.iloc[-1]
            current_long_ma = long_ma.iloc[-1]
            current_rsi = rsi.iloc[-1]

            # Define regime score (0 = bearish, 0.5 = neutral, 1 = bullish)
            regime_score = 0.5  # Start neutral

            # Trend analysis
            if current_short_ma > current_long_ma:
                regime_score += 0.25
            elif current_short_ma < current_long_ma:
                regime_score -= 0.25

            # RSI analysis
            if current_rsi > 70:
                regime_score -= 0.25  # Overbought
            elif current_rsi < 30:
                regime_score += 0.25  # Oversold

            # Ensure score is between 0 and 1
            regime_score = max(0, min(1, regime_score))

            return regime_score

        except Exception as e:
            self.logger.error(f"Error in _detect_market_regime: {str(e)}")
            return 0.5  # Return neutral if there's an error

    def _get_sentiment_features(self, symbol: str) -> pd.DataFrame:
        """Get sentiment features from news data"""
        try:
            # Get news data
            ticker = yf.Ticker(symbol)
            news = ticker.news

            if not news:
                return pd.DataFrame()

            sentiments = []
            for article in news:
                # Analyze sentiment of title
                blob = TextBlob(article.get('title', ''))
                sentiments.append(blob.sentiment.polarity)

            # Calculate aggregate sentiment metrics
            sentiment_df = pd.DataFrame({
                'feature_sentiment_mean': [np.mean(sentiments)],
                'feature_sentiment_std': [np.std(sentiments)],
                'feature_sentiment_max': [max(sentiments)],
                'feature_sentiment_min': [min(sentiments)]
            })

            return sentiment_df

        except Exception as e:
            print(f"Error getting sentiment features: {str(e)}")
            return pd.DataFrame()

    def _update_performance_metrics(self, timeframe: str, actual: float, predicted: float) -> None:
        """Update performance metrics for adaptive weighting"""
        try:
            if timeframe not in self.performance_tracker:
                self.performance_tracker[timeframe] = {
                    'predictions': [],
                    'actuals': [],
                    'rmse': [],
                    'direction_accuracy': [],
                    'weight': 0.5  # Initial weight
                }

            metrics = self.performance_tracker[timeframe]
            metrics['predictions'].append(predicted)
            metrics['actuals'].append(actual)

            # Keep only last 100 predictions
            if len(metrics['predictions']) > 100:
                metrics['predictions'] = metrics['predictions'][-100:]
                metrics['actuals'] = metrics['actuals'][-100:]

            # Update metrics
            rmse = np.sqrt(mean_squared_error(metrics['actuals'], metrics['predictions']))
            direction_accuracy = np.mean(
                np.sign(np.diff(metrics['actuals'])) ==
                np.sign(np.diff(metrics['predictions'])))

            metrics['rmse'].append(rmse)
            metrics['direction_accuracy'].append(direction_accuracy)

            # Update weight based on recent performance
            metrics['weight'] = direction_accuracy * (1 - min(rmse, 1))

        except Exception as e:
            print(f"Error updating performance metrics: {str(e)}")

    def _apply_conformal_prediction(self, X_train: pd.DataFrame, y_train: pd.Series,
                                  X_test: pd.DataFrame, alpha: float = 0.1) -> Dict:
        """Apply conformal prediction to get prediction intervals"""
        try:
            from mapie.regression import MapieRegressor

            # Initialize MAPIE regressor with our LightGBM model
            mapie = MapieRegressor(
                estimator=self.models.get('1d'),  # Use daily model as base
                method="plus",  # Use plus method for prediction intervals
                cv="prefit",   # Model is already fitted
                n_jobs=-1
            )

            # Fit MAPIE on training data
            mapie.fit(X_train, y_train)

            # Get predictions and prediction intervals
            y_pred, y_pis = mapie.predict(X_test, alpha=alpha)

            # Calculate coverage and interval width
            coverage = (y_train >= y_pis[:, 0]) & (y_train <= y_pis[:, 1])
            interval_width = y_pis[:, 1] - y_pis[:, 0]

            return {
                'prediction': y_pred,
                'lower_bound': y_pis[:, 0],
                'upper_bound': y_pis[:, 1],
                'coverage': coverage.mean(),
                'avg_width': interval_width.mean()
            }

        except Exception as e:
            print(f"Error in conformal prediction: {str(e)}")
            return None

    def _calculate_confidence(self, predictions: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate prediction confidence based on historical accuracy"""
        try:
            # Ensure inputs are numpy arrays
            predictions = np.array(predictions)
            y_true = np.array(y_true)

            # Calculate percentage errors
            percentage_errors = np.abs((predictions - y_true) / y_true)

            # Convert to accuracy scores (inverse of errors)
            accuracy_scores = 1 - percentage_errors

            # Calculate confidence score (0-100%)
            confidence = np.mean(accuracy_scores) * 100

            # Clip confidence to reasonable range
            confidence = np.clip(confidence, 0, 100)

            # Add penalty for high volatility
            if len(predictions) > 1:
                volatility = np.std(predictions) / np.mean(predictions)
                volatility_penalty = min(25, volatility * 100)  # Cap penalty at 25%
                confidence = max(0, confidence - volatility_penalty)

            return float(confidence)

        except Exception as e:
            self.logger.error(f"Error calculating confidence: {str(e)}")
            return 25.0  # Return conservative confidence if calculation fails

    def _calculate_regression_coefficient(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate regression coefficient over a rolling window"""
        def reg_coef(y):
            x = np.arange(len(y))
            return np.polyfit(x, y, 1)[0]

        return prices.rolling(window).apply(reg_coef)

    def save_model_version(self, model, timeframe: str):
        """Save model with versioning to Google Drive"""
        try:
            # Create version-specific directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            version_dir = os.path.join(self.model_path, f'version_{timestamp}')
            os.makedirs(version_dir, exist_ok=True)

            # Save model and metadata
            model_path = os.path.join(version_dir, f'model_{timeframe}.txt')
            model.save_model(model_path)

            # Save performance metrics
            if timeframe in self.performance_tracker:
                metrics = self.performance_tracker[timeframe]
                metrics_path = os.path.join(version_dir, f'metrics_{timeframe}.json')
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f)

            return version_dir

        except Exception as e:
            self.logger.error(f"Error saving model version: {str(e)}")
            return None

    def _train_model_walk_forward(self, combined_data: pd.DataFrame, timeframe: str):
        """Train model using walk-forward optimization"""
        try:
            feature_cols = [col for col in combined_data.columns
                           if col.startswith(('feature_', 'reg_coef_', 'normalized_price'))]

            # Create target variable
            combined_data['target'] = combined_data['close'].pct_change().shift(-1)
            combined_data = combined_data.dropna()

            # Initialize TimeSeriesSplit
            tscv = TimeSeriesSplit(
                n_splits=5,
                test_size=int(len(combined_data) * 0.2)
            )

            models = []
            metrics = {
                'rmse': [],
                'direction_accuracy': []
            }

            # Walk-forward training
            for train_idx, val_idx in tscv.split(combined_data):
                train = combined_data.iloc[train_idx]
                val = combined_data.iloc[val_idx]

                train_data = lgb.Dataset(
                    train[feature_cols],
                    label=train['target']
                )

                val_data = lgb.Dataset(
                    val[feature_cols],
                    label=val['target'],
                    reference=train_data
                )

                # Train model
                model = lgb.train(
                    self.base_params,
                    train_data,
                    valid_sets=[val_data],
                    callbacks=[
                        lgb.early_stopping(50),
                        lgb.log_evaluation(100)
                    ]
                )

                # Calculate metrics
                predictions = model.predict(val[feature_cols])
                rmse = np.sqrt(mean_squared_error(val['target'], predictions))
                direction_accuracy = np.mean(
                    np.sign(predictions) == np.sign(val['target'])
                )

                metrics['rmse'].append(rmse)
                metrics['direction_accuracy'].append(direction_accuracy)
                models.append(model)

            # Use ensemble of models
            self.models[timeframe] = models
            self.performance_tracker[timeframe] = {
                'rmse': np.mean(metrics['rmse']),
                'direction_accuracy': np.mean(metrics['direction_accuracy'])
            }

            # Save model versions
            self.save_model_version(models[-1], timeframe)  # Save latest model

            return models

        except Exception as e:
            self.logger.error(f"Error in walk-forward training: {str(e)}")
            return None

    def _calculate_ensemble_confidence(self, predictions: List[float], timeframe: str,
                                    market_regime: float) -> float:
        """Calculate confidence score based on ensemble agreement and market conditions"""
        try:
            if not predictions:
                return 25.0  # Default conservative confidence

            # Calculate prediction agreement
            pred_std = np.std(predictions)
            pred_mean = np.mean(predictions)
            cv = pred_std / (abs(pred_mean) + 1e-8)  # Coefficient of variation

            # Get historical performance metrics
            metrics = self.performance_tracker.get(timeframe, {})
            historical_accuracy = metrics.get('direction_accuracy', 0.5)
            historical_rmse = metrics.get('rmse', 1.0)

            # Calculate base confidence from model metrics
            base_confidence = historical_accuracy * (1.0 / (1.0 + historical_rmse))

            # Calculate ensemble agreement factor (0 to 1)
            agreement_factor = 1.0 / (1.0 + cv)

            # Calculate volatility factor
            if hasattr(self, 'latest_data') and len(self.latest_data) > 0:
                volatility = self.latest_data['close'].pct_change().std() * np.sqrt(252)
                volatility_factor = 1.0 / (1.0 + volatility)
            else:
                volatility_factor = 0.5

            # Timeframe-specific weights
            timeframe_weights = {
                '1d': 0.8,    # Higher weight for daily predictions
                '1wk': 0.9,   # Highest weight for weekly
                '1mo': 0.7    # Lower weight for monthly due to uncertainty
            }
            timeframe_factor = timeframe_weights.get(timeframe, 0.8)

            # Market regime factor (0.5 to 1.5)
            regime_factor = 0.5 + market_regime

            # Combine all factors
            confidence = (
                base_confidence * 0.3 +      # Historical performance
                agreement_factor * 0.3 +     # Ensemble agreement
                volatility_factor * 0.2 +    # Market volatility
                timeframe_factor * 0.1 +     # Timeframe reliability
                regime_factor * 0.1          # Market regime
            ) * 100  # Convert to percentage

            # Bound confidence between 25% and 85%
            confidence = min(85.0, max(25.0, confidence))

            self.logger.info(f"""
            Confidence Calculation Factors:
            - Base Confidence: {base_confidence:.3f}
            - Agreement Factor: {agreement_factor:.3f}
            - Volatility Factor: {volatility_factor:.3f}
            - Timeframe Factor: {timeframe_factor:.3f}
            - Regime Factor: {regime_factor:.3f}
            - Final Confidence: {confidence:.1f}%
            """)

            return float(confidence)

        except Exception as e:
            self.logger.error(f"Error calculating confidence: {str(e)}")
            return 25.0  # Return conservative confidence on error

    def _train_models(self, timeframe: str, data: Dict[str, pd.DataFrame]) -> Dict[str, lgb.LGBMRegressor]:
        """Train models for multiple stocks for a given timeframe"""
        try:
            models = {}
            for symbol, df in data.items():
                # Prepare features and target
                X = df.drop('target', axis=1)
                y = df['target']

                # Train model
                model = self._train_model(X, y)
                models[symbol] = model

                # Clear memory
                gc.collect()

            return models

        except Exception as e:
            self.logger.error(f"Error in training multiple stocks: {str(e)}")
            raise

    def train_all_models(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, lgb.LGBMRegressor]]:
        """Train models for all timeframes and stocks"""
        models = {}
        for timeframe in self.timeframes:
            try:
                timeframe_data = data[timeframe]
                models[timeframe] = self._train_models(timeframe, timeframe_data)
            except Exception as e:
                self.logger.error(f"Failed to train/load models for {timeframe}")
                models[timeframe] = {}
        return models

# Mount Google Drive for model storage
try:
    drive.mount('/content/drive')
    os.makedirs('/content/drive/MyDrive/stock_predictor/long_term', exist_ok=True)
except Exception as e:
    print(f"Warning: Could not mount Google Drive: {str(e)}")

class LongTermStockPredictor(EnhancedStockPredictor):
    def __init__(self, base_params: Dict = None):
        # Initialize parent class
        super().__init__(base_params)

        # Set up model storage path
        self.model_path = '/content/drive/MyDrive/stock_predictor/long_term'

        # Override timeframes to include longer periods
        self.timeframes = ['1d', '1wk', '1mo']
        self.models_by_timeframe = {tf: {} for tf in self.timeframes}

        # Add timeframe-specific parameters
        self.warm_up_size = {
            '1d': 1000,
            '1wk': 200,   # About 4 years of weekly data
            '1mo': 60     # 5 years of monthly data
        }

        self.default_lookback = {
            '1d': 365,    # 1 year
            '1wk': 520,   # 10 years
            '1mo': 120    # 10 years
        }

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize progress bar for Colab
        self.pbar = None

        # Detect hardware
        self.device = self._get_device()
        print(f"Using device: {self.device}")

    def _optimize_for_colab(self, X: pd.DataFrame) -> pd.DataFrame:
        """Optimize data structure for Colab environment"""
        try:
            # Convert to float32 for GPU efficiency
            if self.device == 'gpu':
                float_cols = X.select_dtypes(include=['float64']).columns
                X[float_cols] = X[float_cols].astype('float32')

            # Clear memory
            gc.collect()
            if self.device == 'gpu':
                torch.cuda.empty_cache()

            return X

        except Exception as e:
            self.logger.warning(f"Optimization warning: {str(e)}")
            return X

    def prepare_market_data(self, df: pd.DataFrame, timeframe: str = '1d') -> pd.DataFrame:
        """Convert market data into features with advanced indicators"""
        try:
            # Create a deep copy to avoid warnings
            X = df.copy(deep=True)

            # Handle MultiIndex columns first
            if isinstance(X.columns, pd.MultiIndex):
                X.columns = [col[0] for col in X.columns]
            X.columns = X.columns.str.lower()

            # Verify required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in X.columns for col in required_columns):
                self.logger.error(f"Missing required columns. Available: {X.columns.tolist()}")
                return None

            try:
                # Initialize feature columns list
                feature_cols = []

                # Calculate base features
                high_low_diff = X['high'] - X['low']
                if (high_low_diff <= 0).any():
                    high_low_diff = high_low_diff.replace(0, np.nan).ffill().bfill()
                X.loc[:, 'feature_normalized_price'] = (X['close'] - X['low']) / high_low_diff
                feature_cols.append('feature_normalized_price')

                # Calculate window-based features
                timeframe_windows = {
                    '1d': [5, 10, 20, 50],
                    '1wk': [2, 4, 8, 13, 26],
                    '1mo': [2, 3, 6, 12]
                }
                windows = timeframe_windows.get(timeframe, [5, 10, 20, 50])

                for window in windows:
                    if len(X) > window:
                        # Price-based features with consistent naming
                        feature_names = [
                            f'feature_reg_coef_{window}p',
                            f'feature_volatility_{window}p',
                            f'feature_volume_volatility_{window}p',
                            f'feature_sma_{window}p',
                            f'feature_ema_{window}p'
                        ]

                        X.loc[:, feature_names[0]] = self._calculate_regression_coefficient(X['close'], window)
                        X.loc[:, feature_names[1]] = X['close'].pct_change().rolling(window).std()
                        X.loc[:, feature_names[2]] = X['volume'].pct_change().rolling(window).std()
                        X.loc[:, feature_names[3]] = X['close'].rolling(window=window).mean()
                        X.loc[:, feature_names[4]] = ta.trend.ema_indicator(X['close'], window=window)

                        feature_cols.extend(feature_names)

                # Add technical indicators with consistent naming
                tech_indicators = {
                    'feature_rsi': lambda: ta.momentum.rsi(X['close']),
                    'feature_macd': lambda: ta.trend.macd_diff(X['close']),
                    'feature_adx': lambda: ta.trend.adx(X['high'], X['low'], X['close']),
                    'feature_cci': lambda: ta.trend.cci(X['high'], X['low'], X['close'])
                }

                for name, func in tech_indicators.items():
                    X.loc[:, name] = func()
                    feature_cols.append(name)

                # Handle missing values
                X[feature_cols] = X[feature_cols].ffill().bfill()

                # Store feature columns for validation
                self.feature_cols = feature_cols

                # Return only the feature columns and close price
                return X[feature_cols + ['close']]

            except Exception as e:
                self.logger.error(f"Error in feature calculation: {str(e)}")
                print(f"DataFrame info:")
                print(X.info())
                return None

        except Exception as e:
            self.logger.error(f"Error in prepare_market_data: {str(e)}")
            return None

    def predict_long_term(self, symbol: str, timeframe: str = '1wk', horizon: int = 4) -> Optional[Dict]:
        """Generate long-term prediction for a given symbol"""
        try:
            # Validate timeframe
            if timeframe not in self.timeframes:
                self.logger.error(f"Invalid timeframe: {timeframe}")
                return None

            # Get historical data
            self.latest_data = yf.download(
                symbol,
                start=(datetime.now() - relativedelta(years=10)),
                end=datetime.now(),
                interval=timeframe
            )

            if len(self.latest_data) < self.min_required_points.get(timeframe, 50):
                self.logger.error(f"Insufficient data points for {symbol}")
                return None

            # Prepare features
            X = self.prepare_market_data(self.latest_data, timeframe)
            if X is None:
                return None

            # Train or load models
            if timeframe not in self.models:
                self.models[timeframe] = self.train_on_multiple_stocks(timeframe=timeframe)

            if self.models[timeframe] is None:
                self.logger.error(f"Failed to train/load models for {timeframe}")
                return None

            # Generate prediction
            prediction = self._predict_timeframe(X, timeframe)
            if prediction is None:
                return None

            # Calculate market regime
            market_regime = self._detect_market_regime(self.latest_data)

            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'horizon': horizon,
                'current_price': prediction['current_price'],
                'predicted_price': prediction['price'],
                'confidence': prediction['confidence'],
                'market_regime': market_regime,
                'prediction_date': datetime.now().strftime('%Y-%m-%d'),
                'confidence_interval': prediction['confidence_interval']
            }

        except Exception as e:
            self.logger.error(f"Error in predict_long_term: {str(e)}")
            return None

    def train_on_multiple_stocks(self, timeframe: str) -> Optional[List[lgb.Booster]]:
        """Train models on multiple stocks with optimized parameters"""
        try:
            # Check cache first
            cached_models = self._get_cached_model(timeframe)
            if cached_models is not None:
                return cached_models

            # Get adjusted parameters
            params = self._adjust_params_by_timeframe(timeframe)

            # Store original params
            original_params = self.base_params.copy()

            # Update base params temporarily
            self.base_params.update(params)

            try:
                # Prepare combined data
                all_data = []
                for symbol in tqdm(self.training_symbols, desc="Processing stocks"):
                    try:
                        # Get historical data
                        data = yf.download(
                            symbol,
                            start=(datetime.now() - relativedelta(years=10)),
                            end=datetime.now(),
                            interval=timeframe
                        )

                        if len(data) < self.min_required_points.get(timeframe, 50):
                            continue

                        # Prepare features
                        prepared_data = self.prepare_market_data(data, timeframe)
                        if prepared_data is not None:
                            prepared_data['symbol'] = symbol
                            all_data.append(prepared_data)

                    except Exception as e:
                        self.logger.warning(f"Error processing {symbol}: {str(e)}")
                        continue

                if not all_data:
                    self.logger.error("No valid data collected for training")
                    return None

                # Combine all stock data
                combined_data = pd.concat(all_data, axis=0)
                combined_data = combined_data.sort_index()

                # Train models
                models = self._train_model(combined_data, timeframe)

                return models

            finally:
                # Restore original params
                self.base_params = original_params

        except Exception as e:
            self.logger.error(f"Error in training multiple stocks: {str(e)}")
            return None

    def clear_memory(self):
        """Enhanced memory clearing for Colab"""
        super().clear_memory()
        if self.pbar:
            self.pbar.close()
        gc.collect()
        if self.device == 'gpu':
            torch.cuda.empty_cache()

    def load_model(self, timeframe: str):
        """Load saved model for inference"""
        try:
            model_path = f'/content/drive/MyDrive/stock_predictor/model_{timeframe}.txt'
            if os.path.exists(model_path):
                return lgb.Booster(model_file=model_path)
            return None
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return None

    def save_model_version(self, model, timeframe: str):
        """Save model with versioning to Google Drive"""
        try:
            # Create version-specific directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            version_dir = os.path.join(self.model_path, f'version_{timestamp}')
            os.makedirs(version_dir, exist_ok=True)

            # Save model and metadata
            model_path = os.path.join(version_dir, f'model_{timeframe}.txt')
            model.save_model(model_path)

            # Save performance metrics
            if timeframe in self.performance_tracker:
                metrics = self.performance_tracker[timeframe]
                metrics_path = os.path.join(version_dir, f'metrics_{timeframe}.json')
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f)

            return version_dir

        except Exception as e:
            self.logger.error(f"Error saving model version: {str(e)}")
            return None

    def _train_model(self, X: pd.DataFrame, y: pd.Series) -> lgb.LGBMRegressor:
        """Train a single model with built-in validation split"""
        try:
            # Split data into training and validation sets (80-20 split)
            train_size = int(len(X) * 0.8)

            X_train = X[:train_size]
            y_train = y[:train_size]
            X_val = X[train_size:]
            y_val = y[train_size:]

            # Create and train model
            model = self._create_model()

            # Set up early stopping
            eval_set = [(X_val, y_val)]
            callbacks = [
                lgb.early_stopping(stopping_rounds=50, verbose=True),
                lgb.log_evaluation(period=100)
            ]

            # Train the model
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                callbacks=callbacks
            )

            return model

        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}")
            raise

    def _train_models(self, timeframe: str, data: Dict[str, pd.DataFrame]) -> Dict[str, lgb.LGBMRegressor]:
        """Train models for multiple stocks for a given timeframe"""
        try:
            models = {}
            for symbol, df in data.items():
                # Prepare features and target
                X = df.drop('target', axis=1)
                y = df['target']

                # Train model
                model = self._train_model(X, y)
                models[symbol] = model

                # Clear memory
                gc.collect()

            return models

        except Exception as e:
            self.logger.error(f"Error in training multiple stocks: {str(e)}")
            raise

# Modified main function for Colab
def main():
    """Example usage in Colab"""
    try:
        # Initialize predictor
        predictor = LongTermStockPredictor()

        # Test symbols with progress tracking
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        timeframes = ['1wk', '1mo']
        horizons = {'1wk': 4, '1mo': 6}  # 4 weeks and 6 months

        for symbol in tqdm(symbols, desc="Processing symbols"):
            print(f"\nAnalyzing {symbol}...")

            for timeframe in timeframes:
                horizon = horizons[timeframe]
                prediction = predictor.predict_long_term(
                    symbol=symbol,
                    timeframe=timeframe,
                    horizon=horizon
                )

                if prediction:
                    print(f"\n{timeframe} Prediction (Horizon: {horizon}):")
                    print(f"Current Price: ${prediction['current_price']:.2f}")
                    print(f"Predicted Price: ${prediction['predicted_price']:.2f}")
                    print(f"Confidence: {prediction['confidence']:.1f}%")
                    print(f"Market Regime: {prediction['market_regime']}")
                else:
                    print(f"Failed to generate prediction for {symbol} - {timeframe}")

            # Clear memory after each symbol
            predictor.clear_memory()

    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
