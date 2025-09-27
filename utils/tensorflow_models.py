import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import os
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import json

class AdvancedUrbanStressPredictor:
    """Advanced TensorFlow deep learning models for urban stress prediction."""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.feature_names = [
            'air_quality', 'noise_level', 'temperature', 'humidity',
            'crowd_density', 'light_pollution', 'hour_of_day', 
            'day_of_week', 'is_weekend', 'season'
        ]
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize various deep learning architectures."""
        # Multi-layer Perceptron for basic pattern recognition
        self.models['mlp'] = self._create_mlp_model()
        
        # LSTM for temporal sequence prediction
        self.models['lstm'] = self._create_lstm_model()
        
        # CNN for spatial-temporal pattern recognition
        self.models['cnn'] = self._create_cnn_model()
        
        # Transformer for attention-based pattern recognition
        self.models['transformer'] = self._create_transformer_model()
        
        # Ensemble model combining all approaches
        self.models['ensemble'] = None  # Will be created after training individual models
        
        # Initialize scalers
        self.scalers['standard'] = StandardScaler()
        self.scalers['minmax'] = MinMaxScaler()
    
    def _create_mlp_model(self) -> keras.Model:
        """Create deep Multi-Layer Perceptron model."""
        model = keras.Sequential([
            layers.Input(shape=(len(self.feature_names),)),
            
            # First hidden layer with dropout
            layers.Dense(256, activation='relu', name='hidden_1'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Second hidden layer
            layers.Dense(128, activation='relu', name='hidden_2'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Third hidden layer
            layers.Dense(64, activation='relu', name='hidden_3'),
            layers.BatchNormalization(),
            layers.Dropout(0.1),
            
            # Fourth hidden layer
            layers.Dense(32, activation='relu', name='hidden_4'),
            
            # Output layer for stress prediction (0-10 scale)
            layers.Dense(1, activation='sigmoid', name='output')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def _create_lstm_model(self) -> keras.Model:
        """Create LSTM model for temporal sequence prediction."""
        # For LSTM, we'll reshape data to include time sequence
        model = keras.Sequential([
            layers.Input(shape=(24, len(self.feature_names))),  # 24-hour sequences
            
            # First LSTM layer
            layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            layers.BatchNormalization(),
            
            # Second LSTM layer
            layers.LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
            layers.BatchNormalization(),
            
            # Dense layers
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.1),
            
            # Output layer
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def _create_cnn_model(self) -> keras.Model:
        """Create 1D CNN model for pattern recognition in sensor data."""
        model = keras.Sequential([
            layers.Input(shape=(len(self.feature_names), 1)),
            
            # First conv layer
            layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            
            # Second conv layer
            layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            
            # Flatten and dense layers
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            
            # Output layer
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def _create_transformer_model(self) -> keras.Model:
        """Create Transformer model with attention mechanism."""
        # Define transformer block
        def transformer_block(inputs, head_size, num_heads, ff_dim, dropout=0):
            # Attention and normalization
            x = layers.MultiHeadAttention(
                key_dim=head_size, num_heads=num_heads, dropout=dropout
            )(inputs, inputs)
            x = layers.Dropout(dropout)(x)
            x = layers.LayerNormalization(epsilon=1e-6)(x + inputs)
            
            # Feed forward and normalization
            ffn = keras.Sequential([
                layers.Dense(ff_dim, activation='relu'),
                layers.Dropout(dropout),
                layers.Dense(inputs.shape[-1])
            ])
            x2 = ffn(x)
            return layers.LayerNormalization(epsilon=1e-6)(x + x2)
        
        # Input layer
        inputs = layers.Input(shape=(1, len(self.feature_names)))
        
        # Transformer blocks
        x = transformer_block(inputs, head_size=256, num_heads=4, ff_dim=4, dropout=0.1)
        x = transformer_block(x, head_size=256, num_heads=4, ff_dim=4, dropout=0.1)
        
        # Global pooling and output
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def generate_training_data(self, n_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate comprehensive training data for deep learning."""
        X = []
        y = []
        
        for _ in range(n_samples):
            # Generate time-based features
            hour = np.random.randint(0, 24)
            day_of_week = np.random.randint(0, 7)
            is_weekend = 1 if day_of_week >= 5 else 0
            season = np.random.randint(0, 4)  # 0: Spring, 1: Summer, 2: Fall, 3: Winter
            
            # Environmental features with realistic patterns
            air_quality = self._generate_air_quality(hour, season)
            noise_level = self._generate_noise_level(hour, day_of_week)
            temperature = self._generate_temperature(hour, season)
            humidity = self._generate_humidity(temperature, season)
            crowd_density = self._generate_crowd_density(hour, day_of_week)
            light_pollution = self._generate_light_pollution(hour, crowd_density)
            
            features = [
                air_quality, noise_level, temperature, humidity,
                crowd_density, light_pollution, hour, day_of_week, 
                is_weekend, season
            ]
            
            # Calculate stress level using complex interactions
            stress = self._calculate_complex_stress(features)
            
            X.append(features)
            y.append(stress / 10.0)  # Normalize to 0-1 range
        
        return np.array(X), np.array(y)
    
    def _generate_air_quality(self, hour: int, season: int) -> float:
        """Generate realistic air quality data."""
        base_aqi = 45 + season * 10  # Higher pollution in winter
        
        # Rush hour patterns
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            base_aqi += 30
        elif 22 <= hour or hour <= 6:
            base_aqi -= 15
        
        return max(0, min(500, base_aqi + np.random.normal(0, 20)))
    
    def _generate_noise_level(self, hour: int, day_of_week: int) -> float:
        """Generate realistic noise level data."""
        base_noise = 55
        
        # Time patterns
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            base_noise += 20
        elif 22 <= hour or hour <= 6:
            base_noise -= 25
        
        # Weekend patterns
        if day_of_week >= 5:
            if 12 <= hour <= 24:
                base_noise += 10
        
        return max(30, min(120, base_noise + np.random.normal(0, 15)))
    
    def _generate_temperature(self, hour: int, season: int) -> float:
        """Generate realistic temperature data."""
        seasonal_base = [15, 25, 18, 8][season]  # Spring, Summer, Fall, Winter
        daily_variation = 8 * np.sin((hour - 6) * np.pi / 12)
        
        return seasonal_base + daily_variation + np.random.normal(0, 3)
    
    def _generate_humidity(self, temperature: float, season: int) -> float:
        """Generate realistic humidity data."""
        base_humidity = 60 + season * 5
        temp_effect = (25 - temperature) * 0.8
        
        return max(20, min(95, base_humidity + temp_effect + np.random.normal(0, 10)))
    
    def _generate_crowd_density(self, hour: int, day_of_week: int) -> float:
        """Generate realistic crowd density data."""
        base_crowd = 30
        
        # Time patterns
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            base_crowd += 40
        elif 12 <= hour <= 14:
            base_crowd += 25
        elif 22 <= hour or hour <= 6:
            base_crowd -= 20
        
        # Weekend patterns
        if day_of_week >= 5:
            if 10 <= hour <= 22:
                base_crowd += 20
            else:
                base_crowd -= 15
        
        return max(0, min(100, base_crowd + np.random.normal(0, 15)))
    
    def _generate_light_pollution(self, hour: int, crowd_density: float) -> float:
        """Generate realistic light pollution data."""
        base_light = 100 if 6 <= hour <= 22 else 30
        crowd_effect = crowd_density * 0.5
        
        return max(0, base_light + crowd_effect + np.random.uniform(-20, 20))
    
    def _calculate_complex_stress(self, features: List[float]) -> float:
        """Calculate stress using complex feature interactions."""
        air_quality, noise_level, temperature, humidity, crowd_density, \
        light_pollution, hour, day_of_week, is_weekend, season = features
        
        # Base stress from environmental factors
        stress = 0.0
        
        # Air quality impact (non-linear)
        stress += (air_quality / 100) ** 1.5 * 2.5
        
        # Noise impact with time sensitivity
        noise_impact = (noise_level - 40) / 20
        if 22 <= hour or hour <= 7:  # Noise more stressful at night
            noise_impact *= 1.5
        stress += max(0, noise_impact) * 2.0
        
        # Temperature comfort zone
        optimal_temp = 22
        temp_stress = abs(temperature - optimal_temp) / 15
        stress += temp_stress ** 1.2 * 1.5
        
        # Humidity discomfort
        if humidity > 70 or humidity < 30:
            stress += abs(humidity - 50) / 50 * 1.2
        
        # Crowd density with location sensitivity
        if crowd_density > 60:
            stress += ((crowd_density - 60) / 40) ** 1.3 * 2.0
        
        # Light pollution (especially at night)
        if 22 <= hour or hour <= 6:
            stress += (light_pollution / 200) * 1.5
        
        # Interaction effects
        # High air quality + high noise = amplified stress
        if air_quality > 80 and noise_level > 70:
            stress += 1.5
        
        # Hot + humid + crowded = heat island stress
        if temperature > 28 and humidity > 70 and crowd_density > 70:
            stress += 2.0
        
        # Time-based modifiers
        if hour == 0:  # Monday blues
            stress += 1.0
        elif hour == 4:  # Friday relief
            stress -= 0.5
        elif is_weekend:
            stress *= 0.8
        
        # Add some randomness
        stress += np.random.normal(0, 0.3)
        
        return max(0, min(10, stress))
    
    def prepare_lstm_data(self, X: np.ndarray, sequence_length: int = 24) -> np.ndarray:
        """Prepare data for LSTM model with time sequences."""
        # Create sequences for LSTM
        X_sequences = []
        for i in range(len(X) - sequence_length + 1):
            X_sequences.append(X[i:i + sequence_length])
        
        return np.array(X_sequences)
    
    def train_all_models(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2):
        """Train all deep learning models."""
        print("Training advanced TensorFlow models...")
        
        # Scale the data
        X_scaled = self.scalers['standard'].fit_transform(X)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=validation_split, random_state=42
        )
        
        # Define callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001
        )
        
        # Train MLP
        print("Training MLP model...")
        self.models['mlp'].fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Train CNN
        print("Training CNN model...")
        X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val_cnn = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        
        self.models['cnn'].fit(
            X_train_cnn, y_train,
            validation_data=(X_val_cnn, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Train Transformer
        print("Training Transformer model...")
        X_train_trans = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_val_trans = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
        
        self.models['transformer'].fit(
            X_train_trans, y_train,
            validation_data=(X_val_trans, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Prepare LSTM data (requires sequence)
        if len(X_scaled) >= 24:
            print("Training LSTM model...")
            X_lstm = self.prepare_lstm_data(X_scaled)
            y_lstm = y[23:]  # Adjust y for sequence offset
            
            X_lstm_train, X_lstm_val, y_lstm_train, y_lstm_val = train_test_split(
                X_lstm, y_lstm, test_size=validation_split, random_state=42
            )
            
            self.models['lstm'].fit(
                X_lstm_train, y_lstm_train,
                validation_data=(X_lstm_val, y_lstm_val),
                epochs=50,
                batch_size=16,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
        
        print("All models trained successfully!")
    
    def predict_stress(self, sensor_data: Dict[str, Any]) -> Dict[str, float]:
        """Predict stress using all models."""
        # Extract and prepare features
        features = self._extract_features(sensor_data)
        X = np.array([features])
        X_scaled = self.scalers['standard'].transform(X)
        
        predictions = {}
        
        # MLP prediction
        pred_mlp = self.models['mlp'].predict(X_scaled, verbose=0)[0][0] * 10
        predictions['mlp'] = max(0, min(10, pred_mlp))
        
        # CNN prediction
        X_cnn = X_scaled.reshape(1, X_scaled.shape[1], 1)
        pred_cnn = self.models['cnn'].predict(X_cnn, verbose=0)[0][0] * 10
        predictions['cnn'] = max(0, min(10, pred_cnn))
        
        # Transformer prediction
        X_trans = X_scaled.reshape(1, 1, X_scaled.shape[1])
        pred_trans = self.models['transformer'].predict(X_trans, verbose=0)[0][0] * 10
        predictions['transformer'] = max(0, min(10, pred_trans))
        
        # Ensemble prediction (weighted average)
        ensemble_pred = (
            predictions['mlp'] * 0.4 +
            predictions['cnn'] * 0.3 +
            predictions['transformer'] * 0.3
        )
        predictions['ensemble'] = max(0, min(10, ensemble_pred))
        
        return predictions
    
    def _extract_features(self, sensor_data: Dict[str, Any]) -> List[float]:
        """Extract features from sensor data."""
        current_time = datetime.now()
        hour = current_time.hour
        day_of_week = current_time.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        season = (current_time.month - 1) // 3  # 0-3 for seasons
        
        return [
            sensor_data.get('air_quality', 50),
            sensor_data.get('noise_level', 60),
            sensor_data.get('temperature', 20),
            sensor_data.get('humidity', 60),
            sensor_data.get('crowd_density', 30),
            sensor_data.get('light_pollution', 100),
            hour,
            day_of_week,
            is_weekend,
            season
        ]
    
    def save_models(self):
        """Save all trained models."""
        for name, model in self.models.items():
            if model is not None:
                model.save(os.path.join(self.model_dir, f"{name}_model.h5"))
        
        # Save scalers
        joblib.dump(self.scalers, os.path.join(self.model_dir, "scalers.joblib"))
        
        print(f"Models saved to {self.model_dir}")
    
    def load_models(self):
        """Load pre-trained models."""
        try:
            for name in ['mlp', 'cnn', 'transformer', 'lstm']:
                model_path = os.path.join(self.model_dir, f"{name}_model.h5")
                if os.path.exists(model_path):
                    self.models[name] = keras.models.load_model(model_path)
            
            scaler_path = os.path.join(self.model_dir, "scalers.joblib")
            if os.path.exists(scaler_path):
                self.scalers = joblib.load(scaler_path)
            
            print("Models loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def get_model_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for all models."""
        # Generate test data
        X_test, y_test = self.generate_training_data(1000)
        X_test_scaled = self.scalers['standard'].transform(X_test)
        
        performance = {}
        
        for name, model in self.models.items():
            if model is not None:
                try:
                    if name == 'lstm':
                        # Special handling for LSTM
                        if len(X_test_scaled) >= 24:
                            X_test_lstm = self.prepare_lstm_data(X_test_scaled)
                            y_test_lstm = y_test[23:]
                            loss, mae, mse = model.evaluate(X_test_lstm, y_test_lstm, verbose=0)
                        else:
                            continue
                    elif name == 'cnn':
                        X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
                        loss, mae, mse = model.evaluate(X_test_reshaped, y_test, verbose=0)
                    elif name == 'transformer':
                        X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
                        loss, mae, mse = model.evaluate(X_test_reshaped, y_test, verbose=0)
                    else:  # MLP
                        loss, mae, mse = model.evaluate(X_test_scaled, y_test, verbose=0)
                    
                    performance[name] = {
                        'loss': float(loss),
                        'mae': float(mae) * 10,  # Scale back to 0-10
                        'rmse': float(np.sqrt(mse)) * 10
                    }
                except Exception as e:
                    print(f"Error evaluating {name}: {e}")
                    performance[name] = {'loss': 0, 'mae': 0, 'rmse': 0}
        
        return performance