import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple

class StressPredictionModel:
    """Advanced ML models for predicting urban stress levels from environmental data."""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'neural_network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        self.ensemble_weights = {
            'random_forest': 0.4,
            'gradient_boosting': 0.35,
            'neural_network': 0.25
        }
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importance = {}
        self.model_performance = {}
        
        # Initialize with some training data
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize models with synthetic training data."""
        # Generate synthetic training data
        training_data = self._generate_training_data(1000)
        X = training_data[['air_quality', 'noise_level', 'temperature', 'humidity', 
                          'crowd_density', 'hour_of_day', 'day_of_week']]
        y = training_data['stress_level']
        
        # Train models
        self._train_models(pd.DataFrame(X), pd.Series(y))
    
    def _generate_training_data(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic training data with realistic patterns."""
        data = []
        
        for _ in range(n_samples):
            # Generate realistic environmental data
            hour = np.random.randint(0, 24)
            day_of_week = np.random.randint(0, 7)
            
            # Base environmental readings
            air_quality = max(0, np.random.normal(60, 25))
            noise_level = max(30, np.random.normal(65, 12))
            temperature = np.random.normal(20, 8)
            humidity = max(20, min(95, np.random.normal(65, 15)))
            crowd_density = max(0, min(100, np.random.normal(45, 20)))
            
            # Add time-based patterns
            if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
                air_quality += 20
                noise_level += 15
                crowd_density += 25
            elif 22 <= hour or hour <= 6:  # Night time
                air_quality -= 10
                noise_level -= 20
                crowd_density -= 15
            
            # Weekend patterns
            if day_of_week >= 5:  # Weekend
                crowd_density += 10 if hour >= 10 and hour <= 22 else -10
                noise_level += 5 if hour >= 12 and hour <= 24 else -10
            
            # Calculate stress level based on environmental factors
            stress_level = self._calculate_stress_from_factors(
                air_quality, noise_level, temperature, humidity, crowd_density, hour, day_of_week
            )
            
            data.append({
                'air_quality': air_quality,
                'noise_level': noise_level,
                'temperature': temperature,
                'humidity': humidity,
                'crowd_density': crowd_density,
                'hour_of_day': hour,
                'day_of_week': day_of_week,
                'stress_level': stress_level
            })
        
        return pd.DataFrame(data)
    
    def _calculate_stress_from_factors(self, air_quality: float, noise_level: float, 
                                     temperature: float, humidity: float, 
                                     crowd_density: float, hour: int, day_of_week: int) -> float:
        """Calculate stress level based on environmental factors using domain knowledge."""
        stress = 0.0
        
        # Air quality impact (AQI scale)
        if air_quality < 50:
            stress += 0.5
        elif air_quality < 100:
            stress += 1.5
        elif air_quality < 150:
            stress += 3.0
        else:
            stress += 5.0
        
        # Noise level impact (decibels)
        if noise_level < 50:
            stress += 0.2
        elif noise_level < 70:
            stress += 1.0
        elif noise_level < 85:
            stress += 2.5
        else:
            stress += 4.0
        
        # Temperature comfort (optimal around 20-24°C)
        temp_discomfort = abs(temperature - 22) / 10
        stress += min(temp_discomfort, 2.0)
        
        # Humidity discomfort (optimal around 40-60%)
        if humidity < 30 or humidity > 70:
            stress += 1.0
        elif humidity < 20 or humidity > 80:
            stress += 2.0
        
        # Crowd density stress
        if crowd_density > 80:
            stress += 2.0
        elif crowd_density > 60:
            stress += 1.0
        elif crowd_density < 20:
            stress += 0.5  # Too empty can also be stressful
        
        # Time-based stress factors
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
            stress += 1.5
        elif hour >= 23 or hour <= 5:  # Very late/early hours
            stress += 1.0
        
        # Day of week factors
        if day_of_week == 0:  # Monday
            stress += 0.8
        elif day_of_week == 4:  # Friday
            stress += 0.3
        elif day_of_week >= 5:  # Weekend
            stress -= 0.5
        
        # Add some random variation
        stress += np.random.normal(0, 0.3)
        
        # Clamp to 0-10 scale
        return max(0, min(10, stress))
    
    def _train_models(self, X: pd.DataFrame, y: pd.Series):
        """Train all models on the provided data."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train each model
        for name, model in self.models.items():
            try:
                model.fit(X_train, y_train)
                
                # Evaluate model
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                self.model_performance[name] = {
                    'mae': mae,
                    'r2': r2,
                    'rmse': np.sqrt(np.mean((y_test - y_pred) ** 2))
                }
                
                # Store feature importance for tree-based models
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = dict(zip(X.columns, model.feature_importances_))
                
            except Exception as e:
                print(f"Error training {name}: {e}")
        
        self.is_trained = True
    
    def predict_sensor_stress(self, sensor_data: Dict[str, Any]) -> float:
        """Predict stress level for a single sensor reading."""
        if not self.is_trained:
            return 5.0  # Default moderate stress
        
        # Extract features
        features = self._extract_features(sensor_data)
        
        # Make predictions with all models
        predictions = {}
        for name, model in self.models.items():
            try:
                features_scaled = self.scaler.transform([features])
                pred = model.predict(features_scaled)[0]
                predictions[name] = max(0, min(10, pred))
            except Exception:
                predictions[name] = 5.0
        
        # Ensemble prediction
        ensemble_pred = sum(
            pred * self.ensemble_weights[name] 
            for name, pred in predictions.items()
        )
        
        return round(ensemble_pred, 2)
    
    def predict_city_stress(self, processed_data: Dict[str, Any]) -> float:
        """Predict overall city stress level from aggregated sensor data."""
        if not processed_data or 'sensors' not in processed_data:
            return 5.0
        
        sensor_predictions = []
        for sensor in processed_data['sensors']:
            if sensor.get('status') == 'active':
                stress = self.predict_sensor_stress(sensor)
                sensor_predictions.append(stress)
        
        if not sensor_predictions:
            return 5.0
        
        # Weight by population density if available
        weighted_stress = np.mean(sensor_predictions)
        
        # Apply city-wide factors
        if processed_data.get('rush_hour', False):
            weighted_stress *= 1.2
        
        if processed_data.get('weekend', False):
            weighted_stress *= 0.9
        
        clamped_stress = max(0.0, min(10.0, float(weighted_stress)))
        return round(clamped_stress, 2)
    
    def _extract_features(self, sensor_data: Dict[str, Any]) -> List[float]:
        """Extract ML features from sensor data."""
        current_time = datetime.now()
        
        features = [
            sensor_data.get('air_quality', 50),
            sensor_data.get('noise_level', 60),
            sensor_data.get('temperature', 20),
            sensor_data.get('humidity', 60),
            sensor_data.get('crowd_density', 30),
            current_time.hour,
            current_time.weekday()
        ]
        
        return features
    
    def predict_future_stress(self, sensor_data: Dict[str, Any], hours_ahead: int) -> List[Tuple[datetime, float]]:
        """Predict stress levels for future time periods."""
        predictions = []
        current_time = datetime.now()
        
        for h in range(1, hours_ahead + 1):
            future_time = current_time + timedelta(hours=h)
            
            # Modify sensor data for future time
            future_sensor_data = sensor_data.copy()
            
            # Add time-based variations
            hour = future_time.hour
            if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
                future_sensor_data['air_quality'] = future_sensor_data.get('air_quality', 50) + 15
                future_sensor_data['noise_level'] = future_sensor_data.get('noise_level', 60) + 10
                future_sensor_data['crowd_density'] = future_sensor_data.get('crowd_density', 30) + 20
            elif 22 <= hour or hour <= 6:  # Night
                future_sensor_data['air_quality'] = max(20, future_sensor_data.get('air_quality', 50) - 10)
                future_sensor_data['noise_level'] = max(30, future_sensor_data.get('noise_level', 60) - 15)
                future_sensor_data['crowd_density'] = max(5, future_sensor_data.get('crowd_density', 30) - 15)
            
            # Add random variation for uncertainty
            for key in ['air_quality', 'noise_level', 'temperature', 'humidity', 'crowd_density']:
                if key in future_sensor_data:
                    variation = np.random.normal(0, 0.1 * h)  # Uncertainty increases with time
                    future_sensor_data[key] += variation
            
            stress_prediction = self.predict_sensor_stress(future_sensor_data)
            predictions.append((future_time, stress_prediction))
        
        return predictions
    
    def retrain_model(self, new_data: Dict[str, Any]) -> bool:
        """Retrain models with new data."""
        try:
            # Generate more training data including recent patterns
            training_data = self._generate_training_data(1500)
            
            # If we have real data, incorporate it
            if new_data and 'sensors' in new_data:
                real_data_rows = []
                for sensor in new_data['sensors']:
                    if sensor.get('status') == 'active':
                        features = self._extract_features(sensor)
                        stress = self._calculate_stress_from_factors(
                            features[0], features[1], features[2], features[3], 
                            features[4], int(features[5]), int(features[6])
                        )
                        
                        real_data_rows.append({
                            'air_quality': features[0],
                            'noise_level': features[1],
                            'temperature': features[2],
                            'humidity': features[3],
                            'crowd_density': features[4],
                            'hour_of_day': features[5],
                            'day_of_week': features[6],
                            'stress_level': stress
                        })
                
                if real_data_rows:
                    real_df = pd.DataFrame(real_data_rows)
                    training_data = pd.concat([training_data, real_df], ignore_index=True)
            
            # Retrain models
            X = training_data[['air_quality', 'noise_level', 'temperature', 'humidity', 
                              'crowd_density', 'hour_of_day', 'day_of_week']]
            y = training_data['stress_level']
            
            self._train_models(pd.DataFrame(X), pd.Series(y))
            return True
            
        except Exception as e:
            print(f"Error retraining models: {e}")
            return False
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance for trained models."""
        return self.feature_importance
    
    def get_model_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for all models."""
        return self.model_performance
    
    def predict_intervention_impact(self, sensor_data: Dict[str, Any], 
                                  intervention_type: str) -> Tuple[float, float]:
        """Predict the impact of an intervention on stress levels."""
        current_stress = self.predict_sensor_stress(sensor_data)
        
        # Intervention impact factors
        intervention_impacts = {
            'air_quality': {'air_quality': -30, 'estimated_reduction': 2.1},
            'noise_control': {'noise_level': -25, 'estimated_reduction': 1.8},
            'climate_control': {'temperature': -5, 'estimated_reduction': 2.0},
            'crowd_management': {'crowd_density': -30, 'estimated_reduction': 1.5},
            'lighting': {'estimated_reduction': 1.2},
            'green_space': {'air_quality': -15, 'noise_level': -10, 'estimated_reduction': 2.8}
        }
        
        if intervention_type not in intervention_impacts:
            return current_stress, 0.0
        
        # Apply intervention effects
        modified_sensor_data = sensor_data.copy()
        impact_config = intervention_impacts[intervention_type]
        
        for factor, change in impact_config.items():
            if factor in modified_sensor_data and factor != 'estimated_reduction':
                modified_sensor_data[factor] = max(0, modified_sensor_data[factor] + change)
        
        predicted_stress = self.predict_sensor_stress(modified_sensor_data)
        stress_reduction = current_stress - predicted_stress
        
        return predicted_stress, stress_reduction
    
    def analyze_stress_factors(self, sensor_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze which factors contribute most to current stress levels."""
        base_stress = self.predict_sensor_stress(sensor_data)
        factor_contributions = {}
        
        factors_to_test = ['air_quality', 'noise_level', 'temperature', 'humidity', 'crowd_density']
        
        for factor in factors_to_test:
            if factor in sensor_data:
                # Test impact by improving this factor
                test_data = sensor_data.copy()
                
                if factor == 'air_quality':
                    test_data[factor] = min(sensor_data[factor], 30)  # Good air quality
                elif factor == 'noise_level':
                    test_data[factor] = min(sensor_data[factor], 45)  # Quiet
                elif factor == 'temperature':
                    test_data[factor] = 22  # Optimal temperature
                elif factor == 'humidity':
                    test_data[factor] = 50  # Optimal humidity
                elif factor == 'crowd_density':
                    test_data[factor] = min(sensor_data[factor], 40)  # Moderate crowd
                
                improved_stress = self.predict_sensor_stress(test_data)
                contribution = base_stress - improved_stress
                factor_contributions[factor] = max(0, contribution)
        
        return factor_contributions
