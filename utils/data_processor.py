import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

class DataProcessor:
    """Processes and analyzes sensor data for urban mental health monitoring."""
    
    def __init__(self):
        self.processed_cache = {}
        self.historical_cache = {}
        self.last_processed = None
    
    def process_sensor_data(self, sensor_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process raw sensor data into analyzed format for ML and visualization."""
        if not sensor_data:
            return {'sensors': [], 'summary': {}, 'alerts': []}
        
        current_time = datetime.now()
        processed_sensors = []
        alerts = []
        
        # Process each sensor
        for sensor in sensor_data:
            if sensor.get('status') != 'active':
                continue
                
            processed_sensor = sensor.copy()
            
            # Add derived metrics
            processed_sensor['air_quality_category'] = self._categorize_air_quality(sensor.get('air_quality', 50))
            processed_sensor['noise_category'] = self._categorize_noise_level(sensor.get('noise_level', 60))
            processed_sensor['comfort_index'] = self._calculate_comfort_index(sensor)
            processed_sensor['health_risk_score'] = self._calculate_health_risk(sensor)
            
            # Time-based context
            processed_sensor['hour'] = current_time.hour
            processed_sensor['day_of_week'] = current_time.weekday()
            processed_sensor['is_weekend'] = current_time.weekday() >= 5
            processed_sensor['is_rush_hour'] = self._is_rush_hour(current_time.hour)
            
            # Generate alerts for concerning readings
            sensor_alerts = self._generate_sensor_alerts(sensor)
            alerts.extend(sensor_alerts)
            
            processed_sensors.append(processed_sensor)
        
        # Calculate summary statistics
        summary = self._calculate_summary_statistics(processed_sensors)
        
        # Add temporal context
        summary['timestamp'] = current_time.isoformat()
        summary['rush_hour'] = self._is_rush_hour(current_time.hour)
        summary['weekend'] = current_time.weekday() >= 5
        summary['time_of_day'] = self._get_time_of_day_category(current_time.hour)
        
        processed_data = {
            'sensors': processed_sensors,
            'summary': summary,
            'alerts': alerts,
            'metadata': {
                'processed_at': current_time.isoformat(),
                'total_sensors': len(sensor_data),
                'active_sensors': len(processed_sensors),
                'data_quality': self._assess_data_quality(processed_sensors)
            }
        }
        
        # Cache the processed data
        self.processed_cache = processed_data
        self.last_processed = current_time
        
        return processed_data
    
    def _categorize_air_quality(self, aqi: float) -> str:
        """Categorize air quality index into standard categories."""
        if aqi <= 50:
            return 'Good'
        elif aqi <= 100:
            return 'Moderate'
        elif aqi <= 150:
            return 'Unhealthy for Sensitive Groups'
        elif aqi <= 200:
            return 'Unhealthy'
        elif aqi <= 300:
            return 'Very Unhealthy'
        else:
            return 'Hazardous'
    
    def _categorize_noise_level(self, noise: float) -> str:
        """Categorize noise levels."""
        if noise <= 50:
            return 'Quiet'
        elif noise <= 70:
            return 'Moderate'
        elif noise <= 85:
            return 'Loud'
        else:
            return 'Very Loud'
    
    def _calculate_comfort_index(self, sensor: Dict[str, Any]) -> float:
        """Calculate overall comfort index (0-100) based on environmental factors."""
        # Temperature comfort (optimal 18-24°C)
        temp = sensor.get('temperature', 20)
        temp_comfort = max(0, 100 - abs(temp - 21) * 5)
        
        # Humidity comfort (optimal 40-60%)
        humidity = sensor.get('humidity', 50)
        if 40 <= humidity <= 60:
            humidity_comfort = 100
        elif 30 <= humidity <= 70:
            humidity_comfort = 80
        else:
            humidity_comfort = max(0, 100 - abs(humidity - 50) * 2)
        
        # Air quality comfort
        aqi = sensor.get('air_quality', 50)
        aqi_comfort = max(0, 100 - aqi * 0.8)
        
        # Noise comfort
        noise = sensor.get('noise_level', 60)
        noise_comfort = max(0, 100 - max(0, noise - 50) * 2)
        
        # Crowd comfort (moderate crowds are okay)
        crowd = sensor.get('crowd_density', 30)
        if crowd <= 70:
            crowd_comfort = 100 - crowd * 0.5
        else:
            crowd_comfort = max(0, 100 - crowd * 1.5)
        
        # Weighted average
        comfort_index = (
            temp_comfort * 0.25 +
            humidity_comfort * 0.15 +
            aqi_comfort * 0.3 +
            noise_comfort * 0.2 +
            crowd_comfort * 0.1
        )
        
        return round(comfort_index, 2)
    
    def _calculate_health_risk(self, sensor: Dict[str, Any]) -> float:
        """Calculate health risk score (0-10) where higher is more risky."""
        risk_score = 0.0
        
        # Air quality risk
        aqi = sensor.get('air_quality', 50)
        if aqi > 150:
            risk_score += 4.0
        elif aqi > 100:
            risk_score += 2.5
        elif aqi > 50:
            risk_score += 1.0
        
        # Noise pollution risk
        noise = sensor.get('noise_level', 60)
        if noise > 85:
            risk_score += 3.0
        elif noise > 70:
            risk_score += 1.5
        elif noise > 60:
            risk_score += 0.5
        
        # Temperature extremes
        temp = sensor.get('temperature', 20)
        if temp > 35 or temp < 0:
            risk_score += 2.0
        elif temp > 30 or temp < 5:
            risk_score += 1.0
        
        # High crowd density during health concerns
        crowd = sensor.get('crowd_density', 30)
        if crowd > 80:
            risk_score += 1.0
        
        return min(10.0, round(risk_score, 2))
    
    def _is_rush_hour(self, hour: int) -> bool:
        """Determine if current hour is rush hour."""
        return (7 <= hour <= 9) or (17 <= hour <= 19)
    
    def _get_time_of_day_category(self, hour: int) -> str:
        """Get time of day category."""
        if 6 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 18:
            return 'Afternoon'
        elif 18 <= hour < 22:
            return 'Evening'
        else:
            return 'Night'
    
    def _generate_sensor_alerts(self, sensor: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alerts based on sensor readings."""
        alerts = []
        current_time = datetime.now()
        
        # Air quality alerts
        aqi = sensor.get('air_quality', 50)
        if aqi > 200:
            alerts.append({
                'type': 'CRITICAL',
                'category': 'Air Quality',
                'message': f"Hazardous air quality detected at {sensor.get('location', 'Unknown')}",
                'value': aqi,
                'threshold': 200,
                'sensor_id': sensor.get('sensor_id'),
                'timestamp': current_time.isoformat()
            })
        elif aqi > 150:
            alerts.append({
                'type': 'HIGH',
                'category': 'Air Quality',
                'message': f"Unhealthy air quality at {sensor.get('location', 'Unknown')}",
                'value': aqi,
                'threshold': 150,
                'sensor_id': sensor.get('sensor_id'),
                'timestamp': current_time.isoformat()
            })
        
        # Noise alerts
        noise = sensor.get('noise_level', 60)
        if noise > 90:
            alerts.append({
                'type': 'HIGH',
                'category': 'Noise Pollution',
                'message': f"Dangerous noise levels at {sensor.get('location', 'Unknown')}",
                'value': noise,
                'threshold': 90,
                'sensor_id': sensor.get('sensor_id'),
                'timestamp': current_time.isoformat()
            })
        elif noise > 80:
            alerts.append({
                'type': 'MEDIUM',
                'category': 'Noise Pollution',
                'message': f"High noise levels at {sensor.get('location', 'Unknown')}",
                'value': noise,
                'threshold': 80,
                'sensor_id': sensor.get('sensor_id'),
                'timestamp': current_time.isoformat()
            })
        
        # Temperature alerts
        temp = sensor.get('temperature', 20)
        if temp > 40:
            alerts.append({
                'type': 'CRITICAL',
                'category': 'Extreme Heat',
                'message': f"Extreme heat warning at {sensor.get('location', 'Unknown')}",
                'value': temp,
                'threshold': 40,
                'sensor_id': sensor.get('sensor_id'),
                'timestamp': current_time.isoformat()
            })
        elif temp < -10:
            alerts.append({
                'type': 'CRITICAL',
                'category': 'Extreme Cold',
                'message': f"Extreme cold warning at {sensor.get('location', 'Unknown')}",
                'value': temp,
                'threshold': -10,
                'sensor_id': sensor.get('sensor_id'),
                'timestamp': current_time.isoformat()
            })
        
        # Crowd density alerts
        crowd = sensor.get('crowd_density', 30)
        if crowd > 90:
            alerts.append({
                'type': 'HIGH',
                'category': 'Overcrowding',
                'message': f"Severe overcrowding at {sensor.get('location', 'Unknown')}",
                'value': crowd,
                'threshold': 90,
                'sensor_id': sensor.get('sensor_id'),
                'timestamp': current_time.isoformat()
            })
        
        return alerts
    
    def _calculate_summary_statistics(self, sensors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics for all active sensors."""
        if not sensors:
            return {}
        
        # Extract numeric values
        air_quality_values = [s.get('air_quality', 0) for s in sensors]
        noise_values = [s.get('noise_level', 0) for s in sensors]
        temperature_values = [s.get('temperature', 0) for s in sensors]
        humidity_values = [s.get('humidity', 0) for s in sensors]
        crowd_values = [s.get('crowd_density', 0) for s in sensors]
        comfort_values = [s.get('comfort_index', 0) for s in sensors]
        health_risk_values = [s.get('health_risk_score', 0) for s in sensors]
        
        summary = {
            'air_quality': {
                'mean': round(np.mean(air_quality_values), 2),
                'median': round(np.median(air_quality_values), 2),
                'std': round(np.std(air_quality_values), 2),
                'min': round(np.min(air_quality_values), 2),
                'max': round(np.max(air_quality_values), 2)
            },
            'noise_level': {
                'mean': round(np.mean(noise_values), 2),
                'median': round(np.median(noise_values), 2),
                'std': round(np.std(noise_values), 2),
                'min': round(np.min(noise_values), 2),
                'max': round(np.max(noise_values), 2)
            },
            'temperature': {
                'mean': round(np.mean(temperature_values), 2),
                'median': round(np.median(temperature_values), 2),
                'std': round(np.std(temperature_values), 2),
                'min': round(np.min(temperature_values), 2),
                'max': round(np.max(temperature_values), 2)
            },
            'humidity': {
                'mean': round(np.mean(humidity_values), 2),
                'median': round(np.median(humidity_values), 2),
                'std': round(np.std(humidity_values), 2),
                'min': round(np.min(humidity_values), 2),
                'max': round(np.max(humidity_values), 2)
            },
            'crowd_density': {
                'mean': round(np.mean(crowd_values), 2),
                'median': round(np.median(crowd_values), 2),
                'std': round(np.std(crowd_values), 2),
                'min': round(np.min(crowd_values), 2),
                'max': round(np.max(crowd_values), 2)
            },
            'comfort_index': {
                'mean': round(np.mean(comfort_values), 2),
                'median': round(np.median(comfort_values), 2),
                'std': round(np.std(comfort_values), 2),
                'min': round(np.min(comfort_values), 2),
                'max': round(np.max(comfort_values), 2)
            },
            'health_risk': {
                'mean': round(np.mean(health_risk_values), 2),
                'median': round(np.median(health_risk_values), 2),
                'std': round(np.std(health_risk_values), 2),
                'min': round(np.min(health_risk_values), 2),
                'max': round(np.max(health_risk_values), 2)
            }
        }
        
        # Add categorical summaries
        aqi_categories = [self._categorize_air_quality(aqi) for aqi in air_quality_values]
        noise_categories = [self._categorize_noise_level(noise) for noise in noise_values]
        
        summary['air_quality_distribution'] = {
            category: aqi_categories.count(category)
            for category in ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy', 'Hazardous']
        }
        
        summary['noise_distribution'] = {
            category: noise_categories.count(category)
            for category in ['Quiet', 'Moderate', 'Loud', 'Very Loud']
        }
        
        return summary
    
    def _assess_data_quality(self, sensors: List[Dict[str, Any]]) -> float:
        """Assess overall data quality (0-100)."""
        if not sensors:
            return 0.0
        
        quality_score = 0.0
        total_sensors = len(sensors)
        
        for sensor in sensors:
            sensor_quality = 0.0
            
            # Check for presence of key metrics
            required_fields = ['air_quality', 'noise_level', 'temperature', 'humidity', 'crowd_density']
            present_fields = sum(1 for field in required_fields if field in sensor and sensor[field] is not None)
            sensor_quality += (present_fields / len(required_fields)) * 60  # 60% for completeness
            
            # Check for reasonable values
            reasonable_values = 0
            total_checks = 0
            
            if 'air_quality' in sensor:
                if 0 <= sensor['air_quality'] <= 500:
                    reasonable_values += 1
                total_checks += 1
            
            if 'noise_level' in sensor:
                if 20 <= sensor['noise_level'] <= 120:
                    reasonable_values += 1
                total_checks += 1
            
            if 'temperature' in sensor:
                if -20 <= sensor['temperature'] <= 50:
                    reasonable_values += 1
                total_checks += 1
            
            if 'humidity' in sensor:
                if 0 <= sensor['humidity'] <= 100:
                    reasonable_values += 1
                total_checks += 1
            
            if 'crowd_density' in sensor:
                if 0 <= sensor['crowd_density'] <= 100:
                    reasonable_values += 1
                total_checks += 1
            
            if total_checks > 0:
                sensor_quality += (reasonable_values / total_checks) * 30  # 30% for reasonable values
            
            # Timestamp recency (10% for timing)
            if 'timestamp' in sensor:
                try:
                    sensor_time = datetime.fromisoformat(sensor['timestamp'].replace('Z', '+00:00'))
                    age_minutes = (datetime.now() - sensor_time).total_seconds() / 60
                    if age_minutes <= 5:
                        sensor_quality += 10
                    elif age_minutes <= 15:
                        sensor_quality += 5
                except:
                    pass
            
            quality_score += sensor_quality
        
        return round(quality_score / total_sensors, 2)
    
    def generate_historical_data(self, time_range: str) -> pd.DataFrame:
        """Generate historical data for trend analysis."""
        current_time = datetime.now()
        
        # Determine time range
        if time_range == "Last Hour":
            start_time = current_time - timedelta(hours=1)
            freq = '5T'  # 5-minute intervals
        elif time_range == "Last 6 Hours":
            start_time = current_time - timedelta(hours=6)
            freq = '15T'  # 15-minute intervals
        elif time_range == "Last 24 Hours":
            start_time = current_time - timedelta(hours=24)
            freq = 'H'  # Hourly intervals
        elif time_range == "Last Week":
            start_time = current_time - timedelta(weeks=1)
            freq = '4H'  # 4-hour intervals
        else:
            start_time = current_time - timedelta(hours=24)
            freq = 'H'
        
        # Generate time series
        time_points = pd.date_range(start=start_time, end=current_time, freq=freq)
        
        historical_data = []
        for timestamp in time_points:
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            
            # Base values with realistic patterns
            air_quality = 50 + np.random.normal(0, 15)
            noise_level = 60 + np.random.normal(0, 10)
            temperature = 20 + 8 * np.sin((hour - 6) * np.pi / 12) + np.random.normal(0, 2)
            humidity = 65 - (temperature - 20) * 1.5 + np.random.normal(0, 10)
            crowd_density = 40 + np.random.normal(0, 15)
            
            # Add time-based patterns
            if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
                air_quality += 20
                noise_level += 15
                crowd_density += 25
            elif 22 <= hour or hour <= 6:  # Night
                air_quality -= 10
                noise_level -= 20
                crowd_density -= 20
            
            # Weekend patterns
            if day_of_week >= 5:  # Weekend
                if 12 <= hour <= 22:
                    crowd_density += 15  # More people out during weekend days
                else:
                    crowd_density -= 10
                noise_level += 5 if 12 <= hour <= 24 else -10
            
            # Clamp values to realistic ranges
            air_quality = max(10, min(400, air_quality))
            noise_level = max(30, min(100, noise_level))
            humidity = max(20, min(95, humidity))
            crowd_density = max(0, min(100, crowd_density))
            
            historical_data.append({
                'timestamp': timestamp,
                'air_quality': round(air_quality, 1),
                'noise_level': round(noise_level, 1),
                'temperature': round(temperature, 1),
                'humidity': round(humidity, 1),
                'crowd_density': round(crowd_density, 1),
                'hour': hour,
                'day_of_week': day_of_week,
                'is_weekend': day_of_week >= 5,
                'is_rush_hour': 7 <= hour <= 9 or 17 <= hour <= 19
            })
        
        return pd.DataFrame(historical_data)
    
    def analyze_trends(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends in historical data."""
        if historical_data.empty:
            return {}
        
        trends = {}
        numeric_columns = ['air_quality', 'noise_level', 'temperature', 'humidity', 'crowd_density']
        
        for column in numeric_columns:
            if column in historical_data.columns:
                values = historical_data[column].values
                x = np.arange(len(values))
                
                # Calculate trend (slope)
                slope, intercept = np.polyfit(x, np.array(values), 1)
                
                # Calculate correlation coefficient
                correlation = np.corrcoef(x, np.array(values))[0, 1]
                
                trends[column] = {
                    'slope': round(slope, 4),
                    'correlation': round(correlation, 4),
                    'direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
                    'strength': 'strong' if abs(correlation) > 0.7 else 'moderate' if abs(correlation) > 0.4 else 'weak'
                }
        
        return trends
    
    def get_cached_data(self) -> Optional[Dict[str, Any]]:
        """Get cached processed data if available and recent."""
        if (self.last_processed and 
            datetime.now() - self.last_processed < timedelta(minutes=5) and
            self.processed_cache):
            return self.processed_cache
        return None
    
    def export_data(self, data: Dict[str, Any], format: str = 'json') -> str:
        """Export processed data in specified format."""
        if format.lower() == 'json':
            return json.dumps(data, indent=2, default=str)
        elif format.lower() == 'csv':
            if 'sensors' in data:
                df = pd.DataFrame(data['sensors'])
                return df.to_csv(index=False)
        return str(data)
