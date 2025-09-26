import random
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any

class SensorNetwork:
    """Simulates a network of IoT sensors for urban environmental monitoring."""
    
    def __init__(self, num_sensors: int = 20):
        self.num_sensors = num_sensors
        self.sensors = self._initialize_sensors()
        self.last_update = datetime.now()
    
    def _initialize_sensors(self) -> List[Dict[str, Any]]:
        """Initialize sensor network with realistic urban locations."""
        locations = [
            "Central Park", "Times Square", "Brooklyn Bridge", "Wall Street",
            "Soho District", "Greenwich Village", "Upper East Side", "Chinatown",
            "Financial District", "Tribeca", "Lower East Side", "Midtown",
            "Chelsea Market", "High Line Park", "Bryant Park", "Union Square",
            "Madison Square", "Flatiron District", "East Village", "West Village"
        ]
        
        sensors = []
        base_lat, base_lon = 40.7128, -74.0060  # NYC coordinates
        
        for i in range(self.num_sensors):
            # Generate realistic coordinates around NYC
            lat_offset = np.random.uniform(-0.05, 0.05)
            lon_offset = np.random.uniform(-0.05, 0.05)
            
            sensor = {
                'sensor_id': f"UHN-{str(i+1).zfill(3)}",
                'location': locations[i % len(locations)],
                'latitude': base_lat + lat_offset,
                'longitude': base_lon + lon_offset,
                'status': random.choice(['active', 'active', 'active', 'maintenance', 'offline']),
                'installation_date': datetime.now() - timedelta(days=random.randint(30, 365)),
                'last_maintenance': datetime.now() - timedelta(days=random.randint(1, 90)),
                'battery_level': random.uniform(20, 100),
                'signal_strength': random.uniform(60, 100)
            }
            
            # Generate initial sensor readings
            sensor.update(self._generate_sensor_readings(sensor))
            sensors.append(sensor)
        
        return sensors
    
    def _generate_sensor_readings(self, sensor: Dict[str, Any]) -> Dict[str, Any]:
        """Generate realistic sensor readings based on location and time."""
        current_time = datetime.now()
        hour = current_time.hour
        
        # Base readings with location-specific variations
        location_factors = {
            'Times Square': {'noise_boost': 20, 'crowd_boost': 40, 'air_penalty': 15},
            'Central Park': {'noise_reduction': 15, 'air_improvement': 20, 'crowd_reduction': 20},
            'Wall Street': {'crowd_boost': 25, 'air_penalty': 10},
            'Brooklyn Bridge': {'noise_boost': 15, 'crowd_boost': 30},
            'Financial District': {'crowd_boost': 20, 'air_penalty': 12}
        }
        
        factors = location_factors.get(sensor['location'], {})
        
        # Time-based patterns
        rush_hour = 7 <= hour <= 9 or 17 <= hour <= 19
        night_time = 22 <= hour or hour <= 6
        
        # Air Quality Index (0-500 scale, lower is better)
        base_aqi = 45 + np.random.normal(0, 15)
        if rush_hour:
            base_aqi += 25
        if night_time:
            base_aqi -= 10
        base_aqi += factors.get('air_penalty', 0) - factors.get('air_improvement', 0)
        air_quality = max(0, min(500, base_aqi))
        
        # Noise Level (decibels)
        base_noise = 55 + np.random.normal(0, 8)
        if rush_hour:
            base_noise += 15
        if night_time:
            base_noise -= 20
        base_noise += factors.get('noise_boost', 0) - factors.get('noise_reduction', 0)
        noise_level = max(30, min(120, base_noise))
        
        # Temperature (Celsius) - realistic for urban environment
        base_temp = 20 + 10 * np.sin((hour - 6) * np.pi / 12)  # Daily temperature cycle
        base_temp += np.random.normal(0, 3)
        # Urban heat island effect
        base_temp += 2
        temperature = max(-10, min(45, base_temp))
        
        # Humidity (percentage)
        base_humidity = 60 + np.random.normal(0, 15)
        # Inverse relationship with temperature
        humidity_temp_factor = (25 - temperature) * 0.5
        humidity = max(20, min(95, base_humidity + humidity_temp_factor))
        
        # Crowd Density (percentage of maximum capacity)
        base_crowd = 30 + np.random.normal(0, 15)
        if rush_hour:
            base_crowd += 30
        if night_time:
            base_crowd -= 20
        # Weekend patterns
        if current_time.weekday() >= 5:  # Weekend
            if sensor['location'] in ['Central Park', 'Times Square']:
                base_crowd += 20
            else:
                base_crowd -= 10
        
        base_crowd += factors.get('crowd_boost', 0) - factors.get('crowd_reduction', 0)
        crowd_density = max(0, min(100, base_crowd))
        
        # Light pollution (lux)
        base_light = 50 if night_time else 1000 + np.random.normal(0, 200)
        if sensor['location'] == 'Times Square':
            base_light *= 3
        light_pollution = max(0, base_light)
        
        return {
            'air_quality': round(air_quality, 1),
            'noise_level': round(noise_level, 1),
            'temperature': round(temperature, 1),
            'humidity': round(humidity, 1),
            'crowd_density': round(crowd_density, 1),
            'light_pollution': round(light_pollution, 1),
            'timestamp': current_time.isoformat(),
            'data_quality_score': random.uniform(85, 99)
        }
    
    def update_all_sensors(self) -> None:
        """Update readings for all sensors."""
        for sensor in self.sensors:
            if sensor['status'] == 'active':
                # Occasionally sensors might go offline
                if random.random() < 0.02:  # 2% chance
                    sensor['status'] = 'offline'
                else:
                    sensor.update(self._generate_sensor_readings(sensor))
            elif sensor['status'] == 'maintenance':
                # Sensors in maintenance might come back online
                if random.random() < 0.3:  # 30% chance
                    sensor['status'] = 'active'
                    sensor.update(self._generate_sensor_readings(sensor))
            elif sensor['status'] == 'offline':
                # Offline sensors might be fixed
                if random.random() < 0.1:  # 10% chance
                    sensor['status'] = 'active'
                    sensor.update(self._generate_sensor_readings(sensor))
        
        self.last_update = datetime.now()
    
    def get_all_sensor_data(self) -> List[Dict[str, Any]]:
        """Get current data from all sensors."""
        # Auto-update if data is older than 1 minute
        if datetime.now() - self.last_update > timedelta(minutes=1):
            self.update_all_sensors()
        
        return self.sensors
    
    def get_sensor_by_id(self, sensor_id: str) -> Dict[str, Any]:
        """Get data from a specific sensor."""
        for sensor in self.sensors:
            if sensor['sensor_id'] == sensor_id:
                return sensor
        return {}
    
    def get_sensors_by_location(self, location: str) -> List[Dict[str, Any]]:
        """Get all sensors in a specific location."""
        return [sensor for sensor in self.sensors if sensor['location'] == location]
    
    def get_active_sensors(self) -> List[Dict[str, Any]]:
        """Get only active sensors."""
        return [sensor for sensor in self.sensors if sensor['status'] == 'active']
    
    def get_sensor_statistics(self) -> Dict[str, Any]:
        """Get network-wide statistics."""
        active_count = len([s for s in self.sensors if s['status'] == 'active'])
        maintenance_count = len([s for s in self.sensors if s['status'] == 'maintenance'])
        offline_count = len([s for s in self.sensors if s['status'] == 'offline'])
        
        active_sensors = self.get_active_sensors()
        
        if active_sensors:
            avg_air_quality = np.mean([s['air_quality'] for s in active_sensors])
            avg_noise = np.mean([s['noise_level'] for s in active_sensors])
            avg_temperature = np.mean([s['temperature'] for s in active_sensors])
            avg_humidity = np.mean([s['humidity'] for s in active_sensors])
            avg_crowd = np.mean([s['crowd_density'] for s in active_sensors])
        else:
            avg_air_quality = avg_noise = avg_temperature = avg_humidity = avg_crowd = 0
        
        return {
            'total_sensors': len(self.sensors),
            'active_sensors': active_count,
            'maintenance_sensors': maintenance_count,
            'offline_sensors': offline_count,
            'uptime_percentage': (active_count / len(self.sensors)) * 100,
            'average_readings': {
                'air_quality': round(avg_air_quality, 1),
                'noise_level': round(avg_noise, 1),
                'temperature': round(avg_temperature, 1),
                'humidity': round(avg_humidity, 1),
                'crowd_density': round(avg_crowd, 1)
            },
            'last_update': self.last_update.isoformat()
        }
    
    def get_last_update_time(self) -> str:
        """Get the last update timestamp."""
        return self.last_update.strftime('%Y-%m-%d %H:%M:%S')
    
    def simulate_sensor_failure(self, sensor_id: str) -> bool:
        """Simulate a sensor failure for testing."""
        for sensor in self.sensors:
            if sensor['sensor_id'] == sensor_id:
                sensor['status'] = 'offline'
                return True
        return False
    
    def repair_sensor(self, sensor_id: str) -> bool:
        """Repair a failed sensor."""
        for sensor in self.sensors:
            if sensor['sensor_id'] == sensor_id:
                sensor['status'] = 'active'
                sensor['last_maintenance'] = datetime.now()
                sensor.update(self._generate_sensor_readings(sensor))
                return True
        return False
