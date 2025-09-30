import os
import random
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
from utils.iot_data_aggregator import IoTDataAggregator
from .database_manager import DatabaseManager
from .real_time_api_integration import RealTimeAPIIntegration

# Initialize IoT Data Aggregator with Google Maps API
# NOTE: Replace "YOUR_GOOGLE_MAPS_API_KEY" with your actual API key.
# For security, consider loading this from environment variables or a secrets manager.
IOT_AGGREGATOR = IoTDataAggregator(google_maps_api_key=os.getenv('google_maps_api_key'))

class SensorNetwork:
    """Simulates a network of IoT sensors for urban environmental monitoring."""

    def __init__(self, num_sensors: int = 20):
        self.num_sensors = num_sensors
        self.sensors = self._initialize_sensors()
        self.last_update = datetime.now()
        # Only enable real IoT data if environment variables are available
        self.use_real_data = os.getenv('ENABLE_REAL_IOT_DATA', 'false').lower() == 'true'

        try:
            if self.use_real_data:
                # The IoTDataAggregator is now initialized globally with the API key
                # self.iot_aggregator = IoTDataAggregator() # Removed: Use global IOT_AGGREGATOR
                self.iot_aggregator = IOT_AGGREGATOR
                self.db_manager = DatabaseManager()
                self.api_integration = RealTimeAPIIntegration()
                print("Real IoT data integration enabled")
            else:
                self.iot_aggregator = None
                self.db_manager = None
                self.api_integration = RealTimeAPIIntegration()  # Always initialize for demo
                print("Using simulated sensor data with API integration")
        except Exception as e:
            print(f"Warning: Could not initialize real data sources: {e}")
            self.use_real_data = False
            self.iot_aggregator = None
            self.db_manager = None
            self.api_integration = RealTimeAPIIntegration()  # Fallback to demo mode

    def _initialize_sensors(self) -> List[Dict[str, Any]]:
        """Initialize sensor network with realistic urban locations."""
        locations = [
          "Electronic City", "Koramangala", "Indiranagar", "Jayanagar", "BTM Layout", 
         "Marathahalli", "Banashankari", "Malleshwaram","Basavanagudi", "HSR Layout", 
          "Yeshwantpur", "Hebbal","Bellandur", "Sarjapur Road", "Brigade Road", "MG Road", 
          "Cunningham Road", "Vijayanagar",
        ]
        

        sensors = []
        base_lat, base_lon = 12.9716, 77.5946  # Bangalore coordinates

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
        # First, try to update with real API data
        if hasattr(self, 'api_integration'):
            self.update_sensors_with_real_data()
        
        for sensor in self.sensors:
            if sensor['status'] == 'active':
                # Occasionally sensors might go offline
                if random.random() < 0.02:  # 2% chance
                    sensor['status'] = 'offline'
                else:
                    # Update sensor readings, including crowd density using Google Maps API
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
        """Get current data from all sensors - real or simulated."""
        if self.use_real_data and self.iot_aggregator:
            return self._get_real_sensor_data()
        else:
            # Fallback to simulation
            if datetime.now() - self.last_update > timedelta(minutes=1):
                self.update_all_sensors()
            return self.sensors

    def _get_real_sensor_data(self) -> List[Dict[str, Any]]:
        """Get real sensor data from IoT aggregator and save to database."""
        try:
            real_readings = self.iot_aggregator.collect_all_sensor_data()
            sensor_data = []

            for reading in real_readings:
                # Convert IoT reading to sensor format expected by the app
                sensor = {
                    'sensor_id': reading.sensor_id,
                    'location': reading.location,
                    'latitude': reading.latitude,
                    'longitude': reading.longitude,
                    'air_quality': reading.air_quality,
                    'noise_level': reading.noise_level,
                    'temperature': reading.temperature,
                    'humidity': reading.humidity,
                    'crowd_density': reading.crowd_density,
                    'light_pollution': reading.light_pollution,
                    'timestamp': reading.timestamp.isoformat(),
                    'data_quality_score': reading.data_quality_score,
                    'status': 'active',
                    'battery_level': 85 + np.random.uniform(-15, 15),  # Simulate battery
                    'signal_strength': 80 + np.random.uniform(-20, 20),  # Simulate signal
                    'installation_date': datetime.now() - timedelta(days=random.randint(30, 365)),
                    'last_maintenance': datetime.now() - timedelta(days=random.randint(1, 90))
                }

                # Save to database if available
                if self.db_manager:
                    self.db_manager.save_sensor_data(sensor)
                    # Also save to historical readings
                    self.db_manager.save_sensor_reading_history(
                        reading.sensor_id, sensor,
                        None  # Stress level will be calculated later
                    )

                sensor_data.append(sensor)

            self.last_update = datetime.now()
            print(f"Successfully collected real data from {len(real_readings)} sensors")
            return sensor_data

        except Exception as e:
            print(f"Error collecting real sensor data: {e}")
            # Fallback to simulation
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

    def update_sensors_with_real_data(self) -> None:
        """Update sensor data using real API sources."""
        if not hasattr(self, 'api_integration'):
            return
            
        try:
            # Get real data for all locations
            all_locations_data = self.api_integration.get_all_locations_data()
            
            for sensor in self.sensors:
                location = sensor['location']
                
                # Find matching location data
                location_data = None
                for data in all_locations_data:
                    if data['location'] == location:
                        location_data = data
                        break
                
                if location_data:
                    # Update air quality data
                    air_quality = location_data.get('air_quality', {})
                    if air_quality:
                        sensor['air_quality'] = air_quality.get('aqi', sensor['air_quality'])
                    
                    # Update traffic data (affects noise levels)
                    traffic = location_data.get('traffic', {})
                    if traffic:
                        congestion = traffic.get('congestion_level', 0)
                        # Higher congestion = higher noise levels
                        base_noise = sensor['noise_level']
                        sensor['noise_level'] = base_noise + (congestion * 20)  # Add up to 20dB
                    
                    # Update crowd density
                    crowd = location_data.get('crowd', {})
                    if crowd:
                        sensor['crowd_density'] = crowd.get('density_level', sensor['crowd_density']) * 100
                    
                    # Update timestamp
                    sensor['last_updated'] = datetime.now()
                    
        except Exception as e:
            print(f"Error updating sensors with real data: {e}")

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