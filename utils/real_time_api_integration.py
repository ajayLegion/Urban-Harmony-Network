"""
Real-time API Integration for Urban Harmony Network
Integrates with multiple data sources for Bangalore locations
"""

import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import os
from dataclasses import dataclass
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AirQualityData:
    """Data class for air quality information"""
    location: str
    aqi: float
    pm25: float
    pm10: float
    no2: float
    so2: float
    co: float
    o3: float
    timestamp: datetime
    source: str

@dataclass
class TrafficData:
    """Data class for traffic information"""
    location: str
    congestion_level: float  # 0-1 scale
    travel_time: int  # minutes
    distance: float  # km
    timestamp: datetime
    source: str

@dataclass
class CrowdData:
    """Data class for crowd density information"""
    location: str
    density_level: float  # 0-1 scale
    popular_times: Dict[str, float]  # hourly data
    timestamp: datetime
    source: str

class RealTimeAPIIntegration:
    """Main class for integrating with real-time data APIs"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'UrbanHarmonyNetwork/1.0',
            'Accept': 'application/json'
        })
        
        # API endpoints
        self.apis = {
            'bharat_aqi': 'https://bharataqi-api.herokuapp.com/api/aqi',
            'data_gov': 'https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69',
            'cpcb': 'https://app.cpcbccr.com/ccr/caaqms/caaqms_landing_data',
            'google_places': 'https://maps.googleapis.com/maps/api/place',
            'google_traffic': 'https://maps.googleapis.com/maps/api/distancematrix'
        }
        
        # Bangalore locations with their coordinates
        self.bangalore_locations = {
            "Whitefield": {"lat": 12.9698, "lon": 77.7500},
            "Electronic City": {"lat": 12.8456, "lon": 77.6603},
            "Koramangala": {"lat": 12.9279, "lon": 77.6271},
            "Indiranagar": {"lat": 12.9784, "lon": 77.6408},
            "Jayanagar": {"lat": 12.9249, "lon": 77.5838},
            "HSR Layout": {"lat": 12.9116, "lon": 77.6370},
            "BTM Layout": {"lat": 12.9165, "lon": 77.6101},
            "Marathahalli": {"lat": 12.9591, "lon": 77.6974},
            "Banashankari": {"lat": 12.9081, "lon": 77.5536},
            "Rajajinagar": {"lat": 12.9915, "lon": 77.5526},
            "Malleshwaram": {"lat": 13.0037, "lon": 77.5619},
            "Basavanagudi": {"lat": 12.9395, "lon": 77.5745},
            "Yeshwantpur": {"lat": 13.0284, "lon": 77.5547},
            "Hebbal": {"lat": 13.0362, "lon": 77.5970},
            "Bellandur": {"lat": 12.9249, "lon": 77.6733},
            "Sarjapur Road": {"lat": 12.9010, "lon": 77.6874},
            "Brigade Road": {"lat": 12.9716, "lon": 77.6197},
            "MG Road": {"lat": 12.9759, "lon": 77.6094},
            "Cunningham Road": {"lat": 12.9840, "lon": 77.5949},
            "Vijayanagar": {"lat": 12.9634, "lon": 77.5305}
        }
        
        # Cache for API responses
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
        # API keys (should be set as environment variables)
        self.google_api_key = os.getenv('GOOGLE_MAPS_API_KEY', 'demo_key')
        self.data_gov_api_key = os.getenv('DATA_GOV_API_KEY', 'demo_key')

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache:
            return False
        
        cache_time = self.cache[cache_key].get('timestamp')
        if not cache_time:
            return False
            
        return (datetime.now() - cache_time).seconds < self.cache_duration

    def _make_api_request(self, url: str, params: Dict = None, headers: Dict = None) -> Optional[Dict]:
        """Make API request with error handling and caching"""
        cache_key = f"{url}_{hash(str(params))}"
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            logger.info(f"Using cached data for {url}")
            return self.cache[cache_key]['data']
        
        try:
            request_headers = self.session.headers.copy()
            if headers:
                request_headers.update(headers)
                
            response = self.session.get(url, params=params, headers=request_headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Cache the response
            self.cache[cache_key] = {
                'data': data,
                'timestamp': datetime.now()
            }
            
            logger.info(f"Successfully fetched data from {url}")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for {url}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from {url}: {e}")
            return None

    def get_bharat_aqi_data(self, location: str) -> Optional[AirQualityData]:
        """Get air quality data from Bharat AQI API"""
        try:
            params = {
                'city': 'Bangalore',
                'area': location
            }
            
            data = self._make_api_request(self.apis['bharat_aqi'], params)
            if not data:
                return None
            
            # Parse Bharat AQI response format
            aqi_data = data.get('data', {})
            
            return AirQualityData(
                location=location,
                aqi=float(aqi_data.get('aqi', 0)),
                pm25=float(aqi_data.get('pm25', 0)),
                pm10=float(aqi_data.get('pm10', 0)),
                no2=float(aqi_data.get('no2', 0)),
                so2=float(aqi_data.get('so2', 0)),
                co=float(aqi_data.get('co', 0)),
                o3=float(aqi_data.get('o3', 0)),
                timestamp=datetime.now(),
                source='Bharat AQI'
            )
            
        except Exception as e:
            logger.error(f"Error fetching Bharat AQI data for {location}: {e}")
            return None

    def get_datagov_data(self, location: str) -> Optional[AirQualityData]:
        """Get environmental data from Data.gov.in API"""
        try:
            params = {
                'api-key': self.data_gov_api_key,
                'format': 'json',
                'limit': 10,
                'filters[state]': 'Karnataka',
                'filters[city]': 'Bangalore'
            }
            
            data = self._make_api_request(self.apis['data_gov'], params)
            if not data:
                return None
            
            records = data.get('records', [])
            if not records:
                return None
            
            # Find the most recent record for the location
            latest_record = records[0]  # Assuming records are sorted by date
            
            return AirQualityData(
                location=location,
                aqi=float(latest_record.get('aqi', 0)),
                pm25=float(latest_record.get('pm25', 0)),
                pm10=float(latest_record.get('pm10', 0)),
                no2=float(latest_record.get('no2', 0)),
                so2=float(latest_record.get('so2', 0)),
                co=float(latest_record.get('co', 0)),
                o3=float(latest_record.get('o3', 0)),
                timestamp=datetime.now(),
                source='Data.gov.in'
            )
            
        except Exception as e:
            logger.error(f"Error fetching Data.gov.in data for {location}: {e}")
            return None

    def get_cpcb_data(self, location: str) -> Optional[AirQualityData]:
        """Get air quality data from CPCB API"""
        try:
            # CPCB API might require different parameters
            params = {
                'state': 'Karnataka',
                'city': 'Bangalore',
                'station': location
            }
            
            data = self._make_api_request(self.apis['cpcb'], params)
            if not data:
                return None
            
            # Parse CPCB response format
            station_data = data.get('data', {})
            
            return AirQualityData(
                location=location,
                aqi=float(station_data.get('aqi', 0)),
                pm25=float(station_data.get('pm25', 0)),
                pm10=float(station_data.get('pm10', 0)),
                no2=float(station_data.get('no2', 0)),
                so2=float(station_data.get('so2', 0)),
                co=float(station_data.get('co', 0)),
                o3=float(station_data.get('o3', 0)),
                timestamp=datetime.now(),
                source='CPCB'
            )
            
        except Exception as e:
            logger.error(f"Error fetching CPCB data for {location}: {e}")
            return None

    def get_google_maps_crowd_data(self, location: str) -> Optional[CrowdData]:
        """Get crowd density data from Google Maps Places API"""
        try:
            if self.google_api_key == 'demo_key':
                # Return simulated data for demo
                return self._get_simulated_crowd_data(location)
            
            coords = self.bangalore_locations.get(location)
            if not coords:
                return None
            
            # Search for places in the area
            search_url = f"{self.apis['google_places']}/nearbysearch/json"
            params = {
                'key': self.google_api_key,
                'location': f"{coords['lat']},{coords['lon']}",
                'radius': 1000,  # 1km radius
                'type': 'establishment'
            }
            
            data = self._make_api_request(search_url, params)
            if not data:
                return None
            
            results = data.get('results', [])
            if not results:
                return None
            
            # Calculate crowd density based on number of places and their popularity
            density_level = min(len(results) / 50.0, 1.0)  # Normalize to 0-1
            
            # Generate popular times data (simulated)
            popular_times = {}
            for hour in range(24):
                # Simulate higher density during peak hours
                if 8 <= hour <= 10 or 17 <= hour <= 19:
                    popular_times[str(hour)] = min(density_level * 1.5, 1.0)
                elif 12 <= hour <= 14:
                    popular_times[str(hour)] = min(density_level * 1.2, 1.0)
                else:
                    popular_times[str(hour)] = density_level * 0.3
            
            return CrowdData(
                location=location,
                density_level=density_level,
                popular_times=popular_times,
                timestamp=datetime.now(),
                source='Google Maps Places'
            )
            
        except Exception as e:
            logger.error(f"Error fetching Google Maps crowd data for {location}: {e}")
            return None

    def get_google_maps_traffic_data(self, location: str) -> Optional[TrafficData]:
        """Get traffic data from Google Maps Traffic API"""
        try:
            if self.google_api_key == 'demo_key':
                # Return simulated data for demo
                return self._get_simulated_traffic_data(location)
            
            coords = self.bangalore_locations.get(location)
            if not coords:
                return None
            
            # Use Distance Matrix API to get traffic information
            destination = f"{coords['lat']},{coords['lon']}"
            origin = f"{coords['lat'] + 0.01},{coords['lon'] + 0.01}"  # Nearby point
            
            params = {
                'key': self.google_api_key,
                'origins': origin,
                'destinations': destination,
                'departure_time': int(time.time()),
                'traffic_model': 'best_guess',
                'mode': 'driving'
            }
            
            data = self._make_api_request(self.apis['google_traffic'], params)
            if not data:
                return None
            
            elements = data.get('rows', [{}])[0].get('elements', [])
            if not elements:
                return None
            
            element = elements[0]
            if element.get('status') != 'OK':
                return None
            
            duration_in_traffic = element.get('duration_in_traffic', {})
            duration = element.get('duration', {})
            distance = element.get('distance', {})
            
            travel_time = duration_in_traffic.get('value', 0) // 60  # Convert to minutes
            base_time = duration.get('value', 0) // 60
            distance_km = distance.get('value', 0) / 1000  # Convert to km
            
            # Calculate congestion level
            congestion_level = 0.0
            if base_time > 0:
                congestion_level = min((travel_time - base_time) / base_time, 1.0)
            
            return TrafficData(
                location=location,
                congestion_level=max(0, congestion_level),
                travel_time=travel_time,
                distance=distance_km,
                timestamp=datetime.now(),
                source='Google Maps Traffic'
            )
            
        except Exception as e:
            logger.error(f"Error fetching Google Maps traffic data for {location}: {e}")
            return None

    def _get_simulated_crowd_data(self, location: str) -> CrowdData:
        """Generate simulated crowd data for demo purposes"""
        # Simulate different crowd levels for different locations
        base_density = {
            "MG Road": 0.8, "Brigade Road": 0.9, "Koramangala": 0.7,
            "Electronic City": 0.6, "Whitefield": 0.5, "Indiranagar": 0.8,
            "Jayanagar": 0.6, "HSR Layout": 0.5, "BTM Layout": 0.6,
            "Marathahalli": 0.7, "Banashankari": 0.5, "Rajajinagar": 0.6,
            "Malleshwaram": 0.5, "Basavanagudi": 0.5, "Yeshwantpur": 0.6,
            "Hebbal": 0.5, "Bellandur": 0.6, "Sarjapur Road": 0.4,
            "Cunningham Road": 0.5, "Vijayanagar": 0.5
        }.get(location, 0.5)
        
        # Add some randomness
        density_level = max(0, min(1, base_density + np.random.normal(0, 0.1)))
        
        # Generate popular times
        popular_times = {}
        for hour in range(24):
            if 8 <= hour <= 10 or 17 <= hour <= 19:  # Peak hours
                popular_times[str(hour)] = min(density_level * 1.5, 1.0)
            elif 12 <= hour <= 14:  # Lunch time
                popular_times[str(hour)] = min(density_level * 1.2, 1.0)
            else:  # Off-peak
                popular_times[str(hour)] = density_level * 0.3
        
        return CrowdData(
            location=location,
            density_level=density_level,
            popular_times=popular_times,
            timestamp=datetime.now(),
            source='Simulated'
        )

    def _get_simulated_traffic_data(self, location: str) -> TrafficData:
        """Generate simulated traffic data for demo purposes"""
        # Simulate different traffic levels for different locations
        base_congestion = {
            "MG Road": 0.8, "Brigade Road": 0.9, "Koramangala": 0.7,
            "Electronic City": 0.6, "Whitefield": 0.5, "Indiranagar": 0.8,
            "Jayanagar": 0.6, "HSR Layout": 0.5, "BTM Layout": 0.6,
            "Marathahalli": 0.7, "Banashankari": 0.5, "Rajajinagar": 0.6,
            "Malleshwaram": 0.5, "Basavanagudi": 0.5, "Yeshwantpur": 0.6,
            "Hebbal": 0.5, "Bellandur": 0.6, "Sarjapur Road": 0.4,
            "Cunningham Road": 0.5, "Vijayanagar": 0.5
        }.get(location, 0.5)
        
        # Add some randomness
        congestion_level = max(0, min(1, base_congestion + np.random.normal(0, 0.1)))
        
        # Calculate travel time based on congestion
        base_time = 15  # 15 minutes base travel time
        travel_time = int(base_time * (1 + congestion_level))
        
        # Random distance between 2-10 km
        distance = np.random.uniform(2, 10)
        
        return TrafficData(
            location=location,
            congestion_level=congestion_level,
            travel_time=travel_time,
            distance=distance,
            timestamp=datetime.now(),
            source='Simulated'
        )

    def get_comprehensive_location_data(self, location: str) -> Dict:
        """Get comprehensive data for a location from all available sources"""
        data = {
            'location': location,
            'coordinates': self.bangalore_locations.get(location, {}),
            'air_quality': {},
            'traffic': {},
            'crowd': {},
            'timestamp': datetime.now()
        }
        
        # Try to get air quality data from multiple sources
        aqi_sources = [
            self.get_bharat_aqi_data(location),
            self.get_datagov_data(location),
            self.get_cpcb_data(location)
        ]
        
        # Use the first available source
        for aqi_data in aqi_sources:
            if aqi_data:
                data['air_quality'] = {
                    'aqi': aqi_data.aqi,
                    'pm25': aqi_data.pm25,
                    'pm10': aqi_data.pm10,
                    'no2': aqi_data.no2,
                    'so2': aqi_data.so2,
                    'co': aqi_data.co,
                    'o3': aqi_data.o3,
                    'source': aqi_data.source
                }
                break
        
        # Get traffic data
        traffic_data = self.get_google_maps_traffic_data(location)
        if traffic_data:
            data['traffic'] = {
                'congestion_level': traffic_data.congestion_level,
                'travel_time': traffic_data.travel_time,
                'distance': traffic_data.distance,
                'source': traffic_data.source
            }
        
        # Get crowd data
        crowd_data = self.get_google_maps_crowd_data(location)
        if crowd_data:
            data['crowd'] = {
                'density_level': crowd_data.density_level,
                'popular_times': crowd_data.popular_times,
                'source': crowd_data.source
            }
        
        return data

    def get_all_locations_data(self) -> List[Dict]:
        """Get comprehensive data for all Bangalore locations"""
        all_data = []
        
        for location in self.bangalore_locations.keys():
            location_data = self.get_comprehensive_location_data(location)
            all_data.append(location_data)
            
            # Add a small delay to avoid overwhelming APIs
            time.sleep(0.1)
        
        return all_data

    def get_air_quality_summary(self) -> Dict:
        """Get air quality summary for all locations"""
        all_data = self.get_all_locations_data()
        
        aqi_values = []
        pm25_values = []
        pm10_values = []
        
        for data in all_data:
            aqi = data.get('air_quality', {}).get('aqi', 0)
            pm25 = data.get('air_quality', {}).get('pm25', 0)
            pm10 = data.get('air_quality', {}).get('pm10', 0)
            
            if aqi > 0:
                aqi_values.append(aqi)
            if pm25 > 0:
                pm25_values.append(pm25)
            if pm10 > 0:
                pm10_values.append(pm10)
        
        return {
            'average_aqi': np.mean(aqi_values) if aqi_values else 0,
            'average_pm25': np.mean(pm25_values) if pm25_values else 0,
            'average_pm10': np.mean(pm10_values) if pm10_values else 0,
            'max_aqi': max(aqi_values) if aqi_values else 0,
            'min_aqi': min(aqi_values) if aqi_values else 0,
            'total_locations': len(all_data),
            'locations_with_data': len([d for d in all_data if d.get('air_quality', {}).get('aqi', 0) > 0])
        }

    def get_traffic_summary(self) -> Dict:
        """Get traffic summary for all locations"""
        all_data = self.get_all_locations_data()
        
        congestion_levels = []
        travel_times = []
        
        for data in all_data:
            traffic = data.get('traffic', {})
            congestion = traffic.get('congestion_level', 0)
            travel_time = traffic.get('travel_time', 0)
            
            if congestion > 0:
                congestion_levels.append(congestion)
            if travel_time > 0:
                travel_times.append(travel_time)
        
        return {
            'average_congestion': np.mean(congestion_levels) if congestion_levels else 0,
            'average_travel_time': np.mean(travel_times) if travel_times else 0,
            'max_congestion': max(congestion_levels) if congestion_levels else 0,
            'total_locations': len(all_data),
            'locations_with_data': len([d for d in all_data if d.get('traffic', {}).get('congestion_level', 0) > 0])
        }

    def get_crowd_summary(self) -> Dict:
        """Get crowd density summary for all locations"""
        all_data = self.get_all_locations_data()
        
        density_levels = []
        
        for data in all_data:
            crowd = data.get('crowd', {})
            density = crowd.get('density_level', 0)
            
            if density > 0:
                density_levels.append(density)
        
        return {
            'average_density': np.mean(density_levels) if density_levels else 0,
            'max_density': max(density_levels) if density_levels else 0,
            'total_locations': len(all_data),
            'locations_with_data': len([d for d in all_data if d.get('crowd', {}).get('density_level', 0) > 0])
        }
