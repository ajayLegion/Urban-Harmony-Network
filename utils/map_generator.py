import folium
from folium import plugins
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

class MapGenerator:
    """Generates interactive maps for urban sensor network visualization."""
    
    def __init__(self):
        self.default_center = [40.7128, -74.0060]  # New York City
        self.default_zoom = 12
        self.color_schemes = {
            'stress': {
                'low': 'green',
                'medium': 'orange', 
                'high': 'red'
            },
            'air_quality': {
                'good': 'green',
                'moderate': 'yellow',
                'unhealthy': 'orange',
                'very_unhealthy': 'red',
                'hazardous': 'darkred'
            },
            'noise': {
                'quiet': 'green',
                'moderate': 'yellow',
                'loud': 'orange',
                'very_loud': 'red'
            }
        }
    
    def create_base_map(self, center: Optional[List[float]] = None, 
                       zoom: Optional[int] = None, tile_style: str = 'OpenStreetMap') -> folium.Map:
        """Create a base map with specified parameters."""
        if center is None:
            center = self.default_center
        if zoom is None:
            zoom = self.default_zoom
            
        # Available tile styles
        tile_styles = {
            'OpenStreetMap': 'OpenStreetMap',
            'CartoDB': 'CartoDB positron',
            'Stamen Terrain': 'Stamen Terrain',
            'Stamen Toner': 'Stamen Toner'
        }
        
        tiles = tile_styles.get(tile_style, 'OpenStreetMap')
        
        base_map = folium.Map(
            location=center,
            zoom_start=zoom,
            tiles=tiles
        )
        
        return base_map
    
    def add_sensor_markers(self, map_obj: folium.Map, sensors: List[Dict[str, Any]], 
                          color_by: str = 'status', ml_model=None) -> folium.Map:
        """Add sensor markers to the map with color coding based on specified metric."""
        for sensor in sensors:
            # Determine marker color based on color_by parameter
            if color_by == 'status':
                color = self._get_status_color(sensor.get('status', 'unknown'))
                icon = self._get_status_icon(sensor.get('status', 'unknown'))
            elif color_by == 'stress' and ml_model:
                stress_level = ml_model.predict_sensor_stress(sensor)
                color = self._get_stress_color(stress_level)
                icon = self._get_stress_icon(stress_level)
            elif color_by == 'air_quality':
                aqi = sensor.get('air_quality', 50)
                color = self._get_air_quality_color(aqi)
                icon = self._get_air_quality_icon(aqi)
            elif color_by == 'noise':
                noise = sensor.get('noise_level', 60)
                color = self._get_noise_color(noise)
                icon = self._get_noise_icon(noise)
            else:
                color = 'blue'
                icon = 'info-sign'
            
            # Create popup content
            popup_content = self._create_sensor_popup(sensor, ml_model)
            
            # Create tooltip
            tooltip_text = f"Sensor {sensor.get('sensor_id', 'Unknown')} - {sensor.get('location', 'Unknown Location')}"
            
            # Add marker to map
            folium.Marker(
                location=[sensor.get('latitude', 0), sensor.get('longitude', 0)],
                popup=folium.Popup(popup_content, max_width=400),
                tooltip=tooltip_text,
                icon=folium.Icon(color=color, icon=icon, prefix='fa')
            ).add_to(map_obj)
        
        return map_obj
    
    def add_heatmap_layer(self, map_obj: folium.Map, sensors: List[Dict[str, Any]], 
                         metric: str = 'stress', ml_model=None) -> folium.Map:
        """Add heatmap layer to the map based on specified metric."""
        heatmap_data = []
        
        for sensor in sensors:
            if sensor.get('status') != 'active':
                continue
                
            lat = sensor.get('latitude', 0)
            lon = sensor.get('longitude', 0)
            
            # Get intensity based on metric
            if metric == 'stress' and ml_model:
                intensity = ml_model.predict_sensor_stress(sensor) / 10.0
            elif metric == 'air_quality':
                # Normalize AQI (higher is worse, so we want higher intensity for worse quality)
                intensity = min(1.0, sensor.get('air_quality', 50) / 200.0)
            elif metric == 'noise':
                # Normalize noise level
                intensity = min(1.0, max(0, (sensor.get('noise_level', 60) - 40) / 60.0))
            elif metric == 'temperature':
                # Temperature discomfort (deviation from 22°C)
                temp = sensor.get('temperature', 22)
                intensity = min(1.0, abs(temp - 22) / 20.0)
            elif metric == 'crowd_density':
                intensity = sensor.get('crowd_density', 30) / 100.0
            else:
                intensity = 0.5
            
            heatmap_data.append([lat, lon, intensity])
        
        if heatmap_data:
            # Color scheme based on metric
            if metric == 'stress':
                gradient = {0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'}
            elif metric == 'air_quality':
                gradient = {0.2: 'green', 0.4: 'yellow', 0.6: 'orange', 1: 'red'}
            else:
                gradient = {0.4: 'blue', 0.65: 'lime', 0.8: 'orange', 1.0: 'red'}
            
            plugins.HeatMap(
                heatmap_data,
                min_opacity=0.3,
                max_zoom=18,
                radius=25,
                blur=15,
                gradient=gradient
            ).add_to(map_obj)
        
        return map_obj
    
    def add_intervention_zones(self, map_obj: folium.Map, 
                             intervention_zones: List[Dict[str, Any]]) -> folium.Map:
        """Add intervention zones to the map."""
        for zone in intervention_zones:
            # Default zone if not provided
            if not zone:
                continue
                
            lat = zone.get('lat', self.default_center[0])
            lon = zone.get('lon', self.default_center[1])
            radius = zone.get('radius', 300)
            zone_type = zone.get('type', 'general')
            name = zone.get('name', 'Intervention Zone')
            
            # Color based on intervention type
            colors = {
                'air_quality': 'blue',
                'noise_control': 'purple',
                'cooling': 'lightblue',
                'crowd_management': 'orange',
                'lighting': 'yellow',
                'green_space': 'green',
                'general': 'gray'
            }
            
            color = colors.get(zone_type, 'gray')
            
            # Create circle for intervention zone
            folium.Circle(
                location=[lat, lon],
                radius=radius,
                popup=f"<b>Intervention Zone</b><br>{name}<br>Type: {zone_type.title()}<br>Radius: {radius}m",
                color=color,
                fillColor=color,
                fillOpacity=0.2,
                weight=2
            ).add_to(map_obj)
            
            # Add center marker
            folium.Marker(
                location=[lat, lon],
                popup=f"<b>{name}</b><br>Intervention Type: {zone_type.title()}",
                icon=folium.Icon(color=color, icon='cog', prefix='fa')
            ).add_to(map_obj)
        
        return map_obj
    
    def add_traffic_layer(self, map_obj: folium.Map, 
                         traffic_data: List[Dict[str, Any]]) -> folium.Map:
        """Add traffic congestion indicators to the map."""
        for point in traffic_data:
            lat = point.get('lat', 0)
            lon = point.get('lon', 0)
            congestion = point.get('congestion', 0.5)  # 0-1 scale
            
            # Color based on congestion level
            if congestion > 0.8:
                color = 'red'
            elif congestion > 0.6:
                color = 'orange'
            elif congestion > 0.4:
                color = 'yellow'
            else:
                color = 'green'
            
            # Size based on congestion level
            radius = max(5, int(congestion * 15))
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                popup=f"Traffic Congestion: {congestion*100:.0f}%<br>Status: {'Heavy' if congestion > 0.7 else 'Moderate' if congestion > 0.4 else 'Light'}",
                color=color,
                fillColor=color,
                fillOpacity=0.6,
                weight=2
            ).add_to(map_obj)
        
        return map_obj
    
    def add_air_quality_contours(self, map_obj: folium.Map, 
                                sensors: List[Dict[str, Any]]) -> folium.Map:
        """Add air quality contour lines to the map."""
        # Extract air quality data points
        air_quality_points = []
        for sensor in sensors:
            if sensor.get('status') == 'active' and 'air_quality' in sensor:
                air_quality_points.append({
                    'lat': sensor.get('latitude', 0),
                    'lon': sensor.get('longitude', 0),
                    'aqi': sensor.get('air_quality', 50)
                })
        
        if len(air_quality_points) < 3:
            return map_obj  # Need at least 3 points for contours
        
        # Create contour levels
        aqi_values = [point['aqi'] for point in air_quality_points]
        min_aqi = min(aqi_values)
        max_aqi = max(aqi_values)
        
        # Define contour levels
        if max_aqi - min_aqi > 20:
            levels = [50, 100, 150, 200]  # Standard AQI breakpoints
            colors = ['green', 'yellow', 'orange', 'red']
            
            for i, level in enumerate(levels):
                if min_aqi <= level <= max_aqi:
                    # Simple approximation - draw circles around high AQI areas
                    for point in air_quality_points:
                        if point['aqi'] >= level:
                            folium.Circle(
                                location=[point['lat'], point['lon']],
                                radius=200,
                                color=colors[i],
                                weight=1,
                                opacity=0.3,
                                fillOpacity=0.1
                            ).add_to(map_obj)
        
        return map_obj
    
    def add_legend(self, map_obj: folium.Map, legend_type: str = 'stress') -> folium.Map:
        """Add a legend to the map."""
        legends = {
            'stress': {
                'title': 'Stress Levels',
                'items': [
                    ('Low (0-4)', 'green'),
                    ('Medium (4-7)', 'orange'),
                    ('High (7-10)', 'red')
                ]
            },
            'air_quality': {
                'title': 'Air Quality Index',
                'items': [
                    ('Good (0-50)', 'green'),
                    ('Moderate (51-100)', 'yellow'),
                    ('Unhealthy (101-150)', 'orange'),
                    ('Very Unhealthy (151-200)', 'red'),
                    ('Hazardous (201+)', 'darkred')
                ]
            },
            'noise': {
                'title': 'Noise Levels',
                'items': [
                    ('Quiet (<50 dB)', 'green'),
                    ('Moderate (50-70 dB)', 'yellow'),
                    ('Loud (70-85 dB)', 'orange'),
                    ('Very Loud (85+ dB)', 'red')
                ]
            }
        }
        
        legend_info = legends.get(legend_type, legends['stress'])
        
        # Create legend HTML
        legend_html = f'''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: auto; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <h4>{legend_info['title']}</h4>
        '''
        
        for item, color in legend_info['items']:
            legend_html += f'''
            <div style="margin: 5px 0;">
                <i style="background:{color}; width: 12px; height: 12px; 
                         display: inline-block; margin-right: 8px;"></i>
                {item}
            </div>
            '''
        
        legend_html += '</div>'
        
        # Add legend using folium's HTML method
        folium.Html(legend_html, script=True).add_to(map_obj)
        return map_obj
    
    def _create_sensor_popup(self, sensor: Dict[str, Any], ml_model=None) -> str:
        """Create detailed popup content for a sensor."""
        sensor_id = sensor.get('sensor_id', 'Unknown')
        location = sensor.get('location', 'Unknown Location')
        status = sensor.get('status', 'unknown').title()
        
        # Status indicator
        status_color = 'green' if status == 'Active' else 'red' if status == 'Offline' else 'orange'
        
        popup_content = f'''
        <div style="width: 300px;">
            <h4><b>Sensor {sensor_id}</b></h4>
            <p><b>📍 Location:</b> {location}</p>
            <p><b>Status:</b> <span style="color: {status_color};">●</span> {status}</p>
            <hr>
        '''
        
        if sensor.get('status') == 'active':
            # Environmental readings
            popup_content += '<h5>Environmental Readings:</h5>'
            popup_content += f"<p><b>🌫️ Air Quality:</b> {sensor.get('air_quality', 'N/A'):.1f} AQI</p>"
            popup_content += f"<p><b>🔊 Noise Level:</b> {sensor.get('noise_level', 'N/A'):.1f} dB</p>"
            popup_content += f"<p><b>🌡️ Temperature:</b> {sensor.get('temperature', 'N/A'):.1f}°C</p>"
            popup_content += f"<p><b>💧 Humidity:</b> {sensor.get('humidity', 'N/A'):.1f}%</p>"
            popup_content += f"<p><b>👥 Crowd Density:</b> {sensor.get('crowd_density', 'N/A'):.1f}%</p>"
            
            # Stress prediction if ML model is available
            if ml_model:
                stress_level = ml_model.predict_sensor_stress(sensor)
                stress_color = 'green' if stress_level < 4 else 'orange' if stress_level < 7 else 'red'
                popup_content += f'<hr><h5>AI Analysis:</h5>'
                popup_content += f'<p><b>🧠 Stress Level:</b> <span style="color: {stress_color}; font-weight: bold;">{stress_level:.1f}/10</span></p>'
                
                # Risk assessment
                if stress_level < 4:
                    risk_text = "Low Risk - Optimal conditions"
                elif stress_level < 7:
                    risk_text = "Medium Risk - Monitor closely"
                else:
                    risk_text = "High Risk - Intervention recommended"
                
                popup_content += f'<p><b>⚠️ Risk Level:</b> {risk_text}</p>'
            
            # Timestamp
            timestamp = sensor.get('timestamp', datetime.now().isoformat())
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                formatted_time = timestamp
            
            popup_content += f'<hr><small><b>Last Updated:</b> {formatted_time}</small>'
        else:
            popup_content += '<p><i>Sensor is currently offline or under maintenance.</i></p>'
        
        popup_content += '</div>'
        return popup_content
    
    def _get_status_color(self, status: str) -> str:
        """Get color for sensor status."""
        status_colors = {
            'active': 'green',
            'maintenance': 'orange',
            'offline': 'red',
            'unknown': 'gray'
        }
        return status_colors.get(status.lower(), 'gray')
    
    def _get_status_icon(self, status: str) -> str:
        """Get icon for sensor status."""
        status_icons = {
            'active': 'check',
            'maintenance': 'wrench',
            'offline': 'remove',
            'unknown': 'question'
        }
        return status_icons.get(status.lower(), 'question')
    
    def _get_stress_color(self, stress_level: float) -> str:
        """Get color for stress level."""
        if stress_level < 4:
            return 'green'
        elif stress_level < 7:
            return 'orange'
        else:
            return 'red'
    
    def _get_stress_icon(self, stress_level: float) -> str:
        """Get icon for stress level."""
        if stress_level < 4:
            return 'check'
        elif stress_level < 7:
            return 'exclamation'
        else:
            return 'warning'
    
    def _get_air_quality_color(self, aqi: float) -> str:
        """Get color for air quality index."""
        if aqi <= 50:
            return 'green'
        elif aqi <= 100:
            return 'yellow'
        elif aqi <= 150:
            return 'orange'
        elif aqi <= 200:
            return 'red'
        else:
            return 'darkred'
    
    def _get_air_quality_icon(self, aqi: float) -> str:
        """Get icon for air quality index."""
        if aqi <= 50:
            return 'leaf'
        elif aqi <= 100:
            return 'cloud'
        else:
            return 'warning'
    
    def _get_noise_color(self, noise: float) -> str:
        """Get color for noise level."""
        if noise < 50:
            return 'green'
        elif noise < 70:
            return 'yellow'
        elif noise < 85:
            return 'orange'
        else:
            return 'red'
    
    def _get_noise_icon(self, noise: float) -> str:
        """Get icon for noise level."""
        if noise < 50:
            return 'volume-down'
        elif noise < 85:
            return 'volume-up'
        else:
            return 'warning'
    
    def generate_mini_map(self, sensors: List[Dict[str, Any]], 
                         width: int = 400, height: int = 300) -> folium.Map:
        """Generate a smaller overview map."""
        mini_map = self.create_base_map()
        mini_map = self.add_sensor_markers(mini_map, sensors, color_by='status')
        
        # Mini maps are handled by the display system
        # Size adjustments would be handled at display time
        
        return mini_map
    
    def calculate_map_bounds(self, sensors: List[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
        """Calculate appropriate map bounds for sensor locations."""
        if not sensors:
            return self.default_center, self.default_center
        
        lats = [sensor.get('latitude', 0) for sensor in sensors if sensor.get('latitude')]
        lons = [sensor.get('longitude', 0) for sensor in sensors if sensor.get('longitude')]
        
        if not lats or not lons:
            return self.default_center, self.default_center
        
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        
        # Add padding
        lat_padding = (max_lat - min_lat) * 0.1
        lon_padding = (max_lon - min_lon) * 0.1
        
        southwest = [min_lat - lat_padding, min_lon - lon_padding]
        northeast = [max_lat + lat_padding, max_lon + lon_padding]
        
        return southwest, northeast
    
    def export_map(self, map_obj: folium.Map, filename: str, format: str = 'html') -> str:
        """Export map to file."""
        if format.lower() == 'html':
            map_obj.save(filename)
            return f"Map exported to {filename}"
        else:
            return "Currently only HTML export is supported"
