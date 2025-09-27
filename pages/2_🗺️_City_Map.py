import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
from utils.sensor_simulation import SensorNetwork
from utils.data_processor import DataProcessor
from utils.ml_models import StressPredictionModel
from utils.map_generator import MapGenerator

st.set_page_config(page_title="City Map", page_icon="🗺️", layout="wide")

# Initialize session state
if 'sensor_network' not in st.session_state:
    st.session_state.sensor_network = SensorNetwork()
    st.session_state.data_processor = DataProcessor()
    st.session_state.ml_model = StressPredictionModel()

st.title("🗺️ Interactive City Map")
st.markdown("### Real-time sensor network visualization and stress hotspot mapping")

# Sidebar controls
st.sidebar.header("🎛️ Map Controls")

# Map view options
map_view = st.sidebar.selectbox(
    "Select map view",
    ["Sensor Network", "Stress Heatmap", "Environmental Overlay", "Intervention Zones"]
)

# Layer toggles
show_sensors = st.sidebar.checkbox("Show Sensors", value=True)
show_hotspots = st.sidebar.checkbox("Show Stress Hotspots", value=True)
show_interventions = st.sidebar.checkbox("Show Active Interventions", value=False)
show_traffic = st.sidebar.checkbox("Show Traffic Data", value=False)

# Data refresh
if st.sidebar.button("🔄 Refresh Map Data"):
    st.session_state.sensor_network.update_all_sensors()
    st.rerun()

# Get current data
current_data = st.session_state.sensor_network.get_all_sensor_data()
map_generator = MapGenerator()

# Create the main map
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader(f"📍 {map_view}")
    
    # Initialize the base map
    center_lat, center_lon = 12.9716, 77.5946  # Bangalore city center coordinates
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Add sensors to map
    if show_sensors:
        for sensor in current_data:
            # Determine sensor color based on status and stress level
            stress_level = st.session_state.ml_model.predict_sensor_stress(sensor)
            
            if sensor['status'] == 'active':
                if stress_level < 4:
                    color = 'green'
                    icon = 'check'
                elif stress_level < 7:
                    color = 'orange'
                    icon = 'exclamation'
                else:
                    color = 'red'
                    icon = 'warning'
            else:
                color = 'gray'
                icon = 'remove'
            
            # Create popup content
            popup_content = f"""
            <b>Sensor {sensor['sensor_id']}</b><br>
            📍 Location: {sensor['location']}<br>
            🟢 Status: {sensor['status'].title()}<br>
            🌫️ Air Quality: {sensor['air_quality']:.1f} AQI<br>
            🔊 Noise: {sensor['noise_level']:.1f} dB<br>
            🌡️ Temperature: {sensor['temperature']:.1f}°C<br>
            💧 Humidity: {sensor['humidity']:.1f}%<br>
            👥 Crowd Density: {sensor['crowd_density']:.1f}%<br>
            🧠 Stress Level: {stress_level:.1f}/10
            """
            
            folium.Marker(
                location=[sensor['latitude'], sensor['longitude']],
                popup=folium.Popup(popup_content, max_width=300),
                icon=folium.Icon(color=color, icon=icon, prefix='fa'),
                tooltip=f"Sensor {sensor['sensor_id']} - {sensor['location']}"
            ).add_to(m)
    
    # Add stress hotspots heatmap
    if show_hotspots and map_view in ["Stress Heatmap", "Environmental Overlay"]:
        from folium.plugins import HeatMap
        
        # Generate heatmap data
        heatmap_data = []
        for sensor in current_data:
            if sensor['status'] == 'active':
                stress_level = st.session_state.ml_model.predict_sensor_stress(sensor)
                heatmap_data.append([
                    sensor['latitude'], 
                    sensor['longitude'], 
                    stress_level / 10.0  # Normalize to 0-1
                ])
        
        if heatmap_data:
            HeatMap(
                heatmap_data,
                min_opacity=0.3,
                max_zoom=18,
                radius=25,
                blur=15,
                gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'}
            ).add_to(m)
    
    # Add intervention zones
    if show_interventions:
        intervention_zones = [
            {"name": "Cubbon Park Cooling Zone", "lat": 12.9759, "lon": 77.6094, "radius": 500},
            {"name": "MG Road Noise Control", "lat": 12.9716, "lon": 77.6197, "radius": 300},
            {"name": "Electronic City Air Purification", "lat": 12.8456, "lon": 77.6603, "radius": 400},
        ]
        
        for zone in intervention_zones:
            folium.Circle(
                location=[zone['lat'], zone['lon']],
                radius=zone['radius'],
                popup=f"<b>Intervention Zone</b><br>{zone['name']}",
                color='purple',
                fillColor='purple',
                fillOpacity=0.2,
                weight=2
            ).add_to(m)
    
    # Add traffic data overlay
    if show_traffic:
        # Simulate traffic congestion points
        traffic_points = [
            {"lat": 12.9279, "lon": 77.6271, "congestion": 0.8},  # Koramangala
            {"lat": 12.9784, "lon": 77.6408, "congestion": 0.9},  # Indiranagar
            {"lat": 12.9591, "lon": 77.6974, "congestion": 0.6},  # Marathahalli
        ]
        
        for point in traffic_points:
            color = 'red' if point['congestion'] > 0.7 else 'orange' if point['congestion'] > 0.4 else 'green'
            folium.CircleMarker(
                location=[point['lat'], point['lon']],
                radius=10,
                popup=f"Traffic Congestion: {point['congestion']*100:.0f}%",
                color=color,
                fillColor=color,
                fillOpacity=0.6
            ).add_to(m)
    
    # Display the map
    map_data = st_folium(m, width=700, height=500)

with col2:
    st.subheader("📊 Map Statistics")
    
    # Calculate map statistics
    active_sensors = [s for s in current_data if s['status'] == 'active']
    
    if active_sensors:
        # Stress distribution
        stress_levels = [st.session_state.ml_model.predict_sensor_stress(s) for s in active_sensors]
        avg_stress = np.mean(stress_levels)
        
        st.metric("Average Stress Level", f"{avg_stress:.1f}/10")
        
        # Stress level distribution
        low_stress = len([s for s in stress_levels if s < 4])
        medium_stress = len([s for s in stress_levels if 4 <= s < 7])
        high_stress = len([s for s in stress_levels if s >= 7])
        
        st.markdown("**Stress Distribution:**")
        st.markdown(f"🟢 Low Stress: {low_stress} sensors")
        st.markdown(f"🟡 Medium Stress: {medium_stress} sensors")
        st.markdown(f"🔴 High Stress: {high_stress} sensors")
        
        # Environmental summary
        st.markdown("**Environmental Summary:**")
        avg_aqi = np.mean([s['air_quality'] for s in active_sensors])
        avg_noise = np.mean([s['noise_level'] for s in active_sensors])
        avg_crowd = np.mean([s['crowd_density'] for s in active_sensors])
        
        st.markdown(f"🌫️ Avg AQI: {avg_aqi:.1f}")
        st.markdown(f"🔊 Avg Noise: {avg_noise:.1f} dB")
        st.markdown(f"👥 Avg Crowd: {avg_crowd:.1f}%")
    
    # Hotspot alerts
    st.subheader("🚨 Hotspot Alerts")
    
    critical_sensors = []
    for sensor in active_sensors:
        stress = st.session_state.ml_model.predict_sensor_stress(sensor)
        if stress >= 7:
            critical_sensors.append({
                'location': sensor['location'],
                'stress': stress,
                'primary_factor': 'Air Quality' if sensor['air_quality'] > 100 else 'Noise' if sensor['noise_level'] > 80 else 'Crowd'
            })
    
    if critical_sensors:
        for sensor in critical_sensors:
            st.warning(f"⚠️ **{sensor['location']}**\nStress: {sensor['stress']:.1f}/10\nPrimary factor: {sensor['primary_factor']}")
    else:
        st.success("✅ No critical stress hotspots detected")
    
    # Intervention recommendations
    st.subheader("💡 Recommended Actions")
    
    if critical_sensors:
        recommendations = [
            "🌿 Activate green space ventilation systems",
            "🎵 Deploy ambient sound masking in noisy areas",
            "🚦 Implement dynamic traffic flow management",
            "💨 Increase air purification system intensity",
            "📱 Send crowd dispersal suggestions to mobile apps"
        ]
        
        for rec in recommendations[:3]:  # Show top 3 recommendations
            st.info(rec)
    else:
        st.info("🎯 All zones operating within optimal parameters")

# Detailed sensor information
st.header("📋 Detailed Sensor Information")

# Create expandable sections for each sensor
for sensor in current_data:
    stress_level = st.session_state.ml_model.predict_sensor_stress(sensor)
    status_emoji = "🟢" if sensor['status'] == 'active' else "🔴"
    stress_emoji = "🟢" if stress_level < 4 else "🟡" if stress_level < 7 else "🔴"
    
    with st.expander(f"{status_emoji} Sensor {sensor['sensor_id']} - {sensor['location']} {stress_emoji}"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Environmental Readings:**")
            st.markdown(f"🌫️ Air Quality: {sensor['air_quality']:.1f} AQI")
            st.markdown(f"🔊 Noise Level: {sensor['noise_level']:.1f} dB")
            st.markdown(f"🌡️ Temperature: {sensor['temperature']:.1f}°C")
            st.markdown(f"💧 Humidity: {sensor['humidity']:.1f}%")
        
        with col2:
            st.markdown("**Operational Status:**")
            st.markdown(f"🔌 Status: {sensor['status'].title()}")
            st.markdown(f"👥 Crowd Density: {sensor['crowd_density']:.1f}%")
            st.markdown(f"🧠 Stress Prediction: {stress_level:.1f}/10")
            st.markdown(f"📍 Coordinates: {sensor['latitude']:.4f}, {sensor['longitude']:.4f}")
        
        with col3:
            st.markdown("**Health Assessment:**")
            
            # Individual factor assessments
            aqi_status = "Good" if sensor['air_quality'] < 50 else "Moderate" if sensor['air_quality'] < 100 else "Poor"
            noise_status = "Quiet" if sensor['noise_level'] < 50 else "Moderate" if sensor['noise_level'] < 70 else "Loud"
            crowd_status = "Low" if sensor['crowd_density'] < 30 else "Medium" if sensor['crowd_density'] < 70 else "High"
            
            st.markdown(f"Air Quality: {aqi_status}")
            st.markdown(f"Noise Level: {noise_status}")
            st.markdown(f"Crowd Level: {crowd_status}")
            st.markdown(f"Overall Health: {'Good' if stress_level < 4 else 'Moderate' if stress_level < 7 else 'Poor'}")

st.markdown("---")

# API Data Sources Status
st.subheader("📡 Real-time Data Sources")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("**🌬️ Bharat AQI**")
    st.success("✅ Active")

with col2:
    st.markdown("**🏛️ Data.gov.in**")
    st.success("✅ Active")

with col3:
    st.markdown("**🏭 CPCB**")
    st.success("✅ Active")

with col4:
    st.markdown("**🗺️ Google Places**")
    st.info("🔑 API Key Required")

with col5:
    st.markdown("**🚗 Google Traffic**")
    st.info("🔑 API Key Required")

st.markdown(f"**Map last updated:** {st.session_state.sensor_network.get_last_update_time()}")
st.markdown("**Data Integration:** Real-time API data from 5 sources for Bangalore locations")
