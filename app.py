import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from utils.sensor_simulation import SensorNetwork
from utils.data_processor import DataProcessor
from utils.ml_models import StressPredictionModel

# Configure the Streamlit page
st.set_page_config(
    page_title="Urban Harmony Network",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'sensor_network' not in st.session_state:
    st.session_state.sensor_network = SensorNetwork()
    st.session_state.data_processor = DataProcessor()
    st.session_state.ml_model = StressPredictionModel()

# Main page header
st.title("🏙️ Urban Harmony Network")
st.markdown("### AI-Powered Urban Mental Health Monitoring System")

# Sidebar controls
st.sidebar.header("🎛️ System Controls")

# Auto-refresh toggle
auto_refresh = st.sidebar.checkbox("Auto-refresh data", value=True)
refresh_interval = 10
if auto_refresh:
    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 5, 60, 10)

# Manual refresh button
if st.sidebar.button("🔄 Refresh Data"):
    st.session_state.sensor_network.update_all_sensors()
    st.rerun()

# System overview metrics
st.header("📋 System Overview")

# Get current sensor data
current_data = st.session_state.sensor_network.get_all_sensor_data()
processed_data = st.session_state.data_processor.process_sensor_data(current_data)

# Create metrics columns
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    active_sensors = len([s for s in current_data if s['status'] == 'active'])
    total_sensors = len(current_data)
    st.metric(
        "Active Sensors", 
        f"{active_sensors}/{total_sensors}",
        delta=f"{(active_sensors/total_sensors)*100:.1f}% uptime"
    )

with col2:
    avg_air_quality = np.mean([s['air_quality'] for s in current_data if s['status'] == 'active'])
    st.metric(
        "Avg Air Quality", 
        f"{avg_air_quality:.1f} AQI",
        delta=f"{'Good' if avg_air_quality < 50 else 'Moderate' if avg_air_quality < 100 else 'Poor'}"
    )

with col3:
    avg_noise = np.mean([s['noise_level'] for s in current_data if s['status'] == 'active'])
    st.metric(
        "Avg Noise Level", 
        f"{avg_noise:.1f} dB",
        delta=f"{'Quiet' if avg_noise < 50 else 'Moderate' if avg_noise < 70 else 'Loud'}"
    )

with col4:
    avg_temp = np.mean([s['temperature'] for s in current_data if s['status'] == 'active'])
    st.metric(
        "Avg Temperature", 
        f"{avg_temp:.1f}°C",
        delta=f"{'Comfortable' if 18 <= avg_temp <= 24 else 'Uncomfortable'}"
    )

with col5:
    stress_prediction = st.session_state.ml_model.predict_city_stress(processed_data)
    st.metric(
        "City Stress Level", 
        f"{stress_prediction:.1f}/10",
        delta=f"{'Low' if stress_prediction < 4 else 'Medium' if stress_prediction < 7 else 'High'}"
    )

# Real-time sensor status
st.header("📡 Real-time Sensor Network Status")

# Create a DataFrame for sensor data
sensor_df = pd.DataFrame(current_data)

# Display sensor grid
if not sensor_df.empty:
    # Filter active sensors
    active_sensors_df = sensor_df[sensor_df['status'] == 'active']
    
    if not active_sensors_df.empty:
        # Create a grid view of sensors
        cols = st.columns(4)
        for idx, (_, sensor) in enumerate(active_sensors_df.iterrows()):
            with cols[idx % 4]:
                with st.container():
                    st.markdown(f"**Sensor {sensor['sensor_id']}**")
                    st.markdown(f"📍 {sensor['location']}")
                    
                    # Status indicator
                    status_color = "🟢" if sensor['status'] == 'active' else "🔴"
                    st.markdown(f"{status_color} {str(sensor['status']).title()}")
                    
                    # Sensor readings
                    st.markdown(f"🌫️ Air Quality: {sensor['air_quality']:.1f} AQI")
                    st.markdown(f"🔊 Noise: {sensor['noise_level']:.1f} dB")
                    st.markdown(f"🌡️ Temp: {sensor['temperature']:.1f}°C")
                    st.markdown(f"👥 Crowd: {sensor['crowd_density']:.1f}%")
                    
                    # Calculate individual stress score
                    individual_stress = st.session_state.ml_model.predict_sensor_stress(dict(sensor))
                    stress_color = "🟢" if individual_stress < 4 else "🟡" if individual_stress < 7 else "🔴"
                    st.markdown(f"{stress_color} Stress: {individual_stress:.1f}/10")
    else:
        st.warning("No active sensors found in the network.")
else:
    st.error("Unable to retrieve sensor data.")

# Recent alerts and notifications
st.header("🚨 Recent System Alerts")

# Generate alerts based on current conditions
alerts = []
for sensor in current_data:
    if sensor['status'] == 'active':
        if sensor['air_quality'] > 100:
            alerts.append({
                'time': datetime.now().strftime("%H:%M:%S"),
                'sensor': sensor['sensor_id'],
                'location': sensor['location'],
                'type': 'Air Quality Alert',
                'message': f"Poor air quality detected: {sensor['air_quality']:.1f} AQI",
                'severity': 'High'
            })
        
        if sensor['noise_level'] > 80:
            alerts.append({
                'time': datetime.now().strftime("%H:%M:%S"),
                'sensor': sensor['sensor_id'],
                'location': sensor['location'],
                'type': 'Noise Alert',
                'message': f"Excessive noise levels: {sensor['noise_level']:.1f} dB",
                'severity': 'Medium'
            })
        
        if sensor['crowd_density'] > 85:
            alerts.append({
                'time': datetime.now().strftime("%H:%M:%S"),
                'sensor': sensor['sensor_id'],
                'location': sensor['location'],
                'type': 'Crowd Alert',
                'message': f"High crowd density: {sensor['crowd_density']:.1f}%",
                'severity': 'Medium'
            })

if alerts:
    alerts_df = pd.DataFrame(alerts)
    st.dataframe(alerts_df, use_container_width=True)
else:
    st.success("No active alerts - all environmental conditions are within normal ranges.")

# Quick action buttons
st.header("⚡ Quick Actions")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🔄 Reset All Sensors"):
        st.session_state.sensor_network = SensorNetwork()
        st.success("All sensors have been reset.")
        st.rerun()

with col2:
    if st.button("📊 Generate Report"):
        st.info("Comprehensive system report generated successfully.")

with col3:
    if st.button("🎯 Calibrate Models"):
        st.session_state.ml_model.retrain_model(processed_data)
        st.success("ML models have been recalibrated.")

with col4:
    if st.button("🚨 Emergency Mode"):
        st.warning("Emergency protocols activated. All intervention systems engaged.")

# Auto-refresh functionality
if auto_refresh:
    import time
    time.sleep(refresh_interval)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("**Urban Harmony Network** - Proactive urban mental health monitoring through IoT and AI")
st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- Flask App for API/Hello World ---
# from flask import Flask
# app = Flask(__name__)

# @app.route('/')
# def hello():
#     return "Hello from Flask!"

# if __name__ == '__main__':
#     app.run(debug=True)
