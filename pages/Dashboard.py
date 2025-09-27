import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from utils.sensor_simulation import SensorNetwork
from utils.data_processor import DataProcessor
from utils.ml_models import StressPredictionModel

st.set_page_config(page_title="Dashboard",layout="wide")

# Initialize session state if not already done
if 'sensor_network' not in st.session_state:
    st.session_state.sensor_network = SensorNetwork()
    st.session_state.data_processor = DataProcessor()
    st.session_state.ml_model = StressPredictionModel()

st.title("📊 Real-time Dashboard")
st.markdown("### Comprehensive monitoring of urban environmental factors")

# Time range selector
st.sidebar.header("⏰ Time Range")
time_range = st.sidebar.selectbox(
    "Select time range for analysis",
    ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last Week"]
)

# Refresh controls
if st.sidebar.button("🔄 Refresh Dashboard"):
    st.session_state.sensor_network.update_all_sensors()
    st.rerun()

# Get current and historical data
current_data = st.session_state.sensor_network.get_all_sensor_data()
historical_data = st.session_state.data_processor.generate_historical_data(time_range)

# Main metrics row
st.header("🎯 Key Performance Indicators")

col1, col2, col3, col4, col5, col6 = st.columns(6)

# Calculate aggregated metrics
active_sensors_data = [s for s in current_data if s['status'] == 'active']

if active_sensors_data:
    avg_air_quality = np.mean([s['air_quality'] for s in active_sensors_data])
    avg_noise = np.mean([s['noise_level'] for s in active_sensors_data])
    avg_temp = np.mean([s['temperature'] for s in active_sensors_data])
    avg_humidity = np.mean([s['humidity'] for s in active_sensors_data])
    avg_crowd = np.mean([s['crowd_density'] for s in active_sensors_data])
    city_stress = st.session_state.ml_model.predict_city_stress(
        st.session_state.data_processor.process_sensor_data(current_data)
    )

    with col1:
        aqi_color = "normal" if avg_air_quality < 50 else "off" if avg_air_quality < 100 else "inverse"
        st.metric("Air Quality Index", f"{avg_air_quality:.1f}", 
                 delta=f"{np.random.uniform(-5, 5):.1f} from yesterday")

    with col2:
        noise_color = "normal" if avg_noise < 60 else "off" if avg_noise < 80 else "inverse"
        st.metric("Noise Level (dB)", f"{avg_noise:.1f}", 
                 delta=f"{np.random.uniform(-3, 3):.1f} from yesterday")

    with col3:
        st.metric("Temperature (°C)", f"{avg_temp:.1f}", 
                 delta=f"{np.random.uniform(-2, 2):.1f} from yesterday")

    with col4:
        st.metric("Humidity (%)", f"{avg_humidity:.1f}", 
                 delta=f"{np.random.uniform(-5, 5):.1f} from yesterday")

    with col5:
        st.metric("Crowd Density (%)", f"{avg_crowd:.1f}", 
                 delta=f"{np.random.uniform(-10, 10):.1f} from yesterday")

    with col6:
        stress_color = "normal" if city_stress < 4 else "off" if city_stress < 7 else "inverse"
        st.metric("City Stress Level", f"{city_stress:.1f}/10", 
                 delta=f"{np.random.uniform(-0.5, 0.5):.1f} from yesterday")

# Environmental trends
st.header("📈 Environmental Trends")

# Create subplot figure
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Air Quality Over Time', 'Noise Levels Over Time', 
                   'Temperature Trends', 'Crowd Density Patterns'),
    specs=[[{"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}]]
)

# Generate time series data
time_points = pd.date_range(
    start=datetime.now() - timedelta(hours=24), 
    end=datetime.now(), 
    freq='H'
)

# Air Quality trend
aqi_trend = [np.random.normal(avg_air_quality, 15) for _ in time_points]
fig.add_trace(
    go.Scatter(x=time_points, y=aqi_trend, name="Air Quality", 
              line=dict(color='red', width=2)),
    row=1, col=1
)

# Noise trend
noise_trend = [np.random.normal(avg_noise, 8) for _ in time_points]
fig.add_trace(
    go.Scatter(x=time_points, y=noise_trend, name="Noise Level", 
              line=dict(color='orange', width=2)),
    row=1, col=2
)

# Temperature trend
temp_trend = [np.random.normal(avg_temp, 3) for _ in time_points]
fig.add_trace(
    go.Scatter(x=time_points, y=temp_trend, name="Temperature", 
              line=dict(color='blue', width=2)),
    row=2, col=1
)

# Crowd density trend
crowd_trend = [np.random.normal(avg_crowd, 12) for _ in time_points]
fig.add_trace(
    go.Scatter(x=time_points, y=crowd_trend, name="Crowd Density", 
              line=dict(color='green', width=2)),
    row=2, col=2
)

fig.update_layout(height=600, showlegend=False, title_text="24-Hour Environmental Trends")
fig.update_xaxes(title_text="Time")
fig.update_yaxes(title_text="AQI", row=1, col=1)
fig.update_yaxes(title_text="Decibels", row=1, col=2)
fig.update_yaxes(title_text="°C", row=2, col=1)
fig.update_yaxes(title_text="Density %", row=2, col=2)

st.plotly_chart(fig, use_container_width=True)

# Sensor network status
st.header("🔌 Sensor Network Status")

col1, col2 = st.columns([2, 1])

with col1:
    # Sensor status chart
    sensor_status_data = []
    for sensor in current_data:
        sensor_status_data.append({
            'Sensor ID': sensor['sensor_id'],
            'Location': sensor['location'],
            'Status': sensor['status'],
            'Air Quality': sensor['air_quality'],
            'Noise Level': sensor['noise_level'],
            'Temperature': sensor['temperature'],
            'Crowd Density': sensor['crowd_density'],
            'Last Update': sensor['timestamp']
        })
    
    sensor_df = pd.DataFrame(sensor_status_data)
    st.dataframe(sensor_df, use_container_width=True)

with col2:
    # Status distribution pie chart
    status_counts = sensor_df['Status'].value_counts()
    fig_pie = px.pie(
        values=status_counts.values, 
        names=status_counts.index,
        title="Sensor Status Distribution",
        color_discrete_map={'active': 'green', 'maintenance': 'orange', 'offline': 'red'}
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# Correlation analysis
st.header("🔗 Environmental Factor Correlations")

# Create correlation matrix
correlation_data = pd.DataFrame({
    'Air_Quality': [s['air_quality'] for s in active_sensors_data],
    'Noise_Level': [s['noise_level'] for s in active_sensors_data],
    'Temperature': [s['temperature'] for s in active_sensors_data],
    'Humidity': [s['humidity'] for s in active_sensors_data],
    'Crowd_Density': [s['crowd_density'] for s in active_sensors_data]
})

if not correlation_data.empty:
    correlation_matrix = correlation_data.corr()
    
    fig_corr = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        title="Environmental Factors Correlation Matrix",
        color_continuous_scale='RdBu'
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)

# Real-time alerts
st.header("🚨 Active Alerts and Notifications")

alerts = []
for sensor in active_sensors_data:
    if sensor['air_quality'] > 100:
        alerts.append({
            'Time': datetime.now().strftime("%H:%M:%S"),
            'Sensor': sensor['sensor_id'],
            'Location': sensor['location'],
            'Type': 'Air Quality Alert',
            'Severity': 'High',
            'Value': f"{sensor['air_quality']:.1f} AQI"
        })
    
    if sensor['noise_level'] > 80:
        alerts.append({
            'Time': datetime.now().strftime("%H:%M:%S"),
            'Sensor': sensor['sensor_id'],
            'Location': sensor['location'],
            'Type': 'Noise Alert',
            'Severity': 'Medium',
            'Value': f"{sensor['noise_level']:.1f} dB"
        })

if alerts:
    alerts_df = pd.DataFrame(alerts)
    st.dataframe(alerts_df, use_container_width=True)
else:
    st.success("✅ No active alerts - all environmental conditions are within normal ranges")

# Performance summary
st.header("📋 System Performance Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Network Health")
    uptime_percentage = (len(active_sensors_data) / len(current_data)) * 100
    st.metric("Network Uptime", f"{uptime_percentage:.1f}%")
    st.metric("Data Quality Score", f"{np.random.uniform(85, 99):.1f}%")

with col2:
    st.subheader("Processing Stats")
    st.metric("Data Points/Hour", f"{len(active_sensors_data) * 60:,}")
    st.metric("ML Predictions", f"{len(active_sensors_data):,}")

with col3:
    st.subheader("System Load")
    st.metric("CPU Usage", f"{np.random.uniform(20, 80):.1f}%")
    st.metric("Memory Usage", f"{np.random.uniform(30, 70):.1f}%")

st.markdown("---")
st.markdown(f"**Dashboard last updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
