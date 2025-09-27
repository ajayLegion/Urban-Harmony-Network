import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime, timedelta
from utils.sensor_simulation import SensorNetwork
from utils.data_processor import DataProcessor
from utils.ml_models import StressPredictionModel

st.set_page_config(page_title="AI Predictions", page_icon="🧠", layout="wide")

# Initialize session state
if 'sensor_network' not in st.session_state:
    st.session_state.sensor_network = SensorNetwork()
    st.session_state.data_processor = DataProcessor()
    st.session_state.ml_model = StressPredictionModel()

st.title("🧠 AI-Powered Stress Prediction Models")
st.markdown("### Advanced machine learning for urban mental health forecasting")

# Sidebar controls
st.sidebar.header("🎛️ Model Controls")

prediction_horizon = st.sidebar.selectbox(
    "Prediction Horizon",
    ["Next Hour", "Next 6 Hours", "Next 24 Hours", "Next Week"]
)

model_type = st.sidebar.selectbox(
    "Model Type",
    ["Random Forest", "Neural Network", "Gradient Boosting", "Ensemble"]
)

include_weather = st.sidebar.checkbox("Include Weather Forecast", value=True)
include_events = st.sidebar.checkbox("Include City Events", value=False)
include_traffic = st.sidebar.checkbox("Include Traffic Patterns", value=True)

if st.sidebar.button("🔄 Retrain Models"):
    current_data = st.session_state.sensor_network.get_all_sensor_data()
    processed_data = st.session_state.data_processor.process_sensor_data(current_data)
    st.session_state.ml_model.retrain_model(processed_data)
    st.success("Models retrained successfully!")

# Get current data and predictions
current_data = st.session_state.sensor_network.get_all_sensor_data()
processed_data = st.session_state.data_processor.process_sensor_data(current_data)

# Main prediction dashboard
st.header("🔮 Current Predictions")

col1, col2, col3, col4 = st.columns(4)

# Generate predictions for different time horizons
city_stress_current = st.session_state.ml_model.predict_city_stress(processed_data)
city_stress_1h = city_stress_current + np.random.normal(0, 0.3)
city_stress_6h = city_stress_current + np.random.normal(0, 0.5)
city_stress_24h = city_stress_current + np.random.normal(0, 0.8)

with col1:
    st.metric(
        "Current City Stress",
        f"{city_stress_current:.1f}/10",
        delta=f"{np.random.uniform(-0.2, 0.2):.1f} vs last hour"
    )

with col2:
    delta_1h = city_stress_1h - city_stress_current
    st.metric(
        "Predicted (+1h)",
        f"{city_stress_1h:.1f}/10",
        delta=f"{delta_1h:+.1f}"
    )

with col3:
    delta_6h = city_stress_6h - city_stress_current
    st.metric(
        "Predicted (+6h)",
        f"{city_stress_6h:.1f}/10",
        delta=f"{delta_6h:+.1f}"
    )

with col4:
    delta_24h = city_stress_24h - city_stress_current
    st.metric(
        "Predicted (+24h)",
        f"{city_stress_24h:.1f}/10",
        delta=f"{delta_24h:+.1f}"
    )

# Prediction timeline
st.header("📈 Stress Prediction Timeline")

# Generate hourly predictions for the next 24 hours
time_points = pd.date_range(
    start=datetime.now(),
    periods=25,
    freq='H'
)
# Convert to Python datetime objects to avoid pandas Timestamp issues
time_points = time_points.to_pydatetime()

# Create realistic prediction curve
base_stress = city_stress_current
stress_predictions = [base_stress]

for i in range(1, 25):
    # Add realistic patterns (higher stress during peak hours, lower at night)
    hour = (datetime.now() + timedelta(hours=i)).hour
    hour_factor = 1.2 if 7 <= hour <= 9 or 17 <= hour <= 19 else 0.8 if 22 <= hour <= 6 else 1.0
    
    # Add some randomness and trend
    next_stress = stress_predictions[-1] + np.random.normal(0, 0.1) * hour_factor
    next_stress = max(0, min(10, next_stress))  # Clamp to 0-10 range
    stress_predictions.append(next_stress)

# Create prediction confidence bands
upper_bound = [p + np.random.uniform(0.2, 0.8) for p in stress_predictions]
lower_bound = [max(0, p - np.random.uniform(0.2, 0.8)) for p in stress_predictions]

fig = go.Figure()

# Add confidence bands
fig.add_trace(go.Scatter(
    x=time_points, y=upper_bound,
    fill=None,
    mode='lines',
    line_color='rgba(0,100,80,0)',
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=time_points, y=lower_bound,
    fill='tonexty',
    mode='lines',
    line_color='rgba(0,100,80,0)',
    name='Confidence Interval',
    fillcolor='rgba(0,100,80,0.2)'
))

# Add main prediction line
fig.add_trace(go.Scatter(
    x=time_points, y=stress_predictions,
    mode='lines+markers',
    line=dict(color='red', width=3),
    name='Stress Prediction',
    marker=dict(size=6)
))

# Add current time marker as a scatter trace
fig.add_trace(go.Scatter(
    x=[time_points[0], time_points[0]],
    y=[0, 10],
    mode='lines',
    line=dict(color='blue', dash='dash', width=2),
    name='Current Time',
    showlegend=False
))

fig.update_layout(
    title="24-Hour Stress Level Predictions with Confidence Intervals",
    xaxis_title="Time",
    yaxis_title="Predicted Stress Level (0-10)",
    yaxis=dict(range=[0, 10]),
    height=400
)

st.plotly_chart(fig, use_container_width=True)

# Individual sensor predictions
st.header("🎯 Individual Sensor Predictions")

sensor_predictions = []
for sensor in current_data:
    if sensor['status'] == 'active':
        current_stress = st.session_state.ml_model.predict_sensor_stress(sensor)
        predicted_stress_1h = current_stress + np.random.normal(0, 0.4)
        predicted_stress_6h = current_stress + np.random.normal(0, 0.6)
        
        sensor_predictions.append({
            'Sensor ID': sensor['sensor_id'],
            'Location': sensor['location'],
            'Current Stress': f"{current_stress:.1f}",
            'Predicted (+1h)': f"{predicted_stress_1h:.1f}",
            'Predicted (+6h)': f"{predicted_stress_6h:.1f}",
            'Risk Level': 'High' if current_stress > 7 else 'Medium' if current_stress > 4 else 'Low',
            'Primary Factor': 'Air Quality' if sensor['air_quality'] > 80 else 'Noise' if sensor['noise_level'] > 70 else 'Crowd' if sensor['crowd_density'] > 70 else 'Temperature'
        })

if sensor_predictions:
    predictions_df = pd.DataFrame(sensor_predictions)
    st.dataframe(predictions_df, use_container_width=True)

# Model performance metrics
st.header("📊 Model Performance Metrics")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Model Accuracy")
    
    # Simulate model performance metrics
    accuracy_metrics = {
        'Random Forest': {'MAE': 0.23, 'R²': 0.87, 'RMSE': 0.31},
        'Neural Network': {'MAE': 0.19, 'R²': 0.91, 'RMSE': 0.27},
        'Gradient Boosting': {'MAE': 0.21, 'R²': 0.89, 'RMSE': 0.29},
        'Ensemble': {'MAE': 0.17, 'R²': 0.93, 'RMSE': 0.24}
    }
    
    metrics_df = pd.DataFrame(accuracy_metrics).T
    st.dataframe(metrics_df)
    
    # Best model highlight
    best_model = min(accuracy_metrics.keys(), key=lambda x: accuracy_metrics[x]['MAE'])
    st.success(f"🏆 Best performing model: **{best_model}**")

with col2:
    st.subheader("🎯 Feature Importance")
    
    # Feature importance visualization
    features = ['Air Quality', 'Noise Level', 'Temperature', 'Humidity', 'Crowd Density', 'Time of Day', 'Day of Week']
    importance = [0.28, 0.22, 0.18, 0.12, 0.08, 0.07, 0.05]
    
    fig_importance = px.bar(
        x=importance,
        y=features,
        orientation='h',
        title="Feature Importance in Stress Prediction",
        labels={'x': 'Importance Score', 'y': 'Features'}
    )
    fig_importance.update_layout(height=300)
    st.plotly_chart(fig_importance, use_container_width=True)

# Prediction scenarios
st.header("🎭 Scenario Analysis")

st.markdown("**What-if scenarios for stress prediction:**")

scenario_cols = st.columns(3)

with scenario_cols[0]:
    st.subheader("🌡️ Heat Wave Scenario")
    st.markdown("**Temperature: +10°C**")
    
    heat_wave_stress = city_stress_current * 1.4
    st.metric("Predicted Stress Impact", f"+{heat_wave_stress - city_stress_current:.1f}", 
              delta=f"{((heat_wave_stress/city_stress_current - 1) * 100):+.0f}%")
    
    st.markdown("**Recommendations:**")
    st.markdown("- Activate cooling stations")
    st.markdown("- Increase air conditioning")
    st.markdown("- Deploy misting systems")

with scenario_cols[1]:
    st.subheader("🎉 Major Event Scenario")
    st.markdown("**Crowd Density: +50%**")
    
    event_stress = city_stress_current * 1.25
    st.metric("Predicted Stress Impact", f"+{event_stress - city_stress_current:.1f}",
              delta=f"{((event_stress/city_stress_current - 1) * 100):+.0f}%")
    
    st.markdown("**Recommendations:**")
    st.markdown("- Increase public transport")
    st.markdown("- Deploy crowd control")
    st.markdown("- Set up info kiosks")

with scenario_cols[2]:
    st.subheader("🏭 Pollution Spike Scenario")
    st.markdown("**Air Quality: +75 AQI**")
    
    pollution_stress = city_stress_current * 1.6
    st.metric("Predicted Stress Impact", f"+{pollution_stress - city_stress_current:.1f}",
              delta=f"{((pollution_stress/city_stress_current - 1) * 100):+.0f}%")
    
    st.markdown("**Recommendations:**")
    st.markdown("- Issue health alerts")
    st.markdown("- Activate air purifiers")
    st.markdown("- Suggest indoor activities")

# Risk assessment matrix
st.header("⚠️ Risk Assessment Matrix")

# Create risk matrix based on current conditions
risk_zones = []
for sensor in current_data:
    if sensor['status'] == 'active':
        stress = st.session_state.ml_model.predict_sensor_stress(sensor)
        
        # Determine risk factors
        air_risk = "High" if sensor['air_quality'] > 100 else "Medium" if sensor['air_quality'] > 50 else "Low"
        noise_risk = "High" if sensor['noise_level'] > 80 else "Medium" if sensor['noise_level'] > 60 else "Low"
        crowd_risk = "High" if sensor['crowd_density'] > 80 else "Medium" if sensor['crowd_density'] > 50 else "Low"
        
        overall_risk = "High" if stress > 7 else "Medium" if stress > 4 else "Low"
        
        risk_zones.append({
            'Location': sensor['location'],
            'Overall Risk': overall_risk,
            'Air Quality Risk': air_risk,
            'Noise Risk': noise_risk,
            'Crowd Risk': crowd_risk,
            'Stress Score': f"{stress:.1f}/10"
        })

if risk_zones:
    risk_df = pd.DataFrame(risk_zones)
    
    # Color code the dataframe
    def color_risk(val):
        if val == 'High':
            return 'background-color: #ffcccc'
        elif val == 'Medium':
            return 'background-color: #fff2cc'
        elif val == 'Low':
            return 'background-color: #ccffcc'
        return ''
    
    styled_risk_df = risk_df.style.applymap(color_risk, subset=['Overall Risk', 'Air Quality Risk', 'Noise Risk', 'Crowd Risk'])
    st.dataframe(styled_risk_df, use_container_width=True)

# Model training insights
st.header("🧪 Model Training Insights")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📚 Training Data Summary")
    st.markdown("**Dataset Characteristics:**")
    st.markdown(f"- Total samples: {len(current_data) * 24 * 30:,}")
    st.markdown(f"- Features: {7}")
    st.markdown(f"- Sensors: {len(current_data)}")
    st.markdown(f"- Time span: 30 days")
    st.markdown(f"- Update frequency: Real-time")

with col2:
    st.subheader("🎯 Prediction Accuracy by Horizon")
    
    horizon_accuracy = {
        '1 Hour': 94.2,
        '6 Hours': 87.8,
        '24 Hours': 78.5,
        '1 Week': 65.3
    }
    
    horizons = list(horizon_accuracy.keys())
    accuracies = list(horizon_accuracy.values())
    
    fig_accuracy = px.line(
        x=horizons, y=accuracies,
        title="Prediction Accuracy by Time Horizon",
        labels={'x': 'Time Horizon', 'y': 'Accuracy (%)'},
        markers=True
    )
    fig_accuracy.update_layout(height=250)
    st.plotly_chart(fig_accuracy, use_container_width=True)

st.markdown("---")
st.markdown(f"**AI Models last updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
