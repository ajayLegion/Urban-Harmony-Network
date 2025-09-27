import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils.sensor_simulation import SensorNetwork
from utils.data_processor import DataProcessor
from utils.ml_models import StressPredictionModel

st.set_page_config(page_title="Interventions", layout="wide")

# Initialize session state
if 'sensor_network' not in st.session_state:
    st.session_state.sensor_network = SensorNetwork()
    st.session_state.data_processor = DataProcessor()
    st.session_state.ml_model = StressPredictionModel()

# Initialize intervention tracking
if 'active_interventions' not in st.session_state:
    st.session_state.active_interventions = []

if 'intervention_history' not in st.session_state:
    st.session_state.intervention_history = []

st.title("💡 Smart Urban Interventions")
st.markdown("### AI-driven recommendations for improving urban mental health")

# Sidebar controls
st.sidebar.header("🎛️ Intervention Controls")

intervention_mode = st.sidebar.selectbox(
    "Intervention Mode",
    ["Automatic", "Manual Override", "Recommendation Only"]
)

auto_execute = st.sidebar.checkbox("Auto-execute interventions", value=False)
emergency_override = st.sidebar.checkbox("Emergency override mode", value=False)

if st.sidebar.button("🔄 Refresh Recommendations"):
    st.rerun()

# Get current data and analyze stress levels
current_data = st.session_state.sensor_network.get_all_sensor_data()
processed_data = st.session_state.data_processor.process_sensor_data(current_data)

# Current situation analysis
st.header("📊 Current Situation Analysis")

col1, col2, col3, col4 = st.columns(4)

# Calculate key metrics
active_sensors = [s for s in current_data if s['status'] == 'active']
stress_levels = [st.session_state.ml_model.predict_sensor_stress(s) for s in active_sensors]
city_stress = np.mean(stress_levels) if stress_levels else 0

high_stress_zones = len([s for s in stress_levels if s > 7])
medium_stress_zones = len([s for s in stress_levels if 4 <= s <= 7])
active_interventions_count = len(st.session_state.active_interventions)

with col1:
    st.metric("City Stress Level", f"{city_stress:.1f}/10")

with col2:
    st.metric("High Stress Zones", high_stress_zones, 
              delta=f"{high_stress_zones} requiring intervention")

with col3:
    st.metric("Active Interventions", active_interventions_count)

with col4:
    intervention_effectiveness = np.random.uniform(75, 95)
    st.metric("Intervention Effectiveness", f"{intervention_effectiveness:.1f}%")

# Recommended interventions
st.header("🎯 Recommended Interventions")

# Generate intervention recommendations based on current conditions
recommendations = []

for sensor in active_sensors:
    stress_level = st.session_state.ml_model.predict_sensor_stress(sensor)
    
    if stress_level > 7:  # High stress
        primary_factors = []
        
        if sensor['air_quality'] > 100:
            primary_factors.append('air_quality')
            recommendations.append({
                'location': sensor['location'],
                'sensor_id': sensor['sensor_id'],
                'priority': 'High',
                'type': 'Air Quality Improvement',
                'description': 'Deploy mobile air purification units',
                'estimated_impact': '2.1 stress reduction',
                'implementation_time': '15 minutes',
                'cost': 'Medium',
                'primary_factor': 'Air Quality',
                'stress_level': stress_level
            })
        
        if sensor['noise_level'] > 80:
            primary_factors.append('noise')
            recommendations.append({
                'location': sensor['location'],
                'sensor_id': sensor['sensor_id'],
                'priority': 'High',
                'type': 'Noise Reduction',
                'description': 'Activate sound barriers and ambient masking',
                'estimated_impact': '1.8 stress reduction',
                'implementation_time': '10 minutes',
                'cost': 'Low',
                'primary_factor': 'Noise',
                'stress_level': stress_level
            })
        
        if sensor['crowd_density'] > 85:
            primary_factors.append('crowd')
            recommendations.append({
                'location': sensor['location'],
                'sensor_id': sensor['sensor_id'],
                'priority': 'High',
                'type': 'Crowd Management',
                'description': 'Implement dynamic routing suggestions',
                'estimated_impact': '1.5 stress reduction',
                'implementation_time': '5 minutes',
                'cost': 'Low',
                'primary_factor': 'Crowd Density',
                'stress_level': stress_level
            })
        
        if sensor['temperature'] > 30:
            primary_factors.append('temperature')
            recommendations.append({
                'location': sensor['location'],
                'sensor_id': sensor['sensor_id'],
                'priority': 'High',
                'type': 'Cooling Activation',
                'description': 'Deploy misting systems and shade structures',
                'estimated_impact': '2.0 stress reduction',
                'implementation_time': '20 minutes',
                'cost': 'Medium',
                'primary_factor': 'Temperature',
                'stress_level': stress_level
            })
    
    elif 4 <= stress_level <= 7:  # Medium stress - preventive measures
        recommendations.append({
            'location': sensor['location'],
            'sensor_id': sensor['sensor_id'],
            'priority': 'Medium',
            'type': 'Preventive Enhancement',
            'description': 'Increase green space lighting and nature sounds',
            'estimated_impact': '0.8 stress reduction',
            'implementation_time': '8 minutes',
            'cost': 'Low',
            'primary_factor': 'Prevention',
            'stress_level': stress_level
        })

# Display recommendations
if recommendations:
    recommendations_df = pd.DataFrame(recommendations)
    
    # Sort by priority and stress level
    priority_order = {'High': 3, 'Medium': 2, 'Low': 1}
    recommendations_df['priority_score'] = recommendations_df['priority'].map(priority_order)
    recommendations_df = recommendations_df.sort_values(['priority_score', 'stress_level'], ascending=[False, False])
    
    st.dataframe(recommendations_df.drop(['priority_score'], axis=1), use_container_width=True)
    
    # Intervention execution interface
    st.header("⚡ Execute Interventions")
    
    selected_interventions = []
    for idx, rec in recommendations_df.iterrows():
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown(f"**{rec['type']}** - {rec['location']}")
            st.markdown(f"*{rec['description']}*")
        
        with col2:
            priority_color = "🔴" if rec['priority'] == 'High' else "🟡" if rec['priority'] == 'Medium' else "🟢"
            st.markdown(f"{priority_color} **{rec['priority']}** Priority")
            st.markdown(f"Impact: {rec['estimated_impact']}")
        
        with col3:
            if st.button(f"Execute", key=f"exec_{idx}"):
                # Add to active interventions
                intervention = {
                    'id': len(st.session_state.active_interventions) + 1,
                    'location': rec['location'],
                    'type': rec['type'],
                    'start_time': datetime.now(),
                    'status': 'Active',
                    'estimated_completion': datetime.now() + timedelta(minutes=int(rec['implementation_time'].split()[0]))
                }
                st.session_state.active_interventions.append(intervention)
                
                # Add to history
                st.session_state.intervention_history.append({
                    **intervention,
                    'executed_at': datetime.now(),
                    'estimated_impact': rec['estimated_impact']
                })
                
                st.success(f"✅ Intervention executed: {rec['type']} at {rec['location']}")
                st.rerun()

else:
    st.success("🎉 No interventions needed - all zones operating within optimal stress levels!")

# Active interventions monitoring
st.header("🔄 Active Interventions")

if st.session_state.active_interventions:
    active_interventions_data = []
    
    for intervention in st.session_state.active_interventions:
        elapsed_time = datetime.now() - intervention['start_time']
        remaining_time = intervention['estimated_completion'] - datetime.now()
        
        # Check if intervention should be completed
        if remaining_time.total_seconds() <= 0:
            intervention['status'] = 'Completed'
        
        progress = min(100, (elapsed_time.total_seconds() / (intervention['estimated_completion'] - intervention['start_time']).total_seconds()) * 100)
        
        active_interventions_data.append({
            'ID': intervention['id'],
            'Location': intervention['location'],
            'Type': intervention['type'],
            'Status': intervention['status'],
            'Progress': f"{progress:.0f}%",
            'Remaining Time': f"{max(0, remaining_time.total_seconds() / 60):.0f} min"
        })
    
    active_df = pd.DataFrame(active_interventions_data)
    st.dataframe(active_df, use_container_width=True)
    
    # Remove completed interventions button
    if st.button("🧹 Clear Completed Interventions"):
        st.session_state.active_interventions = [
            i for i in st.session_state.active_interventions 
            if i['status'] != 'Completed'
        ]
        st.rerun()

else:
    st.info("No active interventions currently running.")

# Intervention types and capabilities
st.header("🛠️ Available Intervention Systems")

intervention_systems = {
    'Air Quality Management': {
        'description': 'Mobile air purification units and filtration systems',
        'capabilities': ['PM2.5 reduction', 'CO2 filtering', 'Allergen removal'],
        'coverage': '500m radius',
        'response_time': '10-15 minutes',
        'effectiveness': '85%'
    },
    'Noise Control': {
        'description': 'Sound barriers and ambient audio masking',
        'capabilities': ['Traffic noise reduction', 'Construction dampening', 'Nature sound overlay'],
        'coverage': '300m radius',
        'response_time': '5-10 minutes',
        'effectiveness': '78%'
    },
    'Climate Control': {
        'description': 'Misting systems and portable cooling/heating',
        'capabilities': ['Temperature regulation', 'Humidity control', 'Wind generation'],
        'coverage': '200m radius',
        'response_time': '15-25 minutes',
        'effectiveness': '82%'
    },
    'Crowd Management': {
        'description': 'Dynamic routing and space optimization',
        'capabilities': ['Route suggestions', 'Capacity alerts', 'Flow optimization'],
        'coverage': 'City-wide',
        'response_time': '2-5 minutes',
        'effectiveness': '73%'
    },
    'Lighting Optimization': {
        'description': 'Smart lighting for mood and safety enhancement',
        'capabilities': ['Circadian adjustment', 'Safety improvement', 'Ambiance control'],
        'coverage': '1km radius',
        'response_time': '1-3 minutes',
        'effectiveness': '69%'
    },
    'Green Space Enhancement': {
        'description': 'Mobile gardens and nature exposure systems',
        'capabilities': ['Oxygen generation', 'Stress reduction', 'Biophilia activation'],
        'coverage': '400m radius',
        'response_time': '30-60 minutes',
        'effectiveness': '91%'
    }
}

cols = st.columns(2)
for idx, (system_name, details) in enumerate(intervention_systems.items()):
    with cols[idx % 2]:
        with st.expander(f"🔧 {system_name}"):
            st.markdown(f"**Description:** {details['description']}")
            st.markdown(f"**Coverage:** {details['coverage']}")
            st.markdown(f"**Response Time:** {details['response_time']}")
            st.markdown(f"**Effectiveness:** {details['effectiveness']}")
            st.markdown("**Capabilities:**")
            for capability in details['capabilities']:
                st.markdown(f"- {capability}")

# Intervention effectiveness analysis
st.header("📈 Intervention Effectiveness Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Historical Success Rates")
    
    # Generate historical effectiveness data
    intervention_types = ['Air Quality', 'Noise Control', 'Climate Control', 'Crowd Management', 'Lighting', 'Green Space']
    success_rates = [85, 78, 82, 73, 69, 91]
    
    fig_success = px.bar(
        x=intervention_types,
        y=success_rates,
        title="Intervention Success Rates",
        labels={'x': 'Intervention Type', 'y': 'Success Rate (%)'},
        color=success_rates,
        color_continuous_scale='RdYlGn'
    )
    fig_success.update_layout(height=400)
    st.plotly_chart(fig_success, use_container_width=True)

with col2:
    st.subheader("📉 Stress Reduction Impact")
    
    # Generate stress reduction data
    impact_data = {
        'Intervention': intervention_types,
        'Avg Stress Reduction': [2.1, 1.8, 2.0, 1.5, 1.2, 2.8],
        'Response Time (min)': [12, 7, 20, 3, 2, 45]
    }
    
    fig_impact = px.scatter(
        impact_data,
        x='Response Time (min)',
        y='Avg Stress Reduction',
        size='Avg Stress Reduction',
        color='Intervention',
        title="Stress Reduction vs Response Time",
        hover_data=['Intervention']
    )
    fig_impact.update_layout(height=400)
    st.plotly_chart(fig_impact, use_container_width=True)

# Cost-benefit analysis
st.header("💰 Cost-Benefit Analysis")

cost_benefit_data = {
    'Intervention Type': intervention_types,
    'Implementation Cost': ['Medium', 'Low', 'Medium', 'Low', 'Low', 'High'],
    'Operating Cost/Hour': ['$45', '$12', '$38', '$8', '$5', '$67'],
    'Stress Reduction Value': ['$210', '$180', '$200', '$150', '$120', '$280'],
    'ROI': ['367%', '1400%', '426%', '1775%', '2300%', '318%']
}

cost_df = pd.DataFrame(cost_benefit_data)
st.dataframe(cost_df, use_container_width=True)

# Emergency intervention protocols
st.header("🚨 Emergency Intervention Protocols")

if emergency_override:
    st.error("⚠️ **EMERGENCY OVERRIDE MODE ACTIVE**")
    
    emergency_protocols = [
        "🔴 **Code Red - Extreme Air Quality Event**: Deploy all available air purification units",
        "🟠 **Code Orange - Heat Emergency**: Activate all cooling systems and emergency shelters",
        "🟡 **Code Yellow - Noise Crisis**: Implement emergency sound barriers and traffic rerouting",
        "🔵 **Code Blue - Mass Gathering**: Deploy crowd control and emergency routing systems"
    ]
    
    for protocol in emergency_protocols:
        st.markdown(protocol)
    
    if st.button("🚨 ACTIVATE ALL EMERGENCY PROTOCOLS"):
        st.error("All emergency intervention systems have been activated!")

# Intervention scheduling
st.header("📅 Scheduled Interventions")

st.markdown("**Upcoming Scheduled Interventions:**")

scheduled_interventions = [
    {
        'Date': '2025-09-25 08:00',
        'Type': 'Preventive Air Quality',
        'Location': 'Central Park',
        'Reason': 'Rush hour traffic spike prediction'
    },
    {
        'Date': '2025-09-25 14:00',
        'Type': 'Climate Control',
        'Location': 'Times Square',
        'Reason': 'High temperature forecast'
    },
    {
        'Date': '2025-09-25 18:00',
        'Type': 'Crowd Management',
        'Location': 'Brooklyn Bridge',
        'Reason': 'Evening commute optimization'
    }
]

scheduled_df = pd.DataFrame(scheduled_interventions)
st.dataframe(scheduled_df, use_container_width=True)

st.markdown("---")
st.markdown(f"**Intervention system last updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
