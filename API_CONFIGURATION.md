# API Configuration Guide

## Required Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# Google Maps API Keys (Required for traffic and crowd data)
GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here

# Data.gov.in API Key (Optional - for environmental data)
DATA_GOV_API_KEY=your_datagov_api_key_here

# Real-time IoT Data Integration
ENABLE_REAL_IOT_DATA=true

# Database Configuration
DATABASE_URL=sqlite:///urban_harmony.db

# API Rate Limiting
API_CACHE_DURATION=300

# Logging Level
LOG_LEVEL=INFO

# Demo Mode (uses simulated data when APIs are unavailable)
DEMO_MODE=true
```

## API Sources Integration

### 1. Bharat AQI API
- **Endpoint**: https://bharataqi-api.herokuapp.com/api/aqi
- **Purpose**: Real-time air quality data for Bangalore
- **Data**: AQI, PM2.5, PM10, NO2, SO2, CO, O3
- **Authentication**: None required

### 2. Data.gov.in API
- **Endpoint**: https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69
- **Purpose**: Government environmental data
- **Data**: Air quality, environmental monitoring
- **Authentication**: API key required (get from data.gov.in)

### 3. CPCB API
- **Endpoint**: https://app.cpcbccr.com/ccr/caaqms/caaqms_landing_data
- **Purpose**: Central Pollution Control Board air quality data
- **Data**: Real-time air quality monitoring
- **Authentication**: None required

### 4. Google Maps Places API
- **Endpoint**: https://maps.googleapis.com/maps/api/place
- **Purpose**: Crowd density and popular times data
- **Data**: Place popularity, crowd levels
- **Authentication**: Google API key required

### 5. Google Maps Traffic API
- **Endpoint**: https://maps.googleapis.com/maps/api/distancematrix
- **Purpose**: Real-time traffic conditions
- **Data**: Travel times, congestion levels
- **Authentication**: Google API key required

## Setup Instructions

### 1. Google Maps API Setup
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the following APIs:
   - Places API
   - Distance Matrix API
   - Maps JavaScript API
4. Create API credentials (API key)
5. Add the API key to your `.env` file

### 2. Data.gov.in API Setup
1. Visit [data.gov.in](https://data.gov.in/)
2. Register for an account
3. Request API access
4. Add the API key to your `.env` file

### 3. Running with Real Data
```bash
# Set environment variable to enable real data
export ENABLE_REAL_IOT_DATA=true

# Run the application
streamlit run app.py
```

### 4. Demo Mode (No API Keys Required)
```bash
# Run in demo mode with simulated data
export DEMO_MODE=true
export ENABLE_REAL_IOT_DATA=false

# Run the application
streamlit run app.py
```

## Features Enabled by APIs

### With Real APIs:
- ✅ Real-time air quality data from multiple sources
- ✅ Actual traffic congestion levels
- ✅ Live crowd density data
- ✅ Historical data trends
- ✅ Accurate location-based monitoring

### In Demo Mode:
- ✅ Simulated but realistic data
- ✅ All features functional
- ✅ No API rate limits
- ✅ No external dependencies

## Troubleshooting

### Common Issues:
1. **API Rate Limits**: The system includes caching to minimize API calls
2. **Network Issues**: Automatic fallback to simulated data
3. **Invalid API Keys**: Check your `.env` file configuration
4. **API Changes**: The system includes error handling for API changes

### Logs:
Check the console output for API integration status and any error messages.

## Cost Considerations

- **Google Maps APIs**: Pay-per-use, see Google's pricing
- **Data.gov.in**: Free with registration
- **Bharat AQI**: Free
- **CPCB**: Free

The system is designed to minimize API calls through intelligent caching.
