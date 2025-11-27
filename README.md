# Urban Harmony Network
## Overview

Welcome to **Urban Harmony Network** – an innovative IoT&Api system designed to monitor and improve urban mental health by tracking environmental factors in real-time. Imagine a city where air quality, noise levels, crowd density, and more are constantly analyzed to predict stress hotspots and suggest interventions. This project turns that vision into reality, using sensors, AI, and blockchain to create healthier urban spaces.

Built with a focus on New York City (initially targeting 20 key locations like Times Square, Central Park, and the Financial District), this system aggregates data from various sources, runs predictive models, and even automates urban infrastructure like traffic lights and air purifiers to reduce stress.

Whether you're a developer, urban planner, or just curious about smart cities, this repo has everything you need to get started, deploy, and contribute.

## Key Features

- **Real-Time Monitoring**: Collects data on air quality, noise, temperature, humidity, light, and crowd density.
- **Stress Prediction**: Uses machine learning to forecast mental health impacts from environmental data.
- **Interventions**: Automatically suggests and simulates actions like optimizing traffic or activating noise cancellation.
- **Data Visualization**: Interactive dashboards with maps, charts, and trends.
- **Blockchain Integration**: Secure, decentralized sharing of data across cities while ensuring privacy.
- **Mobile Support**: Push notifications and API access for on-the-go insights.
- **Simulation Mode**: Test with mock data before deploying real hardware.

## Technologies Used

This project leverages a robust stack of modern tools to handle everything from data collection to AI-driven insights. Here's a breakdown:

### Frontend & Visualization
- **Streamlit**: Powers the intuitive web dashboard for real-time views.
- **Plotly**: Creates interactive graphs and charts for data exploration.
- **Folium**: Handles geospatial mapping to visualize stress levels across locations.
- **Streamlit-Folium**: Seamless integration for maps in the dashboard.

### Backend & Data Processing
- **Python 3.11**: The core language driving the logic.
- **NumPy & Pandas**: For efficient data manipulation and analysis.
- **Scikit-learn**: Traditional ML models like Random Forest and Gradient Boosting.
- **TensorFlow**: Deep learning frameworks for advanced neural networks.

### Database & Storage
- **PostgreSQL 16**: Stores sensor data, predictions, and intervention logs.
- **psycopg2**: Python adapter for database interactions.

### IoT & Real-Time Data
- **Requests**: API calls to external services like PurpleAir and EPA AirNow.
- **Web3.py**: Blockchain for secure inter-city data sharing.
- **WebSocket**: Enables real-time streaming of updates.

### Security & Blockchain
- **Cryptography & PyJWT**: Handles encryption, JWT authentication, and secure data.
- **Ethereum Integration**: Smart contracts for decentralized, privacy-focused data exchange.
- **Differential Privacy**: Ensures anonymous sharing compliant with GDPR/CCPA.

### Machine Learning & Algorithms
We use a mix of traditional and deep learning techniques to process and predict from environmental data:

#### Traditional ML Models
- **Random Forest Regressor**: Ensemble of 100 decision trees for stress prediction and feature importance.
- **Gradient Boosting Regressor**: Sequential boosting with 100 estimators for handling non-linear data.
- **Multi-Layer Perceptron (MLP)**: Neural network with layers (100, 50 neurons) for pattern recognition.

#### Deep Learning Models
- **Deep MLP**: 4 hidden layers (256 → 128 → 64 → 32) with ReLU, batch norm, and dropout.
- **LSTM**: Two layers (128, 64 units) for time-series forecasting.
- **1D CNN**: Conv1D layers (64, 32 filters) for stream pattern extraction.
- **Transformer**: Multi-head attention (4 heads) for complex data relationships.

#### Data Processing & Analysis Algorithms
- **Scalers**: StandardScaler and MinMaxScaler for feature normalization.
- **Ensemble Averaging**: Weighted combination of models (e.g., 40% Random Forest).
- **AQI Calculation**: EPA formula for air quality indexing.
- **Comfort & Stress Indices**: Multi-factor algorithms weighing temperature, humidity, noise, etc.
- **Time-Series Patterns**: Sinusoidal modeling with Gaussian noise for realistic simulations.
- **Trend Analysis**: Linear regression, Pearson correlation, and polynomial fitting.
- **Spatial Matching**: Euclidean distance for location-based nearest neighbors.

#### Other Algorithms
- **Health Risk Scoring**: Threshold-based assessment on a 0-10 scale.
- **Data Quality Checks**: Completeness, range validation, and recency scoring.
- **Prediction Horizons**: Forecasts for 1h, 6h, 24h, and 1 week with uncertainty modeling.
- **Caching**: Time-based (10-min) and LRU for efficient API handling.
- **Statistical Tools**: Gaussian distributions, moving averages, and percentiles.

### IoT Infrastructure
- **Sensor APIs**: PurpleAir, EPA AirNow, OpenWeatherMap.
- **Hardware Control**: Smart traffic lights, air purifiers, noise cancellation, and adaptive lighting.
- **Connectivity**: LoRaWAN, cellular, WebSocket, REST APIs.

### Mobile & APIs
- **Firebase**: For push notifications.
- **JWT & REST**: Secure mobile app integration.

This stack ensures scalability, security, and real-time performance for urban-scale deployments.

## Sensors Recommended

To bring this system to life, we recommend the following sensors based on the project's IoT focus:

### Primary Environmental Sensors
1. **Air Quality**:
   - PM2.5/PM10 Sensors (e.g., PurpleAir PA-II-SD for particulates).
   - Gas Sensors (NO2, O3, CO) like Bosch BME680.
   - VOC Sensors for indoor volatiles.

2. **Noise Levels**:
   - Digital Sound Level Meters (MEMS microphones).
   - Frequency-specific sensors for detailed pollution analysis.

3. **Environmental Comfort**:
   - Temperature/Humidity (DHT22 or SHT30).
   - Light Pollution (Photoresistors or lux meters).

4. **Crowd Density**:
   - Infrared Counters.
   - Bluetooth/WiFi Beacons for anonymous estimation.
   - Privacy-focused Camera Sensors with edge AI.

### Hardware 
### building project iot Hardware on 2nd stage
- **Controllers**: ESP32 or Raspberry Pi for integration.
- **Connectivity**: LoRaWAN or cellular for data transmission.
- **Power**: Solar panels with batteries for outdoor sustainability.

The code already supports simulation for these sensors, making hardware swaps seamless. Deploy at high-impact NYC spots for best results!

## Installation

1. Clone the repo:
   ```
   git clone https://github.com/ajayLegion/Urban-Harmony-Network.git
   cd urban-harmony-network
   ```

2. Set up a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Configure the database:
   - Install PostgreSQL and create a database.
   - Update `config.py` with your DB credentials.

5. Set up APIs:
   - Add keys for PurpleAir, EPA AirNow, OpenWeatherMap in `.env`.

6. Run the app:
   ```
   streamlit run app.py
   ```

## Usage

- **Dashboard**: Access via browser for maps, predictions, and interventions.
- **API Endpoints**: Use `/predict`, `/intervene` for integrations.
- **Mobile**: Connect via JWT-authenticated REST APIs.
- **Simulation**: Toggle mock data in `sensor_simulation.py` for testing.

For detailed code walkthroughs, check the source files like `ml_models.py` and `tensorflow_models.py`.

## Contributing

We'd love your help! Fork the repo, create a branch, and submit a PR. Focus on:
- Adding models.
- Improving UI/UX.
- Expanding to other cities.

Please follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

This project is licensed under the   Apache-2.0 license - see the [LICENSE](LICENSE) file for details.

---

# Urban Harmony Network (UHN)

## Overview

The **Urban Harmony Network (UHN)** is a pioneering IoT with using api ecosystem designed to combat urban mental health deterioration by creating adaptive, healthier city environments. UHN integrates real-time environmental and biometric data with AI-driven predictions and active interventions to mitigate stress caused by urban factors like pollution, noise, and isolation. This project addresses both current urban mental health challenges and future issues like climate-induced heatwaves and increasing urbanization, expected to impact over 1 billion people by 2030.

## Problem Statement

Urban living contributes to mental health issues, with air pollution linked to a 10-20% increase in anxiety/depression, noise pollution causing chronic stress, and limited green spaces fostering isolation. By 2030, 68% of the global population will live in cities, amplifying these issues through climate-driven heatwaves and resource strain. UHN shifts from reactive treatments to preventive, real-time environmental tuning for "self-healing" cities.
