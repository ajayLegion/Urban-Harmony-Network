import os
import sqlite3
from typing import Dict, List, Any, Optional
import json
from datetime import datetime, timedelta
import uuid

class DatabaseManager:
    """Database manager for Urban Harmony Network with SQLite integration for local development."""
    
    def __init__(self, db_path: str = "urban_harmony.db"):
        self.db_path = db_path
        self._initialize_database()
    
    def _get_connection(self):
        """Get database connection with proper error handling."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # This allows dict-like access
            return conn
        except Exception as e:
            print(f"Database connection error: {e}")
            return None
    
    def _initialize_database(self):
        """Initialize database tables if they don't exist."""
        conn = self._get_connection()
        if not conn:
            return
        
        try:
            cursor = conn.cursor()
            
            # Create sensor_data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sensor_data (
                    sensor_id TEXT PRIMARY KEY,
                    location_name TEXT,
                    latitude REAL,
                    longitude REAL,
                    air_quality REAL,
                    noise_level REAL,
                    temperature REAL,
                    humidity REAL,
                    crowd_density REAL,
                    light_pollution REAL,
                    status TEXT,
                    battery_level REAL,
                    signal_strength REAL,
                    data_quality_score REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create sensor_readings_history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sensor_readings_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sensor_id TEXT,
                    air_quality REAL,
                    noise_level REAL,
                    temperature REAL,
                    humidity REAL,
                    crowd_density REAL,
                    light_pollution REAL,
                    stress_level REAL,
                    recorded_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create interventions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interventions (
                    intervention_id TEXT PRIMARY KEY,
                    type TEXT,
                    location_name TEXT,
                    latitude REAL,
                    longitude REAL,
                    description TEXT,
                    status TEXT,
                    priority_level INTEGER,
                    estimated_cost REAL,
                    estimated_impact REAL,
                    target_metric TEXT,
                    created_by TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create citizen_feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS citizen_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    location_name TEXT,
                    latitude REAL,
                    longitude REAL,
                    feedback_type TEXT,
                    rating INTEGER,
                    comment TEXT,
                    stress_level REAL,
                    environmental_concerns TEXT,
                    suggestion TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create blockchain_transactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS blockchain_transactions (
                    transaction_hash TEXT PRIMARY KEY,
                    block_number INTEGER,
                    from_city TEXT,
                    to_city TEXT,
                    data_type TEXT,
                    data_hash TEXT,
                    verification_status TEXT,
                    gas_used REAL,
                    transaction_fee REAL,
                    timestamp_shared DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create ml_model_performance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ml_model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT,
                    model_version TEXT,
                    accuracy_metrics TEXT,  -- JSON string
                    training_data_size INTEGER,
                    deployment_status TEXT,
                    performance_notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create real_time_alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS real_time_alerts (
                    alert_id TEXT PRIMARY KEY,
                    sensor_id TEXT,
                    alert_type TEXT,
                    severity_level INTEGER,
                    message TEXT,
                    location_name TEXT,
                    latitude REAL,
                    longitude REAL,
                    status TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            cursor.close()
            
        except Exception as e:
            print(f"Error initializing database: {e}")
        finally:
            conn.close()
    
    def save_sensor_data(self, sensor_data: Dict[str, Any]) -> bool:
        """Save or update sensor data in the database."""
        conn = self._get_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            # Insert or replace sensor data
            query = """
                INSERT OR REPLACE INTO sensor_data (
                    sensor_id, location_name, latitude, longitude, 
                    air_quality, noise_level, temperature, humidity, 
                    crowd_density, light_pollution, status, 
                    battery_level, signal_strength, data_quality_score, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            cursor.execute(query, (
                sensor_data.get('sensor_id'),
                sensor_data.get('location'),
                sensor_data.get('latitude'),
                sensor_data.get('longitude'),
                sensor_data.get('air_quality'),
                sensor_data.get('noise_level'),
                sensor_data.get('temperature'),
                sensor_data.get('humidity'),
                sensor_data.get('crowd_density'),
                sensor_data.get('light_pollution'),
                sensor_data.get('status'),
                sensor_data.get('battery_level'),
                sensor_data.get('signal_strength'),
                sensor_data.get('data_quality_score'),
                sensor_data.get('timestamp', datetime.now().isoformat())
            ))
            
            conn.commit()
            cursor.close()
            return True
        
        except Exception as e:
            print(f"Error saving sensor data: {e}")
            return False
        finally:
            conn.close()
    
    def get_all_sensors(self) -> List[Dict[str, Any]]:
        """Retrieve all sensors from the database."""
        conn = self._get_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM sensor_data 
                ORDER BY location_name, sensor_id
            """)
            
            sensors = cursor.fetchall()
            cursor.close()
            
            # Convert to list of dictionaries
            return [dict(sensor) for sensor in sensors]
        
        except Exception as e:
            print(f"Error retrieving sensors: {e}")
            return []
        finally:
            conn.close()
    
    def save_sensor_reading_history(self, sensor_id: str, reading_data: Dict[str, Any], 
                                   stress_level: float = None) -> bool:
        """Save historical sensor reading."""
        conn = self._get_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            query = """
                INSERT INTO sensor_readings_history (
                    sensor_id, air_quality, noise_level, temperature, 
                    humidity, crowd_density, light_pollution, stress_level
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            cursor.execute(query, (
                sensor_id,
                reading_data.get('air_quality'),
                reading_data.get('noise_level'),
                reading_data.get('temperature'),
                reading_data.get('humidity'),
                reading_data.get('crowd_density'),
                reading_data.get('light_pollution'),
                stress_level
            ))
            
            conn.commit()
            cursor.close()
            return True
        
        except Exception as e:
            print(f"Error saving sensor history: {e}")
            return False
        finally:
            conn.close()
    
    def get_sensor_history(self, sensor_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get historical sensor readings."""
        conn = self._get_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor()
            
            query = """
                SELECT * FROM sensor_readings_history 
                WHERE sensor_id = ? 
                AND recorded_at >= datetime('now', '-{} hours')
                ORDER BY recorded_at DESC
            """.format(hours)
            
            cursor.execute(query, (sensor_id,))
            history = cursor.fetchall()
            cursor.close()
            
            return [dict(record) for record in history]
        
        except Exception as e:
            print(f"Error retrieving sensor history: {e}")
            return []
        finally:
            conn.close()
    
    def save_intervention(self, intervention_data: Dict[str, Any]) -> bool:
        """Save intervention data to database."""
        conn = self._get_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            # Generate intervention ID if not provided
            if 'intervention_id' not in intervention_data:
                intervention_data['intervention_id'] = f"INT-{uuid.uuid4().hex[:8]}"
            
            query = """
                INSERT INTO interventions (
                    intervention_id, type, location_name, latitude, longitude,
                    description, status, priority_level, estimated_cost,
                    estimated_impact, target_metric, created_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            cursor.execute(query, (
                intervention_data.get('intervention_id'),
                intervention_data.get('type'),
                intervention_data.get('location_name'),
                intervention_data.get('latitude'),
                intervention_data.get('longitude'),
                intervention_data.get('description'),
                intervention_data.get('status'),
                intervention_data.get('priority_level'),
                intervention_data.get('estimated_cost'),
                intervention_data.get('estimated_impact'),
                intervention_data.get('target_metric'),
                intervention_data.get('created_by')
            ))
            
            conn.commit()
            cursor.close()
            return True
        
        except Exception as e:
            print(f"Error saving intervention: {e}")
            return False
        finally:
            conn.close()
    
    def get_active_interventions(self) -> List[Dict[str, Any]]:
        """Get all active interventions."""
        conn = self._get_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM interventions 
                WHERE status IN ('planned', 'active', 'in_progress')
                ORDER BY priority_level DESC, created_at DESC
            """)
            
            interventions = cursor.fetchall()
            cursor.close()
            
            return [dict(intervention) for intervention in interventions]
        
        except Exception as e:
            print(f"Error retrieving interventions: {e}")
            return []
        finally:
            conn.close()
    
    def save_citizen_feedback(self, feedback_data: Dict[str, Any]) -> bool:
        """Save citizen feedback from mobile app."""
        conn = self._get_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            query = """
                INSERT INTO citizen_feedback (
                    user_id, location_name, latitude, longitude, 
                    feedback_type, rating, comment, stress_level,
                    environmental_concerns, suggestion
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            cursor.execute(query, (
                feedback_data.get('user_id'),
                feedback_data.get('location_name'),
                feedback_data.get('latitude'),
                feedback_data.get('longitude'),
                feedback_data.get('feedback_type'),
                feedback_data.get('rating'),
                feedback_data.get('comment'),
                feedback_data.get('stress_level'),
                feedback_data.get('environmental_concerns'),
                feedback_data.get('suggestion')
            ))
            
            conn.commit()
            cursor.close()
            return True
        
        except Exception as e:
            print(f"Error saving citizen feedback: {e}")
            return False
        finally:
            conn.close()
    
    def get_citizen_feedback(self, location: str = None, hours: int = 72) -> List[Dict[str, Any]]:
        """Get citizen feedback, optionally filtered by location."""
        conn = self._get_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor()
            
            if location:
                query = """
                    SELECT * FROM citizen_feedback 
                    WHERE location_name = ? 
                    AND created_at >= datetime('now', '-{} hours')
                    ORDER BY created_at DESC
                """.format(hours)
                cursor.execute(query, (location,))
            else:
                query = """
                    SELECT * FROM citizen_feedback 
                    WHERE created_at >= datetime('now', '-{} hours')
                    ORDER BY created_at DESC
                """.format(hours)
                cursor.execute(query)
            
            feedback = cursor.fetchall()
            cursor.close()
            
            return [dict(record) for record in feedback]
        
        except Exception as e:
            print(f"Error retrieving citizen feedback: {e}")
            return []
        finally:
            conn.close()
    
    def save_blockchain_transaction(self, transaction_data: Dict[str, Any]) -> bool:
        """Save blockchain transaction for inter-city data sharing."""
        conn = self._get_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            query = """
                INSERT INTO blockchain_transactions (
                    transaction_hash, block_number, from_city, to_city,
                    data_type, data_hash, verification_status, gas_used, transaction_fee
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            cursor.execute(query, (
                transaction_data.get('transaction_hash'),
                transaction_data.get('block_number'),
                transaction_data.get('from_city'),
                transaction_data.get('to_city'),
                transaction_data.get('data_type'),
                transaction_data.get('data_hash'),
                transaction_data.get('verification_status'),
                transaction_data.get('gas_used'),
                transaction_data.get('transaction_fee')
            ))
            
            conn.commit()
            cursor.close()
            return True
        
        except Exception as e:
            print(f"Error saving blockchain transaction: {e}")
            return False
        finally:
            conn.close()
    
    def get_blockchain_transactions(self, city: str = None) -> List[Dict[str, Any]]:
        """Get blockchain transactions, optionally filtered by city."""
        conn = self._get_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor()
            
            if city:
                query = """
                    SELECT * FROM blockchain_transactions 
                    WHERE from_city = ? OR to_city = ?
                    ORDER BY timestamp_shared DESC
                """
                cursor.execute(query, (city, city))
            else:
                query = """
                    SELECT * FROM blockchain_transactions 
                    ORDER BY timestamp_shared DESC 
                    LIMIT 100
                """
                cursor.execute(query)
            
            transactions = cursor.fetchall()
            cursor.close()
            
            return [dict(record) for record in transactions]
        
        except Exception as e:
            print(f"Error retrieving blockchain transactions: {e}")
            return []
        finally:
            conn.close()
    
    def save_ml_model_performance(self, model_data: Dict[str, Any]) -> bool:
        """Save ML model performance metrics."""
        conn = self._get_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            query = """
                INSERT INTO ml_model_performance (
                    model_name, model_version, accuracy_metrics,
                    training_data_size, deployment_status, performance_notes
                ) VALUES (?, ?, ?, ?, ?, ?)
            """
            
            # Convert accuracy_metrics to JSON string
            accuracy_metrics = model_data.get('accuracy_metrics', {})
            if isinstance(accuracy_metrics, dict):
                accuracy_metrics = json.dumps(accuracy_metrics)
            
            cursor.execute(query, (
                model_data.get('model_name'),
                model_data.get('model_version'),
                accuracy_metrics,
                model_data.get('training_data_size'),
                model_data.get('deployment_status'),
                model_data.get('performance_notes')
            ))
            
            conn.commit()
            cursor.close()
            return True
        
        except Exception as e:
            print(f"Error saving ML model performance: {e}")
            return False
        finally:
            conn.close()
    
    def create_alert(self, alert_data: Dict[str, Any]) -> bool:
        """Create a new real-time alert."""
        conn = self._get_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            # Generate alert ID if not provided
            if 'alert_id' not in alert_data:
                alert_data['alert_id'] = f"ALERT-{uuid.uuid4().hex[:8]}"
            
            query = """
                INSERT INTO real_time_alerts (
                    alert_id, sensor_id, alert_type, severity_level,
                    message, location_name, latitude, longitude, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            cursor.execute(query, (
                alert_data.get('alert_id'),
                alert_data.get('sensor_id'),
                alert_data.get('alert_type'),
                alert_data.get('severity_level'),
                alert_data.get('message'),
                alert_data.get('location_name'),
                alert_data.get('latitude'),
                alert_data.get('longitude'),
                alert_data.get('status')
            ))
            
            conn.commit()
            cursor.close()
            return True
        
        except Exception as e:
            print(f"Error creating alert: {e}")
            return False
        finally:
            conn.close()
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts."""
        conn = self._get_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM real_time_alerts 
                WHERE status = 'active'
                ORDER BY severity_level DESC, created_at DESC
            """)
            
            alerts = cursor.fetchall()
            cursor.close()
            
            return [dict(alert) for alert in alerts]
        
        except Exception as e:
            print(f"Error retrieving alerts: {e}")
            return []
        finally:
            conn.close()
    
    def initialize_demo_data(self) -> bool:
        """Initialize the database with some demo data from our simulation."""
        try:
            from .sensor_simulation import SensorNetwork
            
            # Create a sensor network and save initial data
            network = SensorNetwork()
            sensors = network.get_all_sensor_data()
            
            # Save sensor data to database
            success_count = 0
            for sensor in sensors:
                if self.save_sensor_data(sensor):
                    success_count += 1
            
            print(f"Initialized {success_count}/{len(sensors)} sensors in database")
            return success_count > 0
        except Exception as e:
            print(f"Error initializing demo data: {e}")
            return False
