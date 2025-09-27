import os
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from typing import Dict, List, Any, Optional
import json
from datetime import datetime, timedelta
import uuid

class DatabaseManager:
    """Database manager for Urban Harmony Network with PostgreSQL integration."""
    
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not found")
    
    def _get_connection(self):
        """Get database connection with proper error handling."""
        try:
            return psycopg2.connect(self.database_url, cursor_factory=RealDictCursor)
        except Exception as e:
            print(f"Database connection error: {e}")
            return None
    
    def save_sensor_data(self, sensor_data: Dict[str, Any]) -> bool:
        """Save or update sensor data in the database."""
        conn = self._get_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            # Insert or update sensor data
            query = """
                INSERT INTO sensor_data (
                    sensor_id, location_name, latitude, longitude, 
                    air_quality, noise_level, temperature, humidity, 
                    crowd_density, light_pollution, status, 
                    battery_level, signal_strength, data_quality_score, timestamp
                ) VALUES (
                    %(sensor_id)s, %(location)s, %(latitude)s, %(longitude)s,
                    %(air_quality)s, %(noise_level)s, %(temperature)s, %(humidity)s,
                    %(crowd_density)s, %(light_pollution)s, %(status)s,
                    %(battery_level)s, %(signal_strength)s, %(data_quality_score)s, 
                    %(timestamp)s
                )
                ON CONFLICT (sensor_id) DO UPDATE SET
                    location_name = EXCLUDED.location_name,
                    latitude = EXCLUDED.latitude,
                    longitude = EXCLUDED.longitude,
                    air_quality = EXCLUDED.air_quality,
                    noise_level = EXCLUDED.noise_level,
                    temperature = EXCLUDED.temperature,
                    humidity = EXCLUDED.humidity,
                    crowd_density = EXCLUDED.crowd_density,
                    light_pollution = EXCLUDED.light_pollution,
                    status = EXCLUDED.status,
                    battery_level = EXCLUDED.battery_level,
                    signal_strength = EXCLUDED.signal_strength,
                    data_quality_score = EXCLUDED.data_quality_score,
                    timestamp = EXCLUDED.timestamp
            """
            
            cursor.execute(query, sensor_data)
            conn.commit()
            cursor.close()
            return True
        
        except Exception as e:
            print(f"Error saving sensor data: {e}")
            conn.rollback()
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
                ) VALUES (
                    %(sensor_id)s, %(air_quality)s, %(noise_level)s, 
                    %(temperature)s, %(humidity)s, %(crowd_density)s, 
                    %(light_pollution)s, %(stress_level)s
                )
            """
            
            data = {
                'sensor_id': sensor_id,
                'air_quality': reading_data.get('air_quality'),
                'noise_level': reading_data.get('noise_level'),
                'temperature': reading_data.get('temperature'),
                'humidity': reading_data.get('humidity'),
                'crowd_density': reading_data.get('crowd_density'),
                'light_pollution': reading_data.get('light_pollution'),
                'stress_level': stress_level
            }
            
            cursor.execute(query, data)
            conn.commit()
            cursor.close()
            return True
        
        except Exception as e:
            print(f"Error saving sensor history: {e}")
            conn.rollback()
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
                WHERE sensor_id = %s 
                AND recorded_at >= NOW() - make_interval(hours => %s)
                ORDER BY recorded_at DESC
            """
            
            cursor.execute(query, (sensor_id, hours))
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
                ) VALUES (
                    %(intervention_id)s, %(type)s, %(location_name)s, 
                    %(latitude)s, %(longitude)s, %(description)s, %(status)s,
                    %(priority_level)s, %(estimated_cost)s, %(estimated_impact)s,
                    %(target_metric)s, %(created_by)s
                )
            """
            
            cursor.execute(query, intervention_data)
            conn.commit()
            cursor.close()
            return True
        
        except Exception as e:
            print(f"Error saving intervention: {e}")
            conn.rollback()
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
                ) VALUES (
                    %(user_id)s, %(location_name)s, %(latitude)s, %(longitude)s,
                    %(feedback_type)s, %(rating)s, %(comment)s, %(stress_level)s,
                    %(environmental_concerns)s, %(suggestion)s
                )
            """
            
            cursor.execute(query, feedback_data)
            conn.commit()
            cursor.close()
            return True
        
        except Exception as e:
            print(f"Error saving citizen feedback: {e}")
            conn.rollback()
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
                    WHERE location_name = %s 
                    AND created_at >= NOW() - make_interval(hours => %s)
                    ORDER BY created_at DESC
                """
                cursor.execute(query, (location, hours))
            else:
                query = """
                    SELECT * FROM citizen_feedback 
                    WHERE created_at >= NOW() - make_interval(hours => %s)
                    ORDER BY created_at DESC
                """
                cursor.execute(query, (hours,))
            
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
                ) VALUES (
                    %(transaction_hash)s, %(block_number)s, %(from_city)s, %(to_city)s,
                    %(data_type)s, %(data_hash)s, %(verification_status)s, 
                    %(gas_used)s, %(transaction_fee)s
                )
            """
            
            cursor.execute(query, transaction_data)
            conn.commit()
            cursor.close()
            return True
        
        except Exception as e:
            print(f"Error saving blockchain transaction: {e}")
            conn.rollback()
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
                    WHERE from_city = %s OR to_city = %s
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
                ) VALUES (
                    %(model_name)s, %(model_version)s, %(accuracy_metrics)s,
                    %(training_data_size)s, %(deployment_status)s, %(performance_notes)s
                )
            """
            
            # Convert accuracy_metrics to proper JSON format
            if 'accuracy_metrics' in model_data:
                model_data['accuracy_metrics'] = Json(model_data['accuracy_metrics'])
            
            cursor.execute(query, model_data)
            conn.commit()
            cursor.close()
            return True
        
        except Exception as e:
            print(f"Error saving ML model performance: {e}")
            conn.rollback()
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
                ) VALUES (
                    %(alert_id)s, %(sensor_id)s, %(alert_type)s, %(severity_level)s,
                    %(message)s, %(location_name)s, %(latitude)s, %(longitude)s, %(status)s
                )
            """
            
            cursor.execute(query, alert_data)
            conn.commit()
            cursor.close()
            return True
        
        except Exception as e:
            print(f"Error creating alert: {e}")
            conn.rollback()
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