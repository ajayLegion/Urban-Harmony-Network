import os
import json
import requests
import asyncio
import websocket
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import threading
import time
from .database_manager import DatabaseManager

class InterventionType(Enum):
    AIR_PURIFICATION = "air_purification"
    NOISE_CONTROL = "noise_control"
    CLIMATE_CONTROL = "climate_control"
    TRAFFIC_MANAGEMENT = "traffic_management"
    LIGHTING_ADJUSTMENT = "lighting_adjustment"
    GREEN_SPACE_ACTIVATION = "green_space_activation"
    CROWD_MANAGEMENT = "crowd_management"
    WATER_FEATURE = "water_feature"

class InterventionStatus(Enum):
    PLANNED = "planned"
    ACTIVATING = "activating"
    ACTIVE = "active"
    DEACTIVATING = "deactivating"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class UrbanInfrastructureDevice:
    device_id: str
    device_type: str
    location: str
    latitude: float
    longitude: float
    capabilities: List[str]
    status: str
    api_endpoint: str
    auth_token: str
    last_communication: datetime

@dataclass
class InterventionCommand:
    intervention_id: str
    device_id: str
    action: str
    parameters: Dict[str, Any]
    priority: int
    scheduled_time: datetime
    duration_minutes: int
    expected_impact: float

class RealTimeInterventionSystem:
    """Real-time intervention system for urban infrastructure control."""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.devices = {}
        self.active_interventions = {}
        self.command_queue = []
        self.intervention_callbacks = {}
        self.system_running = False
        
        # Load infrastructure device configurations
        self._load_infrastructure_devices()
        
        # Start the intervention monitoring system
        self.monitoring_thread = None
    
    def _load_infrastructure_devices(self):
        """Load urban infrastructure devices that can be controlled."""
        
        # Air Quality Management Systems
        self.devices['air_purifier_times_square'] = UrbanInfrastructureDevice(
            device_id='air_purifier_times_square',
            device_type='air_purification_unit',
            location='MG Road',
            latitude=12.9716,
            longitude=77.6197,
            capabilities=['air_filtration', 'ozone_generation', 'particle_capture'],
            status='online',
            api_endpoint='https://api.urbantech.city/air-systems/times-square',
            auth_token=os.getenv('URBAN_INFRASTRUCTURE_API_TOKEN', 'demo_token'),
            last_communication=datetime.now()
        )
        
        # Smart Traffic Light Systems
        self.devices['traffic_control_broadway'] = UrbanInfrastructureDevice(
            device_id='traffic_control_broadway',
            device_type='smart_traffic_lights',
            location='Broadway & 42nd St',
            latitude=40.7590,
            longitude=-73.9845,
            capabilities=['timing_adjustment', 'pedestrian_priority', 'emergency_override'],
            status='online',
            api_endpoint='https://api.nyc.gov/traffic/control/broadway-42nd',
            auth_token=os.getenv('NYC_TRAFFIC_API_TOKEN', 'demo_token'),
            last_communication=datetime.now()
        )
        
        # Smart Street Lighting
        self.devices['lighting_central_park'] = UrbanInfrastructureDevice(
            device_id='lighting_central_park',
            device_type='smart_street_lights',
            location='Cubbon Park',
            latitude=12.9759,
            longitude=77.6094,
            capabilities=['brightness_control', 'color_temperature', 'motion_sensing'],
            status='online',
            api_endpoint='https://api.centralpark.org/lighting/control',
            auth_token=os.getenv('PARKS_API_TOKEN', 'demo_token'),
            last_communication=datetime.now()
        )
        
        # Noise Cancellation Systems
        self.devices['noise_barrier_wall_street'] = UrbanInfrastructureDevice(
            device_id='noise_barrier_wall_street',
            device_type='active_noise_control',
            location='Wall Street',
            latitude=40.7074,
            longitude=-74.0113,
            capabilities=['active_cancellation', 'sound_masking', 'directional_filtering'],
            status='online',
            api_endpoint='https://api.wallstreet.district/noise-control',
            auth_token=os.getenv('FINANCIAL_DISTRICT_API_TOKEN', 'demo_token'),
            last_communication=datetime.now()
        )
        
        # Smart HVAC Systems
        self.devices['hvac_subway_union_square'] = UrbanInfrastructureDevice(
            device_id='hvac_subway_union_square',
            device_type='subway_climate_control',
            location='Union Square Subway',
            latitude=40.7359,
            longitude=-73.9911,
            capabilities=['temperature_control', 'humidity_control', 'air_circulation'],
            status='online',
            api_endpoint='https://api.mta.info/hvac/union-square',
            auth_token=os.getenv('MTA_API_TOKEN', 'demo_token'),
            last_communication=datetime.now()
        )
        
        # Water Features and Cooling Systems
        self.devices['fountain_bryant_park'] = UrbanInfrastructureDevice(
            device_id='fountain_bryant_park',
            device_type='water_feature',
            location='Bryant Park',
            latitude=40.7536,
            longitude=-73.9832,
            capabilities=['mist_generation', 'water_display', 'cooling_zone'],
            status='online',
            api_endpoint='https://api.bryantpark.org/water-features',
            auth_token=os.getenv('BRYANT_PARK_API_TOKEN', 'demo_token'),
            last_communication=datetime.now()
        )
        
        # Digital Signage for Crowd Management
        self.devices['signage_brooklyn_bridge'] = UrbanInfrastructureDevice(
            device_id='signage_brooklyn_bridge',
            device_type='digital_signage',
            location='Electronic City',
            latitude=12.8456,
            longitude=77.6603,
            capabilities=['route_guidance', 'crowd_alerts', 'emergency_messaging'],
            status='online',
            api_endpoint='https://api.nyc.gov/signage/brooklyn-bridge',
            auth_token=os.getenv('NYC_DOT_API_TOKEN', 'demo_token'),
            last_communication=datetime.now()
        )
        
        print(f"Loaded {len(self.devices)} urban infrastructure devices")
    
    def start_monitoring_system(self):
        """Start the real-time monitoring and intervention system."""
        if not self.system_running:
            self.system_running = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            print("Real-time intervention monitoring system started")
    
    def stop_monitoring_system(self):
        """Stop the monitoring system."""
        self.system_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        print("Real-time intervention monitoring system stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop for real-time interventions."""
        while self.system_running:
            try:
                # Process command queue
                self._process_command_queue()
                
                # Monitor active interventions
                self._monitor_active_interventions()
                
                # Check device health
                self._check_device_health()
                
                # Sleep for a short interval
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Wait longer on error
    
    def create_intervention(self, intervention_type: InterventionType, 
                          location: str, sensor_data: Dict[str, Any],
                          duration_minutes: int = 60) -> str:
        """Create a new real-time intervention based on sensor data."""
        
        intervention_id = f"INT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{intervention_type.value[:3].upper()}"
        
        # Find appropriate devices for this intervention
        suitable_devices = self._find_devices_for_intervention(intervention_type, location)
        
        if not suitable_devices:
            print(f"No suitable devices found for {intervention_type.value} at {location}")
            return None
        
        # Calculate intervention parameters based on sensor data
        parameters = self._calculate_intervention_parameters(intervention_type, sensor_data)
        
        # Create intervention commands for each device
        commands = []
        for device in suitable_devices:
            command = InterventionCommand(
                intervention_id=intervention_id,
                device_id=device.device_id,
                action=self._get_device_action(intervention_type, device.device_type),
                parameters=parameters,
                priority=self._calculate_priority(sensor_data),
                scheduled_time=datetime.now(),
                duration_minutes=duration_minutes,
                expected_impact=self._estimate_impact(intervention_type, sensor_data)
            )
            commands.append(command)
        
        # Add commands to queue
        self.command_queue.extend(commands)
        
        # Store intervention in database
        self._save_intervention_to_db(intervention_id, intervention_type, location, 
                                     sensor_data, commands)
        
        print(f"Created intervention {intervention_id} with {len(commands)} commands")
        return intervention_id
    
    def _find_devices_for_intervention(self, intervention_type: InterventionType, 
                                     location: str) -> List[UrbanInfrastructureDevice]:
        """Find devices suitable for a specific intervention type and location."""
        suitable_devices = []
        
        intervention_device_map = {
            InterventionType.AIR_PURIFICATION: ['air_purification_unit', 'hvac_system'],
            InterventionType.NOISE_CONTROL: ['active_noise_control', 'traffic_lights'],
            InterventionType.CLIMATE_CONTROL: ['hvac_system', 'water_feature'],
            InterventionType.TRAFFIC_MANAGEMENT: ['smart_traffic_lights', 'digital_signage'],
            InterventionType.LIGHTING_ADJUSTMENT: ['smart_street_lights'],
            InterventionType.CROWD_MANAGEMENT: ['digital_signage', 'smart_traffic_lights'],
            InterventionType.WATER_FEATURE: ['water_feature']
        }
        
        required_types = intervention_device_map.get(intervention_type, [])
        
        for device in self.devices.values():
            # Check if device type matches
            if device.device_type in required_types or any(dtype in device.device_type for dtype in required_types):
                # Check proximity to location (simplified)
                if location.lower() in device.location.lower() or self._is_device_nearby(device, location):
                    if device.status == 'online':
                        suitable_devices.append(device)
        
        return suitable_devices
    
    def _is_device_nearby(self, device: UrbanInfrastructureDevice, location: str) -> bool:
        """Check if device is nearby the intervention location."""
        # This is a simplified proximity check
        # In a real system, you would use proper geospatial calculations
        location_coords = {
            'MG Road': (12.9716, 77.6197),
            'Cubbon Park': (12.9759, 77.6094),
            'Brigade Road': (12.9716, 77.6197),
            'Electronic City': (12.8456, 77.6603),
            'Koramangala': (12.9279, 77.6271)
        }
        
        if location in location_coords:
            loc_lat, loc_lon = location_coords[location]
            distance = ((device.latitude - loc_lat) ** 2 + (device.longitude - loc_lon) ** 2) ** 0.5
            return distance < 0.01  # Roughly within 1km
        
        return False
    
    def _calculate_intervention_parameters(self, intervention_type: InterventionType, 
                                         sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal parameters for intervention based on sensor data."""
        parameters = {}
        
        if intervention_type == InterventionType.AIR_PURIFICATION:
            air_quality = sensor_data.get('air_quality', 50)
            parameters = {
                'filtration_level': min(100, max(20, air_quality)),
                'fan_speed': min(10, max(1, int(air_quality / 10))),
                'ozone_generation': air_quality > 80
            }
        
        elif intervention_type == InterventionType.NOISE_CONTROL:
            noise_level = sensor_data.get('noise_level', 60)
            parameters = {
                'cancellation_strength': min(100, max(10, noise_level - 40)),
                'frequency_focus': 'traffic' if noise_level > 70 else 'general',
                'masking_enabled': noise_level > 75
            }
        
        elif intervention_type == InterventionType.CLIMATE_CONTROL:
            temperature = sensor_data.get('temperature', 20)
            humidity = sensor_data.get('humidity', 60)
            parameters = {
                'target_temperature': max(18, min(26, 22 if temperature > 28 else temperature + 2)),
                'target_humidity': max(40, min(70, 55 if humidity > 80 else humidity)),
                'ventilation_boost': temperature > 30 or humidity > 80
            }
        
        elif intervention_type == InterventionType.TRAFFIC_MANAGEMENT:
            crowd_density = sensor_data.get('crowd_density', 30)
            parameters = {
                'green_time_extension': min(30, max(5, int(crowd_density / 3))),
                'pedestrian_priority': crowd_density > 60,
                'cycle_optimization': True
            }
        
        elif intervention_type == InterventionType.LIGHTING_ADJUSTMENT:
            light_pollution = sensor_data.get('light_pollution', 100)
            hour = datetime.now().hour
            parameters = {
                'brightness_level': min(100, max(20, 100 - light_pollution / 5)),
                'color_temperature': 3000 if hour > 20 or hour < 6 else 5000,
                'motion_sensitivity': 'high' if 22 <= hour or hour <= 6 else 'normal'
            }
        
        elif intervention_type == InterventionType.WATER_FEATURE:
            temperature = sensor_data.get('temperature', 20)
            crowd_density = sensor_data.get('crowd_density', 30)
            parameters = {
                'mist_intensity': min(100, max(0, (temperature - 18) * 5)),
                'cooling_zone_size': min(100, max(20, crowd_density)),
                'cycle_duration': 15 if temperature > 25 else 10
            }
        
        return parameters
    
    def _get_device_action(self, intervention_type: InterventionType, device_type: str) -> str:
        """Get the appropriate action for a device type and intervention."""
        action_map = {
            'air_purification_unit': {
                InterventionType.AIR_PURIFICATION: 'activate_filtration'
            },
            'smart_traffic_lights': {
                InterventionType.TRAFFIC_MANAGEMENT: 'optimize_timing',
                InterventionType.NOISE_CONTROL: 'reduce_stop_time',
                InterventionType.CROWD_MANAGEMENT: 'enable_pedestrian_priority'
            },
            'smart_street_lights': {
                InterventionType.LIGHTING_ADJUSTMENT: 'adjust_brightness'
            },
            'active_noise_control': {
                InterventionType.NOISE_CONTROL: 'activate_cancellation'
            },
            'subway_climate_control': {
                InterventionType.CLIMATE_CONTROL: 'adjust_hvac'
            },
            'water_feature': {
                InterventionType.WATER_FEATURE: 'activate_cooling',
                InterventionType.CLIMATE_CONTROL: 'enhance_cooling'
            },
            'digital_signage': {
                InterventionType.CROWD_MANAGEMENT: 'display_crowd_guidance',
                InterventionType.TRAFFIC_MANAGEMENT: 'show_alternate_routes'
            }
        }
        
        return action_map.get(device_type, {}).get(intervention_type, 'default_action')
    
    def _calculate_priority(self, sensor_data: Dict[str, Any]) -> int:
        """Calculate intervention priority based on sensor data severity."""
        priority = 1
        
        # Higher priority for severe conditions
        air_quality = sensor_data.get('air_quality', 50)
        noise_level = sensor_data.get('noise_level', 60)
        crowd_density = sensor_data.get('crowd_density', 30)
        temperature = sensor_data.get('temperature', 20)
        
        if air_quality > 100: priority += 3
        elif air_quality > 75: priority += 2
        elif air_quality > 50: priority += 1
        
        if noise_level > 80: priority += 3
        elif noise_level > 70: priority += 2
        elif noise_level > 60: priority += 1
        
        if crowd_density > 80: priority += 3
        elif crowd_density > 60: priority += 2
        elif crowd_density > 40: priority += 1
        
        if temperature > 32 or temperature < 0: priority += 3
        elif temperature > 28 or temperature < 5: priority += 2
        
        return min(10, priority)  # Cap at 10
    
    def _estimate_impact(self, intervention_type: InterventionType, 
                        sensor_data: Dict[str, Any]) -> float:
        """Estimate the expected impact of an intervention."""
        base_impact = {
            InterventionType.AIR_PURIFICATION: 2.5,
            InterventionType.NOISE_CONTROL: 2.0,
            InterventionType.CLIMATE_CONTROL: 1.8,
            InterventionType.TRAFFIC_MANAGEMENT: 1.5,
            InterventionType.LIGHTING_ADJUSTMENT: 1.2,
            InterventionType.CROWD_MANAGEMENT: 1.8,
            InterventionType.WATER_FEATURE: 1.0
        }
        
        impact = base_impact.get(intervention_type, 1.0)
        
        # Adjust based on severity
        if intervention_type == InterventionType.AIR_PURIFICATION:
            air_quality = sensor_data.get('air_quality', 50)
            if air_quality > 100: impact *= 1.5
            elif air_quality > 75: impact *= 1.2
        
        return min(5.0, impact)
    
    def _process_command_queue(self):
        """Process pending intervention commands."""
        current_time = datetime.now()
        
        # Sort commands by priority and scheduled time
        self.command_queue.sort(key=lambda x: (-x.priority, x.scheduled_time))
        
        processed_commands = []
        for command in self.command_queue[:]:  # Copy to avoid modification during iteration
            if current_time >= command.scheduled_time:
                success = self._execute_command(command)
                if success:
                    # Move to active interventions
                    if command.intervention_id not in self.active_interventions:
                        self.active_interventions[command.intervention_id] = {
                            'commands': [],
                            'start_time': current_time,
                            'status': InterventionStatus.ACTIVE
                        }
                    
                    self.active_interventions[command.intervention_id]['commands'].append(command)
                    processed_commands.append(command)
        
        # Remove processed commands from queue
        for command in processed_commands:
            if command in self.command_queue:
                self.command_queue.remove(command)
    
    def _execute_command(self, command: InterventionCommand) -> bool:
        """Execute a single intervention command."""
        device = self.devices.get(command.device_id)
        if not device:
            print(f"Device {command.device_id} not found")
            return False
        
        if device.status != 'online':
            print(f"Device {command.device_id} is not online")
            return False
        
        try:
            # For demo purposes, simulate API calls
            # In production, this would make real HTTP requests to device APIs
            if device.auth_token == 'demo_token':
                success = self._simulate_device_command(device, command)
            else:
                success = self._send_real_device_command(device, command)
            
            if success:
                print(f"Successfully executed {command.action} on {device.device_id}")
                device.last_communication = datetime.now()
                
                # Update database
                self._update_intervention_status(command.intervention_id, InterventionStatus.ACTIVE)
                
            return success
            
        except Exception as e:
            print(f"Failed to execute command on {device.device_id}: {e}")
            return False
    
    def _simulate_device_command(self, device: UrbanInfrastructureDevice, 
                                command: InterventionCommand) -> bool:
        """Simulate device command execution for demo purposes."""
        print(f"SIMULATED: {command.action} on {device.device_type} at {device.location}")
        print(f"Parameters: {json.dumps(command.parameters, indent=2)}")
        
        # Simulate processing time
        time.sleep(0.5)
        
        # Simulate 90% success rate
        import random
        return random.random() > 0.1
    
    def _send_real_device_command(self, device: UrbanInfrastructureDevice, 
                                 command: InterventionCommand) -> bool:
        """Send real command to urban infrastructure device."""
        headers = {
            'Authorization': f'Bearer {device.auth_token}',
            'Content-Type': 'application/json',
            'X-Intervention-ID': command.intervention_id
        }
        
        payload = {
            'action': command.action,
            'parameters': command.parameters,
            'duration_minutes': command.duration_minutes,
            'priority': command.priority
        }
        
        try:
            response = requests.post(
                f"{device.api_endpoint}/control",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            return response.status_code == 200
            
        except requests.exceptions.RequestException as e:
            print(f"HTTP request failed: {e}")
            return False
    
    def _monitor_active_interventions(self):
        """Monitor active interventions and handle completion."""
        current_time = datetime.now()
        completed_interventions = []
        
        for intervention_id, intervention_data in self.active_interventions.items():
            start_time = intervention_data['start_time']
            commands = intervention_data['commands']
            
            # Check if any command has exceeded its duration
            for command in commands:
                end_time = start_time + timedelta(minutes=command.duration_minutes)
                if current_time >= end_time:
                    # Deactivate the intervention
                    self._deactivate_intervention(command)
                    completed_interventions.append(intervention_id)
                    break
        
        # Remove completed interventions
        for intervention_id in completed_interventions:
            del self.active_interventions[intervention_id]
            self._update_intervention_status(intervention_id, InterventionStatus.COMPLETED)
    
    def _deactivate_intervention(self, command: InterventionCommand):
        """Deactivate an intervention command."""
        device = self.devices.get(command.device_id)
        if device:
            deactivate_command = InterventionCommand(
                intervention_id=command.intervention_id,
                device_id=command.device_id,
                action=f"deactivate_{command.action}",
                parameters={'restore_default': True},
                priority=1,
                scheduled_time=datetime.now(),
                duration_minutes=1,
                expected_impact=0
            )
            
            self._execute_command(deactivate_command)
    
    def _check_device_health(self):
        """Check health status of all devices."""
        current_time = datetime.now()
        
        for device in self.devices.values():
            # Check if device hasn't communicated recently
            time_since_last_comm = current_time - device.last_communication
            if time_since_last_comm > timedelta(hours=1):
                if device.status == 'online':
                    device.status = 'offline'
                    print(f"Device {device.device_id} marked as offline")
    
    def _save_intervention_to_db(self, intervention_id: str, intervention_type: InterventionType,
                               location: str, sensor_data: Dict[str, Any], 
                               commands: List[InterventionCommand]):
        """Save intervention to database."""
        intervention_data = {
            'intervention_id': intervention_id,
            'type': intervention_type.value,
            'location_name': location,
            'description': f"Real-time {intervention_type.value.replace('_', ' ')} intervention",
            'status': InterventionStatus.PLANNED.value,
            'priority_level': commands[0].priority if commands else 1,
            'estimated_cost': len(commands) * 50.0,  # Estimate $50 per device command
            'estimated_impact': sum(cmd.expected_impact for cmd in commands),
            'target_metric': intervention_type.value.split('_')[0],
            'created_by': 'RealTimeSystem'
        }
        
        self.db_manager.save_intervention(intervention_data)
    
    def _update_intervention_status(self, intervention_id: str, status: InterventionStatus):
        """Update intervention status in database."""
        # This would be implemented with proper database update
        print(f"Intervention {intervention_id} status updated to {status.value}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and statistics."""
        online_devices = len([d for d in self.devices.values() if d.status == 'online'])
        total_devices = len(self.devices)
        
        return {
            'system_running': self.system_running,
            'total_devices': total_devices,
            'online_devices': online_devices,
            'offline_devices': total_devices - online_devices,
            'active_interventions': len(self.active_interventions),
            'queued_commands': len(self.command_queue),
            'device_types': list(set(d.device_type for d in self.devices.values())),
            'supported_interventions': [t.value for t in InterventionType],
            'last_check': datetime.now().isoformat()
        }
    
    def get_device_status(self, device_id: str = None) -> Dict[str, Any]:
        """Get status of specific device or all devices."""
        if device_id:
            device = self.devices.get(device_id)
            if device:
                return {
                    'device_id': device.device_id,
                    'type': device.device_type,
                    'location': device.location,
                    'status': device.status,
                    'capabilities': device.capabilities,
                    'last_communication': device.last_communication.isoformat()
                }
            return {'error': f'Device {device_id} not found'}
        
        return {
            'devices': {
                device_id: {
                    'type': device.device_type,
                    'location': device.location,
                    'status': device.status,
                    'last_communication': device.last_communication.isoformat()
                }
                for device_id, device in self.devices.items()
            }
        }