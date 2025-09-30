import os
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import hashlib
import jwt
from .database_manager import DatabaseManager

class FeedbackType(Enum):
    ENVIRONMENTAL = "environmental"
    WELLNESS = "wellness"
    SAFETY = "safety"
    INFRASTRUCTURE = "infrastructure"
    SOCIAL = "social"
    GENERAL = "general"

class AlertSeverity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5

@dataclass
class CitizenFeedback:
    user_id: str
    location_name: str
    latitude: float
    longitude: float
    feedback_type: str
    rating: int  # 1-10
    comment: str
    stress_level: int  # 1-10
    environmental_concerns: List[str]
    suggestion: str
    timestamp: datetime
    photos: List[str] = None
    verified: bool = False

@dataclass
class MobileAlert:
    alert_id: str
    user_id: str
    title: str
    message: str
    severity: int
    location: str
    latitude: float
    longitude: float
    timestamp: datetime
    expires_at: datetime
    action_required: bool
    deep_link: str

@dataclass
class UserProfile:
    user_id: str
    device_token: str
    notification_preferences: Dict[str, bool]
    location_sharing_enabled: bool
    stress_tracking_enabled: bool
    preferred_language: str
    accessibility_needs: List[str]
    last_active: datetime

class MobileAppIntegration:
    """Mobile app integration system for citizen engagement and feedback collection."""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.api_base_url = os.getenv('MOBILE_API_BASE_URL', 'https://api.urbanharmony.app')
        self.jwt_secret = os.getenv('JWT_SECRET', 'demo_secret_key')
        self.push_notification_key = os.getenv('PUSH_NOTIFICATION_KEY', 'demo_push_key')
        self.active_users = {}
        self.user_sessions = {}
        
        # Initialize push notification services
        self._initialize_push_services()
    
    def _initialize_push_services(self):
        """Initialize push notification services (Firebase, APNs, etc.)."""
        self.firebase_config = {
            'server_key': os.getenv('FIREBASE_SERVER_KEY', 'demo_firebase_key'),
            'project_id': os.getenv('FIREBASE_PROJECT_ID', 'urban-harmony-demo'),
            'api_url': 'https://fcm.googleapis.com/fcm/send'
        }
        
        print("Mobile app push notification services initialized")
    
    def register_user(self, device_info: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new mobile app user."""
        user_id = str(uuid.uuid4())
        
        user_profile = UserProfile(
            user_id=user_id,
            device_token=device_info.get('device_token', ''),
            notification_preferences={
                'environmental_alerts': True,
                'wellness_tips': True,
                'intervention_updates': True,
                'community_reports': True,
                'emergency_alerts': True
            },
            location_sharing_enabled=device_info.get('location_enabled', False),
            stress_tracking_enabled=device_info.get('stress_tracking', True),
            preferred_language=device_info.get('language', 'en'),
            accessibility_needs=device_info.get('accessibility_needs', []),
            last_active=datetime.now()
        )
        
        # Store user profile in database (simplified)
        try:
            # In a real system, you would save to a users table
            self.active_users[user_id] = user_profile
            
            # Generate JWT token for authentication
            token = self._generate_auth_token(user_id)
            
            return {
                'success': True,
                'user_id': user_id,
                'auth_token': token,
                'profile': asdict(user_profile),
                'api_endpoints': {
                    'submit_feedback': f'{self.api_base_url}/feedback',
                    'get_alerts': f'{self.api_base_url}/alerts',
                    'wellness_data': f'{self.api_base_url}/wellness',
                    'community': f'{self.api_base_url}/community'
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _generate_auth_token(self, user_id: str) -> str:
        """Generate JWT authentication token for user."""
        payload = {
            'user_id': user_id,
            'issued_at': datetime.now().timestamp(),
            'expires_at': (datetime.now() + timedelta(days=30)).timestamp()
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
    
    def verify_auth_token(self, token: str) -> Optional[str]:
        """Verify and decode JWT token to get user_id."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            if payload['expires_at'] > datetime.now().timestamp():
                return payload['user_id']
            return None
        except jwt.InvalidTokenError:
            return None
    
    def submit_citizen_feedback(self, auth_token: str, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process citizen feedback submission from mobile app."""
        user_id = self.verify_auth_token(auth_token)
        if not user_id:
            return {'success': False, 'error': 'Invalid authentication token'}
        
        try:
            # Validate feedback data
            if not self._validate_feedback_data(feedback_data):
                return {'success': False, 'error': 'Invalid feedback data'}
            
            # Create feedback object
            feedback = CitizenFeedback(
                user_id=user_id,
                location_name=feedback_data['location_name'],
                latitude=feedback_data.get('latitude', 0.0),
                longitude=feedback_data.get('longitude', 0.0),
                feedback_type=feedback_data['feedback_type'],
                rating=feedback_data['rating'],
                comment=feedback_data.get('comment', ''),
                stress_level=feedback_data.get('stress_level', 5),
                environmental_concerns=feedback_data.get('environmental_concerns', []),
                suggestion=feedback_data.get('suggestion', ''),
                timestamp=datetime.now(),
                photos=feedback_data.get('photos', [])
            )
            
            # Store feedback in database
            db_feedback_data = {
                'user_id': feedback.user_id,
                'location_name': feedback.location_name,
                'latitude': feedback.latitude,
                'longitude': feedback.longitude,
                'feedback_type': feedback.feedback_type,
                'rating': feedback.rating,
                'comment': feedback.comment,
                'stress_level': feedback.stress_level,
                'environmental_concerns': feedback.environmental_concerns,
                'suggestion': feedback.suggestion
            }
            
            success = self.db_manager.save_citizen_feedback(db_feedback_data)
            
            if success:
                # Process feedback for insights
                insights = self._analyze_feedback(feedback)
                
                # Check if intervention is needed
                if self._should_trigger_intervention(feedback):
                    self._trigger_automatic_intervention(feedback)
                
                # Notify community about significant issues
                if feedback.rating <= 3 or feedback.stress_level >= 8:
                    self._notify_community_moderators(feedback)
                
                # Update user activity
                self._update_user_activity(user_id)
                
                return {
                    'success': True,
                    'feedback_id': str(uuid.uuid4()),
                    'insights': insights,
                    'community_impact': self._calculate_community_impact(feedback),
                    'thank_you_message': 'Thank you for helping improve urban wellness!'
                }
            else:
                return {'success': False, 'error': 'Failed to save feedback'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _validate_feedback_data(self, data: Dict[str, Any]) -> bool:
        """Validate feedback data from mobile app."""
        required_fields = ['location_name', 'feedback_type', 'rating']
        
        for field in required_fields:
            if field not in data:
                return False
        
        # Validate rating range
        if not (1 <= data['rating'] <= 10):
            return False
        
        # Validate feedback type
        valid_types = [t.value for t in FeedbackType]
        if data['feedback_type'] not in valid_types:
            return False
        
        return True
    
    def _analyze_feedback(self, feedback: CitizenFeedback) -> Dict[str, Any]:
        """Analyze citizen feedback for insights and patterns."""
        insights = {
            'sentiment': 'positive' if feedback.rating >= 7 else 'negative' if feedback.rating <= 3 else 'neutral',
            'stress_category': 'high' if feedback.stress_level >= 8 else 'medium' if feedback.stress_level >= 5 else 'low',
            'priority_level': self._calculate_feedback_priority(feedback),
            'environmental_impact': len(feedback.environmental_concerns),
            'suggested_interventions': self._suggest_interventions_from_feedback(feedback)
        }
        
        return insights
    
    def _calculate_feedback_priority(self, feedback: CitizenFeedback) -> int:
        """Calculate priority level based on feedback content."""
        priority = 1
        
        # High stress level increases priority
        if feedback.stress_level >= 8: priority += 3
        elif feedback.stress_level >= 6: priority += 2
        
        # Low rating increases priority
        if feedback.rating <= 2: priority += 3
        elif feedback.rating <= 4: priority += 2
        
        # Environmental concerns increase priority
        if len(feedback.environmental_concerns) >= 3: priority += 2
        elif len(feedback.environmental_concerns) >= 1: priority += 1
        
        # Safety issues get highest priority
        if feedback.feedback_type == FeedbackType.SAFETY.value: priority += 4
        
        return min(10, priority)
    
    def _suggest_interventions_from_feedback(self, feedback: CitizenFeedback) -> List[str]:
        """Suggest interventions based on feedback content."""
        suggestions = []
        
        # Analyze environmental concerns
        for concern in feedback.environmental_concerns:
            if 'air' in concern.lower() or 'pollution' in concern.lower():
                suggestions.append('air_purification')
            elif 'noise' in concern.lower() or 'sound' in concern.lower():
                suggestions.append('noise_control')
            elif 'temperature' in concern.lower() or 'heat' in concern.lower():
                suggestions.append('climate_control')
            elif 'crowd' in concern.lower() or 'crowded' in concern.lower():
                suggestions.append('crowd_management')
            elif 'light' in concern.lower() or 'dark' in concern.lower():
                suggestions.append('lighting_adjustment')
        
        # Analyze feedback type
        if feedback.feedback_type == FeedbackType.INFRASTRUCTURE.value:
            suggestions.append('infrastructure_maintenance')
        elif feedback.feedback_type == FeedbackType.SAFETY.value:
            suggestions.extend(['lighting_adjustment', 'crowd_management'])
        
        return list(set(suggestions))  # Remove duplicates
    
    def _should_trigger_intervention(self, feedback: CitizenFeedback) -> bool:
        """Determine if feedback should trigger an automatic intervention."""
        # High-priority feedback triggers intervention
        if self._calculate_feedback_priority(feedback) >= 8:
            return True
        
        # Multiple similar complaints in the area
        similar_feedback = self._get_similar_recent_feedback(feedback)
        if len(similar_feedback) >= 3:
            return True
        
        # Critical safety issues
        if feedback.feedback_type == FeedbackType.SAFETY.value and feedback.rating <= 2:
            return True
        
        return False
    
    def _get_similar_recent_feedback(self, feedback: CitizenFeedback) -> List[Dict[str, Any]]:
        """Get similar feedback from the same area in recent time."""
        # Get feedback from database for the same location and type in the last 2 hours
        recent_feedback = self.db_manager.get_citizen_feedback(
            location=feedback.location_name,
            hours=2
        )
        
        similar = []
        for fb in recent_feedback:
            if (fb['feedback_type'] == feedback.feedback_type and 
                fb['rating'] <= 4):  # Poor ratings
                similar.append(fb)
        
        return similar
    
    def _trigger_automatic_intervention(self, feedback: CitizenFeedback):
        """Trigger automatic intervention based on citizen feedback."""
        from .real_time_intervention_system import RealTimeInterventionSystem, InterventionType
        
        intervention_system = RealTimeInterventionSystem()
        
        # Map feedback to intervention types
        intervention_mapping = {
            'air_purification': InterventionType.AIR_PURIFICATION,
            'noise_control': InterventionType.NOISE_CONTROL,
            'climate_control': InterventionType.CLIMATE_CONTROL,
            'crowd_management': InterventionType.CROWD_MANAGEMENT,
            'lighting_adjustment': InterventionType.LIGHTING_ADJUSTMENT
        }
        
        suggested_interventions = self._suggest_interventions_from_feedback(feedback)
        
        for suggestion in suggested_interventions:
            if suggestion in intervention_mapping:
                # Create synthetic sensor data from feedback
                sensor_data = {
                    'air_quality': 80 if 'air' in feedback.environmental_concerns else 50,
                    'noise_level': 75 if 'noise' in feedback.environmental_concerns else 60,
                    'temperature': 28 if 'heat' in str(feedback.environmental_concerns) else 22,
                    'crowd_density': 80 if 'crowd' in str(feedback.environmental_concerns) else 40,
                    'humidity': 60
                }
                
                intervention_id = intervention_system.create_intervention(
                    intervention_mapping[suggestion],
                    feedback.location_name,
                    sensor_data,
                    duration_minutes=90  # Longer duration for citizen-triggered interventions
                )
                
                if intervention_id:
                    print(f"Triggered intervention {intervention_id} based on citizen feedback")
    
    def send_push_notification(self, user_id: str, notification_data: Dict[str, Any]) -> bool:
        """Send push notification to mobile app user."""
        user_profile = self.active_users.get(user_id)
        if not user_profile or not user_profile.device_token:
            return False
        
        # Check notification preferences
        notification_type = notification_data.get('type', 'general')
        if not user_profile.notification_preferences.get(notification_type, True):
            return False
        
        try:
            # Firebase Cloud Messaging format
            payload = {
                'to': user_profile.device_token,
                'notification': {
                    'title': notification_data['title'],
                    'body': notification_data['message'],
                    'icon': 'ic_notification',
                    'sound': 'default'
                },
                'data': {
                    'type': notification_type,
                    'action': notification_data.get('action', 'none'),
                    'deep_link': notification_data.get('deep_link', ''),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            headers = {
                'Authorization': f'key={self.firebase_config["server_key"]}',
                'Content-Type': 'application/json'
            }
            
            # For demo purposes, simulate the push notification
            if self.firebase_config['server_key'] == 'demo_firebase_key':
                print(f"SIMULATED PUSH: {notification_data['title']} to user {user_id}")
                return True
            
            # Real push notification
            response = requests.post(
                self.firebase_config['api_url'],
                headers=headers,
                json=payload,
                timeout=10
            )
            
            return response.status_code == 200
            
        except Exception as e:
            print(f"Failed to send push notification: {e}")
            return False
    
    def create_community_alert(self, alert_data: Dict[str, Any]) -> str:
        """Create community alert for mobile app users."""
        alert_id = f"ALERT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        alert = MobileAlert(
            alert_id=alert_id,
            user_id='system',
            title=alert_data['title'],
            message=alert_data['message'],
            severity=alert_data.get('severity', AlertSeverity.MEDIUM.value),
            location=alert_data['location'],
            latitude=alert_data.get('latitude', 0.0),
            longitude=alert_data.get('longitude', 0.0),
            timestamp=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=alert_data.get('duration_hours', 2)),
            action_required=alert_data.get('action_required', False),
            deep_link=alert_data.get('deep_link', '')
        )
        
        # Store alert in database
        db_alert_data = {
            'alert_id': alert.alert_id,
            'alert_type': 'community_alert',
            'severity_level': alert.severity,
            'message': alert.message,
            'location_name': alert.location,
            'latitude': alert.latitude,
            'longitude': alert.longitude,
            'status': 'active'
        }
        
        self.db_manager.create_alert(db_alert_data)
        
        # Send to relevant users based on location
        self._send_location_based_notifications(alert)
        
        return alert_id
    
    def _send_location_based_notifications(self, alert: MobileAlert):
        """Send alert notifications to users in the relevant area."""
        for user_id, user_profile in self.active_users.items():
            if user_profile.location_sharing_enabled:
                # In a real system, you would check user's current location
                # For demo, send to all users
                notification_data = {
                    'type': 'environmental_alerts',
                    'title': alert.title,
                    'message': alert.message,
                    'action': 'open_alert',
                    'deep_link': f'urbanharmony://alerts/{alert.alert_id}'
                }
                
                self.send_push_notification(user_id, notification_data)
    
    def get_user_wellness_insights(self, auth_token: str) -> Dict[str, Any]:
        """Get personalized wellness insights for a user."""
        user_id = self.verify_auth_token(auth_token)
        if not user_id:
            return {'error': 'Invalid authentication token'}
        
        # Get user's recent feedback for analysis
        recent_feedback = self.db_manager.get_citizen_feedback(hours=168)  # 1 week
        user_feedback = [fb for fb in recent_feedback if fb['user_id'] == user_id]
        
        if not user_feedback:
            return {
                'wellness_score': 7.0,
                'message': 'Welcome! Start sharing feedback to get personalized insights.',
                'recommendations': []
            }
        
        # Calculate wellness metrics
        avg_stress = sum(fb['stress_level'] for fb in user_feedback) / len(user_feedback)
        avg_rating = sum(fb['rating'] for fb in user_feedback) / len(user_feedback)
        
        wellness_score = (10 - avg_stress + avg_rating) / 2
        
        # Generate recommendations
        recommendations = []
        if avg_stress > 6:
            recommendations.append("Consider visiting quieter areas like parks during high-stress times")
        if avg_rating < 5:
            recommendations.append("You've reported several concerns. We're working on improvements!")
        
        # Environmental pattern analysis
        common_concerns = {}
        for fb in user_feedback:
            for concern in fb.get('environmental_concerns', []):
                common_concerns[concern] = common_concerns.get(concern, 0) + 1
        
        if common_concerns:
            top_concern = max(common_concerns, key=common_concerns.get)
            recommendations.append(f"Your top concern is {top_concern}. Try areas with better conditions.")
        
        return {
            'wellness_score': round(wellness_score, 1),
            'stress_trend': 'improving' if len(user_feedback) > 1 and user_feedback[-1]['stress_level'] < avg_stress else 'stable',
            'total_contributions': len(user_feedback),
            'community_impact': len(user_feedback) * 0.1,  # Simple impact score
            'recommendations': recommendations,
            'environmental_patterns': common_concerns
        }
    
    def get_community_leaderboard(self) -> Dict[str, Any]:
        """Get community leaderboard for gamification."""
        # Get all recent feedback
        recent_feedback = self.db_manager.get_citizen_feedback(hours=168)  # 1 week
        
        # Calculate user contributions
        user_contributions = {}
        for fb in recent_feedback:
            user_id = fb['user_id']
            user_contributions[user_id] = user_contributions.get(user_id, 0) + 1
        
        # Sort users by contributions
        sorted_users = sorted(user_contributions.items(), key=lambda x: x[1], reverse=True)
        
        leaderboard = []
        for i, (user_id, count) in enumerate(sorted_users[:10]):
            # Anonymize user IDs for privacy
            anonymous_id = hashlib.md5(user_id.encode()).hexdigest()[:8]
            leaderboard.append({
                'rank': i + 1,
                'user_id': f"User-{anonymous_id}",
                'contributions': count,
                'impact_score': count * 0.5
            })
        
        return {
            'leaderboard': leaderboard,
            'total_contributors': len(user_contributions),
            'total_feedback': len(recent_feedback),
            'week_range': f"{(datetime.now() - timedelta(weeks=1)).strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}"
        }
    
    def _notify_community_moderators(self, feedback: CitizenFeedback):
        """Notify community moderators about significant issues."""
        if self._calculate_feedback_priority(feedback) >= 7:
            # Create notification for moderators
            notification_data = {
                'type': 'moderation_alert',
                'title': 'High Priority Citizen Feedback',
                'message': f'Urgent feedback received from {feedback.location_name}',
                'action': 'review_feedback',
                'deep_link': f'urbanharmony://moderation/feedback/{feedback.user_id}'
            }
            
            # Send to system administrators (in real system, would have moderator user IDs)
            print(f"MODERATION ALERT: High priority feedback from {feedback.location_name}")
    
    def _update_user_activity(self, user_id: str):
        """Update user's last activity timestamp."""
        if user_id in self.active_users:
            self.active_users[user_id].last_active = datetime.now()
    
    def _calculate_community_impact(self, feedback: CitizenFeedback) -> float:
        """Calculate the potential community impact of feedback."""
        base_impact = 1.0
        
        # Higher impact for critical feedback
        if feedback.rating <= 2: base_impact *= 2.0
        if feedback.stress_level >= 8: base_impact *= 1.5
        
        # More impact for actionable feedback with suggestions
        if feedback.suggestion: base_impact *= 1.3
        
        # Environmental concerns increase impact
        base_impact *= (1 + len(feedback.environmental_concerns) * 0.2)
        
        return round(base_impact, 2)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics for admin dashboard."""
        return {
            'active_users': len(self.active_users),
            'total_feedback_today': len(self.db_manager.get_citizen_feedback(hours=24)),
            'average_wellness_score': 7.2,  # Would calculate from actual data
            'push_notifications_sent_today': 0,  # Would track actual notifications
            'community_alerts_active': len(self.db_manager.get_active_alerts()),
            'top_feedback_locations': ['Times Square', 'Central Park', 'Brooklyn Bridge'],
            'system_health': 'operational'
        }