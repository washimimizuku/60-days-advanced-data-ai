#!/usr/bin/env python3
"""
Real-time Streaming Anomaly Detector
Processes streaming data from Kafka and detects anomalies in real-time
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List, Any
import time

from kafka import KafkaConsumer, KafkaProducer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import redis
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamingAnomalyDetector:
    """Real-time anomaly detection for streaming data"""
    
    def __init__(self):
        self.setup_connections()
        self.setup_models()
        self.data_buffer = []
        self.window_size = int(os.getenv('STREAMING_WINDOW_SIZE', 1000))
        self.update_frequency = int(os.getenv('MODEL_UPDATE_FREQUENCY', 100))
        self.update_counter = 0
        self.anomaly_threshold = float(os.getenv('ANOMALY_THRESHOLD', 0.95))
        
    def setup_connections(self):
        """Setup streaming and storage connections"""
        # Kafka Consumer
        self.consumer = KafkaConsumer(
            'sensor-data',
            bootstrap_servers=os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092').split(','),
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=True
        )
        
        # Kafka Producer for alerts
        self.producer = KafkaProducer(
            bootstrap_servers=os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092').split(','),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        # Redis for caching
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=int(os.getenv('REDIS_DB', 0)),
            decode_responses=True
        )
        
        # InfluxDB for storing anomalies
        self.influx_client = InfluxDBClient(
            url=f"http://{os.getenv('INFLUXDB_HOST', 'localhost')}:8086",
            token=os.getenv('INFLUXDB_TOKEN', 'anomaly-token-12345'),
            org=os.getenv('INFLUXDB_ORG', 'anomaly-org')
        )
        self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
        
    def setup_models(self):
        """Initialize anomaly detection models"""
        self.models = {
            'isolation_forest': IsolationForest(
                contamination=float(os.getenv('DEFAULT_CONTAMINATION', 0.1)),
                random_state=42,
                n_estimators=50  # Smaller for real-time performance
            ),
            'scaler': StandardScaler()
        }
        self.is_model_trained = False
        
    def extract_features(self, data_point: Dict[str, Any]) -> np.ndarray:
        """Extract numeric features from data point"""
        features = []
        
        # Extract numeric values
        for key, value in data_point.items():
            if isinstance(value, (int, float)) and key != 'is_anomaly':
                features.append(float(value))
        
        return np.array(features).reshape(1, -1)
    
    def update_model(self):
        """Update anomaly detection model with recent data"""
        if len(self.data_buffer) < 100:  # Need minimum data for training
            return
        
        try:
            # Prepare training data from buffer
            feature_matrix = []
            for data_point in self.data_buffer[-self.window_size:]:
                features = self.extract_features(data_point['data'])
                if features.size > 0:
                    feature_matrix.append(features.flatten())
            
            if len(feature_matrix) < 50:
                return
            
            X = np.array(feature_matrix)
            
            # Scale features
            X_scaled = self.models['scaler'].fit_transform(X)
            
            # Train isolation forest
            self.models['isolation_forest'].fit(X_scaled)
            self.is_model_trained = True
            
            logger.info(f"Model updated with {len(X)} samples")
            
            # Cache model statistics
            self.redis_client.set('model:last_update', datetime.now().isoformat())
            self.redis_client.set('model:training_samples', len(X))
            
        except Exception as e:
            logger.error(f"Error updating model: {e}")
    
    def detect_anomaly(self, data_point: Dict[str, Any]) -> Dict[str, Any]:
        """Detect if data point is anomalous"""
        if not self.is_model_trained:
            return {
                'is_anomaly': False,
                'anomaly_score': 0.0,
                'confidence': 0.0,
                'reason': 'Model not trained yet'
            }
        
        try:
            # Extract features
            features = self.extract_features(data_point)
            if features.size == 0:
                return {
                    'is_anomaly': False,
                    'anomaly_score': 0.0,
                    'confidence': 0.0,
                    'reason': 'No numeric features found'
                }
            
            # Scale features
            features_scaled = self.models['scaler'].transform(features)
            
            # Get anomaly score
            anomaly_score = self.models['isolation_forest'].decision_function(features_scaled)[0]
            prediction = self.models['isolation_forest'].predict(features_scaled)[0]
            
            # Normalize score to [0, 1]
            normalized_score = 1 / (1 + np.exp(anomaly_score))  # Sigmoid normalization
            
            is_anomaly = prediction == -1
            confidence = abs(normalized_score - 0.5) * 2  # Distance from decision boundary
            
            return {
                'is_anomaly': bool(is_anomaly),
                'anomaly_score': float(normalized_score),
                'confidence': float(confidence),
                'reason': 'Isolation Forest detection'
            }
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return {
                'is_anomaly': False,
                'anomaly_score': 0.0,
                'confidence': 0.0,
                'reason': f'Detection error: {str(e)}'
            }
    
    def store_anomaly(self, data_point: Dict[str, Any], detection_result: Dict[str, Any]):
        """Store detected anomaly to InfluxDB"""
        try:
            point = Point("anomaly_detections")
            
            # Add detection metadata
            point = point.tag("detector", "streaming")
            point = point.tag("is_anomaly", str(detection_result['is_anomaly']))
            
            # Add data fields
            for key, value in data_point.items():
                if isinstance(value, (int, float)):
                    point = point.field(f"data_{key}", float(value))
            
            # Add detection results
            point = point.field("anomaly_score", detection_result['anomaly_score'])
            point = point.field("confidence", detection_result['confidence'])
            
            # Set timestamp
            timestamp = data_point.get('timestamp', datetime.now().isoformat())
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            point = point.time(timestamp)
            
            self.write_api.write(
                bucket=os.getenv('INFLUXDB_BUCKET', 'anomalies'),
                record=point
            )
            
        except Exception as e:
            logger.error(f"Error storing anomaly: {e}")
    
    def send_alert(self, data_point: Dict[str, Any], detection_result: Dict[str, Any]):
        """Send anomaly alert to Kafka topic"""
        try:
            alert = {
                'timestamp': datetime.now().isoformat(),
                'alert_type': 'anomaly_detected',
                'severity': 'high' if detection_result['confidence'] > 0.8 else 'medium',
                'data_point': data_point,
                'detection_result': detection_result,
                'detector': 'streaming_isolation_forest'
            }
            
            self.producer.send('anomaly-alerts', value=alert)
            logger.info(f"Anomaly alert sent: score={detection_result['anomaly_score']:.3f}")
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
    
    def update_statistics(self, detection_result: Dict[str, Any]):
        """Update real-time statistics in Redis"""
        try:
            # Increment counters
            self.redis_client.incr('stats:total_processed')
            
            if detection_result['is_anomaly']:
                self.redis_client.incr('stats:anomalies_detected')
            
            # Update rates (using sliding window)
            current_time = int(time.time())
            window_key = f"stats:window:{current_time // 60}"  # 1-minute windows
            
            self.redis_client.incr(window_key)
            self.redis_client.expire(window_key, 3600)  # Keep for 1 hour
            
            if detection_result['is_anomaly']:
                anomaly_window_key = f"stats:anomalies:{current_time // 60}"
                self.redis_client.incr(anomaly_window_key)
                self.redis_client.expire(anomaly_window_key, 3600)
            
            # Update last seen timestamp
            self.redis_client.set('stats:last_processed', datetime.now().isoformat())
            
        except Exception as e:
            logger.error(f"Error updating statistics: {e}")
    
    def process_stream(self):
        """Main streaming processing loop"""
        logger.info("Starting streaming anomaly detection...")
        
        try:
            for message in self.consumer:
                try:
                    # Parse message
                    data = message.value
                    
                    # Add to buffer
                    self.data_buffer.append(data)
                    
                    # Maintain buffer size
                    if len(self.data_buffer) > self.window_size:
                        self.data_buffer.pop(0)
                    
                    # Update model periodically
                    self.update_counter += 1
                    if self.update_counter >= self.update_frequency:
                        self.update_model()
                        self.update_counter = 0
                    
                    # Detect anomaly
                    detection_result = self.detect_anomaly(data['data'])
                    
                    # Update statistics
                    self.update_statistics(detection_result)
                    
                    # Store and alert if anomaly detected
                    if detection_result['is_anomaly']:
                        self.store_anomaly(data['data'], detection_result)
                        
                        # Send alert for high-confidence anomalies
                        if detection_result['confidence'] > self.anomaly_threshold:
                            self.send_alert(data, detection_result)
                    
                    # Log progress
                    if self.update_counter % 100 == 0:
                        total_processed = self.redis_client.get('stats:total_processed') or 0
                        anomalies_detected = self.redis_client.get('stats:anomalies_detected') or 0
                        logger.info(f"Processed: {total_processed}, Anomalies: {anomalies_detected}")
                
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    continue
                    
        except KeyboardInterrupt:
            logger.info("Stopping streaming detector...")
        except Exception as e:
            logger.error(f"Streaming error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup connections"""
        try:
            self.consumer.close()
            self.producer.close()
            self.redis_client.close()
            self.influx_client.close()
            logger.info("Connections closed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

def main():
    """Main execution function"""
    detector = StreamingAnomalyDetector()
    
    # Wait for services to be ready
    logger.info("Waiting for services to be ready...")
    time.sleep(30)
    
    # Start processing
    detector.process_stream()

if __name__ == "__main__":
    main()