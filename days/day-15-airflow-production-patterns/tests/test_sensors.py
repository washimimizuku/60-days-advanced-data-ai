"""
Unit tests for custom Airflow sensors
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
import sys
import os

# Add the parent directory to the path to import the solution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solution import DataQualitySensor, ExternalAPIHealthSensor, DatabaseConnectionSensor

class TestDataQualitySensor:
    """Test cases for DataQualitySensor"""
    
    def test_sensor_initialization(self):
        """Test sensor can be initialized with required parameters"""
        sensor = DataQualitySensor(
            task_id='test_sensor',
            data_source='test_source',
            quality_checks=[{'type': 'not_null', 'columns': ['id']}],
            min_records=10
        )
        
        assert sensor.data_source == 'test_source'
        assert sensor.min_records == 10
        assert len(sensor.quality_checks) == 1
    
    @patch('solution.DataQualitySensor._check_data_availability')
    @patch('solution.DataQualitySensor._validate_quality')
    def test_poke_success(self, mock_validate, mock_check_data):
        """Test successful poke when data is available and quality is good"""
        mock_check_data.return_value = True
        mock_validate.return_value = True
        
        sensor = DataQualitySensor(
            task_id='test_sensor',
            data_source='test_source',
            quality_checks=[]
        )
        
        context = {'execution_date': datetime.now()}
        result = sensor.poke(context)
        
        assert result is True
        mock_check_data.assert_called_once()
        mock_validate.assert_called_once()
    
    @patch('solution.DataQualitySensor._check_data_availability')
    def test_poke_no_data(self, mock_check_data):
        """Test poke returns False when no data is available"""
        mock_check_data.return_value = False
        
        sensor = DataQualitySensor(
            task_id='test_sensor',
            data_source='test_source',
            quality_checks=[]
        )
        
        context = {'execution_date': datetime.now()}
        result = sensor.poke(context)
        
        assert result is False
        mock_check_data.assert_called_once()
    
    @patch('solution.DataQualitySensor._check_data_availability')
    @patch('solution.DataQualitySensor._validate_quality')
    def test_poke_quality_failure(self, mock_validate, mock_check_data):
        """Test poke returns False when data quality is poor"""
        mock_check_data.return_value = True
        mock_validate.return_value = False
        
        sensor = DataQualitySensor(
            task_id='test_sensor',
            data_source='test_source',
            quality_checks=[]
        )
        
        context = {'execution_date': datetime.now()}
        result = sensor.poke(context)
        
        assert result is False
        mock_check_data.assert_called_once()
        mock_validate.assert_called_once()

class TestExternalAPIHealthSensor:
    """Test cases for ExternalAPIHealthSensor"""
    
    def test_sensor_initialization(self):
        """Test sensor initialization with API URL"""
        sensor = ExternalAPIHealthSensor(
            task_id='test_api_sensor',
            api_url='https://api.example.com',
            expected_status=200
        )
        
        assert sensor.api_url == 'https://api.example.com'
        assert sensor.expected_status == 200
    
    @patch('requests.get')
    def test_poke_healthy_api(self, mock_get):
        """Test poke returns True when API is healthy"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'status': 'healthy'}
        mock_get.return_value = mock_response
        
        sensor = ExternalAPIHealthSensor(
            task_id='test_api_sensor',
            api_url='https://api.example.com'
        )
        
        context = {'execution_date': datetime.now()}
        result = sensor.poke(context)
        
        assert result is True
        mock_get.assert_called_once_with('https://api.example.com/health', timeout=10)
    
    @patch('requests.get')
    def test_poke_unhealthy_api(self, mock_get):
        """Test poke returns False when API is unhealthy"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'status': 'degraded'}
        mock_get.return_value = mock_response
        
        sensor = ExternalAPIHealthSensor(
            task_id='test_api_sensor',
            api_url='https://api.example.com'
        )
        
        context = {'execution_date': datetime.now()}
        result = sensor.poke(context)
        
        assert result is False
    
    @patch('requests.get')
    def test_poke_api_error(self, mock_get):
        """Test poke returns False when API request fails"""
        mock_get.side_effect = Exception('Connection error')
        
        sensor = ExternalAPIHealthSensor(
            task_id='test_api_sensor',
            api_url='https://api.example.com'
        )
        
        context = {'execution_date': datetime.now()}
        result = sensor.poke(context)
        
        assert result is False

class TestDatabaseConnectionSensor:
    """Test cases for DatabaseConnectionSensor"""
    
    def test_sensor_initialization(self):
        """Test sensor initialization with connection parameters"""
        sensor = DatabaseConnectionSensor(
            task_id='test_db_sensor',
            connection_id='postgres_default',
            max_response_time=5
        )
        
        assert sensor.connection_id == 'postgres_default'
        assert sensor.max_response_time == 5
    
    @patch('solution.PostgresHook')
    @patch('solution.datetime')
    def test_poke_healthy_database(self, mock_datetime, mock_hook_class):
        """Test poke returns True when database is healthy and fast"""
        # Mock datetime to control response time calculation
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = datetime(2024, 1, 1, 12, 0, 2)  # 2 seconds later
        mock_datetime.now.side_effect = [start_time, end_time]
        
        # Mock database hook
        mock_hook = Mock()
        mock_hook.get_first.return_value = [1]
        mock_hook_class.return_value = mock_hook
        
        sensor = DatabaseConnectionSensor(
            task_id='test_db_sensor',
            connection_id='postgres_default',
            max_response_time=5
        )
        
        context = {'execution_date': datetime.now()}
        result = sensor.poke(context)
        
        assert result is True
        mock_hook.get_first.assert_called_once_with("SELECT 1")
    
    @patch('solution.PostgresHook')
    @patch('solution.datetime')
    def test_poke_slow_database(self, mock_datetime, mock_hook_class):
        """Test poke returns False when database is too slow"""
        # Mock datetime to simulate slow response
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = datetime(2024, 1, 1, 12, 0, 10)  # 10 seconds later
        mock_datetime.now.side_effect = [start_time, end_time]
        
        # Mock database hook
        mock_hook = Mock()
        mock_hook.get_first.return_value = [1]
        mock_hook_class.return_value = mock_hook
        
        sensor = DatabaseConnectionSensor(
            task_id='test_db_sensor',
            connection_id='postgres_default',
            max_response_time=5
        )
        
        context = {'execution_date': datetime.now()}
        result = sensor.poke(context)
        
        assert result is False
    
    @patch('solution.PostgresHook')
    def test_poke_database_error(self, mock_hook_class):
        """Test poke returns False when database connection fails"""
        mock_hook_class.side_effect = Exception('Connection failed')
        
        sensor = DatabaseConnectionSensor(
            task_id='test_db_sensor',
            connection_id='postgres_default'
        )
        
        context = {'execution_date': datetime.now()}
        result = sensor.poke(context)
        
        assert result is False

if __name__ == '__main__':
    pytest.main([__file__])