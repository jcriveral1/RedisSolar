Lesson 3: Time-Series Data with Redis TimeSeries
Lesson Overview
In this lesson, we'll implement time-series data storage for meter readings using Redis TimeSeries module. We'll learn how to efficiently store and query time-based data, implement analytics features, and handle historical data for our solar monitoring application.
What You'll Learn

Redis TimeSeries module for high-performance time-series data
Time-based data modeling and storage patterns
Time-range queries and data aggregation
Analytics calculations over time periods
Performance optimization with batch operations


Lesson 3 Objectives
By the end of this lesson, you'll have:

✅ Enhanced models to support time-series data
✅ Implemented meter reading storage with Redis TimeSeries
✅ Built time-range queries for historical data
✅ Created analytics calculations and aggregations
✅ Added performance optimization with batch operations


Step 1: Understanding Redis TimeSeries
What is Redis TimeSeries?
Redis TimeSeries is a Redis module that provides native time-series data structures and operations. It's perfect for storing meter readings with timestamps.
Key Features

High Performance: Optimized for time-based data
Compression: Efficient storage with automatic compression
Retention Policies: Automatic data expiration
Aggregation: Built-in downsampling and aggregation functions
Labels: Metadata for organizing time series

Basic TimeSeries Commands
bash# Create a time series with retention policy
TS.CREATE meter:site:1:wh_generated RETENTION 86400000 LABELS site_id 1 metric wh_generated

# Add data points (timestamp in milliseconds, value)
TS.ADD meter:site:1:wh_generated 1642694400000 150.5
TS.ADD meter:site:1:wh_generated 1642697400000 175.3

# Query data range
TS.RANGE meter:site:1:wh_generated 1642694400000 1642780800000

# Get latest value
TS.GET meter:site:1:wh_generated

# Aggregated queries
TS.RANGE meter:site:1:wh_generated 1642694400000 1642780800000 AGGREGATION avg 3600000

Step 2: Enhanced Models for Time-Series Data
Update redisolar/models/site.py
python"""Domain models for solar sites and related data."""

from dataclasses import dataclass
from typing import Optional, Union, List, Dict
from datetime import datetime, timedelta
from enum import Enum


@dataclass(frozen=True, eq=True)
class Coordinate:
    """Geographic coordinate (latitude, longitude)."""
    
    lat: float
    lng: float
    
    def __post_init__(self):
        """Validate coordinate values."""
        if not -90 <= self.lat <= 90:
            raise ValueError(f"Invalid latitude: {self.lat}")
        if not -180 <= self.lng <= 180:
            raise ValueError(f"Invalid longitude: {self.lng}")


@dataclass(frozen=True, eq=True)
class Site:
    """A solar power installation site."""
    
    id: int
    capacity: float  # kW capacity
    panels: int
    address: str
    city: str
    state: str
    postal_code: str
    coordinate: Optional[Coordinate] = None
    
    def __post_init__(self):
        """Validate site data."""
        if self.capacity <= 0:
            raise ValueError("Capacity must be positive")
        if self.panels <= 0:
            raise ValueError("Panels must be positive")
        if not self.address.strip():
            raise ValueError("Address cannot be empty")


@dataclass(frozen=True, eq=True)
class MeterReading:
    """Individual meter reading from a solar site."""
    
    site_id: int
    timestamp: datetime
    wh_generated: float  # Watt-hours generated
    wh_used: float       # Watt-hours used
    temp_c: float        # Temperature in Celsius
    
    def __post_init__(self):
        """Validate meter reading data."""
        if self.wh_generated < 0:
            raise ValueError("Generated energy cannot be negative")
        if self.wh_used < 0:
            raise ValueError("Used energy cannot be negative")
    
    @property
    def net_generation(self) -> float:
        """Calculate net generation (generated - used)."""
        return self.wh_generated - self.wh_used
    
    @property
    def timestamp_millis(self) -> int:
        """Get timestamp in milliseconds for Redis TimeSeries."""
        return int(self.timestamp.timestamp() * 1000)


class MetricType(Enum):
    """Types of metrics we track in time series."""
    WH_GENERATED = "wh_generated"
    WH_USED = "wh_used"
    TEMP_C = "temp_c"
    NET_GENERATION = "net_generation"


@dataclass(frozen=True)
class TimeSeriesData:
    """Time series data point."""
    timestamp: datetime
    value: float
    
    @property
    def timestamp_millis(self) -> int:
        """Get timestamp in milliseconds."""
        return int(self.timestamp.timestamp() * 1000)
    
    def __str__(self) -> str:
        return f"TimeSeriesData(timestamp={self.timestamp.isoformat()}, value={self.value})"

Step 3: Time-Series DAO Implementation
Create redisolar/dao/timeseries_dao.py
python"""Time-Series Data Access Object for meter readings."""

import redis
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import json
import logging

from .redis_dao import get_redis
from .key_manager import key_manager
from redisolar.models.site import MeterReading, TimeSeriesData, MetricType

logger = logging.getLogger(__name__)


class TimeSeriesDao:
    """Redis TimeSeries implementation for meter readings."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """Initialize with Redis client."""
        self.redis = redis_client or get_redis()
        self.key_manager = key_manager
        
        # Default retention: 30 days in milliseconds
        self.default_retention = 30 * 24 * 60 * 60 * 1000
    
    def _get_ts_key(self, site_id: int, metric: MetricType) -> str:
        """Generate TimeSeries key for site and metric."""
        return f"{self.key_manager.key_prefix}ts:site:{site_id}:{metric.value}"
    
    def _ensure_timeseries_exists(self, site_id: int, metric: MetricType) -> None:
        """Ensure TimeSeries exists for site and metric."""
        ts_key = self._get_ts_key(site_id, metric)
        
        try:
            # Check if TimeSeries exists
            self.redis.execute_command('TS.INFO', ts_key)
            logger.debug(f"TimeSeries {ts_key} already exists")
        except redis.ResponseError as e:
            if "TSDB: the key does not exist" in str(e):
                # Create TimeSeries if it doesn't exist
                labels = {
                    'site_id': str(site_id),
                    'metric': metric.value,
                    'type': 'meter_reading'
                }
                
                # Build CREATE command
                cmd = ['TS.CREATE', ts_key, 'RETENTION', str(self.default_retention)]
                cmd.extend(['LABELS'])
                for key, value in labels.items():
                    cmd.extend([key, value])
                
                self.redis.execute_command(*cmd)
                logger.info(f"Created TimeSeries: {ts_key}")
            else:
                raise e
    
    def add_reading(self, reading: MeterReading) -> None:
        """Add a meter reading to time series."""
        logger.debug(f"Adding reading for site {reading.site_id} at {reading.timestamp}")
        
        # Add all metrics for this reading
        metrics = [
            (MetricType.WH_GENERATED, reading.wh_generated),
            (MetricType.WH_USED, reading.wh_used),
            (MetricType.TEMP_C, reading.temp_c),
            (MetricType.NET_GENERATION, reading.net_generation)
        ]
        
        # Use pipeline for atomic operations
        pipe = self.redis.pipeline()
        
        try:
            for metric_type, value in metrics:
                # Ensure TimeSeries exists
                self._ensure_timeseries_exists(reading.site_id, metric_type)
                
                # Add data point
                ts_key = self._get_ts_key(reading.site_id, metric_type)
                pipe.execute_command('TS.ADD', ts_key, reading.timestamp_millis, value)
            
            # Execute pipeline
            pipe.execute()
            logger.debug(f"Successfully added reading for site {reading.site_id}")
            
        except Exception as e:
            pipe.discard()
            logger.error(f"Failed to add reading for site {reading.site_id}: {e}")
            raise RuntimeError(f"Failed to add reading: {e}")
    
    def add_readings_batch(self, readings: List[MeterReading]) -> None:
        """Add multiple readings efficiently using batch operations."""
        if not readings:
            logger.debug("No readings to add")
            return
        
        logger.info(f"Adding batch of {len(readings)} readings")
        
        # Group readings by site for efficiency
        readings_by_site = {}
        for reading in readings:
            if reading.site_id not in readings_by_site:
                readings_by_site[reading.site_id] = []
            readings_by_site[reading.site_id].append(reading)
        
        # Use pipeline for all operations
        pipe = self.redis.pipeline()
        
        try:
            for site_id, site_readings in readings_by_site.items():
                # Ensure all TimeSeries exist for this site
                for metric in MetricType:
                    self._ensure_timeseries_exists(site_id, metric)
                
                # Add all readings for this site
                for reading in site_readings:
                    metrics = [
                        (MetricType.WH_GENERATED, reading.wh_generated),
                        (MetricType.WH_USED, reading.wh_used),
                        (MetricType.TEMP_C, reading.temp_c),
                        (MetricType.NET_GENERATION, reading.net_generation)
                    ]
                    
                    for metric_type, value in metrics:
                        ts_key = self._get_ts_key(reading.site_id, metric_type)
                        pipe.execute_command('TS.ADD', ts_key, reading.timestamp_millis, value)
            
            # Execute all operations
            pipe.execute()
            logger.info(f"Successfully added batch of {len(readings)} readings")
            
        except Exception as e:
            pipe.discard()
            logger.error(f"Failed to add readings batch: {e}")
            raise RuntimeError(f"Failed to add readings: {e}")
    
    def get_readings_range(
        self, 
        site_id: int, 
        metric: MetricType, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[TimeSeriesData]:
        """Get readings for a site and metric within time range."""
        ts_key = self._get_ts_key(site_id, metric)
        
        start_millis = int(start_time.timestamp() * 1000)
        end_millis = int(end_time.timestamp() * 1000)
        
        logger.debug(f"Getting readings for {ts_key} from {start_time} to {end_time}")
        
        try:
            result = self.redis.execute_command('TS.RANGE', ts_key, start_millis, end_millis)
            
            data_points = []
            for timestamp_millis, value in result:
                timestamp = datetime.fromtimestamp(timestamp_millis / 1000)
                data_points.append(TimeSeriesData(timestamp=timestamp, value=float(value)))
            
            logger.debug(f"Retrieved {len(data_points)} data points")
            return data_points
        
        except redis.ResponseError as e:
            if "TSDB: the key does not exist" in str(e):
                logger.debug(f"TimeSeries {ts_key} does not exist")
                return []
            else:
                logger.error(f"Error retrieving readings: {e}")
                raise e
    
    def get_latest_reading(self, site_id: int, metric: MetricType) -> Optional[TimeSeriesData]:
        """Get the latest reading for a site and metric."""
        ts_key = self._get_ts_key(site_id, metric)
        
        try:
            result = self.redis.execute_command('TS.GET', ts_key)
            if result:
                timestamp_millis, value = result
                timestamp = datetime.fromtimestamp(timestamp_millis / 1000)
                return TimeSeriesData(timestamp=timestamp, value=float(value))
        except redis.ResponseError as e:
            if "TSDB: the key does not exist" in str(e):
                logger.debug(f"TimeSeries {ts_key} does not exist")
            else:
                logger.error(f"Error getting latest reading: {e}")
        
        return None
    
    def get_aggregated_data(
        self,
        site_id: int,
        metric: MetricType,
        start_time: datetime,
        end_time: datetime,
        bucket_size_ms: int = 3600000,  # 1 hour
        aggregation: str = 'avg'
    ) -> List[TimeSeriesData]:
        """Get aggregated data for a time range."""
        ts_key = self._get_ts_key(site_id, metric)
        
        start_millis = int(start_time.timestamp() * 1000)
        end_millis = int(end_time.timestamp() * 1000)
        
        logger.debug(f"Getting aggregated data for {ts_key} with {aggregation} over {bucket_size_ms}ms buckets")
        
        try:
            result = self.redis.execute_command(
                'TS.RANGE', ts_key, start_millis, end_millis,
                'AGGREGATION', aggregation, bucket_size_ms
            )
            
            data_points = []
            for timestamp_millis, value in result:
                timestamp = datetime.fromtimestamp(timestamp_millis / 1000)
                data_points.append(TimeSeriesData(timestamp=timestamp, value=float(value)))
            
            logger.debug(f"Retrieved {len(data_points)} aggregated data points")
            return data_points
        
        except redis.ResponseError as e:
            if "TSDB: the key does not exist" in str(e):
                logger.debug(f"TimeSeries {ts_key} does not exist")
                return []
            else:
                logger.error(f"Error getting aggregated data: {e}")
                raise e
    
    def delete_site_data(self, site_id: int) -> None:
        """Delete all time series data for a site."""
        logger.info(f"Deleting all time series data for site {site_id}")
        
        pipe = self.redis.pipeline()
        
        for metric in MetricType:
            ts_key = self._get_ts_key(site_id, metric)
            try:
                pipe.delete(ts_key)
            except Exception as e:
                logger.warning(f"Error deleting TimeSeries {ts_key}: {e}")
        
        pipe.execute()
        logger.info(f"Deleted time series data for site {site_id}")

Step 4: Enhanced Schema Validation
Update redisolar/models/schemas.py
python"""Marshmallow schemas for data validation and serialization."""

from marshmallow import Schema, fields, post_load, validate, ValidationError
from datetime import datetime
from typing import Dict, Any

from .site import Site, MeterReading, Coordinate, TimeSeriesData


class CoordinateSchema(Schema):
    """Schema for Coordinate model."""
    
    lat = fields.Float(required=True, validate=validate.Range(-90, 90))
    lng = fields.Float(required=True, validate=validate.Range(-180, 180))
    
    @post_load
    def make_coordinate(self, data: Dict[str, Any], **kwargs) -> Coordinate:
        return Coordinate(**data)


class SiteSchema(Schema):
    """Schema for Site model."""
    
    id = fields.Integer(required=True, validate=validate.Range(min=1))
    capacity = fields.Float(required=True, validate=validate.Range(min=0.1))
    panels = fields.Integer(required=True, validate=validate.Range(min=1))
    address = fields.String(required=True, validate=validate.Length(min=1, max=200))
    city = fields.String(required=True, validate=validate.Length(min=1, max=100))
    state = fields.String(required=True, validate=validate.Length(min=2, max=50))
    postal_code = fields.String(required=True, validate=validate.Length(min=3, max=20))
    coordinate = fields.Nested(CoordinateSchema, allow_none=True)
    
    @post_load
    def make_site(self, data: Dict[str, Any], **kwargs) -> Site:
        return Site(**data)


class MeterReadingSchema(Schema):
    """Schema for MeterReading model."""
    
    site_id = fields.Integer(required=True, validate=validate.Range(min=1))
    timestamp = fields.DateTime(required=True)
    wh_generated = fields.Float(required=True, validate=validate.Range(min=0))
    wh_used = fields.Float(required=True, validate=validate.Range(min=0))
    temp_c = fields.Float(required=True)
    
    @post_load
    def make_reading(self, data: Dict[str, Any], **kwargs) -> MeterReading:
        return MeterReading(**data)


class TimeSeriesDataSchema(Schema):
    """Schema for TimeSeriesData model."""
    
    timestamp = fields.DateTime(required=True)
    value = fields.Float(required=True)
    
    @post_load
    def make_data(self, data: Dict[str, Any], **kwargs) -> TimeSeriesData:
        return TimeSeriesData(**data)


# Schema instances for reuse
site_schema = SiteSchema()
meter_reading_schema = MeterReadingSchema()
timeseries_data_schema = TimeSeriesDataSchema()
coordinate_schema = CoordinateSchema()

Step 5: API Routes for Time-Series Operations
Create redisolar/api/timeseries_routes.py
python"""API routes for time-series data operations."""

from flask import Blueprint, request, jsonify
from marshmallow import ValidationError
from datetime import datetime, timedelta
from typing import Dict, Any
import logging

from redisolar.dao.timeseries_dao import TimeSeriesDao
from redisolar.dao.site_dao_redis import SiteDaoRedis
from redisolar.models.schemas import meter_reading_schema, timeseries_data_schema
from redisolar.models.site import MetricType

logger = logging.getLogger(__name__)

# Create blueprint
timeseries_bp = Blueprint('timeseries', __name__, url_prefix='/api/timeseries')

# Initialize DAOs
timeseries_dao = TimeSeriesDao()
site_dao = SiteDaoRedis()


@timeseries_bp.route('/readings/<int:site_id>', methods=['POST'])
def add_meter_reading(site_id: int):
    """Add a meter reading for a site."""
    try:
        # Validate request data
        reading_data = request.get_json()
        if not reading_data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Ensure site_id matches URL
        reading_data['site_id'] = site_id
        
        # Validate and deserialize
        reading = meter_reading_schema.load(reading_data)
        
        # Check if site exists
        if not site_dao.exists(site_id):
            return jsonify({'error': f'Site {site_id} not found'}), 404
        
        # Add reading to time series
        timeseries_dao.add_reading(reading)
        
        logger.info(f"Added reading for site {site_id} at {reading.timestamp}")
        
        return jsonify({
            'message': f'Reading added for site {site_id}',
            'reading': meter_reading_schema.dump(reading)
        }), 201
    
    except ValidationError as e:
        logger.warning(f"Validation error adding reading for site {site_id}: {e.messages}")
        return jsonify({'error': 'Validation failed', 'details': e.messages}), 400
    
    except Exception as e:
        logger.error(f"Error adding reading for site {site_id}: {e}")
        return jsonify({'error': f'Failed to add reading: {str(e)}'}), 500


@timeseries_bp.route('/readings/<int:site_id>/batch', methods=['POST'])
def add_meter_readings_batch(site_id: int):
    """Add multiple meter readings for a site using batch operations."""
    try:
        # Validate request data
        readings_data = request.get_json()
        if not readings_data or not isinstance(readings_data, list):
            return jsonify({'error': 'Expected array of readings'}), 400
        
        # Check if site exists
        if not site_dao.exists(site_id):
            return jsonify({'error': f'Site {site_id} not found'}), 404
        
        # Validate and deserialize all readings
        readings = []
        for i, reading_data in enumerate(readings_data):
            reading_data['site_id'] = site_id
            try:
                reading = meter_reading_schema.load(reading_data)
                readings.append(reading)
            except ValidationError as e:
                logger.warning(f"Validation error for reading {i}: {e.messages}")
                return jsonify({
                    'error': f'Validation failed for reading {i}',
                    'details': e.messages
                }), 400
        
        # Add all readings using batch operation
        timeseries_dao.add_readings_batch(readings)
        
        logger.info(f"Added batch of {len(readings)} readings for site {site_id}")
        
        return jsonify({
            'message': f'Added {len(readings)} readings for site {site_id}',
            'count': len(readings)
        }), 201
    
    except Exception as e:
        logger.error(f"Error adding batch readings for site {site_id}: {e}")
        return jsonify({'error': f'Failed to add readings: {str(e)}'}), 500


@timeseries_bp.route('/readings/<int:site_id>/<metric>', methods=['GET'])
def get_site_readings(site_id: int, metric: str):
    """Get time-series readings for a site and metric with time-range queries."""
    try:
        # Validate metric type
        try:
            metric_type = MetricType(metric)
        except ValueError:
            valid_metrics = [m.value for m in MetricType]
            return jsonify({
                'error': f'Invalid metric: {metric}',
                'valid_metrics': valid_metrics
            }), 400
        
        # Check if site exists
        if not site_dao.exists(site_id):
            return jsonify({'error': f'Site {site_id} not found'}), 404
        
        # Get time range from query parameters
        start_str = request.args.get('start')
        end_str = request.args.get('end')
        
        if start_str and end_str:
            try:
                start_time = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
                end_time = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
            except ValueError:
                return jsonify({
                    'error': 'Invalid date format. Use ISO format (e.g., 2024-01-01T12:00:00)'
                }), 400
        else:
            # Default to last 24 hours
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=24)
        
        # Validate time range
        if start_time >= end_time:
            return jsonify({'error': 'Start time must be before end time'}), 400
        
        # Get aggregation parameters
        aggregation = request.args.get('aggregation', 'raw')
        
        if aggregation == 'raw':
            # Get raw data
            data_points = timeseries_dao.get_readings_range(site_id, metric_type, start_time, end_time)
        else:
            # Get aggregated data
            bucket_size_str = request.args.get('bucket_size', '3600000')  # Default 1 hour
            
            # Handle common bucket size names
            bucket_size_mapping = {
                'minute': 60 * 1000,
                'hour': 60 * 60 * 1000,
                'day': 24 * 60 * 60 * 1000
            }
            
            if bucket_size_str in bucket_size_mapping:
                bucket_size_ms = bucket_size_mapping[bucket_size_str]
            else:
                try:
                    bucket_size_ms = int(bucket_size_str)
                except ValueError:
                    return jsonify({
                        'error': 'Invalid bucket_size. Use number in milliseconds or: minute, hour, day'
                    }), 400
            
            valid_aggregations = ['avg', 'sum', 'min', 'max', 'count']
            if aggregation not in valid_aggregations:
                return jsonify({
                    'error': f'Invalid aggregation: {aggregation}',
                    'valid_aggregations': valid_aggregations
                }), 400
            
            data_points = timeseries_dao.get_aggregated_data(
                site_id, metric_type, start_time, end_time, bucket_size_ms, aggregation
            )
        
        # Serialize data
        data = [timeseries_data_schema.dump(dp) for dp in data_points]
        
        logger.debug(f"Retrieved {len(data)} data points for site {site_id}, metric {metric}")
        
        return jsonify({
            'site_id': site_id,
            'metric': metric,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'aggregation': aggregation,
            'count': len(data),
            'data': data
        })
    
    except Exception as e:
        logger.error(f"Error retrieving readings for site {site_id}, metric {metric}: {e}")
        return jsonify({'error': str(e)}), 500


@timeseries_bp.route('/readings/<int:site_id>/latest', methods=['GET'])
def get_latest_readings(site_id: int):
    """Get latest readings for all metrics of a site."""
    try:
        # Check if site exists
        if not site_dao.exists(site_id):
            return jsonify({'error': f'Site {site_id} not found'}), 404
        
        latest_readings = {}
        
        for metric in MetricType:
            latest_data = timeseries_dao.get_latest_reading(site_id, metric)
            if latest_data:
                latest_readings[metric.value] = timeseries_data_schema.dump(latest_data)
        
        logger.debug(f"Retrieved latest readings for site {site_id}: {len(latest_readings)} metrics")
        
        return jsonify({
            'site_id': site_id,
            'latest_readings': latest_readings,
            'retrieved_at': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error retrieving latest readings for site {site_id}: {e}")
        return jsonify({'error': str(e)}), 500


@timeseries_bp.route('/metrics', methods=['GET'])
def get_available_metrics():
    """Get list of available metrics."""
    return jsonify({
        'metrics': [
            {
                'name': metric.value,
                'description': {
                    'wh_generated': 'Watt-hours generated by solar panels',
                    'wh_used': 'Watt-hours consumed/used',
                    'temp_c': 'Temperature in Celsius',
                    'net_generation': 'Net generation (generated - used)'
                }.get(metric.value, 'No description available')
            }
            for metric in MetricType
        ]
    })

Step 6: Update Main Application
Update redisolar/app.py
python"""Main Flask application."""

import os
import logging
from flask import Flask, jsonify
from redisolar.dao.redis_dao import RedisConnection, get_redis

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app(config_name: str = 'dev') -> Flask:
    """Application factory function."""
    app = Flask(__name__, instance_relative_config=True)
    
    # Load configuration
    config_file = f'{config_name}.cfg'
    app.config.from_pyfile(config_file)
    
    # Initialize Redis connection
    from redisolar.config import Config
    redis_config = Config(
        redis_host=app.config.get('REDIS_HOST', 'localhost'),
        redis_port=app.config.get('REDIS_PORT', 6379),
        redis_password=app.config.get('REDIS_PASSWORD'),
        redis_username=app.config.get('REDIS_USERNAME'),
        redis_db=app.config.get('REDIS_DB', 0),
        redis_key_prefix=app.config.get('REDIS_KEY_PREFIX', 'ru102py-app:'),
    )
    
    # Test Redis connection on startup
    RedisConnection.get_connection(redis_config)
    if not RedisConnection.test_connection():
        raise RuntimeError("Could not connect to Redis")
    
    logger.info("Redis connection established successfully")
    
    # Register blueprints
    from redisolar.api.routes import sites_bp
    from redisolar.api.timeseries_routes import timeseries_bp
    app.register_blueprint(sites_bp)
    app.register_blueprint(timeseries_bp)
    
    logger.info("Registered API blueprints")
    
    # Register routes
    @app.route('/')
    def index():
        """Health check endpoint."""
        return jsonify({
            'message': 'RediSolar API is running!',
            'version': '1.0.0',
            'redis_connected': RedisConnection.test_connection(),
            'redis_info': get_redis().info('server')['redis_version'],
            'endpoints': {
                'sites': [
                    'GET /api/sites - Get all sites',
                    'GET /api/sites/<id> - Get specific site',
                    'POST /api/sites - Create new site',
                    'PUT /api/sites/<id> - Update site',
                    'DELETE /api/sites/<id> - Delete site'
                ],
                'timeseries': [
                    '
                    
