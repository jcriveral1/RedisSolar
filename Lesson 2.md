## Lesson 2: Data Modeling and Redis Hashes

### Lesson Overview
In this lesson, we'll dive deep into data modeling with Redis by creating our core domain models and implementing our first Data Access Object (DAO). We'll learn how to use Redis Hashes to store structured data and implement CRUD operations for solar sites.

### What You'll Learn
- How to model domain objects using Python dataclasses
- Understanding Redis Hash data structure
- Implementing the DAO pattern with Redis
- Data validation and serialization with Marshmallow
- CRUD operations (Create, Read, Update, Delete)
- Redis key naming conventions and best practices

---

## Lesson 2 Objectives

By the end of this lesson, you'll have:
- ✅ Created domain models using dataclasses
- ✅ Implemented schema validation with Marshmallow
- ✅ Built a complete Site DAO using Redis Hashes
- ✅ Written comprehensive tests for data operations
- ✅ Understood Redis key patterns and expiration
- ✅ Created API endpoints for site management

---

## Step 1: Understanding Redis Hashes

Redis Hashes are perfect for storing objects with multiple fields. They're similar to Python dictionaries and ideal for our solar site data.

### Redis Hash Commands
```bash
# Basic hash operations in redis-cli
HSET site:1 id 1 capacity 4.5 panels 12
HGET site:1 capacity
HGETALL site:1
HDEL site:1 panels
HEXISTS site:1 capacity
```

### Why Use Hashes for Our Data?
- **Structured Storage**: Multiple related fields in one key
- **Memory Efficient**: Redis optimizes small hashes
- **Atomic Operations**: All fields updated together
- **Selective Retrieval**: Get specific fields without loading entire object

---

## Step 2: Domain Models

### Create redisolar/models/site.py
```python
"""Domain models for solar sites and related data."""

from dataclasses import dataclass
from typing import Optional, Union
from datetime import datetime


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
class SiteStats:
    """Statistics for a solar site."""
    
    site_id: int
    last_reporting_time: datetime
    meter_reading_count: int
    max_capacity: float
    min_capacity: float
    total_energy_wh: float  # Watt-hours
    
    def __post_init__(self):
        """Validate stats data."""
        if self.meter_reading_count < 0:
            raise ValueError("Reading count cannot be negative")
        if self.max_capacity < self.min_capacity:
            raise ValueError("Max capacity cannot be less than min capacity")


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
```

### Create redisolar/models/__init__.py
```python
"""Domain models package."""

from .site import Site, SiteStats, MeterReading, Coordinate

__all__ = ['Site', 'SiteStats', 'MeterReading', 'Coordinate']
```

---

## Step 3: Data Schemas with Marshmallow

### Create redisolar/models/schemas.py
```python
"""Marshmallow schemas for data validation and serialization."""

from marshmallow import Schema, fields, post_load, validate, ValidationError
from datetime import datetime
from typing import Dict, Any

from .site import Site, SiteStats, MeterReading, Coordinate


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


class SiteStatsSchema(Schema):
    """Schema for SiteStats model."""
    
    site_id = fields.Integer(required=True)
    last_reporting_time = fields.DateTime(required=True)
    meter_reading_count = fields.Integer(required=True, validate=validate.Range(min=0))
    max_capacity = fields.Float(required=True)
    min_capacity = fields.Float(required=True)
    total_energy_wh = fields.Float(required=True, validate=validate.Range(min=0))
    
    @post_load
    def make_stats(self, data: Dict[str, Any], **kwargs) -> SiteStats:
        return SiteStats(**data)


class MeterReadingSchema(Schema):
    """Schema for MeterReading model."""
    
    site_id = fields.Integer(required=True)
    timestamp = fields.DateTime(required=True)
    wh_generated = fields.Float(required=True, validate=validate.Range(min=0))
    wh_used = fields.Float(required=True, validate=validate.Range(min=0))
    temp_c = fields.Float(required=True)
    
    @post_load
    def make_reading(self, data: Dict[str, Any], **kwargs) -> MeterReading:
        return MeterReading(**data)


# Create schema instances for reuse
site_schema = SiteSchema()
site_stats_schema = SiteStatsSchema()
meter_reading_schema = MeterReadingSchema()
coordinate_schema = CoordinateSchema()
```

---

## Step 4: Redis Key Management

### Create redisolar/dao/key_manager.py
```python
"""Redis key management and naming conventions."""

from typing import Union
from redisolar.config import Config


class KeyManager:
    """Manages Redis key generation and naming conventions."""
    
    def __init__(self, key_prefix: str = None):
        """Initialize with key prefix."""
        self.key_prefix = key_prefix or Config.from_env().redis_key_prefix
    
    def site_key(self, site_id: Union[int, str]) -> str:
        """Generate key for a site hash."""
        return f"{self.key_prefix}site:{site_id}"
    
    def sites_set_key(self) -> str:
        """Generate key for the set of all site IDs."""
        return f"{self.key_prefix}sites"
    
    def site_stats_key(self, site_id: Union[int, str]) -> str:
        """Generate key for site statistics hash."""
        return f"{self.key_prefix}site_stats:{site_id}"
    
    def meter_reading_key(self, site_id: Union[int, str], timestamp: str) -> str:
        """Generate key for individual meter reading."""
        return f"{self.key_prefix}meter_reading:{site_id}:{timestamp}"
    
    def meter_readings_set_key(self, site_id: Union[int, str]) -> str:
        """Generate key for set of meter reading timestamps for a site."""
        return f"{self.key_prefix}meter_readings:{site_id}"
    
    def capacity_leaderboard_key(self) -> str:
        """Generate key for capacity leaderboard (sorted set)."""
        return f"{self.key_prefix}capacity_leaderboard"
    
    def geo_sites_key(self) -> str:
        """Generate key for geospatial index of sites."""
        return f"{self.key_prefix}geo_sites"


# Global key manager instance
key_manager = KeyManager()
```

---

## Step 5: Base DAO Abstract Class

### Create redisolar/dao/base.py
```python
"""Base Data Access Object abstract class."""

import abc
from typing import Set, Optional, List, Any
from redisolar.models.site import Site


class SiteDaoBase(abc.ABC):
    """Abstract base class for Site Data Access Objects."""
    
    @abc.abstractmethod
    def insert(self, site: Site) -> None:
        """Insert a single site."""
        pass
    
    @abc.abstractmethod
    def insert_many(self, *sites: Site) -> None:
        """Insert multiple sites."""
        pass
    
    @abc.abstractmethod
    def find_by_id(self, site_id: int) -> Optional[Site]:
        """Find a site by its ID."""
        pass
    
    @abc.abstractmethod
    def find_all(self) -> Set[Site]:
        """Find all sites."""
        pass
    
    @abc.abstractmethod
    def update(self, site: Site) -> None:
        """Update an existing site."""
        pass
    
    @abc.abstractmethod
    def delete(self, site_id: int) -> bool:
        """Delete a site by ID. Returns True if deleted, False if not found."""
        pass
    
    @abc.abstractmethod
    def exists(self, site_id: int) -> bool:
        """Check if a site exists."""
        pass
    
    @abc.abstractmethod
    def count(self) -> int:
        """Get total number of sites."""
        pass
```

---

## Step 6: Site DAO Implementation

### Create redisolar/dao/site_dao_redis.py
```python
"""Redis-based Site Data Access Object implementation."""

import redis
from typing import Set, Optional, List, Dict, Any
import json
from datetime import datetime

from .base import SiteDaoBase
from .key_manager import key_manager
from .redis_dao import get_redis
from redisolar.models.site import Site, Coordinate
from redisolar.models.schemas import site_schema


class SiteDaoRedis(SiteDaoBase):
    """Redis implementation of Site DAO using Hashes."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """Initialize with Redis client."""
        self.redis = redis_client or get_redis()
        self.key_manager = key_manager
    
    def _site_to_hash(self, site: Site) -> Dict[str, str]:
        """Convert Site object to Redis hash format."""
        # Serialize site to dict using schema
        site_dict = site_schema.dump(site)
        
        # Convert all values to strings for Redis
        hash_data = {}
        for key, value in site_dict.items():
            if key == 'coordinate' and value is not None:
                # Store coordinate as JSON string
                hash_data[key] = json.dumps(value)
            elif value is not None:
                hash_data[key] = str(value)
        
        return hash_data
    
    def _hash_to_site(self, hash_data: Dict[str, str]) -> Site:
        """Convert Redis hash data to Site object."""
        if not hash_data:
            raise ValueError("Empty hash data")
        
        # Convert string values back to appropriate types
        site_dict = {}
        for key, value in hash_data.items():
            if key == 'coordinate' and value:
                # Parse coordinate JSON
                coord_data = json.loads(value)
                site_dict[key] = coord_data
            elif key in ['id', 'panels']:
                site_dict[key] = int(value)
            elif key == 'capacity':
                site_dict[key] = float(value)
            else:
                site_dict[key] = value
        
        # Deserialize using schema
        return site_schema.load(site_dict)
    
    def insert(self, site: Site) -> None:
        """Insert a single site."""
        if self.exists(site.id):
            raise ValueError(f"Site with ID {site.id} already exists")
        
        # Start a pipeline for atomic operations
        pipe = self.redis.pipeline()
        
        try:
            # Store site data as hash
            site_key = self.key_manager.site_key(site.id)
            hash_data = self._site_to_hash(site)
            pipe.hset(site_key, mapping=hash_data)
            
            # Add site ID to the set of all sites
            sites_set_key = self.key_manager.sites_set_key()
            pipe.sadd(sites_set_key, site.id)
            
            # Execute pipeline
            pipe.execute()
            
        except Exception as e:
            # If there's an error, clean up
            pipe.discard()
            raise RuntimeError(f"Failed to insert site {site.id}: {e}")
    
    def insert_many(self, *sites: Site) -> None:
        """Insert multiple sites atomically."""
        if not sites:
            return
        
        # Check for duplicates
        existing_ids = []
        for site in sites:
            if self.exists(site.id):
                existing_ids.append(site.id)
        
        if existing_ids:
            raise ValueError(f"Sites already exist: {existing_ids}")
        
        # Use pipeline for batch operations
        pipe = self.redis.pipeline()
        
        try:
            sites_set_key = self.key_manager.sites_set_key()
            
            for site in sites:
                # Store each site as hash
                site_key = self.key_manager.site_key(site.id)
                hash_data = self._site_to_hash(site)
                pipe.hset(site_key, mapping=hash_data)
                
                # Add to sites set
                pipe.sadd(sites_set_key, site.id)
            
            # Execute all operations
            pipe.execute()
            
        except Exception as e:
            pipe.discard()
            raise RuntimeError(f"Failed to insert sites: {e}")
    
    def find_by_id(self, site_id: int) -> Optional[Site]:
        """Find a site by its ID."""
        site_key = self.key_manager.site_key(site_id)
        hash_data = self.redis.hgetall(site_key)
        
        if not hash_data:
            return None
        
        try:
            return self._hash_to_site(hash_data)
        except Exception as e:
            raise RuntimeError(f"Failed to deserialize site {site_id}: {e}")
    
    def find_all(self) -> Set[Site]:
        """Find all sites."""
        sites_set_key = self.key_manager.sites_set_key()
        site_ids = self.redis.smembers(sites_set_key)
        
        if not site_ids:
            return set()
        
        # Use pipeline to get all sites efficiently
        pipe = self.redis.pipeline()
        for site_id in site_ids:
            site_key = self.key_manager.site_key(site_id)
            pipe.hgetall(site_key)
        
        results = pipe.execute()
        
        sites = set()
        for hash_data in results:
            if hash_data:  # Skip empty results
                try:
                    site = self._hash_to_site(hash_data)
                    sites.add(site)
                except Exception as e:
                    print(f"Warning: Failed to deserialize site data: {e}")
        
        return sites
    
    def update(self, site: Site) -> None:
        """Update an existing site."""
        if not self.exists(site.id):
            raise ValueError(f"Site with ID {site.id} does not exist")
        
        site_key = self.key_manager.site_key(site.id)
        hash_data = self._site_to_hash(site)
        
        # Update the hash
        self.redis.hset(site_key, mapping=hash_data)
    
    def delete(self, site_id: int) -> bool:
        """Delete a site by ID."""
        if not self.exists(site_id):
            return False
        
        pipe = self.redis.pipeline()
        
        try:
            # Delete the site hash
            site_key = self.key_manager.site_key(site_id)
            pipe.delete(site_key)
            
            # Remove from sites set
            sites_set_key = self.key_manager.sites_set_key()
            pipe.srem(sites_set_key, site_id)
            
            # Execute operations
            results = pipe.execute()
            
            # Check if the site hash was actually deleted
            return results[0] > 0
            
        except Exception as e:
            pipe.discard()
            raise RuntimeError(f"Failed to delete site {site_id}: {e}")
    
    def exists(self, site_id: int) -> bool:
        """Check if a site exists."""
        site_key = self.key_manager.site_key(site_id)
        return self.redis.exists(site_key) > 0
    
    def count(self) -> int:
        """Get total number of sites."""
        sites_set_key = self.key_manager.sites_set_key()
        return self.redis.scard(sites_set_key)
    
    def find_by_capacity_range(self, min_capacity: float, max_capacity: float) -> List[Site]:
        """Find sites within a capacity range."""
        all_sites = self.find_all()
        return [
            site for site in all_sites 
            if min_capacity <= site.capacity <= max_capacity
        ]
    
    def find_by_state(self, state: str) -> List[Site]:
        """Find sites in a specific state."""
        all_sites = self.find_all()
        return [site for site in all_sites if site.state.lower() == state.lower()]
    
    def get_total_capacity(self) -> float:
        """Calculate total capacity across all sites."""
        all_sites = self.find_all()
        return sum(site.capacity for site in all_sites)
```

---

## Step 7: API Routes for Site Management

### Update redisolar/api/routes.py
```python
"""API routes for site management."""

from flask import Blueprint, request, jsonify
from marshmallow import ValidationError
from typing import Dict, Any

from redisolar.dao.site_dao_redis import SiteDaoRedis
from redisolar.models.schemas import site_schema
from redisolar.models.site import Site, Coordinate

# Create blueprint
sites_bp = Blueprint('sites', __name__, url_prefix='/api/sites')

# Initialize DAO
site_dao = SiteDaoRedis()


@sites_bp.route('/', methods=['GET'])
def get_all_sites():
    """Get all sites."""
    try:
        sites = site_dao.find_all()
        sites_data = [site_schema.dump(site) for site in sites]
        
        return jsonify({
            'sites': sites_data,
            'count': len(sites_data),
            'total_capacity': site_dao.get_total_capacity()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@sites_bp.route('/<int:site_id>', methods=['GET'])
def get_site(site_id: int):
    """Get a specific site by ID."""
    try:
        site = site_dao.find_by_id(site_id)
        
        if not site:
            return jsonify({'error': f'Site {site_id} not found'}), 404
        
        return jsonify(site_schema.dump(site))
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@sites_bp.route('/', methods=['POST'])
def create_site():
    """Create a new site."""
    try:
        # Validate request data
        site_data = request.get_json()
        if not site_data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Deserialize and validate using schema
        site = site_schema.load(site_data)
        
        # Insert into database
        site_dao.insert(site)
        
        return jsonify({
            'message': f'Site {site.id} created successfully',
            'site': site_schema.dump(site)
        }), 201
    
    except ValidationError as e:
        return jsonify({'error': 'Validation failed', 'details': e.messages}), 400
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 409  # Conflict - site already exists
    
    except Exception as e:
        return jsonify({'error': f'Failed to create site: {str(e)}'}), 500


@sites_bp.route('/<int:site_id>', methods=['PUT'])
def update_site(site_id: int):
    """Update an existing site."""
    try:
        # Check if site exists
        if not site_dao.exists(site_id):
            return jsonify({'error': f'Site {site_id} not found'}), 404
        
        # Validate request data
        site_data = request.get_json()
        if not site_data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Ensure ID matches URL parameter
        site_data['id'] = site_id
        
        # Deserialize and validate
        site = site_schema.load(site_data)
        
        # Update in database
        site_dao.update(site)
        
        return jsonify({
            'message': f'Site {site_id} updated successfully',
            'site': site_schema.dump(site)
        })
    
    except ValidationError as e:
        return jsonify({'error': 'Validation failed', 'details': e.messages}), 400
    
    except Exception as e:
        return jsonify({'error': f'Failed to update site: {str(e)}'}), 500


@sites_bp.route('/<int:site_id>', methods=['DELETE'])
def delete_site(site_id: int):
    """Delete a site."""
    try:
        deleted = site_dao.delete(site_id)
        
        if not deleted:
            return jsonify({'error': f'Site {site_id} not found'}), 404
        
        return jsonify({'message': f'Site {site_id} deleted successfully'})
    
    except Exception as e:
        return jsonify({'error': f'Failed to delete site: {str(e)}'}), 500


@sites_bp.route('/search', methods=['GET'])
def search_sites():
    """Search sites by various criteria."""
    try:
        # Get query parameters
        state = request.args.get('state')
        min_capacity = request.args.get('min_capacity', type=float)
        max_capacity = request.args.get('max_capacity', type=float)
        
        sites = []
        
        if state:
            sites = site_dao.find_by_state(state)
        elif min_capacity is not None and max_capacity is not None:
            sites = site_dao.find_by_capacity_range(min_capacity, max_capacity)
        else:
            sites = list(site_dao.find_all())
        
        sites_data = [site_schema.dump(site) for site in sites]
        
        return jsonify({
            'sites': sites_data,
            'count': len(sites_data)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@sites_bp.route('/stats', methods=['GET'])
def get_site_stats():
    """Get overall site statistics."""
    try:
        total_sites = site_dao.count()
        total_capacity = site_dao.get_total_capacity()
        
        if total_sites > 0:
            all_sites = site_dao.find_all()
            avg_capacity = total_capacity / total_sites
            capacities = [site.capacity for site in all_sites]
            max_capacity = max(capacities)
            min_capacity = min(capacities)
        else:
            avg_capacity = max_capacity = min_capacity = 0
        
        return jsonify({
            'total_sites': total_sites,
            'total_capacity': total_capacity,
            'average_capacity': avg_capacity,
            'max_capacity': max_capacity,
            'min_capacity': min_capacity
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

---

## Step 8: Update Flask App

### Update redisolar/app.py
```python
"""Main Flask application."""

import os
from flask import Flask, jsonify
from redisolar.dao.redis_dao import RedisConnection, get_redis


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
    
    # Register blueprints
    from redisolar.api.routes import sites_bp
    app.register_blueprint(sites_bp)
    
    # Register routes
    @app.route('/')
    def index():
        """Health check endpoint."""
        return jsonify({
            'message': 'RediSolar API is running!',
            'redis_connected': RedisConnection.test_connection(),
            'redis_info': get_redis().info('server')['redis_version'],
            'endpoints': [
                'GET /api/sites - Get all sites',
                'GET /api/sites/<id> - Get specific site',
                'POST /api/sites - Create new site',
                'PUT /api/sites/<id> - Update site',
                'DELETE /api/sites/<id> - Delete site',
                'GET /api/sites/search - Search sites',
                'GET /api/sites/stats - Get site statistics'
            ]
        })
    
    @app.route('/health')
    def health():
        """Detailed health check."""
        redis_conn = get_redis()
        
        try:
            # Test basic Redis operations
            test_key = f"{app.config['REDIS_KEY_PREFIX']}health_check"
            redis_conn.set(test_key, "ok", ex=60)
            test_value = redis_conn.get(test_key)
            redis_conn.delete(test_key)
            
            redis_healthy = test_value == "ok"
        except Exception as e:
            redis_healthy = False
            
        return jsonify({
            'status': 'healthy' if redis_healthy else 'unhealthy',
            'redis': {
                'connected': redis_healthy,
                'host': app.config.get('REDIS_HOST'),
                'port': app.config.get('REDIS_PORT'),
                'db': app.config.get('REDIS_DB'),
                'key_prefix': app.config.get('REDIS_KEY_PREFIX'),
            }
        })
    
    return app


# For development server
if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=8081, debug=True)
```

---

## Step 9: Comprehensive Tests

### Create tests/test_site_dao.py
```python
"""Tests for Site DAO Redis implementation."""

import pytest
from datetime import datetime
import redis

from redisolar.dao.site_dao_redis import SiteDaoRedis
from redisolar.dao.key_manager import KeyManager
from redisolar.models.site import Site, Coordinate
from redisolar.config import Config


@pytest.fixture
def redis_client():
    """Create Redis client for testing."""
    config = Config(
        redis_host='localhost',
        redis_port=6379,
        redis_db=1,  # Use test database
        redis_key_prefix='test:'
    )
    
    client = redis.Redis(
        host=config.redis_host,
        port=config.redis_port,
        db=config.redis_db,
        decode_responses=True
    )
    
    # Clear test database
    client.flushdb()
    
    yield client
    
    # Clean up after tests
    client.flushdb()


@pytest.fixture
def site_dao(redis_client):
    """Create Site DAO for testing."""
    return SiteDaoRedis(redis_client)


@pytest.fixture
def sample_site():
    """Create a sample site for testing."""
    return Site(
        id=1,
        capacity=4.5,
        panels=12,
        address="123 Solar Street",
        city="Sunnyville",
        state="CA",
        postal_code="90210",
        coordinate=Coordinate(lat=34.0522, lng=-118.2437)
    )


@pytest.fixture
def sample_sites():
    """Create multiple sample sites for testing."""
    return [
        Site(
            id=1,
            capacity=4.5,
            panels=12,
            address="123 Solar Street",
            city="Sunnyville",
            state="CA",
            postal_code="90210",
            coordinate=Coordinate(lat=34.0522, lng=-118.2437)
        ),
        Site(
            id=2,
            capacity=6.0,
            panels=20,
            address="456 Energy Ave",
            city="Power City",
            state="TX",
            postal_code="75001",
            coordinate=Coordinate(lat=32.7767, lng=-96.7970)
        ),
        Site(
            id=3,
            capacity=3.2,
            panels=8,
            address="789 Green Way",
            city="Eco Town",
            state="FL",
            postal_code="33101"
        )
    ]


class TestSiteDaoRedis:
    """Test cases for Site DAO Redis implementation."""
    
    def test_insert_and_find_by_id(self, site_dao, sample_site):
        """Test inserting and retrieving a site."""
        # Insert site
        site_dao.insert(sample_site)
        
        # Retrieve site
        retrieved_site = site_dao.find_by_id(sample_site.id)
        
        assert retrieved_site is not None
        assert retrieved_site == sample_site
        assert retrieved_site.coordinate.lat == sample_site.coordinate.lat
        assert retrieved_site.coordinate.lng == sample_site.coordinate.lng
    
    def test_insert_duplicate_raises_error(self, site_dao, sample_site):
        """Test that inserting duplicate site raises error."""
        # Insert site
        site_dao.insert(sample_site)
        
        # Try to insert same site again
        with pytest.raises(ValueError, match="already exists"):
            site_dao.insert(sample_site)
    
    def test_find_nonexistent_site_returns_none(self, site_dao):
        """Test that finding non-existent site returns None."""
        result = site_dao.find_by_id(999)
        assert result is None
    
    def test_insert_many(self, site_dao, sample_sites):
        """Test inserting multiple sites."""
        site_dao.insert_many(*sample_sites)
        
        # Verify all sites were inserted
        for site in sample_sites:
            retrieved = site_dao.find_by_id(site.id)
            assert retrieved == site
        
        # Check count
        assert site_dao.count() == len(sample_sites)
    
    def test_find_all(self, site_dao, sample_sites):
        """Test retrieving all sites."""
        # Insert sites
        site_dao.insert_many(*sample_sites)
        
        # Retrieve all
        all_sites = site_dao.find_all()
        
        assert len(all_sites) == len(sample_sites)
        assert all_sites == set(sample_sites)
    
    def test_update_site(self, site_dao, sample_site):
        """Test updating a site."""
        # Insert original site
        site_dao.insert(sample_site)
        
        # Create updated site
        updated_site = Site(
            id=sample_site.id,
            capacity=5.0,  # Changed
            panels=15,     # Changed
            address=sample_site.address,
            city=sample_site.city,
            state=sample_site.state,
            postal_code=sample_site.postal_code,
            coordinate=sample_site.coordinate
        )
        
        # Update site
        site_dao.update(updated_site)
        
        # Retrieve and verify
        retrieved = site_dao.find_by_id(sample_site.id)
        assert retrieved == updated_site
        assert retrieved.capacity == 5.0
        assert retrieved.panels == 15
    
    def test_update_nonexistent_site_raises_error(self, site_dao, sample_site):
        """Test that updating non-existent site raises error."""
        with pytest.raises(ValueError, match="does not exist"):
            site_dao.update(sample_site)
    
    def test_delete_site(self, site_dao, sample_site):
        """Test deleting a site."""
        # Insert site
        site_dao.insert(sample_site)
        assert site_dao.exists(sample_site.id)
        
        # Delete site
        deleted = site_dao.delete(sample_site.id)
        assert deleted is True
        
        # Verify deletion
        assert not site_dao.exists(sample_site.id)
        assert site_dao.find_by_id(sample_site.id) is None
        assert site_dao.count() == 0
    
    def test_delete_nonexistent_site_returns_false(self, site_dao):
        """Test that deleting non-existent site returns False."""
        deleted = site_dao.delete(999)
        assert deleted is False
    
    def test_exists(self, site_dao, sample_site):
        """Test checking if site exists."""
        # Should not exist initially
        assert not site_dao.exists(sample_site.id)
        
        # Insert and check
        site_dao.insert(sample_site)
        assert site_dao.exists(sample_site.id)
        
        # Delete and check
        site_dao.delete(sample_site.id)
        assert not site_dao.exists(sample_site.id)
    
    def test_count(self, site_dao, sample_sites):
        """Test counting sites."""
        assert site_dao.count() == 0
        
        # Insert sites one by one
        for i, site in enumerate(sample_sites, 1):
            site_dao.insert(site)
            assert site_dao.count() == i
    
    def test_find_by_capacity_range(self, site_dao, sample_sites):
        """Test finding sites by capacity range."""
        site_dao.insert_many(*sample_sites)
        
        # Find sites with capacity between 4.0 and 5.0
        sites = site_dao.find_by_capacity_range(4.0, 5.0)
        assert len(sites) == 1
        assert sites[0].id == 1  # Only site 1 has capacity 4.5
        
        # Find sites with capacity between 3.0 and 7.0
        sites = site_dao.find_by_capacity_range(3.0, 7.0)
        assert len(sites) == 3  # All sites
    
    def test_find_by_state(self, site_dao, sample_sites):
        """Test finding sites by state."""
        site_dao.insert_many(*sample_sites)
        
        # Find sites in CA
        ca_sites = site_dao.find_by_state("CA")
        assert len(ca_sites) == 1
        assert ca_sites[0].state == "CA"
        
        # Find sites in TX
        tx_sites = site_dao.find_by_state("TX")
        assert len(tx_sites) == 1
        assert tx_sites[0].state == "TX"
        
        # Test case insensitive
        ca_sites_lower = site_dao.find_by_state("ca")
        assert len(ca_sites_lower) == 1
    
    def test_get_total_capacity(self, site_dao, sample_sites):
        """Test calculating total capacity."""
        assert site_dao.get_total_capacity() == 0
        
        site_dao.insert_many(*sample_sites)
        
        expected_total = sum(site.capacity for site in sample_sites)
        assert site_dao.get_total_capacity() == expected_total
    
    def test_site_without_coordinates(self, site_dao):
        """Test site without coordinate data."""
        site = Site(
            id=100,
            capacity=2.5,
            panels=6,
            address="No GPS Street",
            city="Location Unknown",
            state="XX",
            postal_code="00000"
        )
        
        site_dao.insert(site)
        retrieved = site_dao.find_by_id(100)
        
        assert retrieved == site
        assert retrieved.coordinate is None


class TestKeyManager:
    """Test cases for key manager."""
    
    def test_key_generation(self):
        """Test that keys are generated correctly."""
        km = KeyManager(key_prefix="test:")
        
        assert km.site_key(1) == "test:site:1"
        assert km.sites_set_key() == "test:sites"
        assert km.site_stats_key(1) == "test:site_stats:1"
        assert km.capacity_leaderboard_key() == "test:capacity_leaderboard"
        assert km.geo_sites_key() == "test:geo_sites"
```

### Create tests/test_api_sites.py
```python
"""Tests for Sites API endpoints."""

import pytest
import json
from flask import Flask

from redisolar.app import create_app
from redisolar.dao.site_dao_redis import SiteDaoRedis
from redisolar.dao.redis_dao import RedisConnection
from redisolar.models.site import Site, Coordinate


@pytest.fixture
def app():
    """Create test application."""
    app = create_app('testing')
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture
def site_dao():
    """Create Site DAO for testing."""
    return SiteDaoRedis()


@pytest.fixture
def sample_site_data():
    """Sample site data for API testing."""
    return {
        "id": 1,
        "capacity": 4.5,
        "panels": 12,
        "address": "123 Solar Street",
        "city": "Sunnyville",
        "state": "CA",
        "postal_code": "90210",
        "coordinate": {
            "lat": 34.0522,
            "lng": -118.2437
        }
    }


@pytest.fixture
def sample_site_data_no_coords():
    """Sample site data without coordinates."""
    return {
        "id": 2,
        "capacity": 3.2,
        "panels": 8,
        "address": "456 Energy Ave",
        "city": "Power City",
        "state": "TX",
        "postal_code": "75001"
    }


@pytest.fixture(autouse=True)
def cleanup_redis():
    """Clean up Redis before and after each test."""
    # Clean up before test
    redis_conn = RedisConnection.get_connection()
    redis_conn.flushdb()
    
    yield
    
    # Clean up after test
    redis_conn.flushdb()


class TestSitesAPI:
    """Test cases for Sites API."""
    
    def test_get_all_sites_empty(self, client):
        """Test getting all sites when none exist."""
        response = client.get('/api/sites/')
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['sites'] == []
        assert data['count'] == 0
        assert data['total_capacity'] == 0
    
    def test_create_site(self, client, sample_site_data):
        """Test creating a new site."""
        response = client.post(
            '/api/sites/',
            data=json.dumps(sample_site_data),
            content_type='application/json'
        )
        
        assert response.status_code == 201
        data = response.get_json()
        assert 'Site 1 created successfully' in data['message']
        assert data['site']['id'] == 1
        assert data['site']['capacity'] == 4.5
    
    def test_create_site_without_coordinates(self, client, sample_site_data_no_coords):
        """Test creating a site without coordinates."""
        response = client.post(
            '/api/sites/',
            data=json.dumps(sample_site_data_no_coords),
            content_type='application/json'
        )
        
        assert response.status_code == 201
        data = response.get_json()
        assert data['site']['coordinate'] is None
    
    def test_create_site_invalid_data(self, client):
        """Test creating site with invalid data."""
        invalid_data = {
            "id": "invalid",  # Should be integer
            "capacity": -1,   # Should be positive
            "panels": 0       # Should be positive
        }
        
        response = client.post(
            '/api/sites/',
            data=json.dumps(invalid_data),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'Validation failed' in data['error']
    
    def test_create_duplicate_site(self, client, sample_site_data):
        """Test creating duplicate site returns error."""
        # Create first site
        client.post(
            '/api/sites/',
            data=json.dumps(sample_site_data),
            content_type='application/json'
        )
        
        # Try to create same site again
        response = client.post(
            '/api/sites/',
            data=json.dumps(sample_site_data),
            content_type='application/json'
        )
        
        assert response.status_code == 409
        data = response.get_json()
        assert 'already exists' in data['error']
    
    def test_get_site_by_id(self, client, sample_site_data):
        """Test getting a specific site by ID."""
        # Create site first
        client.post(
            '/api/sites/',
            data=json.dumps(sample_site_data),
            content_type='application/json'
        )
        
        # Get site by ID
        response = client.get('/api/sites/1')
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['id'] == 1
        assert data['capacity'] == 4.5
        assert data['coordinate']['lat'] == 34.0522
    
    def test_get_nonexistent_site(self, client):
        """Test getting non-existent site returns 404."""
        response = client.get('/api/sites/999')
        assert response.status_code == 404
        
        data = response.get_json()
        assert 'not found' in data['error']
    
    def test_update_site(self, client, sample_site_data):
        """Test updating an existing site."""
        # Create site first
        client.post(
            '/api/sites/',
            data=json.dumps(sample_site_data),
            content_type='application/json'
        )
        
        # Update site
        updated_data = sample_site_data.copy()
        updated_data['capacity'] = 5.0
        updated_data['panels'] = 15
        
        response = client.put(
            '/api/sites/1',
            data=json.dumps(updated_data),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'updated successfully' in data['message']
        assert data['site']['capacity'] == 5.0
        assert data['site']['panels'] == 15
    
    def test_update_nonexistent_site(self, client, sample_site_data):
        """Test updating non-existent site returns 404."""
        response = client.put(
            '/api/sites/999',
            data=json.dumps(sample_site_data),
            content_type='application/json'
        )
        
        assert response.status_code == 404
        data = response.get_json()
        assert 'not found' in data['error']
    
    def test_delete_site(self, client, sample_site_data):
        """Test deleting a site."""
        # Create site first
        client.post(
            '/api/sites/',
            data=json.dumps(sample_site_data),
            content_type='application/json'
        )
        
        # Delete site
        response = client.delete('/api/sites/1')
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'deleted successfully' in data['message']
        
        # Verify site is gone
        response = client.get('/api/sites/1')
        assert response.status_code == 404
    
    def test_delete_nonexistent_site(self, client):
        """Test deleting non-existent site returns 404."""
        response = client.delete('/api/sites/999')
        assert response.status_code == 404
    
    def test_get_all_sites_with_data(self, client, sample_site_data, sample_site_data_no_coords):
        """Test getting all sites when some exist."""
        # Create two sites
        client.post(
            '/api/sites/',
            data=json.dumps(sample_site_data),
            content_type='application/json'
        )
        client.post(
            '/api/sites/',
            data=json.dumps(sample_site_data_no_coords),
            content_type='application/json'
        )
        
        # Get all sites
        response = client.get('/api/sites/')
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['count'] == 2
        assert data['total_capacity'] == 4.5 + 3.2  # Sum of capacities
        assert len(data['sites']) == 2
    
    def test_search_sites_by_state(self, client, sample_site_data, sample_site_data_no_coords):
        """Test searching sites by state."""
        # Create sites in different states
        client.post(
            '/api/sites/',
            data=json.dumps(sample_site_data),
            content_type='application/json'
        )
        client.post(
            '/api/sites/',
            data=json.dumps(sample_site_data_no_coords),
            content_type='application/json'
        )
        
        # Search for CA sites
        response = client.get('/api/sites/search?state=CA')
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['count'] == 1
        assert data['sites'][0]['state'] == 'CA'
    
    def test_search_sites_by_capacity_range(self, client, sample_site_data, sample_site_data_no_coords):
        """Test searching sites by capacity range."""
        # Create sites with different capacities
        client.post(
            '/api/sites/',
            data=json.dumps(sample_site_data),
            content_type='application/json'
        )
        client.post(
            '/api/sites/',
            data=json.dumps(sample_site_data_no_coords),
            content_type='application/json'
        )
        
        # Search for sites with capacity 4.0-5.0
        response = client.get('/api/sites/search?min_capacity=4.0&max_capacity=5.0')
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['count'] == 1
        assert data['sites'][0]['capacity'] == 4.5
    
    def test_get_site_stats(self, client, sample_site_data, sample_site_data_no_coords):
        """Test getting site statistics."""
        # Create sites
        client.post(
            '/api/sites/',
            data=json.dumps(sample_site_data),
            content_type='application/json'
        )
        client.post(
            '/api/sites/',
            data=json.dumps(sample_site_data_no_coords),
            content_type='application/json'
        )
        
        # Get stats
        response = client.get('/api/sites/stats')
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['total_sites'] == 2
        assert data['total_capacity'] == 4.5 + 3.2
        assert data['average_capacity'] == (4.5 + 3.2) / 2
        assert data['max_capacity'] == 4.5
        assert data['min_capacity'] == 3.2
```

---

## Step 10: Sample Data Loader

### Create scripts/load_sample_data.py
```python
"""Load sample data into Redis for development and testing."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redisolar.dao.site_dao_redis import SiteDaoRedis
from redisolar.models.site import Site, Coordinate
from redisolar.dao.redis_dao import RedisConnection


def load_sample_sites():
    """Load sample solar sites into Redis."""
    
    # Sample sites data
    sites_data = [
        {
            "id": 1,
            "capacity": 4.5,
            "panels": 12,
            "address": "123 Solar Street",
            "city": "Sunnyville",
            "state": "CA",
            "postal_code": "90210",
            "coordinate": Coordinate(lat=34.0522, lng=-118.2437)
        },
        {
            "id": 2,
            "capacity": 6.0,
            "panels": 20,
            "address": "456 Energy Avenue",
            "city": "Power City",
            "state": "TX",
            "postal_code": "75001",
            "coordinate": Coordinate(lat=32.7767, lng=-96.7970)
        },
        {
            "id": 3,
            "capacity": 3.2,
            "panels": 8,
            "address": "789 Green Way",
            "city": "Eco Town",
            "state": "FL",
            "postal_code": "33101",
            "coordinate": Coordinate(lat=25.7617, lng=-80.1918)
        },
        {
            "id": 4,
            "capacity": 7.8,
            "panels": 26,
            "address": "321 Renewable Road",
            "city": "Solar City",
            "state": "AZ",
            "postal_code": "85001",
            "coordinate": Coordinate(lat=33.4484, lng=-112.0740)
        },
        {
            "id": 5,
            "capacity": 2.1,
            "panels": 6,
            "address": "654 Clean Lane",
            "city": "Green Valley",
            "state": "OR",
            "postal_code": "97201",
            "coordinate": Coordinate(lat=45.5152, lng=-122.6784)
        }
    ]
    
    # Create Site objects
    sites = []
    for site_data in sites_data:
        sites.append(Site(**site_data))
    
    # Initialize DAO and load data
    site_dao = SiteDaoRedis()
    
    try:
        # Check if data already exists
        if site_dao.count() > 0:
            print(f"Warning: {site_dao.count()} sites already exist in Redis.")
            response = input("Do you want to clear existing data and reload? (y/N): ")
            if response.lower() != 'y':
                print("Aborted.")
                return
            
            # Clear existing sites
            all_sites = site_dao.find_all()
            for site in all_sites:
                site_dao.delete(site.id)
            print(f"Cleared {len(all_sites)} existing sites.")
        
        # Load new sites
        site_dao.insert_many(*sites)
        print(f"Successfully loaded {len(sites)} sites into Redis.")
        
        # Display summary
        print("\nLoaded sites:")
        for site in sites:
            print(f"  - Site {site.id}: {site.capacity}kW, {site.panels} panels, {site.city}, {site.state}")
        
        print(f"\nTotal capacity: {site_dao.get_total_capacity()}kW")
        
    except Exception as e:
        print(f"Error loading sample data: {e}")
        return False
    
    return True


def main():
    """Main function."""
    print("Loading sample solar sites into Redis...")
    
    # Test Redis connection
    if not RedisConnection.test_connection():
        print("Error: Could not connect to Redis. Please ensure Redis is running.")
        return
    
    success = load_sample_sites()
    
    if success:
        print("\nSample data loaded successfully!")
        print("You can now test the API endpoints:")
        print("  GET http://localhost:8081/api/sites/")
        print("  GET http://localhost:8081/api/sites/stats")
        print("  GET http://localhost:8081/api/sites/search?state=CA")
    else:
        print("Failed to load sample data.")


if __name__ == "__main__":
    main()
```

---

## Step 11: Running and Testing Everything

### Update requirements.txt
```txt
Flask==2.3.3
redis==5.0.1
python-dotenv==1.0.0
marshmallow==3.20.1
marshmallow-dataclass==8.6.0
dataclasses-json==0.6.1
pytest==7.4.2
python-decouple==3.8
```

### Create Makefile (Optional)
```makefile
.PHONY: help env dev test load clean

help:		## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $1, $2}'

env:		## Set up virtual environment and install dependencies
	python3.8 -m venv env
	env/bin/pip install --upgrade pip
	env/bin/pip install -r requirements.txt
	env/bin/pip install -e .

dev:		## Run development server
	env/bin/python redisolar/app.py

test:		## Run tests
	env/bin/pytest tests/ -v

load:		## Load sample data
	env/bin/python scripts/load_sample_data.py

clean:		## Clean up Redis test data
	redis-cli -n 1 FLUSHDB

redis-start:	## Start Redis using Docker
	docker run -d --name redis-redisolar -p 6379:6379 redis/redis-stack:latest

redis-stop:	## Stop Redis Docker container
	docker stop redis-redisolar && docker rm redis-redisolar
```

### Running the Application

1. **Start Redis** (if not already running):
```bash
make redis-start
# or manually:
docker run -d --name redis-redisolar -p 6379:6379 redis/redis-stack:latest
```

2. **Set up environment**:
```bash
make env
# or manually:
python3.8 -m venv env
source env/bin/activate
pip install -r requirements.txt
pip install -e .
```

3. **Load sample data**:
```bash
make load
# or manually:
python scripts/load_sample_data.py
```

4. **Run the application**:
```bash
make dev
# or manually:
python redisolar/app.py
```

5. **Run tests**:
```bash
make test
# or manually:
pytest tests/ -v
```

### Testing the API

With the application running, test these endpoints:

```bash
# Get all sites
curl http://localhost:8081/api/sites/

# Get specific site
curl http://localhost:8081/api/sites/1

# Create new site
curl -X POST http://localhost:8081/api/sites/ \
  -H "Content-Type: application/json" \
  -d '{
    "id": 6,
    "capacity": 5.5,
    "panels": 18,
    "address": "999 New Solar St",
    "city": "Future City",
    "state": "NY",
    "postal_code": "10001"
  }'

# Search by state
curl "http://localhost:8081/api/sites/search?state=CA"

# Search by capacity range
curl "http://localhost:8081/api/sites/search?min_capacity=4.0&max_capacity=6.0"

# Get statistics
curl http://localhost:8081/api/sites/stats

# Update site
curl -X PUT http://localhost:8081/api/sites/1 \
  -H "Content-Type: application/json" \
  -d '{
    "id": 1,
    "capacity": 5.0,
    "panels": 15,
    "address": "123 Solar Street",
    "city": "Sunnyville",
    "state": "CA",
    "postal_code": "90210"
  }'

# Delete site
curl -X DELETE http://localhost:8081/api/sites/6
```

---

## 🎉 Lesson 2 Complete!

You've successfully:
- ✅ Created comprehensive domain models with dataclasses
- ✅ Implemented data validation with Marshmallow schemas
- ✅ Built a complete Site DAO using Redis Hashes
- ✅ Created a full REST API for site management
- ✅ Written comprehensive tests for both DAO and API layers
- ✅ Implemented Redis key management patterns
- ✅ Created a sample data loader for development

### Key Redis Concepts Learned
1. **Redis Hashes**: Perfect for storing structured objects
2. **Redis Sets**: Used for maintaining collections of IDs
3. **Atomic Operations**: Using pipelines for multi-step operations
4. **Key Naming Conventions**: Structured, predictable key patterns
5. **Data Serialization**: Converting between Python objects and Redis storage

### What's Next?
In **Lesson 3**, we'll:
- Implement Redis Sorted Sets for leaderboards
- Add time-series data with Redis TimeSeries
- Create meter reading storage and retrieval
- Build performance analytics and rankings
- Implement data aggregation patterns

### Key Takeaways
1. **Data Modeling**: Use dataclasses for clean, validated domain models
2. **Separation of Concerns**: Keep models, schemas, DAOs, and APIs separate
3. **Testing Strategy**: Test both unit (DAO) and integration (API) levels
4. **Error Handling**: Proper validation and error responses
5. **Redis Patterns**: Hashes for objects, Sets for collections, Pipelines for atomicity
