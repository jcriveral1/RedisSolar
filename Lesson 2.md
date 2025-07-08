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
