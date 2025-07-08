# Load readings in batches for better performance
        batch_size = 100
        total_batches = (len(readings) + batch_size - 1) // batch_size
        
        for i in range(0, len(readings), batch_size):
            batch = readings[i:i + batch_size]
            timeseries_dao.add_readings_batch(batch)
            current_batch = (i // batch_size) + 1
            print(f"  Loaded batch {current_batch}/{total_batches} ({len(batch)} readings)")
        
        print(f"Successfully loaded {len(readings)} meter readings.")
        
        # Build leaderboards
        print("Building leaderboards...")
        leaderboard_dao.rebuild_capacity_leaderboard()
        
        # Trigger leaderboard updates for generation and efficiency
        from redisolar.tasks.leaderboard_updater import leaderboard_updater
        leaderboard_updater.update_generation_leaderboard()
        leaderboard_updater.update_efficiency_leaderboard()
        
        print("Leaderboards updated.")
        
        # Display summary
        print("\n" + "="*50)
        print("SAMPLE DATA LOADED SUCCESSFULLY!")
        print("="*50)
        
        print(f"\nLoaded sites:")
        for site in sites:
            print(f"  - Site {site.id}: {site.capacity}kW, {site.panels} panels, {site.city}, {site.state}")
        
        print(f"\nTotal capacity: {site_dao.get_total_capacity()}kW")
        print(f"Total readings: {len(readings)} (7 days of hourly data)")
        
        # Show some sample analytics
        print(f"\nSample analytics endpoints to try:")
        print(f"  - Get all sites: GET http://localhost:8081/api/sites/")
        print(f"  - Site analytics: GET http://localhost:8081/api/analytics/analytics/1")
        print(f"  - Capacity leaderboard: GET http://localhost:8081/api/analytics/leaderboards/capacity")
        print(f"  - Generation leaderboard: GET http://localhost:8081/api/analytics/leaderboards/generation")
        print(f"  - Latest readings: GET http://localhost:8081/api/analytics/readings/1/latest")
        print(f"  - Overall summary: GET http://localhost:8081/api/analytics/summary")
        
    except Exception as e:
        print(f"Error loading sample data: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()
```

---

## Step 10: Comprehensive Tests

### Create tests/test_timeseries_dao.py
```python
"""Tests for TimeSeries DAO implementation."""

import pytest
from datetime import datetime, timedelta
import redis

from redisolar.dao.timeseries_dao import TimeSeriesDao
from redisolar.models.site import MeterReading, MetricType
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
def timeseries_dao(redis_client):
    """Create TimeSeries DAO for testing."""
    return TimeSeriesDao(redis_client)


@pytest.fixture
def sample_reading():
    """Create a sample meter reading."""
    return MeterReading(
        site_id=1,
        timestamp=datetime(2024, 1, 15, 12, 0, 0),
        wh_generated=150.5,
        wh_used=75.2,
        temp_c=25.5
    )


@pytest.fixture
def sample_readings():
    """Create multiple sample readings."""
    base_time = datetime(2024, 1, 15, 12, 0, 0)
    readings = []
    
    for i in range(24):  # 24 hours of readings
        timestamp = base_time + timedelta(hours=i)
        readings.append(MeterReading(
            site_id=1,
            timestamp=timestamp,
            wh_generated=100.0 + (i * 10),  # Increasing generation
            wh_used=50.0 + (i * 2),        # Increasing usage
            temp_c=20.0 + i                # Increasing temperature
        ))
    
    return readings


class TestTimeSeriesDao:
    """Test cases for TimeSeries DAO."""
    
    def test_add_single_reading(self, timeseries_dao, sample_reading):
        """Test adding a single meter reading."""
        timeseries_dao.add_reading(sample_reading)
        
        # Verify we can get the reading back
        latest = timeseries_dao.get_latest_reading(1, MetricType.WH_GENERATED)
        assert latest is not None
        assert latest.value == sample_reading.wh_generated
    
    def test_add_readings_batch(self, timeseries_dao, sample_readings):
        """Test adding multiple readings in batch."""
        timeseries_dao.add_readings_batch(sample_readings)
        
        # Verify all readings were added
        start_time = sample_readings[0].timestamp
        end_time = sample_readings[-1].timestamp + timedelta(minutes=1)
        
        generation_data = timeseries_dao.get_readings_range(
            1, MetricType.WH_GENERATED, start_time, end_time
        )
        
        assert len(generation_data) == len(sample_readings)
    
    def test_get_readings_range(self, timeseries_dao, sample_readings):
        """Test getting readings within a time range."""
        timeseries_dao.add_readings_batch(sample_readings)
        
        # Get readings for first 12 hours
        start_time = sample_readings[0].timestamp
        mid_time = start_time + timedelta(hours=12)
        
        data = timeseries_dao.get_readings_range(
            1, MetricType.WH_GENERATED, start_time, mid_time
        )
        
        # Should get 12 readings (0-11 hours)
        assert len(data) == 12
        assert data[0].value == 100.0  # First reading
        assert data[-1].value == 210.0  # 11th reading (100 + 11*10)
    
    def test_get_latest_reading(self, timeseries_dao, sample_readings):
        """Test getting the latest reading."""
        timeseries_dao.add_readings_batch(sample_readings)
        
        latest = timeseries_dao.get_latest_reading(1, MetricType.TEMP_C)
        assert latest is not None
        
        # Should be the last reading's temperature
        expected_temp = sample_readings[-1].temp_c
        assert latest.value == expected_temp
    
    def test_get_aggregated_data(self, timeseries_dao, sample_readings):
        """Test getting aggregated data."""
        timeseries_dao.add_readings_batch(sample_readings)
        
        start_time = sample_readings[0].timestamp
        end_time = sample_readings[-1].timestamp + timedelta(minutes=1)
        
        # Get hourly averages
        aggregated = timeseries_dao.get_aggregated_data(
            1, MetricType.WH_GENERATED, start_time, end_time,
            bucket_size_ms=3600000,  # 1 hour
            aggregation='avg'
        )
        
        # Should have aggregated data points
        assert len(aggregated) > 0
    
    def test_get_site_analytics(self, timeseries_dao, sample_readings):
        """Test getting site analytics."""
        timeseries_dao.add_readings_batch(sample_readings)
        
        start_time = sample_readings[0].timestamp
        end_time = sample_readings[-1].timestamp + timedelta(minutes=1)
        
        analytics = timeseries_dao.get_site_analytics(1, start_time, end_time)
        
        assert analytics is not None
        assert analytics.site_id == 1
        assert analytics.reading_count == len(sample_readings)
        assert analytics.total_generated > 0
        assert analytics.avg_generation > 0
    
    def test_nonexistent_site_data(self, timeseries_dao):
        """Test getting data for non-existent site."""
        now = datetime.now()
        yesterday = now - timedelta(days=1)
        
        data = timeseries_dao.get_readings_range(999, MetricType.WH_GENERATED, yesterday, now)
        assert data == []
        
        latest = timeseries_dao.get_latest_reading(999, MetricType.WH_GENERATED)
        assert latest is None
        
        analytics = timeseries_dao.get_site_analytics(999, yesterday, now)
        assert analytics is None


### Create tests/test_leaderboard_dao.py
```python
"""Tests for Leaderboard DAO implementation."""

import pytest
import redis

from redisolar.dao.leaderboard_dao import LeaderboardDao
from redisolar.dao.site_dao_redis import SiteDaoRedis
from redisolar.models.site import Site, Coordinate
from redisolar.config import Config


@pytest.fixture
def redis_client():
    """Create Redis client for testing."""
    config = Config(
        redis_host='localhost',
        redis_port=6379,
        redis_db=1,
        redis_key_prefix='test:'
    )
    
    client = redis.Redis(
        host=config.redis_host,
        port=config.redis_port,
        db=config.redis_db,
        decode_responses=True
    )
    
    client.flushdb()
    yield client
    client.flushdb()


@pytest.fixture
def site_dao(redis_client):
    """Create Site DAO for testing."""
    return SiteDaoRedis(redis_client)


@pytest.fixture
def leaderboard_dao(redis_client, site_dao):
    """Create Leaderboard DAO for testing."""
    return LeaderboardDao(redis_client, site_dao)


@pytest.fixture
def sample_sites():
    """Create sample sites with different capacities."""
    return [
        Site(id=1, capacity=4.5, panels=12, address="123 St", city="City1", state="CA", postal_code="90210"),
        Site(id=2, capacity=6.0, panels=20, address="456 St", city="City2", state="TX", postal_code="75001"),
        Site(id=3, capacity=3.2, panels=8, address="789 St", city="City3", state="FL", postal_code="33101"),
        Site(id=4, capacity=7.8, panels=26, address="321 St", city="City4", state="AZ", postal_code="85001"),
    ]


class TestLeaderboardDao:
    """Test cases for Leaderboard DAO."""
    
    def test_update_capacity_leaderboard(self, leaderboard_dao, site_dao, sample_sites):
        """Test updating capacity leaderboard."""
        # Insert sites
        site_dao.insert_many(*sample_sites)
        
        # Update leaderboard
        for site in sample_sites:
            leaderboard_dao.update_capacity_leaderboard(site.id, site.capacity)
        
        # Get leaders
        leaders = leaderboard_dao.get_capacity_leaders(10)
        
        # Should be ordered by capacity (highest first)
        assert len(leaders) == 4
        assert leaders[0].site.id == 4  # 7.8 capacity
        assert leaders[1].site.id == 2  # 6.0 capacity
        assert leaders[2].site.id == 1  # 4.5 capacity
        assert leaders[3].site.id == 3  # 3.2 capacity
    
    def test_get_site_rank(self, leaderboard_dao, site_dao, sample_sites):
        """Test getting site's rank in leaderboard."""
        site_dao.insert_many(*sample_sites)
        
        # Update capacity leaderboard
        for site in sample_sites:
            leaderboard_dao.update_capacity_leaderboard(site.id, site.capacity)
        
        # Check ranks
        assert leaderboard_dao.get_site_rank(4, 'capacity') == 1  # Highest capacity
        assert leaderboard_dao.get_site_rank(2, 'capacity') == 2
        assert leaderboard_dao.get_site_rank(1, 'capacity') == 3
        assert leaderboard_dao.get_site_rank(3, 'capacity') == 4  # Lowest capacity
    
    def test_get_site_score(self, leaderboard_dao, site_dao, sample_sites):
        """Test getting site's score in leaderboard."""
        site_dao.insert_many(*sample_sites)
        
        # Update capacity leaderboard
        for site in sample_sites:
            leaderboard_dao.update_capacity_leaderboard(site.id, site.capacity)
        
        # Check scores
        assert leaderboard_dao.get_site_score(4, 'capacity') == 7.8
        assert leaderboard_dao.get_site_score(2, 'capacity') == 6.0
        assert leaderboard_dao.get_site_score(1, 'capacity') == 4.5
        assert leaderboard_dao.get_site_score(3, 'capacity') == 3.2
    
    def test_generation_leaderboard(self, leaderboard_dao, site_dao, sample_sites):
        """Test generation leaderboard functionality."""
        site_dao.insert_many(*sample_sites)
        
        # Update generation leaderboard with different values
        generation_scores = {1: 1500.0, 2: 2000.0, 3: 800.0, 4: 2200.0}
        
        for site_id, generation in generation_scores.items():
            leaderboard_dao.update_generation_leaderboard(site_id, generation)
        
        # Get leaders
        leaders = leaderboard_dao.get_generation_leaders(10)
        
        # Should be ordered by generation (highest first)
        assert len(leaders) == 4
        assert leaders[0].site.id == 4  # 2200.0 generation
        assert leaders[1].site.id == 2  # 2000.0 generation
        assert leaders[2].site.id == 1  # 1500.0 generation
        assert leaders[3].site.id == 3  # 800.0 generation
    
    def test_remove_site_from_leaderboards(self, leaderboard_dao, site_dao, sample_sites):
        """Test removing site from all leaderboards."""
        site_dao.insert_many(*sample_sites)
        
        # Add sites to leaderboards
        for site in sample_sites:
            leaderboard_dao.update_capacity_leaderboard(site.id, site.capacity)
            leaderboard_dao.update_generation_leaderboard(site.id, 1000.0)
        
        # Remove site 2
        leaderboard_dao.remove_site_from_leaderboards(2)
        
        # Check that site 2 is removed from capacity leaderboard
        leaders = leaderboard_dao.get_capacity_leaders(10)
        site_ids = [leader.site.id for leader in leaders]
        assert 2 not in site_ids
        assert len(leaders) == 3
    
    def test_rebuild_capacity_leaderboard(self, leaderboard_dao, site_dao, sample_sites):
        """Test rebuilding capacity leaderboard from site data."""
        site_dao.insert_many(*sample_sites)
        
        # Rebuild leaderboard
        count = leaderboard_dao.rebuild_capacity_leaderboard()
        assert count == len(sample_sites)
        
        # Verify leaderboard is correct
        leaders = leaderboard_dao.get_capacity_leaders(10)
        assert len(leaders) == 4
        assert leaders[0].site.id == 4  # Highest capacity
    
    def test_get_leaderboard_stats(self, leaderboard_dao, site_dao, sample_sites):
        """Test getting leaderboard statistics."""
        site_dao.insert_many(*sample_sites)
        
        # Update capacity leaderboard
        for site in sample_sites:
            leaderboard_dao.update_capacity_leaderboard(site.id, site.capacity)
        
        # Get stats
        stats = leaderboard_dao.get_leaderboard_stats('capacity')
        
        assert stats['total_sites'] == 4
        assert stats['highest_score'] == 7.8
        assert stats['lowest_score'] == 3.2
        assert stats['average_score'] == (4.5 + 6.0 + 3.2 + 7.8) / 4
    
    def test_get_sites_around_rank(self, leaderboard_dao, site_dao, sample_sites):
        """Test getting sites around a specific rank."""
        site_dao.insert_many(*sample_sites)
        
        # Update capacity leaderboard
        for site in sample_sites:
            leaderboard_dao.update_capacity_leaderboard(site.id, site.capacity)
        
        # Get sites around rank 2 (site 2 with 6.0 capacity)
        surrounding = leaderboard_dao.get_sites_around_rank(2, 'capacity', context_size=1)
        
        # Should get sites ranked 1, 2, 3
        assert len(surrounding) == 3
        ranks = [entry.rank for entry in surrounding]
        assert 1 in ranks
        assert 2 in ranks
        assert 3 in ranks
```

---

## Step 11: API Testing Examples

### Create test_api_calls.py (for manual testing)
```python
"""Manual API testing examples."""

import requests
import json
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8081"

def test_meter_readings():
    """Test meter reading endpoints."""
    
    # Add a single reading
    reading_data = {
        "timestamp": datetime.now().isoformat(),
        "wh_generated": 150.5,
        "wh_used": 75.2,
        "temp_c": 25.5
    }
    
    response = requests.post(
        f"{BASE_URL}/api/analytics/readings/1",
        json=reading_data
    )
    print(f"Add reading: {response.status_code} - {response.json()}")
    
    # Get latest readings
    response = requests.get(f"{BASE_URL}/api/analytics/readings/1/latest")
    print(f"Latest readings: {response.status_code} - {response.json()}")
    
    # Get time-series data
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=24)
    
    response = requests.get(
        f"{BASE_URL}/api/analytics/readings/1/wh_generated",
        params={
            "start": start_time.isoformat(),
            "end": end_time.isoformat()
        }
    )
    print(f"Time-series data: {response.status_code} - Data points: {len(response.json().get('data', []))}")


def test_leaderboards():
    """Test leaderboard endpoints."""
    
    # Get capacity leaderboard
    response = requests.get(f"{BASE_URL}/api/analytics/leaderboards/capacity")
    print(f"Capacity leaderboard: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"  Top sites: {len(data['entries'])}")
        for entry in data['entries'][:3]:
            print(f"    {entry['rank']}. Site {entry['site']['id']}: {entry['score']}kW")
    
    # Get generation leaderboard
    response = requests.get(f"{BASE_URL}/api/analytics/leaderboards/generation")
    print(f"Generation leaderboard: {response.status_code}")
    
    # Get site's position
    response = requests.get(f"{BASE_URL}/api/analytics/leaderboards/capacity/site/1")
    print(f"Site 1 position: {response.status_code} - {response.json()}")


def test_analytics():
    """Test analytics endpoints."""
    
    # Get site analytics
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=24)
    
    response = requests.get(
        f"{BASE_URL}/api/analytics/analytics/1",
        params={
            "start": start_time.isoformat(),
            "end": end_time.isoformat()
        }
    )
    print(f"Site analytics: {response.status_code} - {response.json()}")
    
    # Get overall summary
    response = requests.get(f"{BASE_URL}/api/analytics/summary")
    print(f"Analytics summary: {response.status_code} - {response.json()}")


if __name__ == "__main__":
    print("Testing Redis Solar Analytics API...")
    print("="*50)
    
    print("\n1. Testing meter readings:")
    test_meter_readings()
    
    print("\n2. Testing leaderboards:")
    test_leaderboards()
    
    print("\n3. Testing analytics:")
    test_analytics()
    
    print("\nDone!")
```

---

## Step 12: Running and Testing

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
requests==2.31.0
```

### Update Makefile
```makefile
.PHONY: help env dev test load clean timeseries-docker

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

load:		## Load sample data with time-series
	env/bin/python scripts/load_sample_data.py

clean:		## Clean up Redis test data
	redis-cli -n 1 FLUSHDB

timeseries-docker:	## Start Redis with TimeSeries module using Docker
	docker run -d \
		--name redis-redisolar-ts \
		-p 6379:6379 \
		redis/redis-stack:latest

redis-stop:	## Stop Redis Docker container
	docker stop redis-redisolar-ts && docker rm redis-redisolar-ts

test-api:	## Test API endpoints manually
	env/bin/python test_api_calls.py
```

### Running Everything

1. **Start Redis with TimeSeries support**:
```bash
make timeseries-docker
```

2. **Set up environment**:
```bash
make env
```

3. **Load sample data with time-series**:
```bash
make load
```

4. **Run the application**:
```bash
make dev
```

5. **Test the new endpoints**:
```bash
# Test analytics summary
curl http://localhost:8081/api/analytics/summary

# Test capacity leaderboard
curl http://localhost:8081/api/analytics/leaderboards/capacity

# Test site analytics for last 24 hours
curl "http://localhost:8081/api/analytics/analytics/1"

# Test latest readings
curl http://localhost:8081/api/analytics/readings/1/latest

# Add a new reading
curl -X POST http://localhost:8081/api/analytics/readings/1 \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2024-01-20T12:00:00",
    "wh_generated": 200.0,
    "wh_used": 80.0,
    "temp_c": 28.5
  }'

# Get time-series data for generated energy
curl "http://localhost:8081/api/analytics/readings/1/wh_generated?start=2024-01-19T00:00:00&end=2024-01-20T00:00:00"
```

6. **Run comprehensive tests**:
```bash
make test
```

---

## 🎉 Lesson 3 Complete!

You've successfully implemented:
- ✅ **Redis TimeSeries**: High-performance time-series data storage for meter readings
- ✅ **Redis Sorted Sets**: Leaderboards for capacity, generation, and efficiency rankings
- ✅ **Real-time Analytics**: Site analytics, aggregations, and statistical calculations
- ✅ **Background Tasks**: Automated leaderboard updates with threading
- ✅ **Advanced Queries**: Time-range queries, aggregations, and latest data retrieval
- ✅ **Comprehensive API**: Full REST API for analytics and leaderboard operations

### Key Redis Concepts Mastered
1. **TimeSeries Module**: Purpose-built for time-based data with retention policies
2. **Sorted Sets (ZSET)**: Perfect for rankings and leaderboards with automatic ordering
3. **Pipelines**: Batch operations for performance optimization
4. **Key Patterns**: Structured naming for different data types and relationships
5. **Data Aggregation**: Built-in TimeSeries aggregation functions (avg, sum, min, max)
6. **Background Processing**: Asynchronous updates and maintenance tasks

### What's Next?
In **Lesson 4**, we'll add:
- **Geospatial Features**: Redis GEO for location-based queries
- **Full-Text Search**: RediSearch for advanced site searching
- **Rate Limiting**: API rate limiting with Redis
- **Caching Strategies**: Smart caching patterns for performance
- **Lua Scripts**: Custom Redis scripts for complex operations

### Key Achievements
- **Performance**: Efficient time-series storage and retrieval
- **Real-time**: Live leaderboards and analytics
- **Scalability**: Batch operations and background processing
- **Analytics**: Rich statistical analysis and reporting
- **Architecture**: Clean separation of concerns with specialized DAOs

Ready for Lesson 4 where we'll dive into geospatial features and advanced Redis patterns?
```

---

## Step 6: Enhanced Schema Validation

### Update redisolar/models/schemas.py
```python
"""Marshmallow schemas for data validation and serialization."""

from marshmallow import Schema, fields, post_load, validate, ValidationError
from datetime import datetime
from typing import Dict, Any

from .site import Site, SiteStats, MeterReading, Coordinate, TimeSeriesData, SiteAnalytics, LeaderboardEntry


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


class SiteAnalyticsSchema(Schema):
    """Schema for SiteAnalytics model."""
    
    site_id = fields.Integer(required=True)
    period_start = fields.DateTime(required=True)
    period_end = fields.DateTime(required=True)
    total_generated = fields.Float(required=True)
    total_used = fields.Float(required=True)
    avg_generation = fields.Float(required=True)
    max_generation = fields.Float(required=True)
    min_generation = fields.Float(required=True)
    avg_temperature = fields.Float(required=True)
    efficiency_percent = fields.Float(required=True)
    reading_count = fields.Integer(required=True)


class LeaderboardEntrySchema(Schema):
    """Schema for LeaderboardEntry model."""
    
    site = fields.Nested(SiteSchema, required=True)
    score = fields.Float(required=True)
    rank = fields.Integer(required=True)


# Schema instances
site_schema = SiteSchema()
meter_reading_schema = MeterReadingSchema()
timeseries_data_schema = TimeSeriesDataSchema()
site_analytics_schema = SiteAnalyticsSchema()
leaderboard_entry_schema = LeaderboardEntrySchema()
coordinate_schema = CoordinateSchema()
```

---

## Step 7: API Routes for Analytics and Leaderboards

### Create redisolar/api/analytics_routes.py
```python
"""API routes for analytics and time-series data."""

from flask import Blueprint, request, jsonify
from marshmallow import ValidationError
from datetime import datetime, timedelta
from typing import Dict, Any

from redisolar.dao.timeseries_dao import TimeSeriesDao
from redisolar.dao.leaderboard_dao import LeaderboardDao
from redisolar.dao.site_dao_redis import SiteDaoRedis
from redisolar.models.schemas import (
    meter_reading_schema, timeseries_data_schema, 
    site_analytics_schema, leaderboard_entry_schema
)
from redisolar.models.site import MetricType
from redisolar.tasks.leaderboard_updater import leaderboard_updater

# Create blueprint
analytics_bp = Blueprint('analytics', __name__, url_prefix='/api/analytics')

# Initialize DAOs
timeseries_dao = TimeSeriesDao()
leaderboard_dao = LeaderboardDao()
site_dao = SiteDaoRedis()


@analytics_bp.route('/readings/<int:site_id>', methods=['POST'])
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
        
        return jsonify({
            'message': f'Reading added for site {site_id}',
            'reading': meter_reading_schema.dump(reading)
        }), 201
    
    except ValidationError as e:
        return jsonify({'error': 'Validation failed', 'details': e.messages}), 400
    
    except Exception as e:
        return jsonify({'error': f'Failed to add reading: {str(e)}'}), 500


@analytics_bp.route('/readings/<int:site_id>/batch', methods=['POST'])
def add_meter_readings_batch(site_id: int):
    """Add multiple meter readings for a site."""
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
                return jsonify({
                    'error': f'Validation failed for reading {i}',
                    'details': e.messages
                }), 400
        
        # Add all readings
        timeseries_dao.add_readings_batch(readings)
        
        return jsonify({
            'message': f'Added {len(readings)} readings for site {site_id}',
            'count': len(readings)
        }), 201
    
    except Exception as e:
        return jsonify({'error': f'Failed to add readings: {str(e)}'}), 500


@analytics_bp.route('/readings/<int:site_id>/<metric>', methods=['GET'])
def get_site_readings(site_id: int, metric: str):
    """Get time-series readings for a site and metric."""
    try:
        # Validate metric type
        try:
            metric_type = MetricType(metric)
        except ValueError:
            return jsonify({'error': f'Invalid metric: {metric}'}), 400
        
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
                return jsonify({'error': 'Invalid date format. Use ISO format.'}), 400
        else:
            # Default to last 24 hours
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=24)
        
        # Get aggregation parameter
        aggregation = request.args.get('aggregation', 'raw')
        
        if aggregation == 'raw':
            # Get raw data
            data_points = timeseries_dao.get_readings_range(site_id, metric_type, start_time, end_time)
        else:
            # Get aggregated data
            bucket_size = int(request.args.get('bucket_size', 3600000))  # 1 hour default
            data_points = timeseries_dao.get_aggregated_data(
                site_id, metric_type, start_time, end_time, bucket_size, aggregation
            )
        
        # Serialize data
        data = [timeseries_data_schema.dump(dp) for dp in data_points]
        
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
        return jsonify({'error': str(e)}), 500


@analytics_bp.route('/readings/<int:site_id>/latest', methods=['GET'])
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
        
        return jsonify({
            'site_id': site_id,
            'latest_readings': latest_readings
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@analytics_bp.route('/analytics/<int:site_id>', methods=['GET'])
def get_site_analytics(site_id: int):
    """Get analytics for a site over a time period."""
    try:
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
                return jsonify({'error': 'Invalid date format. Use ISO format.'}), 400
        else:
            # Default to last 24 hours
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=24)
        
        # Get analytics
        analytics = timeseries_dao.get_site_analytics(site_id, start_time, end_time)
        
        if not analytics:
            return jsonify({
                'message': f'No data available for site {site_id} in the specified time range'
            }), 404
        
        return jsonify(site_analytics_schema.dump(analytics))
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@analytics_bp.route('/leaderboards/<leaderboard_type>', methods=['GET'])
def get_leaderboard(leaderboard_type: str):
    """Get leaderboard rankings."""
    try:
        # Validate leaderboard type
        valid_types = ['capacity', 'generation', 'efficiency']
        if leaderboard_type not in valid_types:
            return jsonify({'error': f'Invalid leaderboard type. Must be one of: {valid_types}'}), 400
        
        # Get limit from query parameter
        limit = min(int(request.args.get('limit', 10)), 100)  # Max 100
        
        # Get leaderboard entries
        if leaderboard_type == 'capacity':
            entries = leaderboard_dao.get_capacity_leaders(limit)
        elif leaderboard_type == 'generation':
            entries = leaderboard_dao.get_generation_leaders(limit)
        elif leaderboard_type == 'efficiency':
            entries = leaderboard_dao.get_efficiency_leaders(limit)
        
        # Serialize entries
        entries_data = [leaderboard_entry_schema.dump(entry) for entry in entries]
        
        # Get leaderboard stats
        stats = leaderboard_dao.get_leaderboard_stats(leaderboard_type)
        
        return jsonify({
            'leaderboard_type': leaderboard_type,
            'entries': entries_data,
            'stats': stats
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@analytics_bp.route('/leaderboards/<leaderboard_type>/site/<int:site_id>', methods=['GET'])
def get_site_leaderboard_info(leaderboard_type: str, site_id: int):
    """Get a site's position and nearby sites in a leaderboard."""
    try:
        # Validate leaderboard type
        valid_types = ['capacity', 'generation', 'efficiency']
        if leaderboard_type not in valid_types:
            return jsonify({'error': f'Invalid leaderboard type. Must be one of: {valid_types}'}), 400
        
        # Check if site exists
        if not site_dao.exists(site_id):
            return jsonify({'error': f'Site {site_id} not found'}), 404
        
        # Get site's rank and score
        rank = leaderboard_dao.get_site_rank(site_id, leaderboard_type)
        score = leaderboard_dao.get_site_score(site_id, leaderboard_type)
        
        if rank is None:
            return jsonify({
                'message': f'Site {site_id} not found in {leaderboard_type} leaderboard'
            }), 404
        
        # Get surrounding sites
        context_size = int(request.args.get('context', 2))
        surrounding = leaderboard_dao.get_sites_around_rank(site_id, leaderboard_type, context_size)
        
        return jsonify({
            'site_id': site_id,
            'leaderboard_type': leaderboard_type,
            'rank': rank,
            'score': score,
            'surrounding_sites': [leaderboard_entry_schema.dump(entry) for entry in surrounding]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@analytics_bp.route('/leaderboards/update', methods=['POST'])
def update_leaderboards():
    """Manually trigger leaderboard update."""
    try:
        leaderboard_updater.update_all_leaderboards()
        return jsonify({'message': 'Leaderboards updated successfully'})
    
    except Exception as e:
        return jsonify({'error': f'Failed to update leaderboards: {str(e)}'}), 500


@analytics_bp.route('/summary', methods=['GET'])
def get_analytics_summary():
    """Get overall analytics summary."""
    try:
        # Get time range
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)
        
        # Get all sites
        sites = site_dao.find_all()
        
        if not sites:
            return jsonify({
                'message': 'No sites available',
                'total_sites': 0
            })
        
        # Calculate summary statistics
        total_capacity = sum(site.capacity for site in sites)
        total_sites = len(sites)
        
        # Get generation data for all sites
        total_generation_24h = 0
        active_sites = 0
        
        for site in sites:
            generation_data = timeseries_dao.get_readings_range(
                site.id, MetricType.WH_GENERATED, start_time, end_time
            )
            if generation_data:
                site_generation = sum(dp.value for dp in generation_data)
                total_generation_24h += site_generation
                active_sites += 1
        
        # Calculate efficiency
        max_possible_generation = total_capacity * 24 * 1000  # 24 hours * 1000 Wh/kWh
        overall_efficiency = (total_generation_24h / max_possible_generation * 100) if max_possible_generation > 0 else 0
        
        return jsonify({
            'summary': {
                'total_sites': total_sites,
                'active_sites_24h': active_sites,
                'total_capacity_kw': total_capacity,
                'total_generation_24h_wh': total_generation_24h,
                'overall_efficiency_percent': overall_efficiency,
                'period_start': start_time.isoformat(),
                'period_end': end_time.isoformat()
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

---

## Step 8: Update Main Application

### Update redisolar/app.py
```python
"""Main Flask application."""

import os
import atexit
from flask import Flask, jsonify
from redisolar.dao.redis_dao import RedisConnection, get_redis
from redisolar.tasks.leaderboard_updater import leaderboard_updater


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
    
    # Start background services
    if config_name == 'dev':
        leaderboard_updater.start(update_interval=300)  # 5 minutes
        
        # Stop background services on app shutdown
        @atexit.register
        def cleanup():
            leaderboard_updater.stop()
    
    # Register blueprints
    from redisolar.api.routes import sites_bp
    from redisolar.api.analytics_routes import analytics_bp
    app.register_blueprint(sites_bp)
    app.register_blueprint(analytics_bp)
    
    # Register routes
    @app.route('/')
    def index():
        """Health check endpoint."""
        return jsonify({
            'message': 'RediSolar API is running!',
            'redis_connected': RedisConnection.test_connection(),
            'redis_info': get_redis().info('server')['redis_version'],
            'endpoints': {
                'sites': [
                    'GET /api/sites - Get all sites',
                    'GET /api/sites/<id> - Get specific site',
                    'POST /api/sites - Create new site',
                    'PUT /api/sites/<id> - Update site',
                    'DELETE /api/sites/<id> - Delete site',
                    'GET /api/sites/search - Search sites',
                    'GET /api/sites/stats - Get site statistics'
                ],
                'analytics': [
                    'POST /api/analytics/readings/<site_id> - Add meter reading',
                    'POST /api/analytics/readings/<site_id>/batch - Add multiple readings',
                    'GET /api/analytics/readings/<site_id>/<metric> - Get time-series data',
                    'GET /api/analytics/readings/<site_id>/latest - Get latest readings',
                    'GET /api/analytics/analytics/<site_id> - Get site analytics',
                    'GET /api/analytics/leaderboards/<type> - Get leaderboard',
                    'GET /api/analytics/summary - Get overall summary'
                ]
            }
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
            },
            'background_services': {
                'leaderboard_updater': leaderboard_updater.is_running
            }
        })
    
    return app


# For development server
if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=8081, debug=True)
```

---

## Step 9: Enhanced Sample Data with Meter Readings

### Update scripts/load_sample_data.py
```python
"""Load sample data into Redis for development and testing."""

import sys
import os
import random
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redisolar.dao.site_dao_redis import SiteDaoRedis
from redisolar.dao.timeseries_dao import TimeSeriesDao
from redisolar.dao.leaderboard_dao import LeaderboardDao
from redisolar.models.site import Site, Coordinate, MeterReading
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
    
    return sites


def generate_meter_readings(sites, days=7):
    """Generate sample meter readings for sites."""
    readings = []
    end_time = datetime.now()
    
    for site in sites:
        # Generate readings for the last N days
        for day in range(days):
            day_start = end_time - timedelta(days=day)
            
            # Generate 24 hourly readings for each day
            for hour in range(24):
                timestamp = day_start.replace(hour=hour, minute=0, second=0, microsecond=0)
                
                # Simulate solar generation pattern (more during day, peak at noon)
                hour_factor = max(0, 1 - abs(hour - 12) / 12)  # Peak at noon
                base_generation = site.capacity * 800 * hour_factor  # Base Wh
                
                # Add some randomness
                generation_variance = random.uniform(0.7, 1.3)
                wh_generated = base_generation * generation_variance
                
                # Usage is more consistent but varies
                base_usage = site.capacity * 200  # Base usage
                usage_variance = random.uniform(0.8, 1.2)
                wh_used = base_usage * usage_variance
                
                # Temperature varies throughout day
                base_temp = 20 + (hour_factor * 15) + random.uniform(-5, 5)
                
                reading = MeterReading(
                    site_id=site.id,
                    timestamp=timestamp,
                    wh_generated=max(0, wh_generated),
                    wh_used=max(0, wh_used),
                    temp_c=base_temp
                )
                
                readings.append(reading)
    
    return readings


def main():
    """Main function."""
    print("Loading sample solar sites and meter readings into Redis...")
    
    # Test Redis connection
    if not RedisConnection.test_connection():
        print("Error: Could not connect to Redis. Please ensure Redis is running.")
        return
    
    # Initialize DAOs
    site_dao = SiteDaoRedis()
    timeseries_dao = TimeSeriesDao()
    leaderboard_dao = LeaderboardDao()
    
    try:
        # Check if data already exists
        if site_dao.count() > 0:
            print(f"Warning: {site_dao.count()} sites already exist in Redis.")
            response = input("Do you want to clear existing data and reload? (y/N): ")
            if response.lower() != 'y':
                print("Aborted.")
                return
            
            # Clear existing sites and their data
            all_sites = site_dao.find_all()
            for site in all_sites:
                timeseries_dao.delete_site_data(site.id)
                leaderboard_dao.remove_site_from_leaderboards(site.id)
                site_dao.delete(site.id)
            print(f"Cleared {len(all_sites)} existing sites and their data.")
        
        # Load sites
        sites = load_sample_sites()
        site_dao.insert_many(*sites)
        print(f"Successfully loaded {len(sites)} sites into Redis.")
        
        # Generate and load meter readings
        print("Generating sample meter readings (this may take a moment)...")
        readings = generate_meter_readings(sites, days=7)
        
        # Load readings in batches for better performance
        batch_size = 100
        total_batches = (len(readings) + batch_size - 1) // batch_size# Redis + Python Development Course
## Lesson 3: Time-Series Data and Leaderboards

### Lesson Overview
In this lesson, we'll implement time-series data storage for meter readings and create leaderboards using Redis Sorted Sets. We'll learn how to efficiently store and query time-based data, implement ranking systems, and build analytics features for our solar monitoring application.

### What You'll Learn
- Redis TimeSeries module for high-performance time-series data
- Redis Sorted Sets for leaderboards and rankings
- Time-based data querying and aggregation
- Performance optimization with Redis pipelines
- Real-time analytics and statistics
- Data retention and cleanup strategies

---

## Lesson 3 Objectives

By the end of this lesson, you'll have:
- ✅ Implemented meter reading storage with Redis TimeSeries
- ✅ Created capacity and performance leaderboards with Sorted Sets
- ✅ Built time-range queries for historical data
- ✅ Implemented real-time analytics endpoints
- ✅ Added data aggregation and statistical functions
- ✅ Created background tasks for leaderboard updates

---

## Step 1: Understanding Redis Data Structures for Analytics

### Redis TimeSeries
Redis TimeSeries is perfect for storing meter readings with timestamps:
```bash
# TimeSeries commands
TS.CREATE meter:site:1:wh_generated RETENTION 86400000 LABELS site_id 1 metric wh_generated
TS.ADD meter:site:1:wh_generated 1642694400000 150.5
TS.RANGE meter:site:1:wh_generated 1642694400000 1642780800000
```

### Redis Sorted Sets
Sorted Sets are ideal for leaderboards and rankings:
```bash
# Sorted Set commands for leaderboard
ZADD capacity_leaderboard 4.5 site:1 6.0 site:2 3.2 site:3
ZREVRANGE capacity_leaderboard 0 9 WITHSCORES  # Top 10
ZRANK capacity_leaderboard site:1  # Get rank
```

---

## Step 2: Enhanced Models for Time-Series Data

### Update redisolar/models/site.py
```python
"""Domain models for solar sites and related data."""

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


@dataclass(frozen=True, eq=True)
class SiteStats:
    """Statistics for a solar site."""
    
    site_id: int
    last_reporting_time: datetime
    meter_reading_count: int
    max_capacity: float
    min_capacity: float
    total_energy_wh: float  # Total Watt-hours generated
    current_capacity: float  # Current capacity utilization %
    
    def __post_init__(self):
        """Validate stats data."""
        if self.meter_reading_count < 0:
            raise ValueError("Reading count cannot be negative")
        if self.max_capacity < self.min_capacity:
            raise ValueError("Max capacity cannot be less than min capacity")


class MetricType(Enum):
    """Types of metrics we track."""
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


@dataclass(frozen=True)
class SiteRanking:
    """Site ranking data for leaderboards."""
    site_id: int
    score: float
    rank: int
    site: Optional[Site] = None


@dataclass(frozen=True)
class LeaderboardEntry:
    """Leaderboard entry with site information."""
    site: Site
    score: float
    rank: int


@dataclass(frozen=True)
class SiteAnalytics:
    """Analytics data for a site over a time period."""
    site_id: int
    period_start: datetime
    period_end: datetime
    total_generated: float
    total_used: float
    avg_generation: float
    max_generation: float
    min_generation: float
    avg_temperature: float
    efficiency_percent: float  # Generated vs capacity utilization
    reading_count: int
```

---

## Step 3: Time-Series DAO Implementation

### Create redisolar/dao/timeseries_dao.py
```python
"""Time-Series Data Access Object for meter readings."""

import redis
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import json

from .redis_dao import get_redis
from .key_manager import key_manager
from redisolar.models.site import MeterReading, TimeSeriesData, MetricType, SiteAnalytics


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
        except redis.ResponseError:
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
    
    def add_reading(self, reading: MeterReading) -> None:
        """Add a meter reading to time series."""
        # Add all metrics for this reading
        metrics = [
            (MetricType.WH_GENERATED, reading.wh_generated),
            (MetricType.WH_USED, reading.wh_used),
            (MetricType.TEMP_C, reading.temp_c),
            (MetricType.NET_GENERATION, reading.net_generation)
        ]
        
        # Use pipeline for atomic operations
        pipe = self.redis.pipeline()
        
        for metric_type, value in metrics:
            # Ensure TimeSeries exists
            self._ensure_timeseries_exists(reading.site_id, metric_type)
            
            # Add data point
            ts_key = self._get_ts_key(reading.site_id, metric_type)
            pipe.execute_command('TS.ADD', ts_key, reading.timestamp_millis, value)
        
        # Store the full reading as JSON for easy retrieval
        reading_key = f"{self.key_manager.key_prefix}reading:{reading.site_id}:{reading.timestamp_millis}"
        reading_data = {
            'site_id': reading.site_id,
            'timestamp': reading.timestamp.isoformat(),
            'wh_generated': reading.wh_generated,
            'wh_used': reading.wh_used,
            'temp_c': reading.temp_c,
            'net_generation': reading.net_generation
        }
        pipe.setex(reading_key, 86400 * 30, json.dumps(reading_data))  # 30 days TTL
        
        pipe.execute()
    
    def add_readings_batch(self, readings: List[MeterReading]) -> None:
        """Add multiple readings efficiently."""
        if not readings:
            return
        
        # Group readings by site for efficiency
        readings_by_site = {}
        for reading in readings:
            if reading.site_id not in readings_by_site:
                readings_by_site[reading.site_id] = []
            readings_by_site[reading.site_id].append(reading)
        
        # Use pipeline for all operations
        pipe = self.redis.pipeline()
        
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
                
                # Store full reading
                reading_key = f"{self.key_manager.key_prefix}reading:{reading.site_id}:{reading.timestamp_millis}"
                reading_data = {
                    'site_id': reading.site_id,
                    'timestamp': reading.timestamp.isoformat(),
                    'wh_generated': reading.wh_generated,
                    'wh_used': reading.wh_used,
                    'temp_c': reading.temp_c,
                    'net_generation': reading.net_generation
                }
                pipe.setex(reading_key, 86400 * 30, json.dumps(reading_data))
        
        pipe.execute()
    
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
        
        try:
            result = self.redis.execute_command('TS.RANGE', ts_key, start_millis, end_millis)
            
            data_points = []
            for timestamp_millis, value in result:
                timestamp = datetime.fromtimestamp(timestamp_millis / 1000)
                data_points.append(TimeSeriesData(timestamp=timestamp, value=float(value)))
            
            return data_points
        
        except redis.ResponseError:
            # TimeSeries doesn't exist
            return []
    
    def get_latest_reading(self, site_id: int, metric: MetricType) -> Optional[TimeSeriesData]:
        """Get the latest reading for a site and metric."""
        ts_key = self._get_ts_key(site_id, metric)
        
        try:
            result = self.redis.execute_command('TS.GET', ts_key)
            if result:
                timestamp_millis, value = result
                timestamp = datetime.fromtimestamp(timestamp_millis / 1000)
                return TimeSeriesData(timestamp=timestamp, value=float(value))
        except redis.ResponseError:
            pass
        
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
        
        try:
            result = self.redis.execute_command(
                'TS.RANGE', ts_key, start_millis, end_millis,
                'AGGREGATION', aggregation, bucket_size_ms
            )
            
            data_points = []
            for timestamp_millis, value in result:
                timestamp = datetime.fromtimestamp(timestamp_millis / 1000)
                data_points.append(TimeSeriesData(timestamp=timestamp, value=float(value)))
            
            return data_points
        
        except redis.ResponseError:
            return []
    
    def get_site_analytics(
        self,
        site_id: int,
        start_time: datetime,
        end_time: datetime
    ) -> Optional[SiteAnalytics]:
        """Calculate analytics for a site over a time period."""
        
        # Get all metrics for the time range
        generated_data = self.get_readings_range(site_id, MetricType.WH_GENERATED, start_time, end_time)
        used_data = self.get_readings_range(site_id, MetricType.WH_USED, start_time, end_time)
        temp_data = self.get_readings_range(site_id, MetricType.TEMP_C, start_time, end_time)
        
        if not generated_data:
            return None
        
        # Calculate statistics
        generated_values = [dp.value for dp in generated_data]
        used_values = [dp.value for dp in used_data] if used_data else [0] * len(generated_values)
        temp_values = [dp.value for dp in temp_data] if temp_data else [0] * len(generated_values)
        
        total_generated = sum(generated_values)
        total_used = sum(used_values)
        avg_generation = total_generated / len(generated_values) if generated_values else 0
        max_generation = max(generated_values) if generated_values else 0
        min_generation = min(generated_values) if generated_values else 0
        avg_temperature = sum(temp_values) / len(temp_values) if temp_values else 0
        
        # Calculate efficiency (this would need site capacity data)
        efficiency_percent = 0  # Placeholder - would need site capacity
        
        return SiteAnalytics(
            site_id=site_id,
            period_start=start_time,
            period_end=end_time,
            total_generated=total_generated,
            total_used=total_used,
            avg_generation=avg_generation,
            max_generation=max_generation,
            min_generation=min_generation,
            avg_temperature=avg_temperature,
            efficiency_percent=efficiency_percent,
            reading_count=len(generated_values)
        )
    
    def delete_site_data(self, site_id: int) -> None:
        """Delete all time series data for a site."""
        pipe = self.redis.pipeline()
        
        for metric in MetricType:
            ts_key = self._get_ts_key(site_id, metric)
            try:
                pipe.delete(ts_key)
            except:
                pass  # Key might not exist
        
        # Delete reading keys (this is a simplified approach)
        # In production, you'd want a more sophisticated cleanup
        pattern = f"{self.key_manager.key_prefix}reading:{site_id}:*"
        keys = self.redis.keys(pattern)
        if keys:
            pipe.delete(*keys)
        
        pipe.execute()
```

---

## Step 4: Leaderboard DAO Implementation

### Create redisolar/dao/leaderboard_dao.py
```python
"""Leaderboard Data Access Object using Redis Sorted Sets."""

import redis
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta

from .redis_dao import get_redis
from .key_manager import key_manager
from .site_dao_redis import SiteDaoRedis
from redisolar.models.site import Site, SiteRanking, LeaderboardEntry


class LeaderboardDao:
    """Redis Sorted Sets implementation for leaderboards."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None, site_dao: Optional[SiteDaoRedis] = None):
        """Initialize with Redis client and Site DAO."""
        self.redis = redis_client or get_redis()
        self.key_manager = key_manager
        self.site_dao = site_dao or SiteDaoRedis(redis_client)
    
    def update_capacity_leaderboard(self, site_id: int, capacity: float) -> None:
        """Update site's position in capacity leaderboard."""
        leaderboard_key = self.key_manager.capacity_leaderboard_key()
        self.redis.zadd(leaderboard_key, {f"site:{site_id}": capacity})
    
    def update_generation_leaderboard(self, site_id: int, total_generation: float) -> None:
        """Update site's position in generation leaderboard."""
        leaderboard_key = f"{self.key_manager.key_prefix}generation_leaderboard"
        self.redis.zadd(leaderboard_key, {f"site:{site_id}": total_generation})
    
    def update_efficiency_leaderboard(self, site_id: int, efficiency_percent: float) -> None:
        """Update site's position in efficiency leaderboard."""
        leaderboard_key = f"{self.key_manager.key_prefix}efficiency_leaderboard"
        self.redis.zadd(leaderboard_key, {f"site:{site_id}": efficiency_percent})
    
    def get_capacity_leaders(self, limit: int = 10) -> List[LeaderboardEntry]:
        """Get top sites by capacity."""
        leaderboard_key = self.key_manager.capacity_leaderboard_key()
        return self._get_leaderboard_entries(leaderboard_key, limit)
    
    def get_generation_leaders(self, limit: int = 10) -> List[LeaderboardEntry]:
        """Get top sites by total generation."""
        leaderboard_key = f"{self.key_manager.key_prefix}generation_leaderboard"
        return self._get_leaderboard_entries(leaderboard_key, limit)
    
    def get_efficiency_leaders(self, limit: int = 10) -> List[LeaderboardEntry]:
        """Get top sites by efficiency."""
        leaderboard_key = f"{self.key_manager.key_prefix}efficiency_leaderboard"
        return self._get_leaderboard_entries(leaderboard_key, limit)
    
    def _get_leaderboard_entries(self, leaderboard_key: str, limit: int) -> List[LeaderboardEntry]:
        """Get leaderboard entries from a sorted set."""
        # Get top entries with scores (highest first)
        results = self.redis.zrevrange(leaderboard_key, 0, limit - 1, withscores=True)
        
        entries = []
        for rank, (member, score) in enumerate(results, 1):
            # Extract site ID from member (format: "site:123")
            site_id = int(member.split(':')[1])
            
            # Get site details
            site = self.site_dao.find_by_id(site_id)
            if site:
                entries.append(LeaderboardEntry(
                    site=site,
                    score=float(score),
                    rank=rank
                ))
        
        return entries
    
    def get_site_rank(self, site_id: int, leaderboard_type: str) -> Optional[int]:
        """Get a site's rank in a specific leaderboard."""
        leaderboard_key = self._get_leaderboard_key(leaderboard_type)
        member = f"site:{site_id}"
        
        # Get rank (Redis returns 0-based, we want 1-based)
        rank = self.redis.zrevrank(leaderboard_key, member)
        return rank + 1 if rank is not None else None
    
    def get_site_score(self, site_id: int, leaderboard_type: str) -> Optional[float]:
        """Get a site's score in a specific leaderboard."""
        leaderboard_key = self._get_leaderboard_key(leaderboard_type)
        member = f"site:{site_id}"
        
        score = self.redis.zscore(leaderboard_key, member)
        return float(score) if score is not None else None
    
    def _get_leaderboard_key(self, leaderboard_type: str) -> str:
        """Get Redis key for leaderboard type."""
        if leaderboard_type == 'capacity':
            return self.key_manager.capacity_leaderboard_key()
        elif leaderboard_type == 'generation':
            return f"{self.key_manager.key_prefix}generation_leaderboard"
        elif leaderboard_type == 'efficiency':
            return f"{self.key_manager.key_prefix}efficiency_leaderboard"
        else:
            raise ValueError(f"Unknown leaderboard type: {leaderboard_type}")
    
    def remove_site_from_leaderboards(self, site_id: int) -> None:
        """Remove a site from all leaderboards."""
        member = f"site:{site_id}"
        
        leaderboards = [
            self.key_manager.capacity_leaderboard_key(),
            f"{self.key_manager.key_prefix}generation_leaderboard",
            f"{self.key_manager.key_prefix}efficiency_leaderboard"
        ]
        
        pipe = self.redis.pipeline()
        for leaderboard in leaderboards:
            pipe.zrem(leaderboard, member)
        pipe.execute()
    
    def rebuild_capacity_leaderboard(self) -> int:
        """Rebuild the capacity leaderboard from site data."""
        # Get all sites
        sites = self.site_dao.find_all()
        
        if not sites:
            return 0
        
        # Clear existing leaderboard
        leaderboard_key = self.key_manager.capacity_leaderboard_key()
        self.redis.delete(leaderboard_key)
        
        # Add all sites to leaderboard
        site_scores = {f"site:{site.id}": site.capacity for site in sites}
        self.redis.zadd(leaderboard_key, site_scores)
        
        return len(sites)
    
    def get_leaderboard_stats(self, leaderboard_type: str) -> Dict[str, Any]:
        """Get statistics about a leaderboard."""
        leaderboard_key = self._get_leaderboard_key(leaderboard_type)
        
        total_sites = self.redis.zcard(leaderboard_key)
        if total_sites == 0:
            return {
                'total_sites': 0,
                'highest_score': 0,
                'lowest_score': 0,
                'average_score': 0
            }
        
        # Get highest and lowest scores
        highest = self.redis.zrevrange(leaderboard_key, 0, 0, withscores=True)
        lowest = self.redis.zrange(leaderboard_key, 0, 0, withscores=True)
        
        highest_score = float(highest[0][1]) if highest else 0
        lowest_score = float(lowest[0][1]) if lowest else 0
        
        # Calculate average (simplified - in production, you might use sampling)
        all_scores = self.redis.zrange(leaderboard_key, 0, -1, withscores=True)
        total_score = sum(float(score) for _, score in all_scores)
        average_score = total_score / total_sites if total_sites > 0 else 0
        
        return {
            'total_sites': total_sites,
            'highest_score': highest_score,
            'lowest_score': lowest_score,
            'average_score': average_score
        }
    
    def get_sites_around_rank(self, site_id: int, leaderboard_type: str, context_size: int = 2) -> List[LeaderboardEntry]:
        """Get sites around a specific site's rank."""
        leaderboard_key = self._get_leaderboard_key(leaderboard_type)
        member = f"site:{site_id}"
        
        # Get the site's rank
        rank = self.redis.zrevrank(leaderboard_key, member)
        if rank is None:
            return []
        
        # Get surrounding entries
        start = max(0, rank - context_size)
        end = rank + context_size
        
        results = self.redis.zrevrange(leaderboard_key, start, end, withscores=True)
        
        entries = []
        for i, (member, score) in enumerate(results):
            site_id_from_member = int(member.split(':')[1])
            site = self.site_dao.find_by_id(site_id_from_member)
            
            if site:
                entries.append(LeaderboardEntry(
                    site=site,
                    score=float(score),
                    rank=start + i + 1
                ))
        
        return entries
```

---

## Step 5: Background Tasks for Leaderboard Updates

### Create redisolar/tasks/leaderboard_updater.py
```python
"""Background tasks for updating leaderboards."""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any

from redisolar.dao.leaderboard_dao import LeaderboardDao
from redisolar.dao.timeseries_dao import TimeSeriesDao
from redisolar.dao.site_dao_redis import SiteDaoRedis
from redisolar.models.site import MetricType


class LeaderboardUpdater:
    """Background service for updating leaderboards."""
    
    def __init__(self):
        """Initialize with DAO instances."""
        self.leaderboard_dao = LeaderboardDao()
        self.timeseries_dao = TimeSeriesDao()
        self.site_dao = SiteDaoRedis()
        self.is_running = False
        self.update_thread = None
    
    def start(self, update_interval: int = 300):  # 5 minutes
        """Start the background updater."""
        if self.is_running:
            return
        
        self.is_running = True
        self.update_thread = threading.Thread(
            target=self._update_loop,
            args=(update_interval,),
            daemon=True
        )
        self.update_thread.start()
        print(f"Leaderboard updater started with {update_interval}s interval")
    
    def stop(self):
        """Stop the background updater."""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join()
        print("Leaderboard updater stopped")
    
    def _update_loop(self, interval: int):
        """Main update loop."""
        while self.is_running:
            try:
                self.update_all_leaderboards()
            except Exception as e:
                print(f"Error updating leaderboards: {e}")
            
            time.sleep(interval)
    
    def update_all_leaderboards(self):
        """Update all leaderboards."""
        print(f"Updating leaderboards at {datetime.now()}")
        
        # Update capacity leaderboard (this is static, only needs updates when sites change)
        self.update_capacity_leaderboard()
        
        # Update generation leaderboard (daily totals)
        self.update_generation_leaderboard()
        
        # Update efficiency leaderboard
        self.update_efficiency_leaderboard()
    
    def update_capacity_leaderboard(self):
        """Update the capacity leaderboard."""
        sites = self.site_dao.find_all()
        
        for site in sites:
            self.leaderboard_dao.update_capacity_leaderboard(site.id, site.capacity)
        
        print(f"Updated capacity leaderboard for {len(sites)} sites")
    
    def update_generation_leaderboard(self):
        """Update generation leaderboard with today's totals."""
        sites = self.site_dao.find_all()
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        tomorrow = today + timedelta(days=1)
        
        updated_count = 0
        
        for site in sites:
            # Get today's generation data
            generation_data = self.timeseries_dao.get_readings_range(
                site.id, MetricType.WH_GENERATED, today, tomorrow
            )
            
            if generation_data:
                total_generation = sum(dp.value for dp in generation_data)
                self.leaderboard_dao.update_generation_leaderboard(site.id, total_generation)
                updated_count += 1
        
        print(f"Updated generation leaderboard for {updated_count} sites")
    
    def update_efficiency_leaderboard(self):
        """Update efficiency leaderboard."""
        sites = self.site_dao.find_all()
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        tomorrow = today + timedelta(days=1)
        
        updated_count = 0
        
        for site in sites:
            # Get today's generation data
            generation_data = self.timeseries_dao.get_readings_range(
                site.id, MetricType.WH_GENERATED, today, tomorrow
            )
            
            if generation_data and site.capacity > 0:
                total_generation = sum(dp.value for dp in generation_data)
                # Calculate efficiency as percentage of capacity utilization
                max_possible = site.capacity * 24 * 1000  # 24 hours * 1000 Wh/kWh
                efficiency = (total_generation / max_possible) * 100 if max_possible > 0 else 0
                
                self.leaderboard_dao.update_efficiency_leaderboard(site.id, efficiency)
                updated_count += 1
        
        print(f"Updated efficiency leaderboard for {updated_count} sites")


# Global instance
leaderboard_updater = LeaderboardUpdater()
