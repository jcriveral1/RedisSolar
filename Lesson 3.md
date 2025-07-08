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


@dataclass(frozen=True)
class LeaderboardEntry:
    """Leaderboard entry with site information."""
    site: Site
    score: float
    rank: int
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
        
        # Delete reading keys
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

from .redis_dao import get_redis
from .key_manager import key_manager
from .site_dao_redis import SiteDaoRedis
from redisolar.models.site import Site, LeaderboardEntry


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
```

---

## Step 5: Background Tasks for Leaderboard Updates

### Create redisolar/tasks/__init__.py
```python
"""Background tasks package."""
```

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
