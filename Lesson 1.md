## Lesson 1 Objectives

By the end of this lesson, you'll have:
- ✅ Redis installed and running locally
- ✅ Python virtual environment set up
- ✅ Project structure created
- ✅ Redis connection established from Python
- ✅ Basic Flask application running
- ✅ Development tools configured

---

## Step 1: Install Redis

### Option A: Using Docker (Recommended)
```bash
# Pull and run Redis with TimeSeries module
docker run -d \
  --name redis-redisolar \
  -p 6379:6379 \
  redis/redis-stack:latest
```

### Option B: Local Installation

#### macOS (using Homebrew):
```bash
brew install redis
brew services start redis
```

#### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install redis-server
sudo systemctl start redis
sudo systemctl enable redis
```

#### Windows:
Use WSL2 with Ubuntu and follow the Ubuntu instructions above.

### Verify Redis Installation
```bash
# Test Redis connection
redis-cli ping
# Should return: PONG
```

---

## Step 2: Project Setup

### Create Project Directory
```bash
mkdir redisolar
cd redisolar
```

### Create Python Virtual Environment
```bash
# Create virtual environment
python3.8 -m venv env

# Activate virtual environment
# On macOS/Linux:
source env/bin/activate

# On Windows:
env\Scripts\activate
```

### Project Structure
Create the following directory structure:
```
redisolar/
├── redisolar/
│   ├── __init__.py
│   ├── app.py
│   ├── config.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── site.py
│   ├── dao/
│   │   ├── __init__.py
│   │   └── redis_dao.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py
│   └── instance/
│       ├── dev.cfg
│       └── testing.cfg
├── tests/
│   ├── __init__.py
│   └── test_basic.py
├── requirements.txt
├── setup.py
└── .env
```

---

## Step 3: Install Dependencies

### Create requirements.txt
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

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Step 4: Configuration Files

### Create .env file
```env
# Redis Configuration
REDISOLAR_REDIS_HOST=localhost
REDISOLAR_REDIS_PORT=6379
REDISOLAR_REDIS_PASSWORD=
REDISOLAR_REDIS_USERNAME=

# Application Configuration
FLASK_APP=redisolar.app
FLASK_ENV=development
REDIS_KEY_PREFIX=ru102py-app:
```

### Create redisolar/instance/dev.cfg
```python
# Development Configuration
DEBUG = True
TESTING = False

# Redis Configuration
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_PASSWORD = None
REDIS_USERNAME = None
REDIS_DB = 0
REDIS_KEY_PREFIX = 'ru102py-app:'

# Application Settings
SECRET_KEY = 'dev-secret-key-change-in-production'
USE_GEO_SITE_API = False  # We'll enable this later
```

### Create redisolar/instance/testing.cfg
```python
# Testing Configuration
DEBUG = False
TESTING = True

# Redis Configuration (separate DB for testing)
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_PASSWORD = None
REDIS_USERNAME = None
REDIS_DB = 1  # Different DB for tests
REDIS_KEY_PREFIX = 'ru102py-test:'

# Application Settings
SECRET_KEY = 'test-secret-key'
USE_GEO_SITE_API = False
```

---

## Step 5: Basic Application Structure

### Create redisolar/__init__.py
```python
"""RediSolar - Solar Energy Monitoring Application"""

__version__ = '1.0.0'
```

### Create redisolar/config.py
```python
"""Application configuration management."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Application configuration class."""
    
    # Redis settings
    redis_host: str = 'localhost'
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_username: Optional[str] = None
    redis_db: int = 0
    redis_key_prefix: str = 'ru102py-app:'
    
    # Flask settings
    secret_key: str = 'dev-secret-key'
    debug: bool = True
    testing: bool = False
    
    # Feature flags
    use_geo_site_api: bool = False
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables."""
        return cls(
            redis_host=os.getenv('REDISOLAR_REDIS_HOST', 'localhost'),
            redis_port=int(os.getenv('REDISOLAR_REDIS_PORT', '6379')),
            redis_password=os.getenv('REDISOLAR_REDIS_PASSWORD'),
            redis_username=os.getenv('REDISOLAR_REDIS_USERNAME'),
            redis_key_prefix=os.getenv('REDIS_KEY_PREFIX', 'ru102py-app:'),
        )
```

---

## Step 6: Redis Connection

### Create redisolar/dao/redis_dao.py
```python
"""Redis Data Access Object base class."""

import redis
from typing import Optional
from redisolar.config import Config


class RedisConnection:
    """Manages Redis connection for the application."""
    
    _instance: Optional[redis.Redis] = None
    _config: Optional[Config] = None
    
    @classmethod
    def get_connection(cls, config: Optional[Config] = None) -> redis.Redis:
        """Get Redis connection instance (singleton pattern)."""
        if cls._instance is None or config != cls._config:
            cls._config = config or Config.from_env()
            
            connection_kwargs = {
                'host': cls._config.redis_host,
                'port': cls._config.redis_port,
                'db': cls._config.redis_db,
                'decode_responses': True,  # Automatically decode bytes to strings
                'socket_connect_timeout': 5,
                'socket_timeout': 5,
                'retry_on_timeout': True,
            }
            
            # Add authentication if provided
            if cls._config.redis_password:
                connection_kwargs['password'] = cls._config.redis_password
            if cls._config.redis_username:
                connection_kwargs['username'] = cls._config.redis_username
                
            cls._instance = redis.Redis(**connection_kwargs)
            
        return cls._instance
    
    @classmethod
    def test_connection(cls) -> bool:
        """Test if Redis connection is working."""
        try:
            conn = cls.get_connection()
            return conn.ping()
        except Exception as e:
            print(f"Redis connection failed: {e}")
            return False


# Convenience function for getting Redis connection
def get_redis() -> redis.Redis:
    """Get Redis connection instance."""
    return RedisConnection.get_connection()
```

---

## Step 7: Basic Flask Application

### Create redisolar/app.py
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
    
    # Register routes
    @app.route('/')
    def index():
        """Health check endpoint."""
        return jsonify({
            'message': 'RediSolar API is running!',
            'redis_connected': RedisConnection.test_connection(),
            'redis_info': get_redis().info('server')['redis_version']
        })
    
    @app.route('/health')
    def health():
        """Detailed health check."""
        redis_conn = get_redis()
        
        try:
            # Test basic Redis operations
            test_key = f"{app.config['REDIS_KEY_PREFIX']}health_check"
            redis_conn.set(test_key, "ok", ex=60)  # Expire in 60 seconds
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

## Step 8: Basic Test

### Create tests/test_basic.py
```python
"""Basic tests for Redis connection and Flask app."""

import pytest
from redisolar.app import create_app
from redisolar.dao.redis_dao import RedisConnection


@pytest.fixture
def app():
    """Create test application."""
    app = create_app('testing')
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


def test_redis_connection():
    """Test Redis connection."""
    assert RedisConnection.test_connection()


def test_app_index(client):
    """Test application index route."""
    response = client.get('/')
    assert response.status_code == 200
    
    data = response.get_json()
    assert data['message'] == 'RediSolar API is running!'
    assert data['redis_connected'] is True


def test_app_health(client):
    """Test application health route."""
    response = client.get('/health')
    assert response.status_code == 200
    
    data = response.get_json()
    assert data['status'] == 'healthy'
    assert data['redis']['connected'] is True
```

---

## Step 9: Run and Test

### Start the Application
```bash
# Make sure Redis is running
# Make sure virtual environment is activated

# Run the Flask application
python redisolar/app.py
```

### Test the Application
```bash
# In another terminal, test the endpoints
curl http://localhost:8081/
curl http://localhost:8081/health

# Run the tests
pytest tests/
```

### Expected Output
- The Flask app should start on `http://localhost:8081`
- The index endpoint should return a JSON response with Redis info
- The health endpoint should show Redis as connected
- All tests should pass

---

## Step 10: Setup Script (Optional)

### Create setup.py
```python
"""Setup script for RediSolar application."""

from setuptools import setup, find_packages

setup(
    name='redisolar',
    version='1.0.0',
    description='Solar Energy Monitoring Application with Redis',
    packages=find_packages(),
    install_requires=[
        'Flask>=2.3.0',
        'redis>=5.0.0',
        'python-dotenv>=1.0.0',
        'marshmallow>=3.20.0',
        'marshmallow-dataclass>=8.6.0',
        'dataclasses-json>=0.6.0',
        'pytest>=7.4.0',
        'python-decouple>=3.8',
    ],
    python_requires='>=3.8',
)
```

### Install in Development Mode
```bash
pip install -e .
```

---

## 🎉 Lesson 1 Complete!

You've successfully:
- ✅ Set up Redis and verified the connection
- ✅ Created a proper Python project structure
- ✅ Established Redis connection from Python
- ✅ Built a basic Flask application with health checks
- ✅ Configured development and testing environments
- ✅ Written and run basic tests

### What's Next?
In **Lesson 2**, we'll:
- Define our data models using Python dataclasses
- Learn about Redis data structures (Hashes, Sets, Lists)
- Create our first Data Access Object (DAO) for solar sites
- Implement CRUD operations for solar site management

### Key Takeaways
1. **Project Structure**: Well-organized code with separation of concerns
2. **Configuration Management**: Environment-specific settings
3. **Connection Management**: Singleton pattern for Redis connections
4. **Testing Strategy**: Separate Redis databases for testing
5. **Health Monitoring**: Always include health check endpoints
