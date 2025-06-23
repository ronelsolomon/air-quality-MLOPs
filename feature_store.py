from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import pandas as pd
from feast import Entity, FeatureView, Field
from feast.types import Int32, Float32, String, Int64
from feast.value_type import ValueType
from feast.infra.offline_stores.file_source import FileSource
from feast.data_source import PushMode
from feast import FeatureStore
from pathlib import Path
import json
import os
import yaml
from datetime import timezone
import pytz

# Define the feature store location
FEAST_REPO = "feature_repo"
os.makedirs(FEAST_REPO, exist_ok=True)

class AirQualityFeatureStore:
    def __init__(self, repo_path: str = FEAST_REPO):
        self.repo_path = Path(repo_path)
        self.store = None
        self._initialize_feature_store()

    def _initialize_feature_store(self):
        if not (self.repo_path / "feature_store.yaml").exists():
            (self.repo_path / "data").mkdir(parents=True, exist_ok=True)

        config = {
            "project": "air_quality",
            "registry": str((self.repo_path / "registry.db").absolute()),
            "provider": "local",
            "online_store": {
                "type": "sqlite",
                "path": str((self.repo_path / "online_store.db").absolute())
            },
            "offline_store": {
                "type": "file"
            },
            "entity_key_serialization_version": 2
        }

        with open(self.repo_path / "feature_store.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        (self.repo_path / "__init__.py").touch()

        self.store = FeatureStore(repo_path=str(self.repo_path))

    def define_entities_and_features(self):
        """Define entities and features for the air quality data"""
        # Define the monitoring station entity
        station = Entity(
            name="station",
            value_type=ValueType.STRING,
            description="Air quality monitoring station",
        )

        # Define the feature view
        air_quality_source = FileSource(
            name="air_quality_source",
            path=str(self.repo_path / "data/air_quality.parquet"),
            timestamp_field="timestamp",
        )

        air_quality_view = FeatureView(
            name="air_quality_metrics",
            entities=[station],
            ttl=timedelta(days=365),
            schema=[
                Field(name="aqi", dtype=Int32),
                Field(name="pm25", dtype=Float32),
                Field(name="pm10", dtype=Float32),
                Field(name="co", dtype=Float32),
                Field(name="no2", dtype=Float32),
                Field(name="o3", dtype=Float32),
                Field(name="temperature", dtype=Float32),
                Field(name="humidity", dtype=Float32),
                Field(name="pressure", dtype=Float32),
                Field(name="dominant_pollutant", dtype=String),
                Field(name="aqi_category", dtype=String),
                Field(name="pm25_trend", dtype=String),
                Field(name="pm25_24h_change", dtype=Float32),
                Field(name="pm25_7d_avg", dtype=Float32),
                Field(name="pm25_pm10_ratio", dtype=Float32),
                Field(name="pm25_no2_ratio", dtype=Float32),
                Field(name="pm25_o3_ratio", dtype=Float32),
                Field(name="heat_index", dtype=Float32),
                Field(name="hour_of_day", dtype=Int32),
                Field(name="day_of_week", dtype=Int32),
                Field(name="is_weekend", dtype=Int32),
                Field(name="month", dtype=Int32),
                Field(name="season", dtype=Int32),
                Field(name="pm25_trend_slope", dtype=Float32),
                Field(name="aqhi", dtype=Float32),
            ],
            source=air_quality_source,
        )

        # Save the feature store configuration
        from feast import FeatureStore
        from pathlib import Path

        with open(self.repo_path / "feature_store.yaml", "w") as f:
            f.write(f"""
project: air_quality
registry: {str((self.repo_path / "registry.db").absolute())}
provider: local
online_store:
    type: sqlite
    path: {str((self.repo_path / "online_store.db").absolute())}
offline_store:
    type: file
""")

        # Create the feature store directory structure
        (self.repo_path / "data").mkdir(exist_ok=True)
        (self.repo_path / "feature_definitions").mkdir(exist_ok=True)

        # Save the feature definitions
        from feast import FeatureStore
        from feast.infra.registry.registry import Registry
        from feast.protos.feast.core.Registry_pb2 import Registry as RegistryProto

        self.store.apply([station, air_quality_view])


    def prepare_feature_data(self, air_quality_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare the air quality data for feature storage"""
        # Basic features
        features = {
            "station": air_quality_data['city']['name'],
            "timestamp": air_quality_data.get('time', {}).get('iso', datetime.utcnow().isoformat()),
            "aqi": air_quality_data.get('aqi', 0),
        "pm25": air_quality_data.get('iaqi', {}).get('pm25', {}).get('v', 0.0),
        "pm10": air_quality_data.get('iaqi', {}).get('pm10', {}).get('v', 0.0),
        "co": air_quality_data.get('iaqi', {}).get('co', {}).get('v', 0.0),
        "no2": air_quality_data.get('iaqi', {}).get('no2', {}).get('v', 0.0),
        "o3": air_quality_data.get('iaqi', {}).get('o3', {}).get('v', 0.0),
        "temperature": air_quality_data.get('iaqi', {}).get('t', {}).get('v', 0.0),
        "humidity": air_quality_data.get('iaqi', {}).get('h', {}).get('v', 0.0),
        "pressure": air_quality_data.get('iaqi', {}).get('p', {}).get('v', 0.0),
        "dominant_pollutant": air_quality_data.get('dominentpol', 'unknown'),
        "latitude": air_quality_data.get('city', {}).get('geo', [None, None])[0],
        "longitude": air_quality_data.get('city', {}).get('geo', [None, None])[1],
        "timezone": air_quality_data.get('time', {}).get('tz', ''),
        "unix_timestamp": air_quality_data.get('time', {}).get('v'),
    }
    
        # Add forecast features
        forecast = air_quality_data.get('forecast', {}).get('daily', {})
        for day_offset in range(7):  # Next 7 days
            day_key = f"day_{day_offset+1}"
            if day_offset < len(forecast.get('pm25', [])):
                features.update({
                    f"{day_key}_pm25_avg": forecast['pm25'][day_offset].get('avg'),
                    f"{day_key}_pm25_max": forecast['pm25'][day_offset].get('max'),
                    f"{day_key}_pm25_min": forecast['pm25'][day_offset].get('min'),
                    f"{day_key}_pm10_avg": forecast['pm10'][day_offset].get('avg'),
                    f"{day_key}_uvi_avg": forecast['uvi'][day_offset].get('avg') if 'uvi' in forecast else None,
                })

        # Add derived features
        features.update(self._calculate_derived_features(air_quality_data))
        return features

    def _calculate_derived_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate derived features from the raw data"""
        derived = {}
        
        # Current data
        iaqi = data.get('iaqi', {})
        
        # 1. Air Quality Index (AQI) Categories
        aqi = data.get('aqi', 0)
        if aqi <= 50:
            derived['aqi_category'] = 'good'
        elif aqi <= 100:
            derived['aqi_category'] = 'moderate'
        elif aqi <= 150:
            derived['aqi_category'] = 'unhealthy_sensitive'
        elif aqi <= 200:
            derived['aqi_category'] = 'unhealthy'
        elif aqi <= 300:
            derived['aqi_category'] = 'very_unhealthy'
        else:
            derived['aqi_category'] = 'hazardous'
        
        # 2. Pollutant Ratios
        pm25 = iaqi.get('pm25', {}).get('v', 0.1)  # Avoid division by zero
        derived.update({
            'pm25_pm10_ratio': pm25 / max(iaqi.get('pm10', {}).get('v', 0.1), 0.1),
            'pm25_no2_ratio': pm25 / max(iaqi.get('no2', {}).get('v', 0.1), 0.1),
            'pm25_o3_ratio': pm25 / max(iaqi.get('o3', {}).get('v', 0.1), 0.1),
        })
        
        # 3. Weather-related features
        temp = iaqi.get('t', {}).get('v', 0)
        humidity = iaqi.get('h', {}).get('v', 0)
        pressure = iaqi.get('p', {}).get('v', 0)
        
        # Heat index (simplified)
        if temp >= 26.7 and humidity >= 40:  # Only calculate for temperatures above 80°F/26.7°C
            c1 = -8.78469475556
            c2 = 1.61139411
            c3 = 2.33854884
            c4 = -0.14611605
            c5 = -0.012308094
            c6 = -0.016424828
            c7 = 0.002211732
            c8 = 0.00072546
            c9 = -0.000003582
            
            t = temp
            r = humidity
            hi = (c1 + c2 * t + c3 * r + c4 * t * r + 
                 c5 * t**2 + c6 * r**2 + c7 * t**2 * r + 
                 c8 * t * r**2 + c9 * t**2 * r**2)
            derived['heat_index'] = hi
        else:
            derived['heat_index'] = temp  # Just use temperature when heat index not applicable
        
        # 4. Time-based features
        time_data = data.get('time', {})
        if 'iso' in time_data:
            try:
                dt = pd.to_datetime(time_data['iso'])
                derived.update({
                    'hour_of_day': dt.hour,
                    'day_of_week': dt.dayofweek,  # Monday=0, Sunday=6
                    'is_weekend': 1 if dt.dayofweek >= 5 else 0,
                    'month': dt.month,
                    'season': (dt.month % 12 + 3) // 3,  # 1=winter, 2=spring, 3=summer, 4=fall
                })
            except:
                pass
        
        # 5. Forecast trends (if available)
        forecast = data.get('forecast', {}).get('daily', {})
        if 'pm25' in forecast and len(forecast['pm25']) > 1:
            # Calculate trend (slope) of PM2.5 forecast
            days = [i for i in range(min(3, len(forecast['pm25'])))]  # Next 3 days
            pm25_avgs = [forecast['pm25'][i].get('avg', 0) for i in range(len(days))]
            
            if len(days) > 1 and any(pm25_avgs):
                # Simple linear regression for trend
                x_mean = sum(days) / len(days)
                y_mean = sum(pm25_avgs) / len(pm25_avgs)
                
                numerator = sum((days[i] - x_mean) * (pm25_avgs[i] - y_mean) for i in range(len(days)))
                denominator = sum((x - x_mean) ** 2 for x in days)
                
                if denominator != 0:
                    trend_slope = numerator / denominator
                    derived['pm25_trend'] = 'increasing' if trend_slope > 0.5 else 'decreasing' if trend_slope < -0.5 else 'stable'
                    derived['pm25_trend_slope'] = trend_slope
        
        # 6. Air Quality Health Index (simplified)
        # This is a simplified version - actual AQHI calculations may vary by region
        no2 = iaqi.get('no2', {}).get('v', 0)
        o3 = iaqi.get('o3', {}).get('v', 0)
        derived['aqhi'] = (pm25 * 0.5 + no2 * 0.3 + o3 * 0.2) / 3
        
        # 7. Air quality change rate (if historical data is available)
        # This would require access to previous measurements
        
        # Add default values for features that might be missing
        derived.setdefault('pm25_7d_avg', None)  # Default to None if not calculated
        derived.setdefault('pm25_24h_change', None)  # Default to None if not calculated
        
        return derived

    def store_features(self, features: Dict[str, Any]):
        """Store features in the feature store"""
        # Create a DataFrame with the features
        df = pd.DataFrame([features])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Ensure the data directory exists
        data_dir = self.repo_path / "data"
        data_dir.mkdir(exist_ok=True)
        
        # Save to parquet
        output_path = data_dir / "air_quality.parquet"
        df.to_parquet(output_path, index=False)
        
        # Apply the feature store
        self.define_entities_and_features()
        
        # Materialize the features
        self.store.materialize_incremental(end_date=datetime.now().replace(tzinfo=timezone.utc))

    def get_historical_features(self, station_id: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Retrieve historical feature data for a specific station within a date range"""
        # Create a range of timestamps for the entity dataframe
        timestamps = pd.date_range(start=start_date, end=end_date, freq='H')
        entity_df = pd.DataFrame(
    {
        "station": [station_id],
        "event_timestamp": [pd.Timestamp(datetime.now(tz=pytz.UTC))],
        "timestamp": [pd.Timestamp(datetime.now(tz=pytz.UTC))]
    }
)
        
        features = [
            "air_quality_metrics:aqi",
            "air_quality_metrics:pm25",
            "air_quality_metrics:pm10",
            "air_quality_metrics:co",
            "air_quality_metrics:no2",
            "air_quality_metrics:o3",
            "air_quality_metrics:temperature",
            "air_quality_metrics:humidity",
            "air_quality_metrics:pressure",
            "air_quality_metrics:dominant_pollutant",
            "air_quality_metrics:aqi_category",
            "air_quality_metrics:pm25_trend",
            "air_quality_metrics:pm25_24h_change",
            "air_quality_metrics:pm25_7d_avg",
            "air_quality_metrics:pm25_pm10_ratio",
            "air_quality_metrics:pm25_no2_ratio",
            "air_quality_metrics:pm25_o3_ratio",
            "air_quality_metrics:heat_index",
            "air_quality_metrics:hour_of_day",
            "air_quality_metrics:day_of_week",
            "air_quality_metrics:is_weekend",
            "air_quality_metrics:month",
            "air_quality_metrics:season",
            "air_quality_metrics:pm25_trend_slope",
            "air_quality_metrics:aqhi",
        ]
        
        # Get historical features
        historical_features = self.store.get_historical_features(
            entity_df=entity_df,
            features=features
        )
        
        # Convert to pandas DataFrame and clean up
        df = historical_features.to_df()
        if not df.empty:
            # Convert timestamps to local timezone for better readability
            df['event_timestamp'] = pd.to_datetime(df['event_timestamp']).dt.tz_convert(None)
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert(None)
            
        return df