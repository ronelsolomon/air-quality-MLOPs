import requests
from datetime import datetime, timedelta
import pandas as pd
import os

def get_air_quality_data(api_token, city='Milpitas', lat=None, lng=None):
    """
    Fetch air quality and weather data from AQICN API
    
    Parameters:
    - api_token: Your AQICN API token
    - city: City name or 'here' for geolocation (default: 'here')
    - lat, lng: Optional coordinates if using geolocation
    """
    base_url = "http://api.waqi.info/feed"
    
    if city == 'Milpitas' and lat and lng:
        # Use geolocation
        url = f"{base_url}/geo:{lat};{lng}/?token={api_token}"
    else:
        # Use city name
        url = f"{base_url}/{city}/?token={api_token}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if data['status'] == 'ok':
            return data['data']
        else:
            print(f"Error: {data.get('data', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def analyze_air_quality(data):
    """
    Analyzes air quality data and returns key insights including:
    - AQI category and health implications
    - Pollutant concentrations
    - Pollutant ratios
    - PM2.5 trend forecast
    - Daily PM2.5 changes
    
    Args:
        data (dict): Air quality data dictionary matching the structure provided
        
    Returns:
        dict: Dictionary containing calculated insights
    """
    insights = {}
    
    # Extract relevant data with safe access
    aqi = data.get('aqi')
    dominant_pollutant = data.get('dominentpol')
    iaqi = data.get('iaqi', {})
    forecast = data.get('forecast', {}).get('daily', {}).get('pm25', [])
    
    # AQI category determination
    if aqi is not None:
        if aqi <= 50:
            category = 'Good'
            health = 'No health precautions needed'
        elif aqi <= 100:
            category = 'Moderate'
            health = 'Unusually sensitive people should reduce exertion'
        elif aqi <= 150:
            category = 'Unhealthy for Sensitive Groups'
            health = 'Children, elderly, and respiratory patients should reduce outdoor activities'
        elif aqi <= 200:
            category = 'Unhealthy'
            health = 'Everyone should reduce prolonged exertion'
        elif aqi <= 300:
            category = 'Very Unhealthy'
            health = 'Health warnings of emergency conditions'
        else:
            category = 'Hazardous'
            health = 'Health alert: everyone may experience serious effects'
            
        insights['AQI Category'] = category
        insights['Health Implications'] = health
        insights['Dominant Pollutant'] = dominant_pollutant

    # Pollutant concentrations
    concentrations = {}
    for pollutant in ['pm25', 'co', 'no2', 'o3', 'h', 'p', 't']:
        if pollutant in iaqi:
            concentrations[pollutant.upper()] = iaqi[pollutant].get('v')
    insights['Pollutant Concentrations'] = concentrations

    # Pollutant ratios (relative to PM2.5)
    ratios = {}
    pm25 = concentrations.get('PM25')
    for pol in ['CO', 'NO2', 'O3']:
        if pol in concentrations and pm25 and concentrations[pol]:
            ratios[f'PM25/{pol}'] = round(pm25 / concentrations[pol], 2)
    insights['Pollutant Ratios'] = ratios

    # Forecast trend analysis
    trend = []
    for day_data in forecast:
        trend.append({
            'date': day_data['day'],
            'avg_pm25': day_data['avg'],
            'max_pm25': day_data['max'],
            'min_pm25': day_data.get('min', 'N/A')
        })
    insights['PM2.5 Forecast Trend'] = trend

    # Daily change calculation
    daily_changes = []
    for i in range(1, len(trend)):
        if 'avg_pm25' in trend[i] and 'avg_pm25' in trend[i-1]:
            change = trend[i]['avg_pm25'] - trend[i-1]['avg_pm25']
            daily_changes.append({
                'date': trend[i]['date'],
                'change': change,
                'trend': 'Increasing' if change > 0 else 'Decreasing'
            })
    insights['Daily PM2.5 Changes'] = daily_changes

    return insights


def main():
    # Initialize the feature store
    # feature_store = AirQualityFeatureStore()
    
    # Get air quality data
    API_TOKEN = os.getenv("AQICN_TOKEN")
    data = get_air_quality_data(API_TOKEN, "Milpitas")
    
    # if data:
    #     # Prepare and store features
    #     features = feature_store.prepare_feature_data(data)
    #     feature_store.store_features(features)
    
    return data

if __name__ == "__main__":
    main()