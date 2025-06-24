import requests
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np
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
    
    # Set page config
    st.set_page_config(
        page_title="Air Quality Dashboard",
        page_icon="🌍",
        layout="wide"
    )

    # Load model and data
    @st.cache_resource
    def load_model():
        try:
            model = joblib.load('models/model.joblib')
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

    @st.cache_data
    def load_data():
        try:
            df = pd.read_csv('training_data.csv', parse_dates=['timestamp'])
            return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return pd.DataFrame()

    # Load data and model
    df = load_data()
    model = load_model()
    
    if df.empty:
        st.warning("No data available. Please run the data pipeline first.")
        return
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Date range selector
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()
    date_range = st.sidebar.date_input(
        "Select date range",
        [max_date - timedelta(days=7), max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    # Filter data by date range
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = df[(df['timestamp'].dt.date >= start_date) & 
                        (df['timestamp'].dt.date <= end_date)]
    else:
        filtered_df = df
    
    # Main metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Data Points", len(filtered_df))
    with col2:
        st.metric("Average AQI", f"{filtered_df['aqi'].mean():.1f}")
    with col3:
        st.metric("Dominant Pollutant", filtered_df['dominant_pollutant'].mode()[0] if not filtered_df.empty else "N/A")
    
    # AQI Time Series
    st.subheader("AQI Over Time")
    if not filtered_df.empty:
        fig = px.line(
            filtered_df, 
            x='timestamp', 
            y='aqi',
            title="AQI Trend",
            labels={'aqi': 'AQI', 'timestamp': 'Date'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Pollutant Analysis
    st.subheader("Pollutant Analysis")
    if not filtered_df.empty:
        pollutant_cols = [col for col in filtered_df.columns if col.startswith('iaqi_')]
        if pollutant_cols:
            pollutant_avg = filtered_df[pollutant_cols].mean().sort_values(ascending=False)
            fig = px.bar(
                x=pollutant_avg.index,
                y=pollutant_avg.values,
                labels={'x': 'Pollutant', 'y': 'Average IAQI'},
                title="Average Pollutant Levels"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Raw Data
    with st.expander("View Raw Data"):
        st.dataframe(filtered_df)
    
    # Predictions (if model is available)
    if model is not None and not filtered_df.empty:
        st.subheader("Model Predictions")
        
        # Get features for prediction (you'll need to adjust this based on your model's features)
        features = filtered_df[['aqi']]  # Adjust with actual feature columns
        
        try:
            # Make predictions
            predictions = model.predict(features)
            
            # Add predictions to dataframe
            results_df = filtered_df[['timestamp', 'aqi']].copy()
            results_df['predicted_aqi'] = predictions
            
            # Plot actual vs predicted
            fig = px.line(
                results_df.melt(id_vars=['timestamp'], 
                              value_vars=['aqi', 'predicted_aqi'],
                              var_name='Type', 
                              value_name='AQI'),
                x='timestamp',
                y='AQI',
                color='Type',
                title="Actual vs Predicted AQI"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error making predictions: {e}")

    # Analyze air quality data
    insights = analyze_air_quality(data)
    
    # Display insights
    st.subheader("Air Quality Insights")
    for key, value in insights.items():
        st.write(f"{key}: {value}")

if __name__ == "__main__":
    main()