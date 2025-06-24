import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from datetime import datetime, timedelta
import os

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

def main():
    st.title("🌍 Air Quality Dashboard")
    
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

if __name__ == "__main__":
    main()
