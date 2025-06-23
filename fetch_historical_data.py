import os
import pytz
import pandas as pd
from datetime import datetime, timedelta
from feature_store import AirQualityFeatureStore
from app import get_air_quality_data

def fetch_historical_data(start_date: str, end_date: str, output_file: str = "training_data.csv", api_token: str = None):
    """
    Fetch historical air quality data for a date range and save to CSV
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        output_file (str): Path to save the output CSV file
        api_token (str): AQICN API token. If None, will try to use AQICN_TOKEN env var
    """
    # Initialize feature store
    feature_store = AirQualityFeatureStore()
    
    # Get API token from environment if not provided
    if api_token is None:
        api_token = os.getenv("AQICN_TOKEN")
        if api_token is None:
            raise ValueError("API token not provided and AQICN_TOKEN environment variable not set")
    
    # Convert string dates to datetime objects
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    # List to store all data points
    all_data = []
    
    # Fetch data for each day in the range
    current_date = start
    while current_date <= end:
        print(f"Fetching data for {current_date.strftime('%Y-%m-%d')}...")
        
        try:
            # Get data for the current date
            data = get_air_quality_data(api_token, "Milpitas")
            # print(data)
            
            if data:
                # Prepare and store features
                features = feature_store.prepare_feature_data(data)
                feature_store.store_features(features)
                
                # Add to our collection
                all_data.append(features)
                
                print(f"Successfully stored data for {current_date.strftime('%Y-%m-%d')}")
            else:
                print(f"No data available for {current_date.strftime('%Y-%m-%d')}")
                
        except Exception as e:
            print(f"Error fetching data for {current_date.strftime('%Y-%m-%d')}: {str(e)}")
        
        # Move to next day
        current_date += timedelta(days=1)
    
    # Convert to DataFrame and save to CSV
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(output_file, index=False)
        print(f"\nSuccessfully saved {len(df)} records to {output_file}")
        return df
    else:
        print("No data was fetched successfully.")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch historical air quality data')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', default='training_data.csv', help='Output CSV file path')
    parser.add_argument('--api-token', help='AQICN API token')
    
    args = parser.parse_args()
    
    fetch_historical_data(
        start_date=args.start_date,
        end_date=args.end_date,
        output_file=args.output,
        api_token=args.api_token
    )
