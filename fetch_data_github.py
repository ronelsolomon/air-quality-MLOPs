import os
import pandas as pd
from datetime import datetime, timedelta
from app import get_air_quality_data
import time

def flatten_air_quality_data(data, date_str=None):
    """Flatten the nested air quality data structure."""
    if not data or 'aqi' not in data:
        return None
        
    # Create a flat dictionary with the data we want to save
    flat_data = {
        'timestamp': date_str or datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'date': date_str or datetime.now().strftime('%Y-%m-%d'),  
        'aqi': data.get('aqi'),
        'dominant_pollutant': data.get('dominentpol'),
        'city': data.get('city', {}).get('name', '')
    }
    
    # Add individual pollutant data
    iaqi = data.get('iaqi', {})
    for key, value in iaqi.items():
        if isinstance(value, dict) and 'v' in value:
            flat_data[f'iaqi_{key}'] = value['v']
    
    return flat_data

def fetch_historical_datas(start_date, end_date, api_token=None, city="Milpitas"):
    if api_token is None:
        api_token = os.getenv("AQICN_TOKEN")
        if api_token is None:
            raise ValueError("API token not provided and AQICN_TOKEN environment variable not set")

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    all_data = []

    current_date = start
    while current_date <= end:
        date_str = current_date.strftime('%Y-%m-%d')
        print(f"Fetching data for {city} on {date_str}...")
        try:
            # Remove date parameter since get_air_quality_data doesn't support it
            data = get_air_quality_data(api_token, city)
            if data:
                # Add the date to the data since we're not getting historical data
                data['date'] = date_str
                flat_data = flatten_air_quality_data(data, date_str)
                if flat_data:
                    all_data.append(flat_data)
                    print(f"Successfully processed data for {date_str}")
                else:
                    print(f"No valid data for {date_str}")
            else:
                print(f"No data returned for {date_str}")
        except Exception as e:
            print(f"Error fetching data for {date_str}: {str(e)}")
        current_date += timedelta(days=1)
        # Add a small delay to avoid hitting rate limits
        time.sleep(1)

    if all_data:
        df = pd.DataFrame(all_data)
        # Ensure consistent column order
        columns = ['timestamp', 'date', 'aqi', 'dominant_pollutant', 'city'] + \
                 [col for col in df.columns if col.startswith('iaqi_') and col not in ['timestamp', 'date', 'aqi', 'dominant_pollutant', 'city']]
        df = df[sorted(columns)]
        print(f"Fetched {len(df)} records.")
        return df
    else:
        print("No data fetched.")
        return pd.DataFrame()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch historical air quality data')
    parser.add_argument("--start-date", required=True, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end-date", required=True, help="End date in YYYY-MM-DD format")
    parser.add_argument("--output", default="training_data.csv", help="Output CSV file path")
    parser.add_argument("--api-token", help="AQICN API token")
    parser.add_argument("--city", default="Milpitas", help="City name")
    args = parser.parse_args()

    # Prefer CLI token, fallback to environment, then hardcoded token
    api_token = args.api_token or os.getenv("AQICN_TOKEN") or "18fed86f6f05b8695480ad90a15a00661e1b28de"
    
    print(f"Fetching data from {args.start_date} to {args.end_date} for {args.city}")
    new_data = fetch_historical_datas(args.start_date, args.end_date, api_token=api_token, city=args.city)

    if new_data.empty:
        print("No new data to save.")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    if os.path.exists(args.output):
        try:
            # Read existing data
            existing = pd.read_csv(args.output)
            # Ensure timestamp is string for comparison
            existing['timestamp'] = existing['timestamp'].astype(str)
            new_data['timestamp'] = new_data['timestamp'].astype(str)
            
            # Find new records (not in existing data)
            existing_timestamps = set(existing['timestamp'])
            new_records = ~new_data['timestamp'].isin(existing_timestamps)
            
            if new_records.any():
                # Only keep new records
                new_data = new_data[new_records]
                # Combine with existing data
                combined = pd.concat([existing, new_data], ignore_index=True)
                # Sort by timestamp
                combined = combined.sort_values('timestamp')
                # Save to file
                combined.to_csv(args.output, index=False)
                print(f"Added {len(new_data)} new records. Total records: {len(combined)}")
            else:
                print("No new records to add.")
                return
                
        except Exception as e:
            print(f"Error processing existing file: {str(e)}")
            print("Creating new file...")
            new_data.to_csv(args.output, index=False)
            print(f"Created new file with {len(new_data)} records")
    else:
        # Save new data to file
        new_data.to_csv(args.output, index=False)
        print(f"Created new file with {len(new_data)} records")

if __name__ == "__main__":
    main()
