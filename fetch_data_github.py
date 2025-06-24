import os
import pandas as pd
from datetime import datetime, timedelta
from app import get_air_quality_data
import time


def flatten_air_quality_data(data, date_str=None):
    """
    Flatten the air quality data structure, preserving all available fields.

    Args:
        data (dict): Raw air quality data from API.
        date_str (str, optional): Date string to use for timestamp fields.

    Returns:
        dict: Flattened air quality data.
    """
    if not data or 'aqi' not in data:
        return None

    now = datetime.now()
    timestamp = date_str + " 00:00:00" if date_str else now.strftime('%Y-%m-%d %H:%M:%S')
    date = date_str or now.strftime('%Y-%m-%d')

    iaqi = data.get('iaqi', {})
    flat_data = {
        'timestamp': timestamp,
        'date': date,
        'aqi': data.get('aqi'),
        'pm25': iaqi.get('pm25', {}).get('v') if isinstance(iaqi.get('pm25'), dict) else None,
        'pm10': iaqi.get('pm10', {}).get('v') if isinstance(iaqi.get('pm10'), dict) else None,
        'co': iaqi.get('co', {}).get('v') if isinstance(iaqi.get('co'), dict) else None,
        'no2': iaqi.get('no2', {}).get('v') if isinstance(iaqi.get('no2'), dict) else None,
        'o3': iaqi.get('o3', {}).get('v') if isinstance(iaqi.get('o3'), dict) else None,
        'temperature': iaqi.get('t', {}).get('v') if isinstance(iaqi.get('t'), dict) else None,
        'humidity': iaqi.get('h', {}).get('v') if isinstance(iaqi.get('h'), dict) else None,
        'pressure': iaqi.get('p', {}).get('v') if isinstance(iaqi.get('p'), dict) else None,
        'dominant_pollutant': data.get('dominentpol'),
        'aqi_category': data.get('category', {}).get('name') if isinstance(data.get('category'), dict) else None,
        'city': data.get('city', {}).get('name', ''),
        'city_geo_lat': data.get('city', {}).get('geo', [None])[0] if isinstance(data.get('city', {}).get('geo'), list) and len(data.get('city', {}).get('geo', [])) > 0 else None,
        'city_geo_lon': data.get('city', {}).get('geo', [None, None])[1] if isinstance(data.get('city', {}).get('geo'), list) and len(data.get('city', {}).get('geo', [])) > 1 else None,
        'station_name': data.get('city', {}).get('name'),
        'station_url': data.get('city', {}).get('url'),
        'station_idx': data.get('idx'),
        'time_iso': data.get('time', {}).get('iso'),
        'time_s': data.get('time', {}).get('s'),
        'time_tz': data.get('time', {}).get('tz'),
        'time_v': data.get('time', {}).get('v'),
        'attributions': '|'.join([attr.get('name', '') for attr in data.get('attributions', []) if isinstance(attr, dict) and 'name' in attr]),
        # New fields with None as default
        'pm25_trend': None,
        'pm25_24h_change': None,
        'pm25_7d_avg': None,
        'heat_index': None,
        'hour_of_day': now.hour,
        'day_of_week': now.weekday(),
        'is_weekend': 1 if now.weekday() >= 5 else 0,
        'month': now.month,
        'season': (now.month % 12 + 3) // 3,
        'pm25_trend_slope': None,
        'aqhi': None,
    }

    # Calculate ratios
    try:
        flat_data['pm25_pm10_ratio'] = flat_data['pm25'] / flat_data['pm10'] if flat_data['pm25'] and flat_data['pm10'] else None
    except Exception:
        flat_data['pm25_pm10_ratio'] = None
    try:
        flat_data['pm25_no2_ratio'] = flat_data['pm25'] / flat_data['no2'] if flat_data['pm25'] and flat_data['no2'] else None
    except Exception:
        flat_data['pm25_no2_ratio'] = None
    try:
        flat_data['pm25_o3_ratio'] = flat_data['pm25'] / flat_data['o3'] if flat_data['pm25'] and flat_data['o3'] else None
    except Exception:
        flat_data['pm25_o3_ratio'] = None

    # Add all individual air quality indices (iaqi)
    for key, value in iaqi.items():
        if isinstance(value, dict) and 'v' in value and f'iaqi_{key}' not in flat_data:
            flat_data[f'iaqi_{key}'] = value['v']

    # Add forecast data if available
    forecast = data.get('forecast', {})
    for day, day_data in forecast.get('daily', {}).items():
        if isinstance(day_data, list):
            for i, item in enumerate(day_data):
                if isinstance(item, dict) and 'avg' in item and 'day' in item:
                    flat_data[f'forecast_{day}_{i}_avg'] = item.get('avg')
                    flat_data[f'forecast_{day}_{i}_day'] = item.get('day')
                    flat_data[f'forecast_{day}_{i}_max'] = item.get('max')
                    flat_data[f'forecast_{day}_{i}_min'] = item.get('min')

    # Add debugging info
    if 'debug' in data:
        flat_data['debug'] = str(data['debug'])

    return flat_data

def fetch_multiple_days_data(start_date, end_date, api_token=None, city="Milpitas"):
    """
    Fetch air quality data for a city for each day in a date range.
    Note: Only fetches current data for each day (not true historical data).

    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        api_token (str, optional): API token for AQICN.
        city (str): City name.

    Returns:
        pd.DataFrame: DataFrame of flattened air quality data.
    """
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
        time.sleep(1)  # Avoid rate limits

    if all_data:
        df = pd.DataFrame(all_data)
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
