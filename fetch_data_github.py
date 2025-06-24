#!/usr/bin/env python3
"""
Lightweight script for GitHub Actions to fetch air quality data.
This version doesn't depend on Feast to avoid dependency issues.
"""
import sys
import os
import pandas as pd
from datetime import datetime, timedelta

def fetch_air_quality_data(start_date: str, end_date: str, output_file: str, api_token: str = None):
    """
    Fetch air quality data from the API between start_date and end_date.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        output_file: Path to save the CSV file
        api_token: Optional API token
    """
    # This is a placeholder - replace with your actual API call
    # For now, we'll just create a dummy DataFrame
    print(f"[INFO] Fetching data from {start_date} to {end_date}")
    
    # Create a simple DataFrame with the date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    data = []
    
    for date in date_range:
        # Replace this with actual API call
        data.append({
            'timestamp': date,
            'aqi': 50,  # Example value
            'pm25': 12.5,  # Example value
            'pm10': 20.0,  # Example value
            'temperature': 22.0,  # Example value
            'humidity': 65.0  # Example value
        })
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"[INFO] Saved {len(df)} records to {output_file}")
    return df

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--api-token", required=True)
    args = parser.parse_args()

    new_data = fetch_new_data(args.start_date, args.end_date, args.api_token)

    if os.path.exists(args.output):
        # Read existing data
        existing = pd.read_csv(args.output)
        # Combine and drop duplicates (if needed)
        combined = pd.concat([existing, new_data]).drop_duplicates()
        combined.to_csv(args.output, index=False)
    else:
        new_data.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()
