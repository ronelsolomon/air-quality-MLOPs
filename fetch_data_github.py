import os
import pandas as pd
from datetime import datetime, timedelta
from app import get_air_quality_data  # Your function that calls the API

def fetch_historical_datas(start_date, end_date, output_file="training_data.csv", api_token=None, city="Milpitas"):
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
            data = get_air_quality_data(api_token, city, date=date_str)
            if data:
                all_data.append(data)
        except Exception as e:
            print(f"Error fetching data for {date_str}: {str(e)}")
        current_date += timedelta(days=1)

    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} records to {output_file}")
    else:
        print("No data fetched.")

# Usage example:
# fetch_historical_data("2025-06-20", "2025-06-23", output_file="aq_data.csv", api_token="YOUR_TOKEN")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--api-token", required=True)
    args = parser.parse_args()

    API_TOKEN = os.getenv("AQICN_TOKEN")

    new_data = fetch_historical_datas(args.start_date, args.end_date, api_token=API_TOKEN)

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
