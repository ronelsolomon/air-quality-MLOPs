import os
import pandas as pd
from datetime import datetime, timedelta
from app import get_air_quality_data  # Your function that calls the API

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
            data = get_air_quality_data(api_token, city, date=date_str)
            if data:
                all_data.append(data)
        except Exception as e:
            print(f"Error fetching data for {date_str}: {str(e)}")
        current_date += timedelta(days=1)

    if all_data:
        df = pd.DataFrame(all_data)
        print(f"Fetched {len(df)} records.")
        return df
    else:
        print("No data fetched.")
        return pd.DataFrame()  # Return empty DataFrame for consistency

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--api-token", required=False)
    parser.add_argument("--city", default="Milpitas")
    args = parser.parse_args()

    # Prefer CLI token, fallback to environment
    api_token = args.api_token or os.getenv("AQICN_TOKEN") or "18fed86f6f05b8695480ad90a15a00661e1b28de"
    if not api_token:
        raise ValueError("API token must be provided via --api-token or AQICN_TOKEN environment variable.")

    new_data = fetch_historical_datas(args.start_date, args.end_date, api_token=api_token, city=args.city)

    if new_data.empty:
        print("No new data to save.")
        return

    if os.path.exists(args.output):
        # Read existing data
        existing = pd.read_csv(args.output)
        # Combine and drop duplicates (by all columns)
        combined = pd.concat([existing, new_data], ignore_index=True).drop_duplicates()
        combined.to_csv(args.output, index=False)
        print(f"Combined with existing data. Saved {len(combined)} records to {args.output}")
    else:
        new_data.to_csv(args.output, index=False)
        print(f"Saved new data to {args.output}")

if __name__ == "__main__":
    main()

