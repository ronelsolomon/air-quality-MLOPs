import os
import time
import pandas as pd
from datetime import datetime, timedelta
from feature_store import AirQualityFeatureStore
from app import get_air_quality_data

def fetch_historical_data(
    start_date: str,
    end_date: str,
    output_file: str = "training_data.csv",
    api_token: str = None,
    city: str = "Milpitas",
    delay_sec: float = 1.0,
    log_errors: bool = True
):
    """
    Fetch historical air quality data for a date range and save to CSV

    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        output_file (str): Path to save the output CSV file
        api_token (str): AQICN API token. If None, will try to use AQICN_TOKEN env var
        city (str): City name for air quality data
        delay_sec (float): Seconds to wait between API calls
        log_errors (bool): Whether to log errors to a file
    """
    # Initialize feature store (optional)
    feature_store = AirQualityFeatureStore()

    # Get API token from environment if not provided
    if api_token is None:
        api_token = os.getenv("AQICN_TOKEN")
        if api_token is None:
            raise ValueError("API token not provided and AQICN_TOKEN environment variable not set")

    # Validate and convert string dates to datetime objects
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Dates must be in YYYY-MM-DD format")

    if start > end:
        raise ValueError("Start date must not be after end date")

    # Warn if output file exists
    if os.path.exists(output_file):
        print(f"Warning: Output file '{output_file}' already exists and will be overwritten.")

    # Prepare error log file if needed
    error_log_file = None
    if log_errors:
        error_log_file = output_file.rsplit('.', 1)[0] + "_errors.log"

    # List to store all data points
    all_data = []

    # Fetch data for each day in the range
    current_date = start
    while current_date <= end:
        date_str = current_date.strftime('%Y-%m-%d')
        print(f"Fetching data for {city} on {date_str}...")

        try:
            # Pass date to get_air_quality_data (assumes function supports this)
            data = get_air_quality_data(api_token, city, date=date_str)

            if data:
                # Prepare and store features
                features = feature_store.prepare_feature_data(data)
                # Optionally store features (warn if store is not persistent)
                try:
                    feature_store.store_features(features)
                except Exception as fs_exc:
                    print(f"Warning: Could not store features in feature store: {fs_exc}")

                # Add to our collection for CSV
                all_data.append(features)
                print(f"Successfully stored data for {date_str}")
            else:
                print(f"No data available for {date_str}")

        except Exception as e:
            err_msg = f"Error fetching data for {date_str}: {str(e)}"
            print(err_msg)
            if log_errors and error_log_file:
                with open(error_log_file, "a") as f:
                    f.write(err_msg + "\n")

        # Move to next day
        current_date += timedelta(days=1)
        time.sleep(delay_sec)  # Respect API rate limits

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
    parser.add_argument('--start-date', required=True, type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', default='training_data.csv', type=str, help='Output CSV file path')
    parser.add_argument('--api-token', type=str, help='AQICN API token')
    parser.add_argument('--city', default='Milpitas', type=str, help='City name')
    parser.add_argument('--delay', default=1.0, type=float, help='Delay (seconds) between API calls')
    parser.add_argument('--no-log-errors', action='store_true', help='Do not log errors to a file')

    args = parser.parse_args()

    fetch_historical_data(
        start_date=args.start_date,
        end_date=args.end_date,
        output_file=args.output,
        api_token=args.api_token,
        city=args.city,
        delay_sec=args.delay,
        log_errors=not args.no_log_errors
    )

