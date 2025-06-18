import os
import pandas as pd
from pandas_datareader import data as pdr

# Output directory inside the Airflow container
output_dir = '/opt/airflow/data'

def ensure_dir():
    """Ensure the output directory exists."""
    os.makedirs(output_dir, exist_ok=True)

def fetch_fred(series_id, filename, start_date='1900-01-01'):
    """
    Fetch a FRED series, resample to monthly frequency, and save to CSV.

    Args:
        series_id (str): FRED series ID
        filename (str): Path to save the CSV file
        start_date (str): Start date for the data (default: '1900-01-01')
    """
    # Fetch data from FRED
    df = pdr.DataReader(series_id, 'fred', start=start_date)
    original_count = len(df)

    # Check if the data is already monthly
    is_monthly = False
    if len(df) > 1:
        # Get the first two dates to check frequency
        dates = df.index.sort_values()[:2]
        days_diff = (dates[1] - dates[0]).days
        # If difference is approximately a month (28-31 days)
        is_monthly = 28 <= days_diff <= 31

    # Resample to monthly frequency if not already monthly
    if not is_monthly:
        # Convert index to datetime if it's not already
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Resample to end of month (EOM) and take the last value of each month
        df = df.resample('ME').last()
        print(f"Resampled {series_id} from {original_count} rows to {len(df)} monthly rows")
    else:
        print(f"{series_id} is already monthly with {len(df)} rows")

    # Print date range
    print(f"Date range for {series_id}: {df.index.min()} to {df.index.max()}")

    # Save to CSV
    df.to_csv(os.path.join(output_dir, filename))

# Dictionary of FRED series to fetch
fred_series = {
    'fetch_fedfunds':      ('FEDFUNDS',      'FEDFUNDS.csv'),
    'fetch_ppiaco':        ('PPIACO',        'PPIACO.csv'),
    'fetch_crude_oil':     ('DCOILWTICO',    'DCOILWTICO.csv'),
    'fetch_m2':            ('M2SL',          'M2SL.csv'),
    'fetch_permit':        ('PERMIT',        'PERMIT.csv'),
    'fetch_cpi':           ('CPALTT01USM657N','CPI.csv'),
}