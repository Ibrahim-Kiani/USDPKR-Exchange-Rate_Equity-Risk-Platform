import os
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
import time
import logging
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Output directory inside the Airflow container
output_dir = '/opt/airflow/dataKse'

def ensure_dir(output_dir):
    """
    Ensure the output directory exists.
    
    Args:
        output_dir (str): Directory to create if it doesn't exist
        
    Returns:
        str: Path to the created directory
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    except Exception as e:
        logging.error(f"Failed to create directory {output_dir}: {str(e)}")
        raise

def create_session_with_retries():
    """
    Create a requests session with retry strategy.
    
    Returns:
        requests.Session: Session with retry configuration
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def fetch_yahoo_finance(ticker, output_dir, filename, start_date='1900-01-01'):
    """
    Fetch financial data from Yahoo Finance, resample it to a monthly frequency,
    and save it to a CSV file.

    Args:
        ticker (str): The ticker symbol for the data on Yahoo Finance.
        output_dir (str): Directory to save the output file.
        filename (str): The name of the CSV file to save the data.
        start_date (str): The start date for fetching data (default: '1900-01-01').
        
    Returns:
        str: Path to the saved file if successful, None otherwise
    """
    try:
        logging.info(f"Fetching {ticker} from Yahoo Finance...")
        
        # Add retry logic for yfinance
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                df = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logging.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2

        if df.empty:
            raise ValueError(f"No data downloaded for ticker '{ticker}'")

        # Use 'Close' price since auto_adjust=True handles adjustments
        price_col = "Close"
        df = df[[price_col]].resample('M').last()

        # Validate data
        if len(df) < 12:  # At least 1 year of data
            raise ValueError(f"Insufficient data points for {ticker}")

        # Map tickers to friendly column names
        column_map = {
            '^KSE': 'KSE100',
            'PKR=X': 'USDPKR',
            'CL=F': 'OIL',
            'GC=F': 'GOLD'
        }
        df.columns = [column_map.get(ticker, ticker)]
        df.index.name = 'DATE'

        out_path = os.path.join(output_dir, filename)
        df.to_csv(out_path, date_format='%Y-%m-%d')
        
        logging.info(f"Saved {ticker} data to {out_path}")
        logging.info(f"Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
        
        return out_path

    except Exception as e:
        logging.error(f"Error fetching {ticker}: {str(e)}")
        raise

def fetch_imf_data(indicator_id, output_dir, filename, indicator_name, frequency='M'):
    """
    Fetch IMF data for a given frequency, process it, and save to CSV.

    Args:
        indicator_id (str): The IMF series ID for the indicator.
        output_dir (str): Directory to save the output file.
        filename (str): The name of the CSV file to save the data.
        indicator_name (str): The desired column name for the data.
        frequency (str): The data frequency ('M' for monthly, 'A' for annual).
        
    Returns:
        str: Path to the saved file if successful, None otherwise
    """
    try:
        logging.info(f"Fetching '{indicator_name}' from IMF (ID: {indicator_id})...")
        url = f"http://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/IFS/{frequency}.PK.{indicator_id}"

        session = create_session_with_retries()
        response = session.get(url)
        response.raise_for_status()
        data = response.json()

        try:
            series = data['CompactData']['DataSet']['Series']
            if not isinstance(series, list):
                series = [series]
        except (KeyError, TypeError):
            raise ValueError(f"No data or invalid data structure for indicator '{indicator_id}'")

        records = []
        for s in series:
            obs_list = s.get('Obs', [])
            for item in obs_list:
                val = item.get('@OBS_VALUE')
                if val is not None:
                    records.append({
                        'DATE': pd.to_datetime(item['@TIME_PERIOD']),
                        indicator_name: float(val)
                    })

        if not records:
            raise ValueError(f"No valid records found for {indicator_id}")

        df = pd.DataFrame(records).set_index('DATE').sort_index()
        df.dropna(inplace=True)
        
        # Validate data
        if len(df) < 12:  # At least 1 year of data
            raise ValueError(f"Insufficient data points for {indicator_name}")
            
        df.index.name = 'DATE'

        out_path = os.path.join(output_dir, filename)
        df.to_csv(out_path, date_format='%Y-%m-%d')
        
        logging.info(f"Saved {indicator_name} data to {out_path}")
        logging.info(f"Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
        
        return out_path

    except Exception as e:
        logging.error(f"Error fetching IMF data for {indicator_name}: {str(e)}")
        raise

def fetch_sbp_placeholder(series_id, output_dir, filename):
    """
    Generates placeholder data to simulate fetching from the State Bank of Pakistan (SBP).
    
    Args:
        series_id (str): The identifier for the series to generate.
        output_dir (str): Directory to save the output file.
        filename (str): The name of the CSV file to save the data.
        
    Returns:
        str: Path to the saved file if successful, None otherwise
    """
    try:
        logging.info(f"Generating placeholder data for SBP series: {series_id}...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 20)
        date_range = pd.date_range(start=start_date, end=end_date, freq='M')
        df = pd.DataFrame(index=date_range)
        df.index.name = 'DATE'

        if series_id == 'INTEREST_RATE':
            df['INTEREST_RATE'] = np.random.uniform(5.0, 22.0, len(date_range)).round(2)
        elif series_id == 'M2':
            m2_values = 15000 * np.exp(np.linspace(0, 2, len(date_range)))
            m2_values *= 1 + np.random.normal(0, 0.02, len(date_range))
            df['M2'] = m2_values.round(2)
        elif series_id == 'FOREX_RESERVES':
            df['FOREX_RESERVES'] = np.random.uniform(7.0, 25.0, len(date_range)).round(2)
        else:
            raise ValueError(f"Unknown series ID: {series_id}")

        out_path = os.path.join(output_dir, filename)
        df.to_csv(out_path, date_format='%Y-%m-%d')
        
        logging.info(f"Saved placeholder {series_id} data to {out_path}")
        logging.info(f"Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
        
        return out_path

    except Exception as e:
        logging.error(f"Error generating placeholder data for {series_id}: {str(e)}")
        raise

def run_ingestion(output_dir):
    """
    Main function to run the data ingestion pipeline.
    
    Args:
        output_dir (str): Directory to save the output files.
        
    Returns:
        dict: Dictionary of file paths for each data source
    """
    try:
        ensure_dir(output_dir)
        results = {}

        data_sources = {
            # Yahoo Finance Data (Monthly)
            'fetch_kse100': ('^KSE', 'KSE100.csv', 'yahoo'),
            'fetch_usdpkr': ('PKR=X', 'USDPKR.csv', 'yahoo'),
            'fetch_oil_prices': ('CL=F', 'OIL.csv', 'yahoo'),
            'fetch_gold_prices': ('GC=F', 'GOLD.csv', 'yahoo'),

            # IMF Data (Monthly & Annual)
            'fetch_inflation': ('PCPI_IX', 'INFLATION.csv', 'imf_monthly', 'INFLATION_YOY'),

            # SBP Placeholder Data (Monthly)
            'fetch_interest_rate': ('INTEREST_RATE', 'INTEREST_RATE.csv', 'sbp'),
            'fetch_m2': ('M2', 'M2.csv', 'sbp'),
            'fetch_forex_reserves': ('FOREX_RESERVES', 'FOREX_RESERVES.csv', 'sbp'),
        }

        for name, params in data_sources.items():
            logging.info(f"\n----- Running: {name} -----")
            source_type = params[2]

            try:
                if source_type == 'yahoo':
                    results[name] = fetch_yahoo_finance(ticker=params[0], output_dir=output_dir, filename=params[1])
                elif source_type == 'imf_monthly':
                    results[name] = fetch_imf_data(indicator_id=params[0], output_dir=output_dir, filename=params[1], 
                                                 indicator_name=params[3], frequency='M')
                elif source_type == 'sbp':
                    results[name] = fetch_sbp_placeholder(series_id=params[0], output_dir=output_dir, filename=params[1])
                
                # Add delay between API calls
                time.sleep(2)
                
            except Exception as e:
                logging.error(f"Failed to fetch {name}: {str(e)}")
                raise

        logging.info("\n----- Data fetching complete. -----")
        return results

    except Exception as e:
        logging.error(f"Error in run_ingestion: {str(e)}")
        raise
