import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

def load_data(data_dir):
    """
    Load all CSV files from the data directory.

    Args:
        data_dir (str): Path to the directory containing the CSV files

    Returns:
        dict: Dictionary of dataframes with filenames as keys
    """
    data_files = {}
    # First try to load the new files with more historical data
    new_files = [
        ('FEDFUNDS_new.csv', 'FEDFUNDS'),
        ('CPI_new.csv', 'CPI')
    ]

    for filename, series_name in new_files:
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            # Read the CSV file
            df = pd.read_csv(file_path)
            data_files[series_name] = df
            print(f"Loaded {filename} with {len(df)} rows from {df['DATE'].min()} to {df['DATE'].max()}")

    # Load the rest of the files
    for file in os.listdir(data_dir):
        if (file.endswith('.csv') and
            not file.startswith('transformed_data') and
            not file in [f for f, _ in new_files]):

            file_path = os.path.join(data_dir, file)
            # Extract the series name from the filename (without .csv extension)
            series_name = os.path.splitext(file)[0]
            # Read the CSV file
            df = pd.read_csv(file_path)
            data_files[series_name] = df
            print(f"Loaded {file} with {len(df)} rows from {df['DATE'].min()} to {df['DATE'].max()}")

    return data_files

def preprocess_dataframes(dataframes):
    """
    Preprocess each dataframe:
    - Convert DATE to datetime
    - Set DATE as index
    - Resample to monthly frequency (last day of month)
    - Rename value column to match the series name

    Args:
        dataframes (dict): Dictionary of dataframes

    Returns:
        dict: Dictionary of preprocessed dataframes
    """
    processed_dfs = {}

    for name, df in dataframes.items():
        # Convert DATE to datetime
        df['DATE'] = pd.to_datetime(df['DATE'])

        # Set DATE as index
        df = df.set_index('DATE')

        # Rename the value column to the series name
        value_col = df.columns[0]  # First column after DATE (which is now the index)
        df = df.rename(columns={value_col: name})

        # Resample to monthly frequency (last day of month)
        # Use last value of the month for each series
        df = df.resample('ME').last()  # 'ME' means month end

        # Create a Month column in format YYYY-MM for potential future use
        df['Month'] = df.index.strftime('%Y-%m')

        processed_dfs[name] = df

    return processed_dfs

def merge_dataframes(dataframes):
    """
    Merge all dataframes on the DATE index.

    Args:
        dataframes (dict): Dictionary of preprocessed dataframes

    Returns:
        pd.DataFrame: Merged dataframe
    """
    # Start with the first dataframe
    merged_df = None

    for name, df in dataframes.items():
        if merged_df is None:
            # Include both the value column and the Month column
            merged_df = df[[name, 'Month']].copy()
        else:
            # Merge on index (DATE)
            merged_df = merged_df.join(
                df[[name]],
                how='inner'
            )

    # Sort by index (DATE)
    merged_df = merged_df.sort_index()

    return merged_df

def time_series_preprocessing(df):
    """
    Perform time series preprocessing:
    - Handle missing values
    - Handle outliers
    - Sort by date
    - Create lag features
    - Create rolling statistics

    Args:
        df (pd.DataFrame): Merged dataframe

    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    # Sort by index (DATE)
    df = df.sort_index()

    # Handle missing values with forward fill, then backward fill
    df = df.ffill().bfill()

    # Handle outliers using IQR method
    for col in df.columns:
        if col != 'Month':  # Skip the Month column
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            # Define bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Replace outliers with bounds
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

    # Create lag features (1, 3, 6, 12 months) for each column
    for col in df.columns:
        if col != 'Month':  # Skip the Month column
            for lag in [1, 3, 6, 12]:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    # Create rolling statistics (mean, std) with windows of 3, 6, 12 months
    for col in df.columns:
        if col != 'Month' and not col.endswith(('_lag_1', '_lag_3', '_lag_6', '_lag_12')):
            for window in [3, 6, 12]:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()

    # Drop rows with NaN values (due to lag and rolling features)
    df = df.dropna()

    return df

def save_transformed_data(df, output_path):
    """
    Save the transformed dataframe to a CSV file.

    Args:
        df (pd.DataFrame): Transformed dataframe
        output_path (str): Path to save the CSV file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path)
    print(f"Transformed data saved to {output_path}")

def main():
    # Define paths
    base_dir = Path(__file__).parent.parent  # Project root directory
    data_dir = base_dir / 'data'
    output_path = data_dir / 'transformed_data.csv'

    # Load data
    print("Loading data...")
    dataframes = load_data(data_dir)

    # Preprocess dataframes
    print("Preprocessing dataframes...")
    processed_dfs = preprocess_dataframes(dataframes)

    # Merge dataframes
    print("Merging dataframes...")
    merged_df = merge_dataframes(processed_dfs)

    # Time series preprocessing
    print("Performing time series preprocessing...")
    transformed_df = time_series_preprocessing(merged_df)

    # Save transformed data
    print("Saving transformed data...")
    save_transformed_data(transformed_df, output_path)

    print("Transformation complete!")

if __name__ == "__main__":
    main()