import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
import pickle

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
        df = df.resample('M').last()  # 'M' means month end

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
                how='outer'
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

def transform_data(data_dir='/opt/airflow/data', output_filename='transformed_data.csv'):
    """
    Main function to transform the data.

    Args:
        data_dir (str): Path to the directory containing the CSV files
        output_filename (str): Name of the output file
    """
    output_path = os.path.join(data_dir, output_filename)

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

def load_and_merge_data(data_dir):
    """
    Load all CSV files from the data directory and merge them on DATE using the user's custom logic.
    Args:
        data_dir (str): Directory containing the CSV files.
    Returns:
        str: Path to the saved merged DataFrame.
    """
    try:
        # Load each file
        gold = pd.read_csv(os.path.join(data_dir, 'GOLD.csv'))
        forex = pd.read_csv(os.path.join(data_dir, 'FOREX_RESERVES.csv'))
        inflation = pd.read_csv(os.path.join(data_dir, 'INFLATION.csv'))
        oil = pd.read_csv(os.path.join(data_dir, 'OIL.csv'))
        interest = pd.read_csv(os.path.join(data_dir, 'INTEREST_RATE.csv'))
        kse = pd.read_csv(os.path.join(data_dir, 'KSE100.csv'))
        usdpkr = pd.read_csv(os.path.join(data_dir, 'USDPKR.csv'))
        m2 = pd.read_csv(os.path.join(data_dir, 'M2.csv'))
        arr = [gold, forex, inflation, oil, interest, kse, usdpkr, m2]
        for i in arr:
            i['DATE'] = pd.to_datetime(i['DATE']).dt.to_period('M')
        del arr[0]
        df = gold.copy()
        for i in arr:
            df = pd.merge(df, i, how='outer', on='DATE')
        df['YEAR'] = df['DATE'].dt.year
        df['MONTH'] = df['DATE'].dt.month
        df['QUARTER'] = df['DATE'].dt.quarter
        df = df.set_index('DATE')
        df = df.query('YEAR > 2004')
        for col in df.columns:
            if df[col].isna().sum() > 0:
                df[col].fillna(0, inplace=True)
                df[f'{col}_naflag'] = df[col] == 0
        # Save the merged DataFrame to a temporary file
        temp_file = os.path.join(data_dir, 'merged_data.csv')
        df.to_csv(temp_file)
        return temp_file
    except Exception as e:
        logging.error(f"Error in load_and_merge_data: {str(e)}")
        raise

def engineer_features(df_path):
    """
    Engineer features from the merged dataframe.
    
    Args:
        df_path (str): Path to the merged DataFrame CSV file.
        
    Returns:
        str: Path to the saved engineered features DataFrame.
    """
    try:
        # Load the DataFrame from the CSV file
        logging.info(f"Loading DataFrame from path: {df_path}")
        df = pd.read_csv(df_path)
        
        logging.info(f"Starting feature engineering. Input DataFrame shape: {df.shape}")
        logging.info(f"Input DataFrame columns: {df.columns.tolist()}")
        logging.info(f"Input DataFrame dtypes:\n{df.dtypes}")
        
        # Convert DATE to datetime
        logging.info("Converting DATE to datetime")
        df['DATE'] = pd.to_datetime(df['DATE'])
        
        # Create time-based features
        logging.info("Creating time-based features")
        df['YEAR'] = df['DATE'].dt.year
        df['MONTH'] = df['DATE'].dt.month
        df['QUARTER'] = df['DATE'].dt.quarter
        
        # Set DATE as index and filter years
        logging.info("Setting DATE as index and filtering years > 2004")
        df = df.set_index('DATE')
        df = df.query('YEAR > 2004')
        logging.info(f"DataFrame shape after year filtering: {df.shape}")
        
        # Handle missing values and create NA flags
        logging.info("Handling missing values and creating NA flags")
        for col in df.columns:
            na_count = df[col].isna().sum()
            if na_count > 0:
                logging.info(f"Column {col} has {na_count} missing values")
                df[col].fillna(0, inplace=True)
                df[f'{col}_naflag'] = df[col] == 0
        
        # Create cyclical month features
        logging.info("Creating cyclical month features")
        df['MONTH_SIN'] = np.sin(2 * np.pi * df['MONTH']/12)
        df['MONTH_COS'] = np.cos(2 * np.pi * df['MONTH']/12)
        
        # Create economic features
        logging.info("Creating economic features")
        df['USDPKR_diff'] = df['USDPKR'].diff().shift(1)
        df['REAL_INTEREST'] = df['INTEREST_RATE'] - df['INFLATION_YOY']
        df['OIL_IN_PKR'] = df['OIL'] * df['USDPKR']
        
        # Create lag features
        logging.info("Creating lag features")
        for col in ['KSE100', 'INFLATION_YOY', 'INTEREST_RATE', 'USDPKR', 'OIL', 'FOREX_RESERVES', 'GOLD', 'M2']:
            if col not in df.columns:
                logging.warning(f"Column {col} not found in DataFrame")
                continue
            df[f'{col}_lag1'] = df[col].shift(1)
            df[f'{col}_lag3'] = df[col].shift(3)
            df[f'{col}_lag12'] = df[col].shift(12)
        
        # Create rolling statistics
        logging.info("Creating rolling statistics")
        for col in ['KSE100', 'OIL', 'GOLD']:
            if col not in df.columns:
                logging.warning(f"Column {col} not found in DataFrame")
                continue
            df[f'{col}_roll_mean3'] = df[col].rolling(window=3).mean().shift(1)
            df[f'{col}_roll_std3'] = df[col].rolling(window=3).std().shift(1)
        
        # Reset index to make DATE a column again
        logging.info("Resetting index")
        df = df.reset_index()
        
        # Drop any rows with NaN values (e.g., from lagged features)
        df.dropna(inplace=True)
        
        # Save the engineered features to a file
        output_path = os.path.join(os.path.dirname(df_path), 'engineered_features.csv')
        df.to_csv(output_path, index=False)
        
        logging.info(f"Feature engineering complete. Final DataFrame shape: {df.shape}")
        logging.info(f"Final DataFrame columns: {df.columns.tolist()}")
        
        return output_path
    except Exception as e:
        logging.error(f"Error in engineer_features: {str(e)}")
        logging.error(f"Error type: {type(e)}")
        logging.error(f"Input type: {type(df_path)}")
        logging.error(f"Input value: {df_path}")
        raise

def prepare_data_for_modeling(df_path):
    """
    Prepare data for modeling by splitting into train/test sets and saving necessary files.
    
    Args:
        df_path (str): Path to the CSV file containing the engineered features.
        
    Returns:
        tuple: (X_train_path, y_train_path, selected_features_path, train_test_data_path)
    """
    try:
        # 1) Load raw features
        df = pd.read_csv(df_path)
        target = 'USDPKR'
        feature_columns = [c for c in df.columns if c not in (target, 'DATE')]

        # 2) Split into train/test
        train_size = int(len(df) * 0.8)
        X_train = df.loc[:train_size-1, feature_columns]
        X_test  = df.loc[train_size:,     feature_columns]
        y_train = df.loc[:train_size-1, target]
        y_test  = df.loc[train_size:,     target]

        # 3) Ensure models dir exists
        models_dir = '/opt/airflow/data/models'
        os.makedirs(models_dir, exist_ok=True)

        # 4) Dump pickles in **binary** mode
        selected_features_path = os.path.join(models_dir, 'selected_features.pkl')
        with open(selected_features_path, 'wb') as f:
            pickle.dump({
                'X_train_selected': X_train,
                'X_test_selected' : X_test
            }, f)

        train_test_data_path = os.path.join(models_dir, 'train_test_data.pkl')
        with open(train_test_data_path, 'wb') as f:
            pickle.dump({
                'y_train': y_train,
                'y_test' : y_test
            }, f)

        # 5) (Optional) also save raw CSV splits if you need them elsewhere:
        X_train_path = os.path.join(models_dir, 'X_train.csv')
        y_train_path = os.path.join(models_dir, 'y_train.csv')
        X_train.to_csv(X_train_path, index=False)
        y_train.to_csv(y_train_path, index=False)

        return X_train_path, y_train_path, selected_features_path, train_test_data_path

    except Exception as e:
        logging.error(f"Error in prepare_data_for_modeling: {e}")
        raise

def run_transformation(data_dir, output_dir):
    """
    Main function to run the data transformation pipeline.
    
    Args:
        data_dir (str): Directory containing input CSV files.
        output_dir (str): Directory to save the transformed data.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and merge data
    print("Loading and merging data...")
    df_path = load_and_merge_data(data_dir)
    
    # Engineer features
    print("Engineering features...")
    features_path = engineer_features(df_path)
    
    # Prepare data for modeling
    print("Preparing data for modeling...")
    X_train_path, y_train_path, selected_features_path, train_test_data_path = prepare_data_for_modeling(features_path)
    
    # Save transformed data
    print("Saving transformed data...")
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)
    pd.DataFrame({'selected_features': pd.read_csv(selected_features_path)['feature']}).to_csv(os.path.join(output_dir, 'selected_features.csv'))
    pd.DataFrame({'train_test_data': pd.read_csv(train_test_data_path).to_dict()}).to_csv(os.path.join(output_dir, 'train_test_data.csv'))
    
    print("Transformation complete!")
    return X_train, y_train
