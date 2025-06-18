import os
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, Lasso, Ridge, ElasticNet, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import joblib
import warnings
import logging

warnings.filterwarnings('ignore')

# Global variables for data paths
DATA_DIR = '/opt/airflow/data'
OUTPUT_DIR = '/opt/airflow/data/model_output'
TRANSFORMED_DATA_PATH = os.path.join(DATA_DIR, 'transformed_data.csv')
SELECTED_FEATURES_PATH = os.path.join(OUTPUT_DIR, 'selected_features.pkl')
TRAIN_TEST_DATA_PATH = os.path.join(OUTPUT_DIR, 'train_test_data.pkl')
SCALER_PATH = os.path.join(OUTPUT_DIR, 'Scaler.pkl')
MLR_MODEL_PATH = os.path.join(OUTPUT_DIR, 'MLR.pkl')
RF_MODEL_PATH = os.path.join(OUTPUT_DIR, 'RandomForest.pkl')
SVR_MODEL_PATH = os.path.join(OUTPUT_DIR, 'SVR.pkl')
MODEL_COMPARISON_PATH = os.path.join(OUTPUT_DIR, 'model_comparison.csv')
MODEL_PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, 'model_predictions.csv')
MODEL_COMPARISON_JSON_PATH = os.path.join(OUTPUT_DIR, 'model_comparison.json')

def ensure_output_dir():
    """Ensure the output directory exists."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR

def load_data():
    """
    Load the transformed data from CSV.

    Returns:
        pd.DataFrame: Loaded dataframe
    """
    df = pd.read_csv(TRANSFORMED_DATA_PATH, index_col=0, parse_dates=True)
    print(f"Loaded data with shape: {df.shape}")
    return df

def prepare_data(df=None, target_col='KSE-100', test_size=0.2):
    """
    Prepare data for modeling by splitting into features and target,
    and creating train/test splits.

    Args:
        df (pd.DataFrame, optional): Input dataframe. If None, will load from TRANSFORMED_DATA_PATH
        target_col (str): Name of the target column
        test_size (float): Proportion of data to use for testing

    Returns:
        str: Path to the saved train/test data
    """
    try:
        # If no dataframe is provided, load it from the transformed data path
        if df is None:
            df = load_data()

        # Ensure the dataframe is sorted by date
        df = df.sort_index()

        # Print column names for debugging
        print(f"Columns in dataframe: {df.columns.tolist()}")

        # Create a list of columns to drop
        cols_to_drop = [col for col in ['Month', target_col] if col in df.columns]

        # Separate features and target
        # Check if target_col exists in the dataframe
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe. Available columns: {df.columns.tolist()}")

        # Drop columns safely
        X = df.drop(cols_to_drop, axis=1)
        y = df[target_col]

        # Calculate the split point
        split_idx = int(len(df) * (1 - test_size))
        print(f"Split index: {split_idx} (total rows: {len(df)})")

        # Split the data
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
        if X_train.shape[0] == 0 or y_train.shape[0] == 0:
            raise ValueError("Training set is empty! Check your data and split parameters.")
        if X_test.shape[0] == 0 or y_test.shape[0] == 0:
            raise ValueError("Test set is empty! Check your data and split parameters.")

        # Save the data for later steps
        data_dict = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(TRAIN_TEST_DATA_PATH), exist_ok=True)

        # Save the data
        with open(TRAIN_TEST_DATA_PATH, 'wb') as f:
            pickle.dump(data_dict, f)

        print(f"Data saved to {TRAIN_TEST_DATA_PATH}")

        # Return the path to the saved data instead of the data itself
        return TRAIN_TEST_DATA_PATH

    except Exception as e:
        print(f"Error in prepare_data: {str(e)}")
        # Re-raise the exception to make sure Airflow knows the task failed
        raise

def select_features_lasso(data_path=None, alpha_range=np.logspace(-4, 1, 20)):
    """
    Select important features using LassoCV with improved convergence settings.

    Args:
        data_path (str, optional): Path to the saved train/test data. If None, will use TRAIN_TEST_DATA_PATH
        alpha_range (np.array): Range of alpha values to try (reduced for faster computation)

    Returns:
        str: Path to the saved selected features
    """
    try:
        # Use the provided data path or the default one
        data_path = data_path or TRAIN_TEST_DATA_PATH

        print(f"Loading data from {data_path}")

        # Load the data
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)

        X_train = data_dict['X_train']
        y_train = data_dict['y_train']
        X_test = data_dict['X_test']

        # Initialize and fit LassoCV with improved convergence settings
        lasso_cv = LassoCV(
            alphas=alpha_range,  # Reduced number of alphas
            cv=TimeSeriesSplit(n_splits=3),  # Reduced number of splits
            max_iter=100000,  # Increased max iterations
            tol=0.001,  # Increased tolerance for faster convergence
            random_state=42,
            n_jobs=-1  # Use all available cores
        )

        print("Fitting LassoCV model (this may take a moment)...")
        lasso_cv.fit(X_train, y_train)

        # Get the best alpha
        best_alpha = lasso_cv.alpha_
        print(f"Best alpha: {best_alpha:.6f}")

        # Get feature importances
        feature_importances = pd.DataFrame({
            'Feature': X_train.columns,
            'Coefficient': lasso_cv.coef_
        })

        # Sort by absolute coefficient value
        feature_importances['Abs_Coefficient'] = np.abs(feature_importances['Coefficient'])
        feature_importances = feature_importances.sort_values('Abs_Coefficient', ascending=False)

        # Select features with non-zero coefficients
        selected_features = feature_importances[feature_importances['Coefficient'] != 0]['Feature'].tolist()

        # If too few features are selected, take the top 20 features
        if len(selected_features) < 10:
            print("Too few features selected, using top 20 features instead")
            selected_features = feature_importances.nlargest(20, 'Abs_Coefficient')['Feature'].tolist()

        print(f"Selected {len(selected_features)} features out of {X_train.shape[1]}")

        # Create datasets with selected features
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]

        # Save the selected features and data
        selected_data = {
            'X_train_selected': X_train_selected,
            'X_test_selected': X_test_selected,
            'selected_features': selected_features,
            'lasso_model': lasso_cv
        }

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(SELECTED_FEATURES_PATH), exist_ok=True)

        with open(SELECTED_FEATURES_PATH, 'wb') as f:
            pickle.dump(selected_data, f)

        print(f"Selected features saved to {SELECTED_FEATURES_PATH}")

        # Return the path to the saved selected features
        return SELECTED_FEATURES_PATH

    except Exception as e:
        print(f"Error in select_features_lasso: {str(e)}")
        # Re-raise the exception to make sure Airflow knows the task failed
        raise

def scale_features(selected_features_path=None):
    """
    Scale the selected features.

    Args:
        selected_features_path (str, optional): Path to the saved selected features. If None, will use SELECTED_FEATURES_PATH

    Returns:
        str: Path to the saved scaled features
    """
    try:
        # Use the provided path or the default one
        selected_features_path = selected_features_path or SELECTED_FEATURES_PATH

        print(f"Loading selected features from {selected_features_path}")

        # Load the selected features data
        with open(selected_features_path, 'rb') as f:
            selected_data = pickle.load(f)

        X_train_selected = selected_data['X_train_selected']
        X_test_selected = selected_data['X_test_selected']

        # Scale the features
        scaler = StandardScaler()
        X_train_selected_scaled = scaler.fit_transform(X_train_selected)
        X_test_selected_scaled = scaler.transform(X_test_selected)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)

        # Save the scaler
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)

        print(f"Scaler saved to {SCALER_PATH}")

        # Update the selected data with scaled features
        selected_data['X_train_selected_scaled'] = X_train_selected_scaled
        selected_data['X_test_selected_scaled'] = X_test_selected_scaled

        with open(SELECTED_FEATURES_PATH, 'wb') as f:
            pickle.dump(selected_data, f)

        print(f"Scaled features saved to {SELECTED_FEATURES_PATH}")

        # Return the path to the saved selected features with scaled data
        return SELECTED_FEATURES_PATH

    except Exception as e:
        print(f"Error in scale_features: {str(e)}")
        # Re-raise the exception to make sure Airflow knows the task failed
        raise

def train_mlr(selected_features_path=None, train_test_data_path=None):
    """
    Train a Multiple Linear Regression model with optimized settings.

    Args:
        selected_features_path (str, optional): Path to the saved selected features. If None, will use SELECTED_FEATURES_PATH
        train_test_data_path (str, optional): Path to the saved train/test data. If None, will use TRAIN_TEST_DATA_PATH

    Returns:
        str: Path to the saved MLR model
    """
    try:
        # Use the provided paths or the default ones
        selected_features_path = selected_features_path or SELECTED_FEATURES_PATH
        train_test_data_path = train_test_data_path or TRAIN_TEST_DATA_PATH

        print(f"Loading selected features from {selected_features_path}")
        print(f"Loading train/test data from {train_test_data_path}")

        # Load the selected features data
        with open(selected_features_path, 'rb') as f:
            selected_data = pickle.load(f)

        # Load the original data to get y_train
        with open(train_test_data_path, 'rb') as f:
            data_dict = pickle.load(f)

        X_train_selected = selected_data['X_train_selected']
        y_train = data_dict['y_train']

        print("Training MLR model...")

        # Initialize and fit the model
        # LinearRegression is already quite fast, but we can set n_jobs for consistency
        model = LinearRegression(n_jobs=-1)  # Use all available cores
        model.fit(X_train_selected, y_train)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(MLR_MODEL_PATH), exist_ok=True)

        # Save the model
        with open(MLR_MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)

        print(f"MLR model saved to {MLR_MODEL_PATH}")

        # Return the path to the saved model
        return MLR_MODEL_PATH

    except Exception as e:
        print(f"Error in train_mlr: {str(e)}")
        # Re-raise the exception to make sure Airflow knows the task failed
        raise

def train_random_forest(selected_features_path=None, train_test_data_path=None):
    """
    Train a Random Forest model with pre-optimized parameters.

    Args:
        selected_features_path (str, optional): Path to the saved selected features. If None, will use SELECTED_FEATURES_PATH
        train_test_data_path (str, optional): Path to the saved train/test data. If None, will use TRAIN_TEST_DATA_PATH

    Returns:
        str: Path to the saved Random Forest model
    """
    try:
        # Use the provided paths or the default ones
        selected_features_path = selected_features_path or SELECTED_FEATURES_PATH
        train_test_data_path = train_test_data_path or TRAIN_TEST_DATA_PATH

        print(f"Loading selected features from {selected_features_path}")
        print(f"Loading train/test data from {train_test_data_path}")

        # Load the selected features data
        with open(selected_features_path, 'rb') as f:
            selected_data = pickle.load(f)

        # Load the original data to get y_train
        with open(train_test_data_path, 'rb') as f:
            data_dict = pickle.load(f)

        X_train_selected = selected_data['X_train_selected']
        y_train = data_dict['y_train']

        # Use pre-optimized parameters based on previous runs
        # These parameters were found to be optimal in previous grid searches
        best_params = {
            'max_depth': 10,
            'min_samples_leaf': 4,
            'min_samples_split': 2,
            'n_estimators': 50
        }

        print(f"Using pre-optimized Random Forest parameters: {best_params}")

        # Initialize and train model with best parameters
        model = RandomForestRegressor(
            max_depth=best_params['max_depth'],
            min_samples_leaf=best_params['min_samples_leaf'],
            min_samples_split=best_params['min_samples_split'],
            n_estimators=best_params['n_estimators'],
            random_state=42,
            n_jobs=-1  # Use all available cores for faster training
        )

        # Fit the model
        model.fit(X_train_selected, y_train)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(RF_MODEL_PATH), exist_ok=True)

        # Save the model
        with open(RF_MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)

        print(f"Random Forest model saved to {RF_MODEL_PATH}")

        # Return the path to the saved model
        return RF_MODEL_PATH

    except Exception as e:
        print(f"Error in train_random_forest: {str(e)}")
        # Re-raise the exception to make sure Airflow knows the task failed
        raise

def train_svr(selected_features_path=None, train_test_data_path=None):
    """
    Train an SVR model with linear kernel and pre-optimized parameters.

    Args:
        selected_features_path (str, optional): Path to the saved selected features. If None, will use SELECTED_FEATURES_PATH
        train_test_data_path (str, optional): Path to the saved train/test data. If None, will use TRAIN_TEST_DATA_PATH

    Returns:
        str: Path to the saved SVR model
    """
    try:
        # Use the provided paths or the default ones
        selected_features_path = selected_features_path or SELECTED_FEATURES_PATH
        train_test_data_path = train_test_data_path or TRAIN_TEST_DATA_PATH

        print(f"Loading selected features from {selected_features_path}")
        print(f"Loading train/test data from {train_test_data_path}")

        # Load the selected features data
        with open(selected_features_path, 'rb') as f:
            selected_data = pickle.load(f)

        # Load the original data to get y_train
        with open(train_test_data_path, 'rb') as f:
            data_dict = pickle.load(f)

        X_train_selected_scaled = selected_data['X_train_selected_scaled']
        y_train = data_dict['y_train']

        # Use pre-optimized parameters based on previous runs
        # These parameters were found to be optimal in previous grid searches
        best_params = {
            'C': 0.1,
            'epsilon': 0.01
        }

        print(f"Using pre-optimized SVR parameters: {best_params}")

        # Initialize and train model with best parameters
        model = SVR(
            kernel='linear',
            C=best_params['C'],
            epsilon=best_params['epsilon']
        )

        # Fit the model
        model.fit(X_train_selected_scaled, y_train)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(SVR_MODEL_PATH), exist_ok=True)

        # Save the model
        with open(SVR_MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)

        print(f"SVR model saved to {SVR_MODEL_PATH}")

        # Return the path to the saved model
        return SVR_MODEL_PATH

    except Exception as e:
        print(f"Error in train_svr: {str(e)}")
        # Re-raise the exception to make sure Airflow knows the task failed
        raise

def evaluate_mlr(model_path=None, selected_features_path=None, train_test_data_path=None):
    """
    Evaluate the MLR model.

    Args:
        model_path (str, optional): Path to the saved MLR model. If None, will use MLR_MODEL_PATH
        selected_features_path (str, optional): Path to the saved selected features. If None, will use SELECTED_FEATURES_PATH
        train_test_data_path (str, optional): Path to the saved train/test data. If None, will use TRAIN_TEST_DATA_PATH

    Returns:
        str: Path to the saved MLR metrics
    """
    try:
        # Use the provided paths or the default ones
        model_path = model_path or MLR_MODEL_PATH
        selected_features_path = selected_features_path or SELECTED_FEATURES_PATH
        train_test_data_path = train_test_data_path or TRAIN_TEST_DATA_PATH

        print(f"Loading MLR model from {model_path}")
        print(f"Loading selected features from {selected_features_path}")
        print(f"Loading train/test data from {train_test_data_path}")

        # Load the model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Load the selected features data
        with open(selected_features_path, 'rb') as f:
            selected_data = pickle.load(f)

        # Load the original data to get y_test
        with open(train_test_data_path, 'rb') as f:
            data_dict = pickle.load(f)

        X_test_selected = selected_data['X_test_selected']
        y_test = data_dict['y_test']

        # Make predictions
        y_pred = model.predict(X_test_selected)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"MLR - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

        # Create metrics
        metrics = {
            'model_name': 'MLR',
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred.tolist() if isinstance(y_pred, np.ndarray) else y_pred.tolist()
        }

        # Ensure the output directory exists
        metrics_path = os.path.join(OUTPUT_DIR, 'mlr_metrics.pkl')
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

        # Save metrics
        with open(metrics_path, 'wb') as f:
            pickle.dump(metrics, f)

        print(f"MLR metrics saved to {metrics_path}")

        # Return the path to the saved metrics
        return metrics_path

    except Exception as e:
        print(f"Error in evaluate_mlr: {str(e)}")
        # Re-raise the exception to make sure Airflow knows the task failed
        raise

def evaluate_random_forest(model_path=None, selected_features_path=None, train_test_data_path=None):
    """
    Evaluate the Random Forest model.

    Args:
        model_path (str, optional): Path to the saved Random Forest model. If None, will use RF_MODEL_PATH
        selected_features_path (str, optional): Path to the saved selected features. If None, will use SELECTED_FEATURES_PATH
        train_test_data_path (str, optional): Path to the saved train/test data. If None, will use TRAIN_TEST_DATA_PATH

    Returns:
        str: Path to the saved Random Forest metrics
    """
    try:
        # Use the provided paths or the default ones
        model_path = model_path or RF_MODEL_PATH
        selected_features_path = selected_features_path or SELECTED_FEATURES_PATH
        train_test_data_path = train_test_data_path or TRAIN_TEST_DATA_PATH

        print(f"Loading Random Forest model from {model_path}")
        print(f"Loading selected features from {selected_features_path}")
        print(f"Loading train/test data from {train_test_data_path}")

        # Load the model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Load the selected features data
        with open(selected_features_path, 'rb') as f:
            selected_data = pickle.load(f)

        # Load the original data to get y_test
        with open(train_test_data_path, 'rb') as f:
            data_dict = pickle.load(f)

        X_test_selected = selected_data['X_test_selected']
        y_test = data_dict['y_test']

        # Make predictions
        y_pred = model.predict(X_test_selected)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"RandomForest - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

        # Create metrics
        metrics = {
            'model_name': 'RandomForest',
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred.tolist() if isinstance(y_pred, np.ndarray) else y_pred.tolist()
        }

        # Ensure the output directory exists
        metrics_path = os.path.join(OUTPUT_DIR, 'rf_metrics.pkl')
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

        # Save metrics
        with open(metrics_path, 'wb') as f:
            pickle.dump(metrics, f)

        print(f"Random Forest metrics saved to {metrics_path}")

        # Return the path to the saved metrics
        return metrics_path

    except Exception as e:
        print(f"Error in evaluate_random_forest: {str(e)}")
        # Re-raise the exception to make sure Airflow knows the task failed
        raise

def evaluate_svr(model_path=None, selected_features_path=None, train_test_data_path=None):
    """
    Evaluate the SVR model.

    Args:
        model_path (str, optional): Path to the saved SVR model. If None, will use SVR_MODEL_PATH
        selected_features_path (str, optional): Path to the saved selected features. If None, will use SELECTED_FEATURES_PATH
        train_test_data_path (str, optional): Path to the saved train/test data. If None, will use TRAIN_TEST_DATA_PATH

    Returns:
        str: Path to the saved SVR metrics
    """
    try:
        # Use the provided paths or the default ones
        model_path = model_path or SVR_MODEL_PATH
        selected_features_path = selected_features_path or SELECTED_FEATURES_PATH
        train_test_data_path = train_test_data_path or TRAIN_TEST_DATA_PATH

        print(f"Loading SVR model from {model_path}")
        print(f"Loading selected features from {selected_features_path}")
        print(f"Loading train/test data from {train_test_data_path}")

        # Load the model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Load the selected features data
        with open(selected_features_path, 'rb') as f:
            selected_data = pickle.load(f)

        # Load the original data to get y_test
        with open(train_test_data_path, 'rb') as f:
            data_dict = pickle.load(f)

        X_test_selected_scaled = selected_data['X_test_selected_scaled']
        y_test = data_dict['y_test']

        # Make predictions
        y_pred = model.predict(X_test_selected_scaled)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"SVR - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

        # Create metrics
        metrics = {
            'model_name': 'SVR',
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred.tolist() if isinstance(y_pred, np.ndarray) else y_pred.tolist()
        }

        # Ensure the output directory exists
        metrics_path = os.path.join(OUTPUT_DIR, 'svr_metrics.pkl')
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

        # Save metrics
        with open(metrics_path, 'wb') as f:
            pickle.dump(metrics, f)

        print(f"SVR metrics saved to {metrics_path}")

        # Return the path to the saved metrics
        return metrics_path

    except Exception as e:
        print(f"Error in evaluate_svr: {str(e)}")
        # Re-raise the exception to make sure Airflow knows the task failed
        raise

def serialize_model_info(mlr_metrics_path=None, rf_metrics_path=None, svr_metrics_path=None, train_test_data_path=None):
    """
    Serialize model information for Power BI.

    Args:
        mlr_metrics_path (str, optional): Path to the saved MLR metrics. If None, will use default path
        rf_metrics_path (str, optional): Path to the saved Random Forest metrics. If None, will use default path
        svr_metrics_path (str, optional): Path to the saved SVR metrics. If None, will use default path
        train_test_data_path (str, optional): Path to the saved train/test data. If None, will use TRAIN_TEST_DATA_PATH

    Returns:
        str: Path to the saved model comparison
    """
    try:
        # Use the provided paths or the default ones
        mlr_metrics_path = mlr_metrics_path or os.path.join(OUTPUT_DIR, 'mlr_metrics.pkl')
        rf_metrics_path = rf_metrics_path or os.path.join(OUTPUT_DIR, 'rf_metrics.pkl')
        svr_metrics_path = svr_metrics_path or os.path.join(OUTPUT_DIR, 'svr_metrics.pkl')
        train_test_data_path = train_test_data_path or TRAIN_TEST_DATA_PATH

        print(f"Loading MLR metrics from {mlr_metrics_path}")
        print(f"Loading Random Forest metrics from {rf_metrics_path}")
        print(f"Loading SVR metrics from {svr_metrics_path}")
        print(f"Loading train/test data from {train_test_data_path}")

        # Load metrics
        with open(mlr_metrics_path, 'rb') as f:
            mlr_metrics = pickle.load(f)

        with open(rf_metrics_path, 'rb') as f:
            rf_metrics = pickle.load(f)

        with open(svr_metrics_path, 'rb') as f:
            svr_metrics = pickle.load(f)

        # Load the original data to get actual CPI values
        with open(train_test_data_path, 'rb') as f:
            data_dict = pickle.load(f)

        y_test = data_dict['y_test']

        model_results = [mlr_metrics, rf_metrics, svr_metrics]

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(MODEL_COMPARISON_PATH), exist_ok=True)

        # Save model comparison results
        comparison_df = pd.DataFrame([
            {
                'Model': result['model_name'],
                'RMSE': result['rmse'],
                'MAE': result['mae'],
                'R_squared': result['r2']
            }
            for result in model_results
        ])

        comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)
        print(f"Model comparison saved to {MODEL_COMPARISON_PATH}")

        # Save model predictions with actual CPI values
        predictions_df = pd.DataFrame()

        # Add actual CPI values
        predictions_df['Actual_CPI'] = y_test.values

        # Add model predictions
        for result in model_results:
            predictions_df[result['model_name']] = result['predictions']

        # Add date index as a column
        predictions_df['Date'] = y_test.index.strftime('%Y-%m-%d')

        # Reorder columns to put Date first
        cols = ['Date', 'Actual_CPI'] + [result['model_name'] for result in model_results]
        predictions_df = predictions_df[cols]

        predictions_df.to_csv(MODEL_PREDICTIONS_PATH, index=False)
        print(f"Model predictions saved to {MODEL_PREDICTIONS_PATH}")

        # Save as JSON for Power BI
        comparison_json = comparison_df.to_dict(orient='records')
        with open(MODEL_COMPARISON_JSON_PATH, 'w') as f:
            json.dump(comparison_json, f)
        print(f"Model comparison JSON saved to {MODEL_COMPARISON_JSON_PATH}")

        print(f"Model information serialized to {OUTPUT_DIR}")

        # Return the path to the saved model comparison
        return MODEL_COMPARISON_PATH

    except Exception as e:
        print(f"Error in serialize_model_info: {str(e)}")
        # Re-raise the exception to make sure Airflow knows the task failed
        raise

def train_lasso_model(selected_features_path=None, train_test_data_path=None, output_dir=None):
    """
    Train a Lasso model, evaluate it, and save metrics.

    Args:
        selected_features_path (str): Path to the selected features pickle file
        train_test_data_path (str): Path to the train/test data pickle file
        output_dir (str): Directory to save the model and metrics

    Returns:
        tuple: (model_path, metrics_path)
    """
    try:
        selected_features_path = selected_features_path or SELECTED_FEATURES_PATH
        train_test_data_path = train_test_data_path or TRAIN_TEST_DATA_PATH
        output_dir = output_dir or OUTPUT_DIR

        # Load selected features
        with open(selected_features_path, 'rb') as f:
            selected_data = pickle.load(f)
        X_train = selected_data['X_train_selected']
        X_test = selected_data['X_test_selected']

        # Load train/test targets
        with open(train_test_data_path, 'rb') as f:
            data_dict = pickle.load(f)
        y_train = data_dict['y_train']
        y_test = data_dict['y_test']

        # Train Lasso model
        model = LassoCV(cv=5, n_alphas=100, max_iter=10000, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        # Save the model
        model_path = os.path.join(output_dir, 'kse_lasso_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Evaluate
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Lasso: R2={r2:.4f}, MSE={mse:.4f}, MAE={mae:.4f}")

        # Save metrics
        metrics_df = pd.DataFrame([{'Model': 'Lasso', 'R_squared': r2, 'MSE': mse, 'MAE': mae}])
        metrics_path = os.path.join(output_dir, 'kse_lasso_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)

        # Predict on the entire dataset and save
        try:
            # Load the full dataset
            df_full = pd.read_csv(TRANSFORMED_DATA_PATH, index_col=0, parse_dates=True)
            # Drop target and Month columns to get features
            feature_cols = X_train.columns.tolist()
            X_full = df_full[feature_cols]
            y_full = df_full['KSE-100'] if 'KSE-100' in df_full.columns else df_full.iloc[:,0]
            y_full_pred = model.predict(X_full)
            pred_df = pd.DataFrame({
                'Date': df_full.index,
                'Actual': y_full,
                'Predicted': y_full_pred
            })
            pred_df.to_csv(os.path.join(output_dir, 'kse_lasso_full_predictions.csv'), index=False)
            print('Full dataset predictions saved to kse_lasso_full_predictions.csv')
        except Exception as e:
            print(f'Error predicting on full dataset (Lasso): {e}')

        return model_path, metrics_path
    except Exception as e:
        logging.error(f"Error in train_lasso_model: {str(e)}")
        raise

def train_stacked_model(selected_features_path=None, train_test_data_path=None, output_dir=None):
    """
    Train a stacked model, evaluate it, and save metrics.

    Args:
        selected_features_path (str): Path to the selected features pickle file
        train_test_data_path (str): Path to the train/test data pickle file
        output_dir (str): Directory to save the model and metrics

    Returns:
        tuple: (model_path, metrics_path)
    """
    try:
        selected_features_path = selected_features_path or SELECTED_FEATURES_PATH
        train_test_data_path = train_test_data_path or TRAIN_TEST_DATA_PATH
        output_dir = output_dir or OUTPUT_DIR

        # Load selected features
        with open(selected_features_path, 'rb') as f:
            selected_data = pickle.load(f)
        X_train = selected_data['X_train_selected']
        X_test = selected_data['X_test_selected']

        # Load train/test targets
        with open(train_test_data_path, 'rb') as f:
            data_dict = pickle.load(f)
        y_train = data_dict['y_train']
        y_test = data_dict['y_test']

        # Initialize base models
        base_models = [
            ('lasso', Lasso(alpha=0.1)),
            ('ridge', Ridge(alpha=1.0)),
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ]
        meta_model = LinearRegression()
        stacked_model = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5
        )
        stacked_model.fit(X_train, y_train)

        # Save the model
        model_path = os.path.join(output_dir, 'kse_stacked_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(stacked_model, f)

        # Evaluate
        y_pred = stacked_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Stacked: R2={r2:.4f}, MSE={mse:.4f}, MAE={mae:.4f}")

        # Save metrics
        metrics_df = pd.DataFrame([{'Model': 'Stacked', 'R_squared': r2, 'MSE': mse, 'MAE': mae}])
        metrics_path = os.path.join(output_dir, 'kse_stacked_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)

        # Predict on the entire dataset and save
        try:
            # Load the full dataset
            df_full = pd.read_csv(TRANSFORMED_DATA_PATH, index_col=0, parse_dates=True)
            feature_cols = X_train.columns.tolist()
            X_full = df_full[feature_cols]
            y_full = df_full['KSE-100'] if 'KSE-100' in df_full.columns else df_full.iloc[:,0]
            y_full_pred = stacked_model.predict(X_full)
            pred_df = pd.DataFrame({
                'Date': df_full.index,
                'Actual': y_full,
                'Predicted': y_full_pred
            })
            pred_df.to_csv(os.path.join(output_dir, 'kse_stacked_full_predictions.csv'), index=False)
            print('Full dataset predictions saved to kse_stacked_full_predictions.csv')
        except Exception as e:
            print(f'Error predicting on full dataset (Stacked): {e}')

        return model_path, metrics_path
    except Exception as e:
        logging.error(f"Error in train_stacked_model: {str(e)}")
        raise

def run_modeling(X, y, output_dir):
    """
    Main function to run the modeling pipeline.
    
    Args:
        X: Feature matrix
        y: Target variable
        output_dir: Directory to save models and results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )
    
    # Train models
    lasso_metrics = train_lasso_model(X_train, y_train, output_dir)
    stacked_metrics = train_stacked_model(X_train, y_train, output_dir)
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'Model': ['Lasso', 'Stacked'],
        'MSE': [lasso_metrics['mse'], stacked_metrics['mse']],
        'R²': [lasso_metrics['r2'], stacked_metrics['r2']]
    })
    
    metrics_path = os.path.join(output_dir, 'model_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    
    print("\nModeling complete! Results saved to:", output_dir)
    
    return metrics_df
