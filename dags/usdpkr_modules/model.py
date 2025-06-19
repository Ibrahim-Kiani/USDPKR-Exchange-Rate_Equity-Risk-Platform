import os
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_data(data_path):
    """
    Load the transformed data from CSV.

    Args:
        data_path (str): Path to the transformed data CSV file

    Returns:
        pd.DataFrame: Loaded dataframe
    """
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"Loaded data with shape: {df.shape}")
    return df

def prepare_data(df, target_col='CPI', test_size=0.2):
    """
    Prepare data for modeling by splitting into features and target,
    and creating train/test splits.

    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Name of the target column
        test_size (float): Proportion of data to use for testing

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    # Ensure the dataframe is sorted by date
    df = df.sort_index()

    # Separate features and target
    X = df.drop(['Month', target_col], axis=1)
    y = df[target_col]

    # Calculate the split point
    split_idx = int(len(df) * (1 - test_size))

    # Split the data
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test

def select_features_lasso(X_train, y_train, X_test, alpha_range=np.logspace(-4, 1, 50)):
    """
    Select important features using LassoCV.

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_test (pd.DataFrame): Test features
        alpha_range (np.array): Range of alpha values to try

    Returns:
        tuple: X_train_selected, X_test_selected, feature_names, lasso_model
    """
    # Initialize and fit LassoCV
    lasso_cv = LassoCV(
        alphas=alpha_range,
        cv=TimeSeriesSplit(n_splits=5),
        max_iter=10000,
        tol=0.0001,
        random_state=42
    )
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
    print(f"Selected {len(selected_features)} features out of {X_train.shape[1]}")

    # Create datasets with selected features
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    return X_train_selected, X_test_selected, selected_features, lasso_cv

def train_mlr(X_train, y_train):
    """
    Train a Multiple Linear Regression model.

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target

    Returns:
        LinearRegression: Trained model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model





def train_random_forest(X_train, y_train):
    """
    Train a Random Forest model with GridSearchCV.

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target

    Returns:
        tuple: Best model, best parameters
    """
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Initialize model
    rf = RandomForestRegressor(random_state=42)

    # Grid search with time series cross-validation
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=TimeSeriesSplit(n_splits=5),
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )

    # Fit grid search
    grid_search.fit(X_train, y_train)

    # Get best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"Best Random Forest parameters: {best_params}")
    return best_model, best_params

def train_svr(X_train_scaled, y_train):
    """
    Train an SVR model with linear kernel and GridSearchCV.

    Args:
        X_train_scaled (np.array): Scaled training features
        y_train (np.array): Training target

    Returns:
        tuple: Best model, best parameters
    """
    # Define parameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.2]
    }

    # Initialize model
    svr = SVR(kernel='linear')

    # Grid search with time series cross-validation
    grid_search = GridSearchCV(
        estimator=svr,
        param_grid=param_grid,
        cv=TimeSeriesSplit(n_splits=5),
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )

    # Fit grid search
    grid_search.fit(X_train_scaled, y_train)

    # Get best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"Best SVR parameters: {best_params}")
    return best_model, best_params

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a model and return performance metrics.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_name (str): Name of the model

    Returns:
        dict: Performance metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

    # Return metrics
    return {
        'model_name': model_name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': y_pred.tolist() if isinstance(y_pred, np.ndarray) else y_pred.tolist()
    }

def serialize_model_info(model_results, models, output_dir):
    """
    Serialize model information for Power BI and save trained models in pickle format.

    Args:
        model_results (list): List of model evaluation results
        models (dict): Dictionary of trained models
        output_dir (str): Directory to save serialized data
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

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

    comparison_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)

    # Save model predictions
    predictions_df = pd.DataFrame()

    for result in model_results:
        predictions_df[result['model_name']] = result['predictions']

    predictions_df.to_csv(os.path.join(output_dir, 'model_predictions.csv'), index=False)

    # Save as JSON for Power BI
    comparison_json = comparison_df.to_dict(orient='records')
    with open(os.path.join(output_dir, 'model_comparison.json'), 'w') as f:
        json.dump(comparison_json, f)

    # Save trained models in pickle format
    for model_name, model in models.items():
        model_path = os.path.join(output_dir, f"{model_name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model {model_name} saved to {model_path}")

    print(f"Model information serialized to {output_dir}")

def run_modeling(data_path, output_dir):
    """
    Run the complete modeling pipeline.

    Args:
        data_path (str): Path to the transformed data
        output_dir (str): Directory to save model outputs
    """
    print("Starting modeling pipeline...")

    # Load data
    df = load_data(data_path)

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Select features with LassoCV
    X_train_selected, X_test_selected, _, _ = select_features_lasso(X_train, y_train, X_test)

    # Scale selected features - create a new scaler for the selected features only
    scaler = StandardScaler()
    X_train_selected_scaled = scaler.fit_transform(X_train_selected)
    X_test_selected_scaled = scaler.transform(X_test_selected)

    # Train models
    print("\nTraining Multiple Linear Regression...")
    mlr_model = train_mlr(X_train_selected, y_train)

    print("\nTraining Random Forest...")
    rf_model, _ = train_random_forest(X_train_selected, y_train)

    print("\nTraining SVR...")
    svr_model, _ = train_svr(X_train_selected_scaled, y_train)

    # Evaluate models
    print("\nEvaluating models...")
    model_results = [
        evaluate_model(mlr_model, X_test_selected, y_test, 'MLR'),
        evaluate_model(rf_model, X_test_selected, y_test, 'RandomForest'),
        evaluate_model(svr_model, X_test_selected_scaled, y_test, 'SVR')
    ]

    # Create a dictionary of models for serialization
    models = {
        'MLR': mlr_model,
        'RandomForest': rf_model,
        'SVR': svr_model,
        'Scaler': scaler  # Also save the scaler for future predictions
    }

    # Serialize model information and trained models
    serialize_model_info(model_results, models, output_dir)

    print("Modeling pipeline complete!")
    return model_results

def model_inflation(data_dir='/opt/airflow/data', output_dir='/opt/airflow/data/model_output'):
    """
    Main function to model inflation data.

    Args:
        data_dir (str): Directory containing the transformed data
        output_dir (str): Directory to save model outputs
    """
    # Define paths
    data_path = os.path.join(data_dir, 'transformed_data.csv')

    # Run modeling pipeline
    model_results = run_modeling(data_path, output_dir)

    return model_results

if __name__ == "__main__":
    # Define paths for local execution
    base_dir = Path(__file__).parent.parent.parent  # Project root directory
    data_dir = base_dir / 'data'
    output_dir = data_dir / 'model_output'

    # Run modeling
    model_inflation(data_dir, output_dir)
