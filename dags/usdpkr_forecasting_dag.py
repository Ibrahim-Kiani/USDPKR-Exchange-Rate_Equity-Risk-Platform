from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from inflation_modules import ingest, transform, model_steps
import logging

# Default arguments
default_args = {
    'owner': 'airflow',
    'retries': 3,  # Increased retries
    'retry_delay': timedelta(minutes=2),  # Shorter retry delay
    'retry_exponential_backoff': True,  # Enable exponential backoff
    'max_retry_delay': timedelta(minutes=10),  # Maximum retry delay
}

data_raw_dir = '/opt/airflow/data/raw'
data_transformed_dir = '/opt/airflow/data/transformed'
data_models_dir = '/opt/airflow/data/models'

def handle_ingestion_error(context):
    """Handle ingestion task errors."""
    task_instance = context['task_instance']
    error = context['exception']
    logging.error(f"Task {task_instance.task_id} failed with error: {str(error)}")
    # You can add additional error handling logic here

with DAG(
    'usdpkr_forecasting',
    default_args=default_args,
    description='Modular USD/PKR Exchange Rate Forecasting Pipeline',
    start_date=datetime(2024, 1, 1),
    schedule_interval='@monthly',
    catchup=False,
    tags=['exchange_rate', 'modular'],
    on_failure_callback=handle_ingestion_error,
) as dag:
    # Directory setup
    ensure_data_dir = PythonOperator(
        task_id='ensure_data_dir',
        python_callable=ingest.ensure_dir,
        op_kwargs={'output_dir': data_raw_dir},
        retries=1,  # Directory creation should not need many retries
    )
    ensure_output_dir = PythonOperator(
        task_id='ensure_output_dir',
        python_callable=ingest.ensure_dir,
        op_kwargs={'output_dir': data_models_dir},
        retries=1,
    )

    # Ingestion tasks (one per source)
    fetch_kse100 = PythonOperator(
        task_id='fetch_kse100',
        python_callable=ingest.fetch_yahoo_finance,
        op_kwargs={'ticker': '^KSE', 'output_dir': data_raw_dir, 'filename': 'KSE100.csv'},
        retries=5,  # More retries for external API
        retry_delay=timedelta(minutes=1),
        retry_exponential_backoff=True,
    )
    fetch_usdpkr = PythonOperator(
        task_id='fetch_usdpkr',
        python_callable=ingest.fetch_yahoo_finance,
        op_kwargs={'ticker': 'PKR=X', 'output_dir': data_raw_dir, 'filename': 'USDPKR.csv'},
        retries=5,
        retry_delay=timedelta(minutes=1),
        retry_exponential_backoff=True,
    )
    fetch_oil = PythonOperator(
        task_id='fetch_oil',
        python_callable=ingest.fetch_yahoo_finance,
        op_kwargs={'ticker': 'CL=F', 'output_dir': data_raw_dir, 'filename': 'OIL.csv'},
        retries=5,
        retry_delay=timedelta(minutes=1),
        retry_exponential_backoff=True,
    )
    fetch_gold = PythonOperator(
        task_id='fetch_gold',
        python_callable=ingest.fetch_yahoo_finance,
        op_kwargs={'ticker': 'GC=F', 'output_dir': data_raw_dir, 'filename': 'GOLD.csv'},
        retries=5,
        retry_delay=timedelta(minutes=1),
        retry_exponential_backoff=True,
    )
    fetch_inflation = PythonOperator(
        task_id='fetch_inflation',
        python_callable=ingest.fetch_imf_data,
        op_kwargs={
            'indicator_id': 'PCPI_IX',
            'output_dir': data_raw_dir,
            'filename': 'INFLATION.csv',
            'indicator_name': 'INFLATION_YOY',
            'frequency': 'M'
        },
        retries=5,
        retry_delay=timedelta(minutes=1),
        retry_exponential_backoff=True,
    )
    fetch_interest_rate = PythonOperator(
        task_id='fetch_interest_rate',
        python_callable=ingest.fetch_sbp_placeholder,
        op_kwargs={'series_id': 'INTEREST_RATE', 'output_dir': data_raw_dir, 'filename': 'INTEREST_RATE.csv'},
        retries=2,  # Fewer retries for local data generation
    )
    fetch_m2 = PythonOperator(
        task_id='fetch_m2',
        python_callable=ingest.fetch_sbp_placeholder,
        op_kwargs={'series_id': 'M2', 'output_dir': data_raw_dir, 'filename': 'M2.csv'},
        retries=2,
    )
    fetch_forex_reserves = PythonOperator(
        task_id='fetch_forex_reserves',
        python_callable=ingest.fetch_sbp_placeholder,
        op_kwargs={'series_id': 'FOREX_RESERVES', 'output_dir': data_raw_dir, 'filename': 'FOREX_RESERVES.csv'},
        retries=2,
    )

    # Merge and feature engineering
    merge_data = PythonOperator(
        task_id='merge_data',
        python_callable=transform.load_and_merge_data,
        op_kwargs={'data_dir': data_raw_dir},
        retries=2,
    )
    feature_engineering = PythonOperator(
        task_id='feature_engineering',
        python_callable=transform.engineer_features,
        op_kwargs={
            'df_path': "{{ ti.xcom_pull(task_ids='merge_data', key='return_value') }}"
        },
        provide_context=True,
        retries=3,
        retry_delay=timedelta(minutes=5),
        dag=dag
    )
    prepare_data = PythonOperator(
        task_id='prepare_data',
        python_callable=transform.prepare_data_for_modeling,
        op_kwargs={
            'df_path': "{{ ti.xcom_pull(task_ids='feature_engineering', key='return_value') }}"
        },
        provide_context=True,
        retries=3,
        retry_delay=timedelta(minutes=5),
        dag=dag
    )

    # Modeling
    train_lasso = PythonOperator(
        task_id='train_lasso',
        python_callable=model_steps.train_lasso_model,
        op_kwargs={
            'X_train_path': "{{ ti.xcom_pull(task_ids='prepare_data')[0] }}",
            'y_train_path': "{{ ti.xcom_pull(task_ids='prepare_data')[1] }}",
            'selected_features_path': "{{ ti.xcom_pull(task_ids='prepare_data')[2] }}",
            'train_test_data_path': "{{ ti.xcom_pull(task_ids='prepare_data')[3] }}",
            'output_dir': data_models_dir
        },
        retries=2,
    )
    train_stacked = PythonOperator(
        task_id='train_stacked',
        python_callable=model_steps.train_stacked_model,
        op_kwargs={
            'X_train_path': "{{ ti.xcom_pull(task_ids='prepare_data')[0] }}",
            'y_train_path': "{{ ti.xcom_pull(task_ids='prepare_data')[1] }}",
            'selected_features_path': "{{ ti.xcom_pull(task_ids='prepare_data')[2] }}",
            'train_test_data_path': "{{ ti.xcom_pull(task_ids='prepare_data')[3] }}",
            'output_dir': data_models_dir
        },
        retries=2,
    )

   

    # Set dependencies
    ensure_data_dir >> [fetch_kse100, fetch_usdpkr, fetch_oil, fetch_gold, fetch_inflation, fetch_interest_rate, fetch_m2, fetch_forex_reserves]
    [fetch_kse100, fetch_usdpkr, fetch_oil, fetch_gold, fetch_inflation, fetch_interest_rate, fetch_m2, fetch_forex_reserves] >> merge_data
    merge_data >> feature_engineering >> prepare_data
    prepare_data >> [train_lasso, train_stacked]
    