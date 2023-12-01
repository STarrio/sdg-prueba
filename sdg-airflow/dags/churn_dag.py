from datetime import datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.sensors.filesystem import FileSensor
from airflow.models import Variable
import os
from data_preprocessing import data_preprocessing
from model_fitting import model_fitting
from model_evaluation import model_evaluation

CURRENT_ESTIMATOR = 'gb' 

AIRFLOW_HOME = os.getenv('AIRFLOW_HOME')
RAW_DATASET_PATH = AIRFLOW_HOME + '/dags/data/dataset.csv'
PROC_DATASET_PATH = AIRFLOW_HOME + '/dags/data/dataset_proc.csv'
HOLDOUT_SET_PATH = AIRFLOW_HOME + '/dags/data/holdout.csv' 
MODEL_STORE_PATH = f'{AIRFLOW_HOME}/dags/model/model_{CURRENT_ESTIMATOR}.joblib'
    
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 11, 25),
    'retries': 1,
}

dag = DAG('churn', default_args=default_args, schedule_interval=None)

data_preprocessing_task = PythonOperator(
    task_id='data_preprocessing',
    python_callable=data_preprocessing,
    op_kwargs={'origin_csv_path':RAW_DATASET_PATH,
                'dest_csv_path': PROC_DATASET_PATH},
    dag=dag,
)

fit_model_task = PythonOperator(
    task_id='fit_model',
    python_callable=model_fitting,
    op_kwargs={'origin_csv_path':PROC_DATASET_PATH,
                'dest_csv_path': HOLDOUT_SET_PATH,
                'model_store_path': MODEL_STORE_PATH,
                'estimator': CURRENT_ESTIMATOR},
    dag=dag,
)

evaluate_model_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=model_evaluation,
    op_kwargs={'origin_csv_path':HOLDOUT_SET_PATH,
                'model_store_path': MODEL_STORE_PATH,
                'estimator': CURRENT_ESTIMATOR,
                'airflow_tld': AIRFLOW_HOME},
    dag=dag,
)

file_sensor_task = FileSensor(
    task_id='detect_dataset',
    filepath=RAW_DATASET_PATH,
    poke_interval=10,
    timeout=300,
    dag=dag
)

file_sensor_task >> data_preprocessing_task >> fit_model_task >> evaluate_model_task
