from tqdm import tqdm
import mlflow
from mlflow.tracking import MlflowClient

import pandas as pd


def export_data():
    experiment_id = "2"
    df_completed = None
    
    df = mlflow.search_runs([experiment_id], filter_string="")
    
    client = MlflowClient()
    for index, run in enumerate(tqdm(df['run_id'].values)):
        metrics = client.get_metric_history(run, "Relative optimality gap")

        x = list()
        y = list()

        temporary_series = df[df['run_id'] == run]
        temporary_series = temporary_series.reset_index()
        for metric in metrics:
            x.append(metric.timestamp)
            y.append(metric.value)
            metric_series = pd.DataFrame({'timestamp':metric.timestamp, 'relative_optimality_gap':metric.value}, index=[0])         
            filled_series  = pd.concat([temporary_series, metric_series], axis=1)   
            if df_completed is None:
                df_completed = filled_series
            else:    
                df_completed = pd.concat([df_completed, filled_series], axis=0, join="outer", ignore_index=True)

            
    df_completed.to_csv("mlflow.csv")

export_data()