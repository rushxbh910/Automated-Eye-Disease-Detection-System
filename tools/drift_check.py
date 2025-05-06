import os, requests
import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def load_reference():
    return pd.read_csv("/mnt/data/offline_sample.csv")

def load_current():
    return pd.read_csv("/mnt/data/inference_logs/latest.csv")

if __name__ == "__main__":
    ref, curr = load_reference(), load_current()
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=curr,
               column_mapping=ColumnMapping())
    drift = report.as_dict()["metrics"][0]["result"]["dataset_drift"]
    if drift:
        requests.post(os.environ["SLACK_WEBHOOK"], json={
            "text": "⚠️ Data drift detected – trigger retraining."
        })
