import os
import json
import mlflow
import base64
import pandas as pd
from typing import Any, Dict
from datetime import datetime
from dotenv import load_dotenv
from json import JSONDecodeError
from io import StringIO, BytesIO
from mlflow import MlflowClient, MlflowException

load_dotenv()


def flatten_dict(d: dict, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class Client:
    def __init__(self):
        os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME')
        os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD')
        os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID')
        os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY')
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv('MLFLOW_S3_ENDPOINT_URL')
        os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = os.getenv('MLFLOW_TRACKING_INSECURE_TLS')
        os.environ['MLFLOW_HTTP_REQUEST_TIMEOUT'] = "1000"

        mlflow.set_tracking_uri("http://62.72.21.79:5000")
        self.client = MlflowClient()

    def models(self):
        returns = []
        models_list = self.client.search_registered_models()

        if not models_list:
            return None

        for model in models_list:
            timestamp_s = model.creation_timestamp / 1000
            returns.append({
                "name": model.name,
                "creation_date": datetime.utcfromtimestamp(timestamp_s).strftime('%Y-%m-%d %H:%M:%S')
            })

        return returns

    def model_parameters(self, name: str) -> Dict[str, Any] | None:
        run_id = self.client.get_registered_model(name).latest_versions[0].run_id
        if not run_id:
            return None
        parameters = self.client.get_run(run_id).data.params
        try:
            parameters = {k: json.loads(v.replace("'", '"')) for k, v in parameters.items()}
        except JSONDecodeError:
            pass
        return flatten_dict(parameters)

    def model_metrics(self, name: str) -> Dict[str, Any] | None:
        run_id = self.client.get_registered_model(name).latest_versions[0].run_id
        metrics = self.client.get_run(run_id).data.metrics
        return metrics

    def model_dataset(self, name: str):
        run_id = self.client.get_registered_model(name).latest_versions[0].run_id
        if not run_id:
            return None

        try:
            dataset = mlflow.artifacts.load_text(f"runs:/{run_id}/dataset")
            return pd.read_csv(StringIO(dataset))
        except MlflowException:
            return None

    def model_images(self, name: str):
        run_id = self.client.get_registered_model(name).latest_versions[0].run_id
        if not run_id:
            return None
        try:
            artifacts = self.client.list_artifacts(run_id, path="model")
            images = {}
            for artifact in artifacts:
                if ".png" in artifact.path:
                    image = mlflow.artifacts.load_image(f"runs:/{run_id}/{artifact.path}")
                    buffered = BytesIO()
                    image_format = image.format if image.format else 'PNG'
                    image.save(buffered, format=image_format)
                    images[str(artifact.path).split("/")[-1]] = (base64.b64encode(buffered.getvalue()).decode('utf-8'))
        except MlflowException:
            return None

        return images