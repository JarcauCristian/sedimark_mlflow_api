import os
import yaml
import json
import mlflow
import base64
import pandas as pd
from typing import Any, Dict, List
from datetime import datetime
from json import JSONDecodeError
from io import StringIO, BytesIO
from mlflow import MlflowClient, MlflowException
from models import model_handlers

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

    def model_versions(self, name: str) -> List[Dict[str, str]] | None:
        versions = self.client.search_model_versions(f"name='{name}'")
        return [{"version": version.version,
                 "type": "" if version.tags.get("model_type") is None else version.tags["model_type"]}
                for version in versions]

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
                    images[str(artifact.path).split("/")[-1]] = (f"data:image/png;base64,"
                                                                 f"{(base64.b64encode(buffered.getvalue()).decode('utf-8'))}")
        except MlflowException:
            return None

        return images

    def model_predict(self, name: str, df: pd.DataFrame):
        run_id = self.client.get_registered_model(name).latest_versions[0].run_id
        if not run_id:
            return None

        artifacts = self.client.list_artifacts(run_id)

        artifacts = [artifact.path for artifact in artifacts if "model" in artifact.path and artifact.is_dir]

        content = mlflow.artifacts.load_text(f"runs:/{run_id}/{artifacts[0]}/MLmodel")

        content = yaml.safe_load(StringIO(content))

        model_type = content["flavors"]["python_function"]["loader_module"].split(".")[1]

        if model_type in list(model_handlers.keys()):
            handler = model_handlers[model_type]
            handler.load_model(model_uri=f"runs:/{run_id}/{artifacts[0]}")
            return handler.predict(df)

        return None

    @staticmethod
    def model_register(run_id: str, model_name: str):
        return mlflow.register_model(f"runs:/{run_id}", model_name)
