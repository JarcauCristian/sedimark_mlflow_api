import io
import os
import torch
import pickle
import tempfile


class BaseModelHandler:
    def load_model(self, model_uri):
        raise NotImplementedError

    def save_model(self, model_uri):
        raise NotImplementedError
    
    def predict(self, input_data):
        raise NotImplementedError


class SklearnModelHandler(BaseModelHandler):
    def __init__(self):
        super(SklearnModelHandler).__init__()
        self.model = None

    def load_model(self, model_uri):
        import mlflow.sklearn
        self.model = mlflow.sklearn.load_model(model_uri)

    def save_model(self, model_uri):
        self.load_model(model_uri)

        buffer = io.BytesIO()
        pickle.dump(self.model, buffer)
        buffer.seek(0)
        return buffer

    def predict(self, input_data):
        return self.model.predict(input_data)


class TensorflowModelHandler(BaseModelHandler):
    def __init__(self):
        super(TensorflowModelHandler).__init__()
        self.model = None

    def load_model(self, model_uri):
        import mlflow.tensorflow
        self.model = mlflow.tensorflow.load_model(model_uri)

    def save_model(self, model_uri):
        self.load_model(model_uri)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            temp_path = tmp.name

        try:
            self.model.save(temp_path, save_format=".h5")

            with open(temp_path, 'rb') as f:
                buffer = io.BytesIO(f.read())
            buffer.seek(0)
        finally:
            os.remove(temp_path)

        return buffer

    def predict(self, input_data):
        return self.model.predict(input_data)


class KerasModelHandler(BaseModelHandler):
    def __init__(self):
        super(KerasModelHandler).__init__()
        self.model = None

    def load_model(self, model_uri):
        import mlflow.keras
        self.model = mlflow.keras.load_model(model_uri)

    def save_model(self, model_uri):
        self.load_model(model_uri)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            temp_path = tmp.name

        try:
            self.model.save(temp_path, save_format=".h5")

            with open(temp_path, 'rb') as f:
                buffer = io.BytesIO(f.read())
            buffer.seek(0)
        finally:
            os.remove(temp_path)

        return buffer

    def predict(self, input_data):
        return self.model.predict(input_data)


class PytorchModelHandler(BaseModelHandler):
    def __init__(self):
        super(PytorchModelHandler).__init__()
        self.model = None

    def load_model(self, model_uri):
        import mlflow.pytorch
        self.model = mlflow.pytorch.load_model(model_uri)
        self.model.eval()

    def save_model(self, model_uri):
        self.load_model(model_uri)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            temp_path = tmp.name

        try:
            torch.jit.save(self.model, temp_path)
            with open(temp_path, 'rb') as f:
                buffer = io.BytesIO(f.read())

            buffer.seek(0)
        finally:
            os.remove(temp_path)

        return buffer

    def predict(self, input_data):
        if not isinstance(input_data, torch.Tensor):
            input_tensor = torch.tensor(input_data)
        else:
            input_tensor = input_data

        with torch.no_grad():
            output = self.model.predict(input_tensor)
        return output


class AnyModelHandler(BaseModelHandler):
    def __init__(self):
        super(AnyModelHandler).__init__()
        self.model = None

    def load_model(self, model_uri):
        import mlflow.pyfunc
        self.model = mlflow.pyfunc.load_model(model_uri)

    def save_model(self, model_uri):
        self.load_model(model_uri)

        buffer = io.BytesIO()
        pickle.dump(self.model, buffer)
        buffer.seek(0)
        return buffer

    def predict(self, input_data):
        return self.model.predict(input_data)


model_handlers = {
    "sklearn": SklearnModelHandler(),
    "tensorflow": TensorflowModelHandler(),
    "keras": KerasModelHandler(),
    "pytorch": PytorchModelHandler(),
    "pyfunc": AnyModelHandler()
}
