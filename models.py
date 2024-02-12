import pandas as pd


class BaseModelHandler:
    def load_model(self, model_uri):
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

    def predict(self, input_data):
        return self.model.predict(input_data)


class TensorflowModelHandler(BaseModelHandler):
    def __init__(self):
        super(TensorflowModelHandler).__init__()
        self.model = None

    def load_model(self, model_uri):
        import mlflow.tensorflow
        self.model = mlflow.tensorflow.load_model(model_uri)

    def predict(self, input_data):
        return self.model.predict(input_data)


class KerasModelHandler(BaseModelHandler):
    def __init__(self):
        super(KerasModelHandler).__init__()
        self.model = None

    def load_model(self, model_uri):
        import mlflow.keras
        self.model = mlflow.keras.load_model(model_uri)

    def predict(self, input_data):
        return self.model.predict(input_data)


class PytorchModelHandler(BaseModelHandler):
    def __init__(self):
        super(PytorchModelHandler).__init__()
        self.model = None

    def load_model(self, model_uri):
        import mlflow.pytorch
        self.model = mlflow.pytorch.load_model(model_uri)

    def predict(self, input_data):
        return self.model.predict(input_data)


class AnyModelHandler(BaseModelHandler):
    def __init__(self):
        super(AnyModelHandler).__init__()
        self.model = None

    def load_model(self, model_uri):
        import mlflow.pyfunc
        self.model = mlflow.pyfunc.load_model(model_uri)

    def predict(self, input_data):
        return self.model.predict(input_data)


model_handlers = {
    "sklearn": SklearnModelHandler,
    "tensorflow": TensorflowModelHandler,
    "keras": KerasModelHandler,
    "pytorch": PytorchModelHandler,
    "pyfunc": AnyModelHandler
}
