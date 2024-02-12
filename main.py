import pandas as pd
import uvicorn
from fastapi import FastAPI, Response, UploadFile
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from mlflow_client import Client
from schemas import Models, Parameters, Metrics, Dataset, Images

app = FastAPI()
client = Client()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Server check endpoint"])
async def hello():
    return JSONResponse("Server is alive!", status_code=200)


@app.get("/models", tags=["Endpoint that loads all the registered models from MlFlow"],
         response_model=Models)
async def models():
    models_list = client.models()
    if models_list is None:
        return JSONResponse("Error getting the models!", status_code=500)
    return JSONResponse(models_list, status_code=200)


@app.get("/model/parameters", tags=["Endpoints that gets all the parameters of a specified register model"],
         response_model=Parameters)
async def model_parameters(name: str):
    parameters = client.model_parameters(name)
    if parameters is None:
        return JSONResponse("Error getting the parameters!", status_code=500)
    return JSONResponse(parameters, status_code=200)


@app.get("/model/metrics", tags=["Endpoints that gets all the metrics of a specified register model"],
         response_model=Metrics)
async def model_metrics(name: str):
    metrics = client.model_metrics(name)
    if metrics is None:
        return JSONResponse("Error getting the metrics!", status_code=500)
    return JSONResponse(metrics, status_code=200)


@app.get("/model/dataset", tags=["Endpoints that gets the train dataset of a specified register model"],
         response_model=Dataset)
async def model_dataset(name: str):
    dataset = client.model_dataset(name)
    if dataset is None:
        return JSONResponse("Error getting the dataset!", status_code=500)

    csv_content = dataset.to_csv(index=False)
    return Response(content=csv_content, media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=dataset.csv"})


@app.get("/model/images", tags=["Endpoints that gets all the images of a specified register model"],
         response_model=Images)
async def model_images(name: str):
    images = client.model_images(name)
    if images is None:
        return JSONResponse("Error getting the images!", status_code=500)
    return JSONResponse(images, status_code=200)


@app.post("/model/predict")
async def model_predict(name: str, file: UploadFile):
    df = pd.read_csv(file.file)
    predictions = client.model_predict(name, df)
    if predictions is None:
        return JSONResponse("Error making the predictions!", status_code=500)
    return JSONResponse(predictions, status_code=200)


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0")
