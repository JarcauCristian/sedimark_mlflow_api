import pandas as pd
import uvicorn
from fastapi import FastAPI, Response, UploadFile
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from mlflow_client import Client
from pydantic import BaseModel
from schemas import Models, Parameters, Metrics, Dataset, Images, Versions
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
client = Client()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Register(BaseModel):
    run_id: str
    model_name: str


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
async def model_parameters(name: str, version: str = None):
    parameters = client.model_parameters(name, version)
    if parameters is None:
        return JSONResponse("Error getting the parameters!", status_code=500)
    return JSONResponse(parameters, status_code=200)


@app.get("/model/metrics", tags=["Endpoints that gets all the metrics of a specified register model"],
         response_model=Metrics)
async def model_metrics(name: str, version: str = None):
    metrics = client.model_metrics(name, version)
    if metrics is None:
        return JSONResponse("Error getting the metrics!", status_code=500)
    return JSONResponse(metrics, status_code=200)


@app.get("/model/dataset", tags=["Endpoints that gets the train dataset of a specified register model"],
         response_model=Dataset)
async def model_dataset(name: str, version: str = None):
    dataset = client.model_dataset(name, version)
    if dataset is None:
        return JSONResponse("Error getting the dataset!", status_code=500)

    csv_content = dataset.to_csv(index=False)
    return Response(content=csv_content, media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=dataset.csv"})


@app.get("/model/images", tags=["Endpoints that gets all the images of a specified register model"],
         response_model=Images)
async def model_images(name: str, version: str = None):
    images = client.model_images(name, version)
    if images is None:
        return JSONResponse("Error getting the images!", status_code=500)
    return JSONResponse(images, status_code=200)


@app.get("/model/versions", tags=["Endpoints that gets all the versions of a specified register model"],
         response_model=Versions)
async def model_versions(name: str):
    versions = client.model_versions(name)
    if versions is None:
        return JSONResponse("Error getting the model versions!", status_code=500)
    return JSONResponse(versions, status_code=200)


@app.get("/model/package", tags=["Endpoints that packages the model to be saved in Mage AI."])
async def model_package(name: str):
    try:
        packaged_model = client.model_package(name)
    except Exception as _:
        return JSONResponse("Error packeing the model!", status_code=500)
    if packaged_model is None:
        return JSONResponse("Error packeing the model!", status_code=500)

    return Response(
        content=packaged_model.getvalue(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={name}.bin"}
    )


@app.post("/model/predict")
async def model_predict(name: str, file: UploadFile):
    df = pd.read_csv(file.file)
    predictions = client.model_predict(name, df)
    if predictions is None:
        return JSONResponse("Error making the predictions!", status_code=500)
    return JSONResponse(None, status_code=200)


@app.post("/model/register")
async def model_register(register: Register):
    if "/" not in register.run_id:
        return JSONResponse(status_code=400, content="The run_id needs to have at least one slash in it!")

    result = client.model_register(register.run_id, register.model_name)

    if not result:
        return JSONResponse(status_code=500, content="Error registering model!")

    return JSONResponse(status_code=200, content=result.name)


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0")
