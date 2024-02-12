from fastapi import File
from pydantic import BaseModel
from typing import Dict, Any


class Models(BaseModel):
    name: str
    creation_date: str


class Parameters(BaseModel):
    parameters: Dict[str, Any]


class Metrics(BaseModel):
    metrics: Dict[str, Any]


class Dataset(BaseModel):
    dataset: str


class Images(BaseModel):
    images: Dict[str, str]
