from pydantic import BaseModel
from typing import Dict, Any, List


class Models(BaseModel):
    name: str
    creation_date: str


class Parameters(BaseModel):
    parameters: Dict[str, Any]


class Metrics(BaseModel):
    metrics: Dict[str, Any]


class Versions(BaseModel):
    versions: List[str]


class Dataset(BaseModel):
    dataset: str


class Images(BaseModel):
    images: Dict[str, str]
