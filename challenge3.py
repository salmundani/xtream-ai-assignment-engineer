from fastapi import BackgroundTasks, FastAPI
import asyncio
from typing import List, Annotated
from pydantic import BaseModel, Field
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from contextlib import asynccontextmanager
import logging
from enum import Enum


DATASET_PATH = "datasets/diamonds/diamonds.csv"
CUT_MAP = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
COLOR_MAP = {"J": 0, "I": 1, "H": 2, "G": 3, "F": 4, "E": 5, "D": 6}
CLARITY_MAP = {
    "I1": 0,
    "SI2": 1,
    "SI1": 2,
    "VS2": 3,
    "VS1": 4,
    "VVS2": 5,
    "VVS1": 6,
    "IF": 7,
}

database_lock = asyncio.Lock()
model_lock = asyncio.Lock()
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    await clean_dataset()
    await retrain_model()
    yield


async def clean_dataset():
    dataset = pd.read_csv(DATASET_PATH)
    dataset.drop(dataset[dataset.price <= 0].index, inplace=True)
    dataset.drop(dataset[dataset.x <= 0].index, inplace=True)
    dataset.drop(dataset[dataset.y <= 0].index, inplace=True)
    dataset.drop(dataset[dataset.z <= 0].index, inplace=True)
    dataset.to_csv(DATASET_PATH, index=False)


app = FastAPI(lifespan=lifespan)


class CutEnum(str, Enum):
    fair = "Fair"
    good = "Good"
    very_good = "Very Good"
    premium = "Premium"
    ideal = "Ideal"


class ColorEnum(str, Enum):
    j = "J"
    i = "I"
    h = "H"
    g = "G"
    f = "F"
    e = "E"
    d = "D"


class ClarityEnum(str, Enum):
    i1 = "I1"
    si2 = "SI2"
    si1 = "SI1"
    vs2 = "VS2"
    vs1 = "VS1"
    vvs2 = "VVS2"
    vvs1 = "VVS1"
    if_ = "IF"


class Diamond(BaseModel):
    carat: float = Field(gt=0)
    cut: CutEnum
    color: ColorEnum
    clarity: ClarityEnum
    depth: float = Field(gt=0)
    table: float = Field(gt=0)
    price: int = Field(gt=0)
    x: float = Field(gt=0)
    y: float = Field(gt=0)
    z: float = Field(gt=0)


async def retrain_model():
    global model
    async with database_lock:
        data = pd.read_csv(DATASET_PATH)
    data["cut"] = data["cut"].map(CUT_MAP)
    data["color"] = data["color"].map(COLOR_MAP)
    data["clarity"] = data["clarity"].map(CLARITY_MAP)
    X = data.drop(columns="price")
    y = data["price"]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    new_model = TransformedTargetRegressor(
        regressor=SVR(), transformer=RobustScaler()
    ).fit(X_train, y_train)
    score = new_model.score(X_test, y_test)
    logging.info(f"Model retrained with score: {score}")
    async with model_lock:
        model = new_model


@app.get("/estimate-price")
async def estimate_price(
    carat: Annotated[float, Field(gt=0)],
    cut: CutEnum,
    color: ColorEnum,
    clarity: ClarityEnum,
    depth: Annotated[float, Field(gt=0)],
    table: Annotated[float, Field(gt=0)],
    x: Annotated[float, Field(gt=0)],
    y: Annotated[float, Field(gt=0)],
    z: Annotated[float, Field(gt=0)],
):
    async with model_lock:
        cut = CUT_MAP[cut]
        color = COLOR_MAP[color]
        clarity = CLARITY_MAP[clarity]
        return round(
            model.predict([[carat, cut, color, clarity, depth, table, x, y, z]])[0]
        )


# Assuming we get diamonds in batches. This endpoint will automatically retrain the model.
@app.post("/add-diamonds")
async def add_diamonds(diamonds: List[Diamond], background_tasks: BackgroundTasks):
    background_tasks.add_task(retrain_model)
    async with database_lock:
        await write_to_dataset(diamonds)
    return {"message": "Diamonds added successfully. Model retraining in progress."}


async def write_to_dataset(diamonds):
    with open(DATASET_PATH, "a") as f:
        for diamond in diamonds:
            f.write(
                f"{diamond.carat},{diamond.cut},{diamond.color},{diamond.clarity},{diamond.depth},{diamond.table},{diamond.price},{diamond.x},{diamond.y},{diamond.z}\n"
            )
