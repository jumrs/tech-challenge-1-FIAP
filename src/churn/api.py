"""API FastAPI para servir as predicoes de churn do modelo campeao."""

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, get_args

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field
from sklearn.pipeline import Pipeline
from starlette.middleware.base import BaseHTTPMiddleware

from churn.config import CHAMPION_MODEL_PATH, PREDICTION_THRESHOLD
from churn.logger import get_logger
from churn.schema import CustomerFeatures

logger = get_logger(__name__)

MODEL_PATH = CHAMPION_MODEL_PATH
MAX_BATCH_SIZE = 1000

# Colunas numericas: quando chegam todas como null, precisam virar NaN (float) para o pipeline.
NUMERIC_FIELDS = [
    name
    for name, field in CustomerFeatures.model_fields.items()
    if {field.annotation, *get_args(field.annotation)} & {int, float}
]

# Cliente real da base (CustomerID 3000002, que cancelou), usado como exemplo no Swagger.
EXAMPLE_CUSTOMER = {
    "CustomerID": 3000002,
    "MonthlyRevenue": 24.0,
    "MonthlyMinutes": 219,
    "TotalRecurringCharge": 22,
    "DirectorAssistedCalls": 0.25,
    "OverageMinutes": 0,
    "RoamingCalls": 0.0,
    "PercChangeMinutes": -157,
    "PercChangeRevenues": -19.0,
    "DroppedCalls": 0.7,
    "BlockedCalls": 0.7,
    "UnansweredCalls": 6.3,
    "CustomerCareCalls": 0.0,
    "ThreewayCalls": 0.0,
    "ReceivedCalls": 97.2,
    "OutboundCalls": 0.0,
    "InboundCalls": 0.0,
    "PeakCallsInOut": 58.0,
    "OffPeakCallsInOut": 24.0,
    "DroppedBlockedCalls": 1.3,
    "CallForwardingCalls": 0,
    "CallWaitingCalls": 0.3,
    "MonthsInService": 61,
    "UniqueSubs": 2,
    "ActiveSubs": 1,
    "ServiceArea": "SEAPOR503",
    "Handsets": 2,
    "HandsetModels": 2,
    "CurrentEquipmentDays": 361,
    "AgeHH1": 62,
    "AgeHH2": 0,
    "ChildrenInHH": "No",
    "HandsetRefurbished": "No",
    "HandsetWebCapable": "Yes",
    "TruckOwner": "No",
    "RVOwner": "No",
    "Homeownership": "Known",
    "BuysViaMailOrder": "Yes",
    "RespondsToMailOffers": "Yes",
    "OptOutMailings": "No",
    "NonUSTravel": "No",
    "OwnsComputer": "Yes",
    "HasCreditCard": "Yes",
    "NewCellphoneUser": "No",
    "NotNewCellphoneUser": "No",
    "ReferralsMadeBySubscriber": 0,
    "IncomeGroup": 4,
    "OwnsMotorcycle": "No",
    "AdjustmentsToCreditRating": 0,
    "HandsetPrice": "30",
    "CreditRating": "1-Highest",
    "PrizmCode": "Suburban",
    "Occupation": "Professional",
    "MaritalStatus": "No",
}


class CustomerPayload(CustomerFeatures):
    """Dados de um cliente. Campos desconhecidos (ex.: variaveis de retencao) sao recusados."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_CUSTOMER]})

    CustomerID: int | None = Field(default=None, description="Opcional, devolvido na resposta.")


class BatchPredictionRequest(BaseModel):
    """Lote de clientes para pontuar de uma vez (ex.: campanha de retencao)."""

    customers: list[CustomerPayload] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class PredictionResponse(BaseModel):
    """Propensao de churn de um cliente."""

    CustomerID: int | None
    churn_probability: float = Field(ge=0, le=1)
    churn_prediction: Literal[0, 1]
    threshold: float


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model: str | None


def load_model(path: Path) -> Pipeline | None:
    """Carrega o pipeline campeao; sem o arquivo, a API sobe em modo degradado."""

    if not path.exists():
        logger.warning(f"Modelo nao encontrado em {path}. Rode `uv run python -m churn.main`.")
        return None
    model = joblib.load(path)
    logger.info(f"Modelo carregado de {path}: {_model_name(model)}.")
    return model


def _model_name(model: Pipeline) -> str:
    return type(model.named_steps["classifier"]).__name__


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Carrega o modelo uma unica vez, na subida, e nao a cada requisicao.
    app.state.model = load_model(MODEL_PATH)
    yield


app = FastAPI(
    title="Churn Prediction API",
    description="Propensao de cancelamento de clientes (Cell2Cell) com o modelo campeao.",
    version="1.0.0",
    lifespan=lifespan,
)


class LatencyLoggingMiddleware(BaseHTTPMiddleware):
    """Registra a latencia de cada requisicao no log e no header X-Process-Time-ms."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000
        response.headers["X-Process-Time-ms"] = f"{duration_ms:.2f}"
        logger.info(
            f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms:.2f} ms)"
        )
        return response


app.add_middleware(LatencyLoggingMiddleware)


def _get_model(request: Request) -> Pipeline:
    model = request.app.state.model
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo indisponivel. Treine com `uv run python -m churn.main` e reinicie a API.",
        )
    return model


def _predict(model: Pipeline, customers: list[CustomerPayload]) -> list[PredictionResponse]:
    """Monta o DataFrame bruto e aplica o pipeline completo (feature engineering incluso)."""

    frame = pd.DataFrame([customer.model_dump(exclude={"CustomerID"}) for customer in customers])
    frame[NUMERIC_FIELDS] = frame[NUMERIC_FIELDS].astype(float)
    probabilities = model.predict_proba(frame)[:, 1]
    return [
        PredictionResponse(
            CustomerID=customer.CustomerID,
            churn_probability=round(float(probability), 4),
            churn_prediction=int(probability >= PREDICTION_THRESHOLD),
            threshold=PREDICTION_THRESHOLD,
        )
        for customer, probability in zip(customers, probabilities)
    ]


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Indica se a API esta no ar e se o modelo foi carregado."""

    model = request.app.state.model
    return HealthResponse(
        status="ok" if model is not None else "degraded",
        model_loaded=model is not None,
        model=_model_name(model) if model is not None else None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerPayload, request: Request) -> PredictionResponse:
    """Retorna a propensao de churn de um cliente."""

    prediction = _predict(_get_model(request), [customer])[0]
    logger.info(
        f"Predicao: CustomerID={prediction.CustomerID}, "
        f"probabilidade={prediction.churn_probability}, classe={prediction.churn_prediction}"
    )
    return prediction


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(batch: BatchPredictionRequest, request: Request) -> BatchPredictionResponse:
    """Retorna a propensao de churn de varios clientes (ate MAX_BATCH_SIZE por chamada)."""

    predictions = _predict(_get_model(request), batch.customers)
    logger.info(f"Predicao em lote: {len(predictions)} clientes.")
    return BatchPredictionResponse(predictions=predictions)
