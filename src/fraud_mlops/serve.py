"""FastAPI scoring service.

Model resolution order at startup:
  1. ``MODEL_PATH`` - an exported bundle directory (used in the container image);
  2. otherwise the MLflow registry alias ``models:/<MODEL_NAME>@<CHAMPION_ALIAS>``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from fraud_mlops import __version__
from fraud_mlops.config import get_settings
from fraud_mlops.data import FEATURE_COLUMNS, MERCHANT_CATEGORIES
from fraud_mlops.registry import ModelBundle, load_alias, load_exported, setup_mlflow

MerchantCategory = Literal[tuple(MERCHANT_CATEGORIES)]  # type: ignore[valid-type]


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload)


def _configure_logging() -> logging.Logger:
    logger = logging.getLogger("fraud_mlops.api")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
        logger.propagate = False
    return logger


log = _configure_logging()


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction_id: str | None = Field(default=None, max_length=64)
    amount: float = Field(gt=0, le=1_000_000, description="Ticket size in account currency")
    merchant_category: MerchantCategory
    hour: int = Field(ge=0, le=23)
    device_mismatch: bool
    geo_mismatch: bool
    txn_count_1h: int = Field(ge=0, le=10_000)
    txn_count_24h: int = Field(ge=0, le=100_000)
    account_age_days: int = Field(ge=0, le=36_500)

    @model_validator(mode="after")
    def _velocity_consistent(self) -> Transaction:
        if self.txn_count_24h < self.txn_count_1h:
            raise ValueError("txn_count_24h must be >= txn_count_1h")
        return self


class BatchRequest(BaseModel):
    transactions: list[Transaction] = Field(min_length=1, max_length=1_000)


class Score(BaseModel):
    transaction_id: str | None
    fraud_probability: float
    decision: Literal["decline", "approve"]
    threshold: float
    model_version: str


class BatchScore(BaseModel):
    scores: list[Score]
    model_version: str
    latency_ms: float


class SingleScore(Score):
    latency_ms: float


def load_bundle_from_env() -> ModelBundle:
    path = os.getenv("MODEL_PATH")
    if path:
        return load_exported(path)
    settings = get_settings()
    client = setup_mlflow(settings)
    return load_alias(client, settings.model_name, settings.champion_alias)


def create_app(bundle: ModelBundle | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "bundle", None) is None:
            app.state.bundle = load_bundle_from_env()
        b = app.state.bundle
        log.info(
            "model loaded",
            extra={
                "extra_fields": {"model": b.name, "version": b.version, "source": b.source, "threshold": b.threshold}
            },
        )
        yield

    app = FastAPI(title="Card Fraud Scoring API", version=__version__, lifespan=lifespan)
    app.state.bundle = bundle

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        request_id = request.headers.get("x-request-id", uuid.uuid4().hex)
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        response.headers["x-request-id"] = request_id
        response.headers["x-latency-ms"] = f"{elapsed:.2f}"
        log.info(
            "request",
            extra={
                "extra_fields": {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "latency_ms": round(elapsed, 2),
                }
            },
        )
        return response

    def _score(transactions: list[Transaction]) -> list[Score]:
        b: ModelBundle | None = app.state.bundle
        if b is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        frame = pd.DataFrame([t.model_dump(exclude={"transaction_id"}) for t in transactions])[FEATURE_COLUMNS]
        frame[["device_mismatch", "geo_mismatch"]] = frame[["device_mismatch", "geo_mismatch"]].astype(int)
        probs = b.predict_proba(frame)
        return [
            Score(
                transaction_id=t.transaction_id,
                fraud_probability=round(float(p), 6),
                decision="decline" if p >= b.threshold else "approve",
                threshold=round(b.threshold, 6),
                model_version=b.version,
            )
            for t, p in zip(transactions, probs, strict=True)
        ]

    @app.get("/health")
    def health() -> dict:
        b: ModelBundle | None = app.state.bundle
        return {
            "status": "ok" if b is not None else "degraded",
            "model_name": b.name if b else None,
            "model_version": b.version if b else None,
            "model_source": b.source if b else None,
            "threshold": b.threshold if b else None,
            "api_version": __version__,
        }

    @app.post("/score", response_model=SingleScore)
    def score(txn: Transaction) -> SingleScore:
        start = time.perf_counter()
        result = _score([txn])[0]
        return SingleScore(**result.model_dump(), latency_ms=round((time.perf_counter() - start) * 1000, 3))

    @app.post("/score/batch", response_model=BatchScore)
    def score_batch(req: BatchRequest) -> BatchScore:
        start = time.perf_counter()
        scores = _score(req.transactions)
        return BatchScore(
            scores=scores,
            model_version=scores[0].model_version,
            latency_ms=round((time.perf_counter() - start) * 1000, 3),
        )

    return app


def main(host: str = "0.0.0.0", port: int = 8000) -> None:  # noqa: S104 - container entrypoint
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port, log_level="warning")
