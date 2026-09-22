"""REST routing without logging customer messages."""

import logging
import os
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException

from intentlab.service import RoutingRequest, RoutingService

logger = logging.getLogger(__name__)


def create_app(service=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.service = service
        if service is None:
            try:
                app.state.service = RoutingService(
                    os.getenv("MODEL_DIR", "artifacts"),
                    os.getenv("ENCODER_DIR", "data/encoder"),
                )
            except (OSError, ValueError, KeyError):
                logger.exception("Model could not be loaded")
        yield

    app = FastAPI(title="Banking intent routing API", version="1.0.0", lifespan=lifespan)

    def current():
        value = getattr(app.state, "service", None)
        if value is None:
            raise HTTPException(503, "Model unavailable. Run python -m intentlab.train first.")
        return value

    @app.get("/health")
    def health():
        return {"alive": True}

    @app.get("/ready")
    def ready():
        return {"ready": True, "model_version": current().version}

    @app.post("/predict")
    def predict(request: RoutingRequest):
        result = current().predict(request)
        result["request_id"] = str(uuid4())
        logger.info(
            "route request_id=%s model=%s action=%s",
            result["request_id"],
            result["model_version"],
            result["action"],
        )
        return result

    return app


app = create_app()
