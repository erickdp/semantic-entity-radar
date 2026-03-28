import uvicorn
from fastapi import FastAPI

from src.entrypoints.api.error_handlers import install_error_handlers
from src.infrastructure.adapters.inbound.http_query_controller import (
    router as query_router,
)


def create_app() -> FastAPI:
    app = FastAPI(title="Entity Radar API", version="0.1.0")
    app.include_router(query_router)
    install_error_handlers(app)
    return app


if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
