from __future__ import annotations

from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from visionflow.api_service import execute_component, get_component, list_components


class ComponentExecutionRequest(BaseModel):
    init_args: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    image_base64: str | None = None
    payload: dict[str, Any] | None = None
    data: Any = None
    batch: list[Any] | None = None


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Vision Flow API",
        version="0.1.0",
        description="FastAPI endpoints for AI Vision Flow components.",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/catalog")
    def catalog() -> dict[str, Any]:
        return {"categories": list_components()}

    @app.get("/api/v1/catalog/{category}/{name}")
    def catalog_item(category: str, name: str) -> dict[str, Any]:
        try:
            component_cls = get_component(category, name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "category": category,
            "name": name,
            "description": (component_cls.__doc__ or "").strip(),
        }

    @app.post("/api/v1/{category}/{name}")
    def run_component(category: str, name: str, request: ComponentExecutionRequest) -> dict[str, Any]:
        try:
            return execute_component(
                category=category,
                name=name,
                init_args=request.init_args,
                config=request.config,
                image_base64=request.image_base64,
                payload=request.payload,
                data=request.data,
                batch=request.batch,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app()


def run(host: str = "0.0.0.0", port: int = 8300, reload: bool = False) -> None:
    uvicorn.run("visionflow.api:app", host=host, port=port, reload=reload)
