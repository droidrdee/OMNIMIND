"""Query and analytics FastAPI routes."""

import time

from fastapi import APIRouter, HTTPException, Request
from loguru import logger

from models.request_models import QueryRequest
from models.response_models import QueryResponse

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def run_query(req: QueryRequest, request: Request) -> QueryResponse:
    """Route a user query through the QueryRouter and return the answer."""
    try:
        start = time.time()
        result = request.app.state.router.route(req.query, force_route=req.force_route)
        latency_ms = (time.time() - start) * 1000

        return QueryResponse(
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            route=result.get("route", ""),
            routing_reason=result.get("routing_reason", ""),
            confidence=result.get("confidence"),
            latency_ms=latency_ms,
        )
    except Exception as exc:
        logger.exception("Query execution failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/analytics")
async def get_analytics(request: Request) -> dict:
    """Return cumulative routing analytics."""
    try:
        return request.app.state.router.get_analytics()
    except Exception as exc:
        logger.exception("Analytics fetch failed")
        raise HTTPException(status_code=500, detail=str(exc))
