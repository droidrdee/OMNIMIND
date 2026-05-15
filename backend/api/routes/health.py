"""Health-check route for service liveness probes."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Simple health probe."""
    return {"status": "ok", "service": "omnimind-backend"}
