from fastapi import APIRouter
from app.api.health import router as health_router
from app.api.aptos import router as aptos_router
from app.api.vessel import router as vessel_router
from app.api.idrid import router as idrid_router
from app.api.unified import router as unified_router
from app.api.model_info import router as model_info_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(aptos_router)
api_router.include_router(vessel_router)
api_router.include_router(idrid_router)
api_router.include_router(unified_router)
api_router.include_router(model_info_router)

__all__ = ["api_router"]
