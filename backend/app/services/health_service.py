from typing import Dict, Any
import logging
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

class HealthService:
    """Centralized health check service to avoid duplicate health check logic"""
    
    def __init__(self):
        self._health_checks = {}
    
    def register_health_check(self, name: str, check_func):
        """Register a health check function"""
        self._health_checks[name] = check_func
    
    async def check_health(self) -> Dict[str, Any]:
        """Perform all registered health checks"""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {}
        }
        
        overall_healthy = True
        
        for name, check_func in self._health_checks.items():
            try:
                # Check if it's a coroutine function
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    # For regular functions, check if callable
                    if callable(check_func):
                        result = check_func()
                    else:
                        result = {"healthy": True, "message": "Service registered"}
                
                health_status["services"][name] = result
                if not result.get("healthy", True):
                    overall_healthy = False
                    
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                health_status["services"][name] = {
                    "healthy": False,
                    "error": str(e)
                }
                overall_healthy = False
        
        health_status["status"] = "healthy" if overall_healthy else "unhealthy"
        return health_status

# Global instance
health_service = HealthService()