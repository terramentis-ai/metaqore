"""AI Gateway service implementation."""

import logging
import time
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx

from metaqore_governance_core.cache import Cache
from metaqore_governance_core.event_bus import event_bus, Event, EventTypes, EventHandler
from metaqore_governance_core.security import SecureGateway
from metaqore_governance_core.psmp import PSMPEngine
from metaqore_governance_core.hmcp_policy import HierarchicalChainingPolicy
from metaqore_governance_core.audit import ComplianceAuditor

from metaqore.llm.client.factory import LLMClientFactory

from .hypie_router import HyPIERouter, InferenceRequest, RoutingPolicy

logger = logging.getLogger(__name__)


class LLMRequest(BaseModel):
    """LLM request payload."""
    prompt: str
    model: Optional[str] = None
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7
    agent_name: str


class AIGatewayService(EventHandler):
    """AI Gateway service for MetaQore."""

    def __init__(self):
        self.app = FastAPI(title="MetaQore AI Gateway")
        self.secure_gateway = SecureGateway()
        self.http_client = httpx.AsyncClient(timeout=30.0)

        # Initialize governance engines
        self.psmp_engine = PSMPEngine()
        self.hmcp_policy_engine = HierarchicalChainingPolicy.from_config()  # TODO: Load from config
        self.compliance_auditor = ComplianceAuditor()

        # Initialize LLM client factory
        self.llm_factory = LLMClientFactory()

        # Initialize HyPIE Router
        self.hypie_router = HyPIERouter(self.psmp_engine, self.hmcp_policy_engine, self.llm_factory, self.compliance_auditor)

        # Initialize cache (Redis in production, in-memory for development)
        self.cache = Cache.create_in_memory_cache()  # TODO: Configure Redis for production

        # Specialist routing registry
        self._specialists: Dict[str, Dict[str, Any]] = {}  # skill_id -> specialist config
        
        # Register as event handler for specialist lifecycle events
        event_bus.register_handler(EventTypes.SPECIALIST_DEPLOYED, self)
        
        self._setup_routes()

    async def handle_event(self, event: Event) -> None:
        """Handle incoming events for specialist lifecycle management."""
        try:
            if event.event_type == EventTypes.SPECIALIST_DEPLOYED:
                await self._handle_specialist_deployed(event)
        except Exception as e:
            logger.error(f"Failed to handle event {event.event_type}: {e}")

    async def _handle_specialist_deployed(self, event: Event) -> None:
        """Handle specialist deployment by registering it for routing."""
        data = event.data
        specialist_id = data["specialist_id"]
        skill_id = data["skill_id"]
        confidence = data["confidence"]
        routing_config = data["routing_config"]

        # Register specialist for routing
        self._specialists[skill_id] = {
            "specialist_id": specialist_id,
            "skill_id": skill_id,
            "confidence": confidence,
            "routing_rules": routing_config,
            "deployment_timestamp": event.timestamp,
        }

        logger.info(f"Registered specialist {specialist_id} for skill {skill_id} with confidence {confidence}")

    def _setup_routes(self):
        """Set up FastAPI routes."""

        @self.app.post("/api/v1/llm/generate")
        async def generate(request: LLMRequest, req: Request):
            """Generate LLM response using HyPIE Router with full governance."""
            start_time = time.time()

            try:
                # Extract client info for security routing
                client_ip = req.client.host if req.client else "unknown"
                user_agent = req.headers.get("user-agent", "")

                # Classify request sensitivity
                sensitivity = await self.secure_gateway.classify_request_sensitivity(
                    request.dict(), client_ip, user_agent
                )

                # Check cache first
                cache_key = f"{request.prompt}:{request.model}:{request.max_tokens}:{request.temperature}"
                cached_response = await self.cache.get(cache_key)
                if cached_response is not None:
                    logger.info(f"Cache hit for request from {request.agent_name}")
                    return cached_response

                # Create HyPIE inference request
                hypie_request = InferenceRequest(
                    prompt=request.prompt,
                    model=request.model,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    agent_name=request.agent_name,
                    sensitivity=sensitivity,
                    routing_policy=RoutingPolicy(
                        data_sensitivity=sensitivity,
                        prefer_local=sensitivity in ["SENSITIVE", "CRITICAL"]
                    )
                )

                # Route through HyPIE Router
                result = await self.hypie_router.route_inference(hypie_request)

                # Cache the response (TTL: 1 hour for LLM responses)
                response_data = {
                    "content": result.content,
                    "provider": result.provider.value,
                    "model": result.model,
                    "tokens_used": result.tokens_used,
                    "processing_time": result.processing_time_ms / 1000,  # Convert to seconds
                    "cost_estimate_usd": result.cost_estimate_usd,
                    "artifact_id": result.artifact_id,
                    "compliance_validated": True
                }
                await self.cache.set(cache_key, response_data, ttl=3600)

                return response_data

            except Exception as e:
                logger.error(f"LLM request failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/v1/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "service": "ai-gateway"}


# Global service instance
service = AIGatewayService()

# Export the FastAPI app for running the service
app = service.app