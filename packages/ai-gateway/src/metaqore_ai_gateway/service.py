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
            """Proxy LLM generation requests with governance."""
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

                # Publish request started event
                event = Event(
                    event_type=EventTypes.LLM_REQUEST_STARTED,
                    source="ai-gateway",
                    data={
                        "agent_name": request.agent_name,
                        "sensitivity": sensitivity,
                        "model": request.model
                    }
                )
                await event_bus.publish(event)

                # Route to appropriate LLM provider based on sensitivity
                provider_endpoint = await self._route_to_provider(sensitivity, request.model, request.agent_name, request.prompt)

                # Make the actual LLM call
                response_data = await self._call_llm_provider(provider_endpoint, request)

                # Cache the response (TTL: 1 hour for LLM responses)
                await self.cache.set(cache_key, response_data, ttl=3600)

                # Publish completion event
                processing_time = time.time() - start_time
                event = Event(
                    event_type=EventTypes.LLM_REQUEST_COMPLETED,
                    source="ai-gateway",
                    data={
                        "agent_name": request.agent_name,
                        "processing_time": processing_time,
                        "sensitivity": sensitivity
                    }
                )
                await event_bus.publish(event)

                return response_data

            except Exception as e:
                # Publish failure event
                processing_time = time.time() - start_time
                event = Event(
                    event_type=EventTypes.LLM_REQUEST_FAILED,
                    source="ai-gateway",
                    data={
                        "agent_name": request.agent_name,
                        "error": str(e),
                        "processing_time": processing_time
                    }
                )
                await event_bus.publish(event)

                logger.error(f"LLM request failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/v1/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "service": "ai-gateway"}

    async def _route_to_provider(self, sensitivity: str, model: Optional[str], agent_name: str = "", prompt: str = "") -> str:
        """Route request to appropriate LLM provider or specialist based on context."""
        
        # First, check if we have a specialist that can handle this request
        specialist_route = await self._route_to_specialist(agent_name, prompt)
        if specialist_route:
            return specialist_route
        
        # Fall back to LLM provider routing based on sensitivity
        if sensitivity == "CRITICAL":
            return "https://secure-llm-provider.com/v1/chat/completions"
        else:
            return "https://standard-llm-provider.com/v1/chat/completions"

    async def _route_to_specialist(self, agent_name: str, prompt: str) -> Optional[str]:
        """Check if a specialist should handle this request."""
        
        # Simple skill detection based on agent name and prompt content
        # In a real implementation, this would use more sophisticated NLP
        detected_skill = None
        
        if "akkadian" in prompt.lower() or "cuneiform" in prompt.lower():
            detected_skill = "clean_akkadian_dates"
        elif "translate" in prompt.lower() and "language" in prompt.lower():
            detected_skill = "translation"
        # Add more skill detection logic as needed
        
        if detected_skill and detected_skill in self._specialists:
            specialist_config = self._specialists[detected_skill]
            
            # Check if specialist meets confidence requirements
            confidence_threshold = specialist_config["routing_rules"].get("confidence_min", 0.8)
            
            # For now, assume we can estimate confidence from prompt complexity
            # In reality, this would be determined by the specialist's actual capabilities
            prompt_complexity = min(1.0, len(prompt.split()) / 100.0)  # Simple heuristic
            
            if prompt_complexity >= confidence_threshold:
                specialist_id = specialist_config["specialist_id"]
                logger.info(f"Routing to specialist {specialist_id} for skill {detected_skill}")
                return f"specialist://{specialist_id}"
        
        return None

    async def _call_llm_provider(self, endpoint: str, request: LLMRequest) -> Dict[str, Any]:
        """Make actual call to LLM provider or specialist."""
        
        # Check if this is a specialist route
        if endpoint.startswith("specialist://"):
            specialist_id = endpoint.split("specialist://")[1]
            return await self._call_specialist(specialist_id, request)
        
        # Regular LLM provider call
        # This would integrate with the LLM adapter system
        # For now, return a mock response
        return {
            "content": f"Mock LLM response for: {request.prompt[:50]}...",
            "provider": "mock-llm",
            "model": request.model or "gpt-3.5-turbo",
            "usage": {
                "prompt_tokens": len(request.prompt.split()),
                "completion_tokens": 50
            }
        }

    async def _call_specialist(self, specialist_id: str, request: LLMRequest) -> Dict[str, Any]:
        """Call a specialist for processing."""
        
        # Find the specialist config
        specialist_config = None
        for skill_config in self._specialists.values():
            if skill_config["specialist_id"] == specialist_id:
                specialist_config = skill_config
                break
        
        if not specialist_config:
            raise ValueError(f"Specialist {specialist_id} not found in routing registry")
        
        # In a real implementation, this would make an HTTP call to the specialist service
        # For now, simulate specialist processing
        skill_id = specialist_config["skill_id"]
        
        # Simulate different specialist behaviors based on skill
        if skill_id == "clean_akkadian_dates":
            # Simulate Akkadian date cleaning
            response_content = f"Specialist {specialist_id} processed: Cleaned Akkadian dates in '{request.prompt[:30]}...'"
        else:
            response_content = f"Specialist {specialist_id} processed: {request.prompt[:50]}..."
        
        return {
            "content": response_content,
            "provider": f"specialist-{specialist_id}",
            "model": f"specialist-{skill_id}",
            "confidence": specialist_config["confidence"],
            "usage": {
                "prompt_tokens": len(request.prompt.split()),
                "completion_tokens": len(response_content.split())
            }
        }


# Global service instance
service = AIGatewayService()

# Export the FastAPI app for running the service
app = service.app