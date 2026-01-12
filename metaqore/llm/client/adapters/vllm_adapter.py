from typing import Any, Dict, Optional
import time
from ..interface import LLMClient, LLMProvider, LLMResponse

try:
    import requests
except ImportError:
    requests = None

# Import governance components for metrics and events
try:
    from metaqore_governance_core.event_bus import event_bus, Event, EventTypes
    from metaqore.metrics.aggregator import get_metrics_aggregator
except ImportError:
    # Fallback for when governance-core is not available
    event_bus = None
    Event = None
    EventTypes = None
    get_metrics_aggregator = lambda: None


class VLLMAdapter(LLMClient):
    """
    Adapter for vLLM inference server (OpenAI-compatible API).
    """

    def __init__(self):
        self._initialized = False
        self._endpoint = None
        self._api_key = None
        self._model = None

    @property
    def provider(self):
        return LLMProvider.VLLM

    def initialize(self, config: Dict[str, Any]) -> None:
        self._endpoint = config.get("endpoint")
        self._api_key = config.get("api_key")
        self._model = config.get("model")
        if not self._endpoint or not self._model:
            raise ValueError("endpoint and model are required for VLLMAdapter")
        self._initialized = True

    def generate(
        self, prompt: str, *, agent_name: str, metadata: Dict[str, Any], **kwargs
    ) -> LLMResponse:
        if not self._initialized:
            raise RuntimeError("VLLMAdapter not initialized")
        if requests is None:
            raise RuntimeError("requests library not available")
        
        start_time = time.time()
        request_id = metadata.get("request_id", f"req_{int(start_time)}")
        
        # Publish request started event
        if event_bus and Event and EventTypes:
            event_bus.publish(Event(
                event_type=EventTypes.LLM_REQUEST_STARTED,
                source=f"llm_adapter.{self.provider.value}",
                data={
                    "request_id": request_id,
                    "provider": self.provider.value,
                    "model": self._model,
                    "agent_name": agent_name,
                    "prompt_length": len(prompt),
                    "metadata": metadata,
                },
                correlation_id=metadata.get("correlation_id"),
            ))
        
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {
            "model": self._model,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 256),
            "temperature": kwargs.get("temperature", 0.7),
        }
        try:
            resp = requests.post(
                f"{self._endpoint}/v1/completions", json=payload, headers=headers, timeout=30
            )
            resp.raise_for_status()
            
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            data = resp.json()
            content = data["choices"][0]["text"]
            usage = data.get("usage", {})
            
            # Record metrics
            if get_metrics_aggregator():
                aggregator = get_metrics_aggregator()
                aggregator.record_api_latency(f"llm_{self.provider.value}", latency_ms)
            
            # Prepare artifact context
            artifact_context = self.prepare_artifact_context(
                LLMResponse(
                    content=content,
                    provider=self.provider,
                    model=self._model,
                    success=True,
                    usage=usage,
                ),
                metadata
            )
            artifact_context.update({
                "latency_ms": latency_ms,
                "request_id": request_id,
                "agent_name": agent_name,
                "endpoint": self._endpoint,
            })
            
            llm_response = LLMResponse(
                content=content,
                provider=self.provider,
                model=self._model,
                success=True,
                metadata={
                    "agent": agent_name,
                    "endpoint": self._endpoint,
                    "generation_config": payload,
                    "latency_ms": latency_ms,
                },
                usage=usage,
                artifact_context=artifact_context,
            )
            
            # Publish completion event
            if event_bus and Event and EventTypes:
                event_bus.publish(Event(
                    event_type=EventTypes.LLM_REQUEST_COMPLETED,
                    source=f"llm_adapter.{self.provider.value}",
                    data={
                        "request_id": request_id,
                        "provider": self.provider.value,
                        "model": self._model,
                        "latency_ms": latency_ms,
                        "tokens_used": usage.get("completion_tokens", 0),
                        "success": True,
                        "artifact_context": artifact_context,
                    },
                    correlation_id=metadata.get("correlation_id"),
                ))
            
            return llm_response
            
        except Exception as e:
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            # Record failed request metrics
            if get_metrics_aggregator():
                aggregator = get_metrics_aggregator()
                aggregator.record_api_latency(f"llm_{self.provider.value}_failed", latency_ms)
            
            # Publish failure event
            if event_bus and Event and EventTypes:
                event_bus.publish(Event(
                    event_type=EventTypes.LLM_REQUEST_FAILED,
                    source=f"llm_adapter.{self.provider.value}",
                    data={
                        "request_id": request_id,
                        "provider": self.provider.value,
                        "model": self._model or "vllm-model",
                        "latency_ms": latency_ms,
                        "error": str(e),
                        "success": False,
                    },
                    correlation_id=metadata.get("correlation_id"),
                ))
            
            return LLMResponse(
                content="",
                provider=self.provider,
                model=self._model or "vllm-model",
                success=False,
                error=f"vLLM generation failed: {str(e)}",
                metadata={"agent": agent_name, "error_details": str(e), "latency_ms": latency_ms},
                artifact_context={
                    "provider": self.provider.value,
                    "model": self._model or "vllm-model",
                    "error": str(e),
                    "latency_ms": latency_ms,
                    "request_id": request_id,
                    "timestamp": metadata.get("timestamp"),
                }
            )

    def validate_config(self, config: Dict[str, Any]) -> bool:
        return bool(config.get("endpoint") and config.get("model"))
