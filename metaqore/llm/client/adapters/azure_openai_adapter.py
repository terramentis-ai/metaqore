from typing import Any, Dict, Optional
import time
from ..interface import LLMClient, LLMProvider, LLMResponse

try:
    import openai
except ImportError:
    openai = None

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


class AzureOpenAIAdapter(LLMClient):
    """
    Adapter for Azure OpenAI API.
    """

    def __init__(self):
        self._initialized = False
        self._api_key = None
        self._endpoint = None
        self._deployment = None
        self._api_version = None

    @property
    def provider(self):
        return LLMProvider.AZURE_OPENAI

    def initialize(self, config: Dict[str, Any]) -> None:
        self._api_key = config.get("api_key")
        self._endpoint = config.get("endpoint")
        self._deployment = config.get("deployment")
        self._api_version = config.get("api_version", "2023-12-01-preview")
        if not self._api_key:
            raise ValueError("api_key is required for AzureOpenAIAdapter")
        if not self._endpoint:
            raise ValueError("endpoint is required for AzureOpenAIAdapter")
        if not self._deployment:
            raise ValueError("deployment is required for AzureOpenAIAdapter")
        self._initialized = True

    def generate(
        self, prompt: str, *, agent_name: str, metadata: Dict[str, Any], **kwargs
    ) -> LLMResponse:
        if not self._initialized:
            raise RuntimeError("AzureOpenAIAdapter not initialized")
        if openai is None:
            raise RuntimeError("openai library not available")

        start_time = time.time()
        request_id = metadata.get("request_id", f"req_{int(start_time)}")

        # Publish request started event
        if event_bus and Event and EventTypes:
            event_bus.publish(
                Event(
                    event_type=EventTypes.LLM_REQUEST_STARTED,
                    source=f"llm_adapter.{self.provider.value}",
                    data={
                        "request_id": request_id,
                        "provider": self.provider.value,
                        "model": self._deployment,
                        "agent_name": agent_name,
                        "prompt_length": len(prompt),
                        "metadata": metadata,
                    },
                    correlation_id=metadata.get("correlation_id"),
                )
            )

        client = openai.AzureOpenAI(
            api_key=self._api_key,
            azure_endpoint=self._endpoint,
            api_version=self._api_version,
        )
        try:
            response = client.chat.completions.create(
                model=self._deployment,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", 256),
                temperature=kwargs.get("temperature", 0.7),
            )

            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000

            content = response.choices[0].message.content
            usage = response.usage.model_dump() if response.usage else {}

            # Record metrics
            if get_metrics_aggregator():
                aggregator = get_metrics_aggregator()
                aggregator.record_api_latency(f"llm_{self.provider.value}", latency_ms)

            # Prepare artifact context
            artifact_context = self.prepare_artifact_context(
                LLMResponse(
                    content=content,
                    provider=self.provider,
                    model=self._deployment,
                    success=True,
                    usage=usage,
                ),
                metadata,
            )
            artifact_context.update(
                {
                    "latency_ms": latency_ms,
                    "request_id": request_id,
                    "agent_name": agent_name,
                    "endpoint": self._endpoint,
                }
            )

            llm_response = LLMResponse(
                content=content,
                provider=self.provider,
                model=self._deployment,
                success=True,
                metadata={
                    "agent": agent_name,
                    "endpoint": self._endpoint,
                    "generation_config": kwargs,
                    "latency_ms": latency_ms,
                },
                usage=usage,
                artifact_context=artifact_context,
            )

            # Publish completion event
            if event_bus and Event and EventTypes:
                event_bus.publish(
                    Event(
                        event_type=EventTypes.LLM_REQUEST_COMPLETED,
                        source=f"llm_adapter.{self.provider.value}",
                        data={
                            "request_id": request_id,
                            "provider": self.provider.value,
                            "model": self._deployment,
                            "latency_ms": latency_ms,
                            "tokens_used": usage.get("total_tokens", 0),
                            "success": True,
                            "artifact_context": artifact_context,
                        },
                        correlation_id=metadata.get("correlation_id"),
                    )
                )

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
                event_bus.publish(
                    Event(
                        event_type=EventTypes.LLM_REQUEST_FAILED,
                        source=f"llm_adapter.{self.provider.value}",
                        data={
                            "request_id": request_id,
                            "provider": self.provider.value,
                            "model": self._deployment,
                            "latency_ms": latency_ms,
                            "error": str(e),
                            "success": False,
                        },
                        correlation_id=metadata.get("correlation_id"),
                    )
                )

            return LLMResponse(
                content="",
                provider=self.provider,
                model=self._deployment,
                success=False,
                error=f"Azure OpenAI generation failed: {str(e)}",
                metadata={
                    "agent": agent_name,
                    "endpoint": self._endpoint,
                    "error_details": str(e),
                    "latency_ms": latency_ms,
                },
                artifact_context={
                    "provider": self.provider.value,
                    "model": self._deployment,
                    "error": str(e),
                    "latency_ms": latency_ms,
                    "request_id": request_id,
                    "timestamp": metadata.get("timestamp"),
                },
            )

    def validate_config(self, config: Dict[str, Any]) -> bool:
        return bool(config.get("api_key") and config.get("endpoint") and config.get("deployment"))
