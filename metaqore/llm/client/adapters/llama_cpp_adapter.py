"""Llama.cpp adapter implementing the LLMClient interface."""

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional
from metaqore.llm.client.interface import LLMClient, LLMProvider, LLMResponse

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


class LlamaCppAdapter(LLMClient):
    """Adapter for llama.cpp local LLM inference."""

    def __init__(self):
        self._initialized = False
        self._model: Optional[Any] = None
        self._model_path: Optional[str] = None
        self._model_config: Dict[str, Any] = {}

    @property
    def provider(self) -> LLMProvider:
        return LLMProvider.LLAMA_CPP

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the Llama.cpp adapter.

        Expected config:
        - model_path: Path to GGUF model file
        - n_ctx: Context window size (default: 2048)
        - n_threads: Number of threads (default: -1 for auto)
        - n_gpu_layers: Number of layers to offload to GPU (default: 0)
        - temperature: Sampling temperature (default: 0.7)
        - max_tokens: Maximum tokens to generate (default: 256)
        """
        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python not installed. Install with: pip install llama-cpp-python"
            )

        model_path = config.get("model_path")
        if not model_path:
            raise ValueError("model_path is required for LlamaCppAdapter")

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Store config for later use
        self._model_path = model_path
        self._model_config = {
            "n_ctx": config.get("n_ctx", 2048),
            "n_threads": config.get("n_threads", -1),
            "n_gpu_layers": config.get("n_gpu_layers", 0),
            "temperature": config.get("temperature", 0.7),
            "max_tokens": config.get("max_tokens", 256),
            "verbose": config.get("verbose", False),
        }

        # Initialize the model
        self._model = Llama(model_path=model_path, **self._model_config)

        self._initialized = True

    def generate(
        self, prompt: str, *, agent_name: str, metadata: Dict[str, Any], **kwargs
    ) -> LLMResponse:
        """
        Generate a completion using llama.cpp.

        Args:
            prompt: The prompt to complete
            agent_name: Name of the agent (for metadata)
            metadata: Additional context
            **kwargs: Override default generation parameters
        """
        if not self._initialized or not self._model:
            raise RuntimeError("LlamaCppAdapter not initialized")
        
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
                    "model": self._model_path or "llama-cpp-model",
                    "agent_name": agent_name,
                    "prompt_length": len(prompt),
                    "metadata": metadata,
                },
                correlation_id=metadata.get("correlation_id"),
            ))

        # Merge kwargs with default config
        gen_config = {**self._model_config, **kwargs}

        try:
            # Generate completion
            output = self._model(
                prompt,
                max_tokens=gen_config.get("max_tokens", 256),
                temperature=gen_config.get("temperature", 0.7),
                echo=False,  # Don't include prompt in response
            )

            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000

            # Extract the generated text
            if isinstance(output, dict) and "choices" in output:
                # OpenAI-style response format
                content = output["choices"][0]["text"]
                usage = output.get("usage", {})
            elif isinstance(output, list) and len(output) > 0:
                # List format
                content = output[0]["text"] if isinstance(output[0], dict) else str(output[0])
                usage = {}
            else:
                content = str(output)
                usage = {}

            # Record metrics
            if get_metrics_aggregator():
                aggregator = get_metrics_aggregator()
                aggregator.record_api_latency(f"llm_{self.provider.value}", latency_ms)

            model_name = (
                self._model.model_path.split("/")[-1]
                if hasattr(self._model, "model_path") and self._model.model_path
                else "llama-cpp-model"
            )
            
            # Prepare artifact context
            artifact_context = self.prepare_artifact_context(
                LLMResponse(
                    content=content,
                    provider=self.provider,
                    model=model_name,
                    success=True,
                    usage=usage,
                ),
                metadata
            )
            artifact_context.update({
                "latency_ms": latency_ms,
                "request_id": request_id,
                "agent_name": agent_name,
                "model_path": self._model_path,
            })
            
            llm_response = LLMResponse(
                content=content,
                provider=self.provider,
                model=model_name,
                success=True,
                metadata={
                    "agent": agent_name,
                    "model_path": self._model_path,
                    "generation_config": gen_config,
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
                        "model": model_name,
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
                        "model": self._model_path or "llama-cpp-model",
                        "latency_ms": latency_ms,
                        "error": str(e),
                        "success": False,
                    },
                    correlation_id=metadata.get("correlation_id"),
                ))
            
            return LLMResponse(
                content="",
                provider=self.provider,
                model="llama-cpp-model",
                success=False,
                error=f"Llama.cpp generation failed: {str(e)}",
                metadata={"agent": agent_name, "error_details": str(e), "latency_ms": latency_ms},
                artifact_context={
                    "provider": self.provider.value,
                    "model": self._model_path or "llama-cpp-model",
                    "error": str(e),
                    "latency_ms": latency_ms,
                    "request_id": request_id,
                    "timestamp": metadata.get("timestamp"),
                }
            )

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate Llama.cpp configuration."""
        if not config.get("model_path"):
            return False

        model_path = Path(config["model_path"])
        if not model_path.exists():
            return False

        # Check if it's a valid GGUF file (basic check)
        if model_path.suffix.lower() not in [".gguf", ".bin"]:
            return False

        return True
