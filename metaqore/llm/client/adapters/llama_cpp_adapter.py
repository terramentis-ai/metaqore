"""Llama.cpp adapter implementing the LLMClient interface."""

import os
from pathlib import Path
from typing import Any, Dict, Optional
from metaqore.llm.client.interface import LLMClient, LLMProvider, LLMResponse


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

            return LLMResponse(
                content=content,
                provider=self.provider,
                model=(
                    self._model.model_path.split("/")[-1]
                    if hasattr(self._model, "model_path") and self._model.model_path
                    else "llama-cpp-model"
                ),
                success=True,
                metadata={
                    "agent": agent_name,
                    "model_path": self._model_path,
                    "generation_config": gen_config,
                },
                usage=usage,
            )

        except Exception as e:
            return LLMResponse(
                content="",
                provider=self.provider,
                model="llama-cpp-model",
                success=False,
                error=f"Llama.cpp generation failed: {str(e)}",
                metadata={"agent": agent_name, "error_details": str(e)},
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
