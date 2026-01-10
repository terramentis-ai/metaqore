from typing import Any, Dict, Optional
from ..interface import LLMClient, LLMProvider, LLMResponse

try:
    import requests
except ImportError:
    requests = None


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
            data = resp.json()
            content = data["choices"][0]["text"]
            usage = data.get("usage", {})
            return LLMResponse(
                content=content,
                provider=self.provider,
                model=self._model,
                success=True,
                metadata={
                    "agent": agent_name,
                    "endpoint": self._endpoint,
                    "generation_config": payload,
                },
                usage=usage,
            )
        except Exception as e:
            return LLMResponse(
                content="",
                provider=self.provider,
                model=self._model or "vllm-model",
                success=False,
                error=f"vLLM generation failed: {str(e)}",
                metadata={"agent": agent_name, "error_details": str(e)},
            )

    def validate_config(self, config: Dict[str, Any]) -> bool:
        return bool(config.get("endpoint") and config.get("model"))
