from typing import Any, Dict, Optional
from ..interface import LLMClient, LLMProvider, LLMResponse

try:
    import anthropic
except ImportError:
    anthropic = None


class AnthropicAdapter(LLMClient):
    """
    Adapter for Anthropic API.
    """

    def __init__(self):
        self._initialized = False
        self._api_key = None
        self._model = None

    @property
    def provider(self):
        return LLMProvider.ANTHROPIC

    def initialize(self, config: Dict[str, Any]) -> None:
        self._api_key = config.get("api_key")
        self._model = config.get("model", "claude-3-sonnet-20240229")
        if not self._api_key:
            raise ValueError("api_key is required for AnthropicAdapter")
        self._initialized = True

    def generate(
        self, prompt: str, *, agent_name: str, metadata: Dict[str, Any], **kwargs
    ) -> LLMResponse:
        if not self._initialized:
            raise RuntimeError("AnthropicAdapter not initialized")
        if anthropic is None:
            raise RuntimeError("anthropic library not available")
        client = anthropic.Anthropic(api_key=self._api_key)
        try:
            response = client.messages.create(
                model=self._model,
                max_tokens=kwargs.get("max_tokens", 256),
                temperature=kwargs.get("temperature", 0.7),
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
            return LLMResponse(
                content=content,
                provider=self.provider,
                model=self._model,
                success=True,
                metadata={"agent": agent_name, "generation_config": kwargs},
                usage=usage,
            )
        except Exception as e:
            return LLMResponse(
                content="",
                provider=self.provider,
                model=self._model,
                success=False,
                error=f"Anthropic generation failed: {str(e)}",
                metadata={"agent": agent_name, "error_details": str(e)},
            )

    def validate_config(self, config: Dict[str, Any]) -> bool:
        return bool(config.get("api_key"))
