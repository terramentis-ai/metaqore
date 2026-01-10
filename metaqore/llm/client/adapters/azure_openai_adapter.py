from typing import Any, Dict, Optional
from ..interface import LLMClient, LLMProvider, LLMResponse

try:
    import openai
except ImportError:
    openai = None


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
            content = response.choices[0].message.content
            usage = response.usage.model_dump() if response.usage else {}
            return LLMResponse(
                content=content,
                provider=self.provider,
                model=self._deployment,
                success=True,
                metadata={
                    "agent": agent_name,
                    "endpoint": self._endpoint,
                    "generation_config": kwargs,
                },
                usage=usage,
            )
        except Exception as e:
            return LLMResponse(
                content="",
                provider=self.provider,
                model=self._deployment,
                success=False,
                error=f"Azure OpenAI generation failed: {str(e)}",
                metadata={"agent": agent_name, "endpoint": self._endpoint, "error_details": str(e)},
            )

    def validate_config(self, config: Dict[str, Any]) -> bool:
        return bool(config.get("api_key") and config.get("endpoint") and config.get("deployment"))
