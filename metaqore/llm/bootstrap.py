"""Bootstrap LLM adapter system on application startup."""

import yaml
from pathlib import Path
from metaqore.llm.client.factory import LLMClientFactory
from metaqore.llm.client.interface import LLMProvider
from metaqore.llm.client.adapters.mock_adapter import MockAdapter


def bootstrap_llm_system(config_path: str = "./config/llm_providers.yaml") -> None:
    """
    Initialize the entire LLM adapter ecosystem.
    Call this during MetaQore application startup.
    """
    # Register all available adapters
    LLMClientFactory.register_adapter(LLMProvider.MOCK, MockAdapter)

    # Try to register optional adapters (will be added later)
    try:
        from metaqore.llm.client.adapters.llama_cpp_adapter import LlamaCppAdapter

        LLMClientFactory.register_adapter(LLMProvider.LLAMA_CPP, LlamaCppAdapter)
    except ImportError:
        pass

    try:
        from metaqore.llm.client.adapters.vllm_adapter import VLLMAdapter

        LLMClientFactory.register_adapter(LLMProvider.VLLM, VLLMAdapter)
    except ImportError:
        pass

    try:
        from metaqore.llm.client.adapters.openai_adapter import OpenAIAdapter

        LLMClientFactory.register_adapter(LLMProvider.OPENAI, OpenAIAdapter)
    except ImportError:
        pass

    try:
        from metaqore.llm.client.adapters.anthropic_adapter import AnthropicAdapter

        LLMClientFactory.register_adapter(LLMProvider.ANTHROPIC, AnthropicAdapter)
    except ImportError:
        pass

    try:
        from metaqore.llm.client.adapters.azure_openai_adapter import AzureOpenAIAdapter

        LLMClientFactory.register_adapter(LLMProvider.AZURE_OPENAI, AzureOpenAIAdapter)
    except ImportError:
        pass

    # Load configuration if it exists
    config_file = Path(config_path)
    if config_file.exists():
        with config_file.open("r") as f:
            config = yaml.safe_load(f) or {}

        # Initialize enabled providers
        provider_configs = config.get("llm_providers", {})

        for provider_name, provider_config in provider_configs.items():
            if provider_name == "routing_policy":
                continue

            if provider_config.get("enabled", False):
                try:
                    provider = LLMProvider(provider_name)
                    client = LLMClientFactory.get_client(provider)
                    client.initialize(provider_config)
                except Exception as e:
                    # Log warning but don't fail startup
                    print(f"Warning: Failed to initialize {provider_name}: {e}")
    else:
        # Initialize mock adapter by default
        mock_client = LLMClientFactory.get_client(LLMProvider.MOCK)
        mock_client.initialize({})
