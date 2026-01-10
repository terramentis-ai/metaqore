"""Tests for LLM adapters."""

import pytest
from unittest.mock import Mock, patch
from metaqore.llm.client.interface import LLMProvider, LLMResponse
from metaqore.llm.client.adapters.mock_adapter import MockAdapter
from metaqore.llm.client.adapters.llama_cpp_adapter import LlamaCppAdapter
from metaqore.llm.client.factory import LLMClientFactory


class TestMockAdapter:
    """Test the MockAdapter."""

    def test_provider_property(self):
        adapter = MockAdapter()
        assert adapter.provider == LLMProvider.MOCK

    def test_initialize(self):
        adapter = MockAdapter()
        config = {}
        adapter.initialize(config)
        assert adapter._initialized is True

    def test_generate_success(self):
        adapter = MockAdapter()
        adapter.initialize({})

        response = adapter.generate(prompt="Test prompt", agent_name="test_agent", metadata={})

        assert isinstance(response, LLMResponse)
        assert response.success is True
        assert response.provider == LLMProvider.MOCK
        assert "test_agent" in response.content

    def test_generate_error(self):
        adapter = MockAdapter()
        adapter.initialize({})

        response = adapter.generate(prompt="Error prompt", agent_name="test_agent", metadata={})

        assert isinstance(response, LLMResponse)
        assert response.success is False
        assert response.error == "Simulated error for testing"

    def test_validate_config(self):
        adapter = MockAdapter()
        assert adapter.validate_config({}) is True


class TestLlamaCppAdapter:
    """Test the LlamaCppAdapter."""

    def test_provider_property(self):
        adapter = LlamaCppAdapter()
        assert adapter.provider == LLMProvider.LLAMA_CPP

    def test_initialize_missing_model_path(self):
        adapter = LlamaCppAdapter()
        with pytest.raises(ValueError, match="model_path is required"):
            adapter.initialize({})

    @patch("pathlib.Path.exists", return_value=False)
    def test_initialize_model_not_found(self, mock_exists):
        adapter = LlamaCppAdapter()
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            adapter.initialize({"model_path": "/nonexistent/model.gguf"})

    @patch("llama_cpp.Llama")
    @patch("pathlib.Path.exists", return_value=True)
    def test_initialize_success(self, mock_exists, mock_llama_class):
        mock_model = Mock()
        mock_llama_class.return_value = mock_model

        adapter = LlamaCppAdapter()
        config = {"model_path": "/path/to/model.gguf"}
        adapter.initialize(config)

        assert adapter._initialized is True
        assert adapter._model_path == "/path/to/model.gguf"
        mock_llama_class.assert_called_once()

    @patch("llama_cpp.Llama")
    @patch("pathlib.Path.exists", return_value=True)
    def test_generate_success(self, mock_exists, mock_llama_class):
        # Mock the llama model to return a subscriptable dict when called

        def fake_llama_call(*args, **kwargs):
            return {
                "choices": [{"text": "Generated response"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

        mock_model = Mock(side_effect=fake_llama_call)
        # Set model_path attribute to mimic real Llama object
        mock_model.model_path = "/path/to/model.gguf"
        mock_llama_class.return_value = mock_model

        adapter = LlamaCppAdapter()
        adapter.initialize({"model_path": "/path/to/model.gguf"})

        response = adapter.generate(prompt="Test prompt", agent_name="test_agent", metadata={})

        assert isinstance(response, LLMResponse)
        assert response.success is True
        assert response.content == "Generated response"
        assert response.provider == LLMProvider.LLAMA_CPP
        assert response.usage == {"prompt_tokens": 10, "completion_tokens": 5}

    @patch("llama_cpp.Llama")
    @patch("pathlib.Path.exists", return_value=True)
    def test_generate_with_exception(self, mock_exists, mock_llama_class):
        # Mock the llama model to raise an exception
        mock_model = Mock()
        mock_model.side_effect = Exception("Model error")
        mock_llama_class.return_value = mock_model

        adapter = LlamaCppAdapter()
        adapter.initialize({"model_path": "/path/to/model.gguf"})

        response = adapter.generate(prompt="Test prompt", agent_name="test_agent", metadata={})

        assert isinstance(response, LLMResponse)
        assert response.success is False
        assert "Model error" in response.error

    def test_validate_config_valid(self):
        adapter = LlamaCppAdapter()

        # Test with valid GGUF file
        with patch("pathlib.Path.exists", return_value=True):
            assert adapter.validate_config({"model_path": "/path/to/model.gguf"}) is True

        # Test with valid BIN file
        with patch("pathlib.Path.exists", return_value=True):
            assert adapter.validate_config({"model_path": "/path/to/model.bin"}) is True

    def test_validate_config_invalid(self):
        adapter = LlamaCppAdapter()

        # Test missing model_path
        assert adapter.validate_config({}) is False

        # Test nonexistent file
        assert adapter.validate_config({"model_path": "/nonexistent/model.gguf"}) is False

        # Test invalid extension
        with patch("pathlib.Path.exists", return_value=True):
            assert adapter.validate_config({"model_path": "/path/to/model.txt"}) is False


class TestLLMClientFactory:
    """Test the LLMClientFactory."""

    def test_register_adapter(self):
        factory = LLMClientFactory()
        factory.register_adapter(LLMProvider.MOCK, MockAdapter)

        assert LLMProvider.MOCK in factory._adapters
        assert factory._adapters[LLMProvider.MOCK] == MockAdapter

    def test_get_client(self):
        factory = LLMClientFactory()
        factory.register_adapter(LLMProvider.MOCK, MockAdapter)

        client = factory.get_client(LLMProvider.MOCK)
        assert isinstance(client, MockAdapter)
        assert client.provider == LLMProvider.MOCK

    def test_get_client_not_registered(self):
        factory = LLMClientFactory()

        with pytest.raises(ValueError, match="model_path is required for LlamaCppAdapter"):
            factory.get_client(LLMProvider.LLAMA_CPP)


class TestVLLMAdapter:
    """Test the VLLMAdapter."""

    @patch("metaqore.llm.client.adapters.vllm_adapter.requests")
    def test_generate_success(self, mock_requests):
        from metaqore.llm.client.adapters.vllm_adapter import VLLMAdapter

        # Mock response from vLLM server
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"text": "vLLM generated response"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7},
        }
        mock_requests.post.return_value = mock_response

        adapter = VLLMAdapter()
        adapter.initialize({"endpoint": "http://localhost:8000", "model": "test-vllm-model"})
        response = adapter.generate(prompt="Test prompt", agent_name="test_agent", metadata={})
        assert isinstance(response, LLMResponse)
        assert response.success is True
        assert response.content == "vLLM generated response"
        assert response.provider == LLMProvider.VLLM
        assert response.usage == {"prompt_tokens": 12, "completion_tokens": 7}

    @patch("metaqore.llm.client.adapters.vllm_adapter.requests")
    def test_generate_error(self, mock_requests):
        from metaqore.llm.client.adapters.vllm_adapter import VLLMAdapter

        mock_requests.post.side_effect = Exception("Connection error")
        adapter = VLLMAdapter()
        adapter.initialize({"endpoint": "http://localhost:8000", "model": "test-vllm-model"})
        response = adapter.generate(prompt="Test prompt", agent_name="test_agent", metadata={})
        assert isinstance(response, LLMResponse)
        assert response.success is False
        assert "Connection error" in response.error
        assert response.provider == LLMProvider.VLLM

    def test_validate_config(self):
        from metaqore.llm.client.adapters.vllm_adapter import VLLMAdapter

        adapter = VLLMAdapter()
        valid = {"endpoint": "http://localhost:8000", "model": "test-vllm-model"}
        invalid = {"endpoint": "http://localhost:8000"}
        assert adapter.validate_config(valid) is True
        assert adapter.validate_config(invalid) is False


class TestOpenAIAdapter:
    """Test the OpenAIAdapter."""

    @patch("metaqore.llm.client.adapters.openai_adapter.openai")
    def test_generate_success(self, mock_openai):
        from metaqore.llm.client.adapters.openai_adapter import OpenAIAdapter

        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "OpenAI generated response"
        mock_response.usage = Mock()
        mock_response.usage.model_dump.return_value = {"prompt_tokens": 10, "completion_tokens": 5}
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        adapter = OpenAIAdapter()
        adapter.initialize({"api_key": "test-key", "model": "gpt-3.5-turbo"})

        response = adapter.generate(prompt="Test prompt", agent_name="test_agent", metadata={})

        assert isinstance(response, LLMResponse)
        assert response.success is True
        assert response.content == "OpenAI generated response"
        assert response.provider == LLMProvider.OPENAI
        assert response.usage == {"prompt_tokens": 10, "completion_tokens": 5}

    @patch("metaqore.llm.client.adapters.openai_adapter.openai")
    def test_generate_error(self, mock_openai):
        from metaqore.llm.client.adapters.openai_adapter import OpenAIAdapter

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        mock_openai.OpenAI.return_value = mock_client

        adapter = OpenAIAdapter()
        adapter.initialize({"api_key": "test-key"})

        response = adapter.generate(prompt="Test prompt", agent_name="test_agent", metadata={})

        assert isinstance(response, LLMResponse)
        assert response.success is False
        assert "API error" in response.error
        assert response.provider == LLMProvider.OPENAI

    def test_validate_config(self):
        from metaqore.llm.client.adapters.openai_adapter import OpenAIAdapter

        adapter = OpenAIAdapter()
        valid = {"api_key": "test-key"}
        invalid = {}
        assert adapter.validate_config(valid) is True
        assert adapter.validate_config(invalid) is False


class TestAnthropicAdapter:
    """Test the AnthropicAdapter."""

    @patch("metaqore.llm.client.adapters.anthropic_adapter.anthropic")
    def test_generate_success(self, mock_anthropic):
        from metaqore.llm.client.adapters.anthropic_adapter import AnthropicAdapter

        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Anthropic generated response"
        mock_response.usage = Mock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        adapter = AnthropicAdapter()
        adapter.initialize({"api_key": "test-key", "model": "claude-3-sonnet-20240229"})

        response = adapter.generate(prompt="Test prompt", agent_name="test_agent", metadata={})

        assert isinstance(response, LLMResponse)
        assert response.success is True
        assert response.content == "Anthropic generated response"
        assert response.provider == LLMProvider.ANTHROPIC
        assert response.usage == {"input_tokens": 10, "output_tokens": 5}

    @patch("metaqore.llm.client.adapters.anthropic_adapter.anthropic")
    def test_generate_error(self, mock_anthropic):
        from metaqore.llm.client.adapters.anthropic_adapter import AnthropicAdapter

        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API error")
        mock_anthropic.Anthropic.return_value = mock_client

        adapter = AnthropicAdapter()
        adapter.initialize({"api_key": "test-key"})

        response = adapter.generate(prompt="Test prompt", agent_name="test_agent", metadata={})

        assert isinstance(response, LLMResponse)
        assert response.success is False
        assert "API error" in response.error
        assert response.provider == LLMProvider.ANTHROPIC

    def test_validate_config(self):
        from metaqore.llm.client.adapters.anthropic_adapter import AnthropicAdapter

        adapter = AnthropicAdapter()
        valid = {"api_key": "test-key"}
        invalid = {}
        assert adapter.validate_config(valid) is True
        assert adapter.validate_config(invalid) is False


class TestAzureOpenAIAdapter:
    """Test the AzureOpenAIAdapter."""

    @patch("metaqore.llm.client.adapters.azure_openai_adapter.openai")
    def test_generate_success(self, mock_openai):
        from metaqore.llm.client.adapters.azure_openai_adapter import AzureOpenAIAdapter

        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Azure OpenAI generated response"
        mock_response.usage = Mock()
        mock_response.usage.model_dump.return_value = {"prompt_tokens": 10, "completion_tokens": 5}
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.AzureOpenAI.return_value = mock_client

        adapter = AzureOpenAIAdapter()
        adapter.initialize(
            {
                "api_key": "test-key",
                "endpoint": "https://test.openai.azure.com/",
                "deployment": "test-deployment",
            }
        )

        response = adapter.generate(prompt="Test prompt", agent_name="test_agent", metadata={})

        assert isinstance(response, LLMResponse)
        assert response.success is True
        assert response.content == "Azure OpenAI generated response"
        assert response.provider == LLMProvider.AZURE_OPENAI
        assert response.usage == {"prompt_tokens": 10, "completion_tokens": 5}

    @patch("metaqore.llm.client.adapters.azure_openai_adapter.openai")
    def test_generate_error(self, mock_openai):
        from metaqore.llm.client.adapters.azure_openai_adapter import AzureOpenAIAdapter

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        mock_openai.AzureOpenAI.return_value = mock_client

        adapter = AzureOpenAIAdapter()
        adapter.initialize(
            {
                "api_key": "test-key",
                "endpoint": "https://test.openai.azure.com/",
                "deployment": "test-deployment",
            }
        )

        response = adapter.generate(prompt="Test prompt", agent_name="test_agent", metadata={})

        assert isinstance(response, LLMResponse)
        assert response.success is False
        assert "API error" in response.error
        assert response.provider == LLMProvider.AZURE_OPENAI

    def test_validate_config(self):
        from metaqore.llm.client.adapters.azure_openai_adapter import AzureOpenAIAdapter

        adapter = AzureOpenAIAdapter()
        valid = {
            "api_key": "test-key",
            "endpoint": "https://test.openai.azure.com/",
            "deployment": "test-deployment",
        }
        invalid_missing_key = {
            "endpoint": "https://test.openai.azure.com/",
            "deployment": "test-deployment",
        }
        invalid_missing_endpoint = {"api_key": "test-key", "deployment": "test-deployment"}
        invalid_missing_deployment = {
            "api_key": "test-key",
            "endpoint": "https://test.openai.azure.com/",
        }
        assert adapter.validate_config(valid) is True
        assert adapter.validate_config(invalid_missing_key) is False
        assert adapter.validate_config(invalid_missing_endpoint) is False
        assert adapter.validate_config(invalid_missing_deployment) is False
