import pytest
from metaqore.llm.bootstrap import bootstrap_llm_system


@pytest.fixture(autouse=True)
def bootstrap_llm_adapters():
    """Bootstrap LLM adapters for all tests."""
    bootstrap_llm_system()


@pytest.fixture
def sample_fixture():
    return "Hello, World!"
