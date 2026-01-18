"""Simplified configuration for BERTA Meta-Orchestrator."""

from pydantic_settings import BaseSettings


class MetaQoreConfig(BaseSettings):
    """Simple configuration for BERTA Meta-Orchestrator."""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8001
    debug: bool = False

    # BERTA settings
    max_agents: int = 20
    hidden_dim: int = 256
    max_tasks: int = 100
    confidence_threshold: float = 0.8
    pruning_threshold: float = 0.1

    class Config:
        env_prefix = "METAQORE_"
