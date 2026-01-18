"""Simplified configuration for MetaQore orchestrator."""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class MetaQoreConfig(BaseSettings):
    """Simple configuration for MetaQore orchestrator."""

    host: str = "0.0.0.0"
    port: int = 8001
    debug: bool = False
    max_agents: int = 20

    class Config:
        env_prefix = "METAQORE_"
        default=None,
        description="Secret key for JWT token verification. Enables JWT auth when set.",
    )
    jwt_algorithm: str = Field(
        default="HS256",
        description="Algorithm for JWT signing/verification.",
    )
    enable_rate_limit: bool = Field(default=True, description="Enable rate limiting.")
    rate_limit_per_minute: int = Field(
        default=120, ge=1, description="Requests allowed per minute per client"
    )
    rate_limit_burst: int = Field(
        default=240, ge=1, description="Burst ceiling within the same window"
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0", description="Redis URL for distributed rate limiting."
    )
    privileged_token: Optional[str] = Field(
        default=None,
        description="Optional shared secret to mark privileged MetaQore clients.",
    )

    @field_validator("max_parallel_branches")
    @classmethod
    def _validate_parallel_branches(cls, value: int, info: ValidationInfo) -> int:  # noqa: D401
        governance_mode = (info.data or {}).get("governance_mode")
        if governance_mode == GovernanceMode.STRICT and value > 1:
            raise ValueError("STRICT mode allows max_parallel_branches <= 1")
        return value

    @field_validator("secure_gateway_policy")
    @classmethod
    def _validate_secure_gateway_policy(cls, value: str) -> str:
        from metaqore.core.security import resolve_routing_policy

        policy = resolve_routing_policy(value)
        return policy.name

    @classmethod
    def from_yaml(
        cls, path: str | Path, *, overrides: Optional[Dict[str, Any]] = None
    ) -> "MetaQoreConfig":
        """Load configuration from a YAML file with optional overrides."""

        with Path(path).expanduser().open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if overrides:
            data.update(overrides)
        return cls(**data)

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation of the config."""

        return self.model_dump()
