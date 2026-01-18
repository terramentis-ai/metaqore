"""
Simplified MetaQore - Orchestration Engine for TerraQore Studio

MetaQore provides lightweight orchestration and coordination services for TerraQore Studio agents.
It handles agent conflict resolution, execution coordination, and basic governance.

Now enhanced with BERTA (Bidirectional Encoder-Represented Task-centric Agents) orchestration.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# Import BERTA orchestrator
from .berta_orchestrator import get_berta_orchestrator, BERTAOrchestrator

logger = logging.getLogger(__name__)


class AgentConflict:
    """Represents a conflict between agents."""

    def __init__(self, agent_name: str, conflict_type: str, description: str):
        self.agent_name = agent_name
        self.conflict_type = conflict_type
        self.description = description
        self.timestamp = datetime.now()


class OrchestratorResult:
    """Result of orchestration operation."""

    def __init__(self, success: bool, data: Any = None, conflicts: Optional[List[AgentConflict]] = None):
        self.success = success
        self.data = data or {}
        self.conflicts = conflicts or []


class MetaQoreOrchestrator:
    """
    Lightweight orchestrator for TerraQore Studio agents.

    Now powered by BERTA (Bidirectional Encoder-Represented Task-centric Agents) for advanced orchestration.

    Provides:
    - Agent execution coordination with bidirectional encoding
    - Vectorized agent capabilities and semantic matching
    - Masked task completion for parallel inference
    - Speculative execution for efficiency
    - Dynamic pruning of inactive components
    - Zero-handoff state management
    """

    def __init__(self):
        # Use BERTA orchestrator as the core engine
        self.berta = get_berta_orchestrator()

        # Legacy compatibility attributes
        self.active_agents = set()
        self.conflict_history = []

        logger.info("MetaQore orchestrator initialized with BERTA engine")

    def register_agent(self, agent_name: str, capabilities: List[str] = None) -> bool:
        """Register an agent with BERTA orchestration."""
        # Use BERTA orchestrator for registration
        capabilities = capabilities or ["general"]  # Default capability
        success = self.berta.register_agent(agent_name, capabilities)

        if success:
            # Maintain legacy compatibility
            self.active_agents.add(agent_name)

        return success

    def unregister_agent(self, agent_name: str) -> bool:
        """Unregister an agent from BERTA orchestration."""
        # BERTA orchestrator doesn't have unregister, so just update legacy tracking
        if agent_name in self.active_agents:
            self.active_agents.remove(agent_name)
            logger.info(f"Unregistered agent: {agent_name}")
            return True
        return False

    def check_conflicts(self, agent_name: str, operation: str) -> List[AgentConflict]:
        """Check for potential conflicts using BERTA context."""
        conflicts = []

        # BERTA handles conflicts through bidirectional encoding
        # Check for basic conflicts as fallback
        if operation == "code_generation" and len(self.active_agents) > 1:
            # Check if multiple agents are trying the same operation simultaneously
            conflicts.append(AgentConflict(
                agent_name=agent_name,
                conflict_type="concurrent_operation",
                description=f"Multiple agents active, BERTA will coordinate {operation}"
            ))

        return conflicts

    def orchestrate_execution(self, agent_name: str, operation: str, **kwargs) -> OrchestratorResult:
        """Orchestrate agent execution using BERTA bidirectional encoding."""
        try:
            # Use BERTA orchestrator for advanced coordination
            berta_result = self.berta.orchestrate_execution(agent_name, operation, **kwargs)

            # Convert BERTA result to legacy OrchestratorResult format
            result_data = {
                "berta_orchestrated": True,
                "task_id": berta_result.get("task_id"),
                "result": berta_result.get("result"),
                "masked_completions": berta_result.get("masked_completions", []),
                "speculative_executions": berta_result.get("speculative_executions", []),
                "pruned_agents": berta_result.get("pruned_agents", []),
                "pruned_tasks": berta_result.get("pruned_tasks", []),
                "global_context_encoded": berta_result.get("global_context_encoded", False),
                "agent": agent_name,
                "operation": operation,
                "timestamp": datetime.now().isoformat(),
                "coordinated": True
            }

            # Add operation-specific coordination
            if operation == "code_generation":
                result_data["priority"] = "high"
            elif operation == "security_scan":
                result_data["priority"] = "critical"

            logger.info(f"BERTA orchestrated execution: {agent_name} -> {operation}")

            return OrchestratorResult(
                success=berta_result.get("success", True),
                data=result_data
            )

        except Exception as e:
            logger.error(f"BERTA orchestration failed: {e}")
            # Fallback to basic conflict checking
            conflicts = self.check_conflicts(agent_name, operation)

            if conflicts:
                return OrchestratorResult(
                    success=False,
                    conflicts=conflicts
                )

            # Basic result for fallback
            result_data = {
                "agent": agent_name,
                "operation": operation,
                "timestamp": datetime.now().isoformat(),
                "coordinated": False,
                "error": str(e)
            }

            return OrchestratorResult(
                success=False,
                data=result_data
            )

    def get_active_agents(self) -> List[str]:
        """Get list of currently active agents from BERTA orchestrator."""
        return self.berta.get_active_agents()

    def get_conflict_history(self) -> List[AgentConflict]:
        """Get history of conflicts."""
        return self.conflict_history.copy()


# Global orchestrator instance
_orchestrator = None

def get_orchestrator() -> MetaQoreOrchestrator:
    """Get the global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MetaQoreOrchestrator()
    return _orchestrator
