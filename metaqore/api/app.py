"""
BERTA Meta-Orchestrator API - Advanced Orchestration Services for TerraQore Studio

Now powered by BERTA (Bidirectional Encoder-Represented Task-centric Agents) for advanced orchestration.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from metaqore.core.orchestrator import get_orchestrator, OrchestratorResult, AgentConflict

# Import BERTA components lazily to avoid startup issues
# from metaqore.core.berta_orchestrator import get_berta_orchestrator

app = FastAPI(title="BERTA Meta-Orchestrator", version="2.0.0-berta")


class OrchestrationRequest(BaseModel):
    agent_name: str
    operation: str
    parameters: Dict[str, Any] = {}
    capabilities: List[str] = []


class OrchestrationResponse(BaseModel):
    success: bool
    data: Dict[str, Any] = {}
    conflicts: List[Dict[str, Any]] = []


class AgentRegistration(BaseModel):
    agent_name: str
    capabilities: List[str] = []


class TaskStatusResponse(BaseModel):
    task_id: Optional[str]
    agent_name: Optional[str]
    operation: Optional[str]
    status: Optional[str]
    priority: Optional[float]
    dependencies: List[str]
    dependents: List[str]
    confidence_score: Optional[float]
    created_at: Optional[str]


class BERTAContextResponse(BaseModel):
    total_agents: int
    total_tasks: int
    active_tasks: int
    completed_tasks: int
    masked_tasks: int
    speculative_tasks: int
    global_state_encoded: bool
    bidirectional_context_active: bool


@app.post("/orchestrate", response_model=OrchestrationResponse)
async def orchestrate_execution(request: OrchestrationRequest):
    """Orchestrate agent execution with BERTA bidirectional encoding."""
    orchestrator = get_orchestrator()

    # Register agent if not already registered
    if request.capabilities:
        orchestrator.register_agent(request.agent_name, request.capabilities)

    result = orchestrator.orchestrate_execution(
        agent_name=request.agent_name, operation=request.operation, **request.parameters
    )

    return OrchestrationResponse(
        success=result.success,
        data=result.data,
        conflicts=[
            {
                "agent_name": c.agent_name,
                "conflict_type": c.conflict_type,
                "description": c.description,
                "timestamp": c.timestamp.isoformat(),
            }
            for c in result.conflicts
        ],
    )


@app.post("/agents/register")
async def register_agent(request: AgentRegistration):
    """Register an agent with BERTA capabilities."""
    orchestrator = get_orchestrator()

    success = orchestrator.register_agent(request.agent_name, request.capabilities)
    if not success:
        raise HTTPException(status_code=409, detail="Agent already registered")

    return {"message": f"Agent {request.agent_name} registered with BERTA orchestrator"}


@app.delete("/agents/{agent_name}")
async def unregister_agent(agent_name: str):
    """Unregister an agent from BERTA orchestrator."""
    orchestrator = get_orchestrator()

    success = orchestrator.unregister_agent(agent_name)
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found")

    return {"message": f"Agent {agent_name} unregistered from BERTA orchestrator"}


@app.get("/agents", response_model=List[str])
async def get_active_agents():
    """Get list of active agents in BERTA orchestrator."""
    orchestrator = get_orchestrator()
    return orchestrator.get_active_agents()


@app.get("/berta/context", response_model=BERTAContextResponse)
async def get_berta_context():
    """Get BERTA global orchestration context summary."""
    from metaqore.core.berta_orchestrator import get_berta_orchestrator

    berta = get_berta_orchestrator()
    context = berta.get_global_context_summary()

    return BERTAContextResponse(**context)


@app.get("/berta/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get status of a specific BERTA task."""
    from metaqore.core.berta_orchestrator import get_berta_orchestrator

    berta = get_berta_orchestrator()
    status = berta.get_task_status(task_id)

    if status is None:
        raise HTTPException(status_code=404, detail="Task not found")

    return TaskStatusResponse(**status)


@app.get("/berta/tasks", response_model=List[TaskStatusResponse])
async def get_all_tasks():
    """Get status of all BERTA tasks."""
    from metaqore.core.berta_orchestrator import get_berta_orchestrator

    berta = get_berta_orchestrator()
    tasks = []

    for task_id in berta.tasks.keys():
        status = berta.get_task_status(task_id)
        if status:
            tasks.append(TaskStatusResponse(**status))

    return tasks


@app.get("/conflicts")
async def get_conflict_history():
    """Get conflict history from BERTA orchestrator."""
    orchestrator = get_orchestrator()
    conflicts = orchestrator.get_conflict_history()

    return [
        {
            "agent_name": c.agent_name,
            "conflict_type": c.conflict_type,
            "description": c.description,
            "timestamp": c.timestamp.isoformat(),
        }
        for c in conflicts
    ]


@app.get("/health")
async def health_check():
    """BERTA orchestrator health check."""
    from metaqore.core.berta_orchestrator import get_berta_orchestrator

    berta = get_berta_orchestrator()
    context = berta.get_global_context_summary()

    return {
        "status": "healthy",
        "service": "BERTA Meta-Orchestrator",
        "berta_active": True,
        "global_state_encoded": context["global_state_encoded"],
        "bidirectional_context_active": context["bidirectional_context_active"],
        "total_agents": context["total_agents"],
        "total_tasks": context["total_tasks"],
    }
