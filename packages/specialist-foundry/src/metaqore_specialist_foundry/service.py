"""Specialist Foundry service implementation."""

import logging
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from metaqore_governance_core.event_bus import event_bus, Event, EventTypes
from metaqore_governance_core.hmcp_registry import SkillRegistry
from metaqore_governance_core.hmcp_policy import HierarchicalChainingPolicy
from metaqore_governance_core.hmcp_validation_gate import ValidationGateRunner
from metaqore_governance_core.config_loader import load_hmcp_config
from metaqore_governance_core.training import MOPDTrainingLoop, TrainingOutcome
from metaqore_governance_core.models import SpecialistModel, SpecialistLifecycle

logger = logging.getLogger(__name__)


class SpecialistProposal(BaseModel):
    """Proposal for a new specialist."""
    name: str
    description: str
    capabilities: List[str]
    training_data_requirements: Dict[str, Any]
    performance_targets: Dict[str, Any]


class TrainingRequest(BaseModel):
    """Request to train a specialist."""
    specialist_id: str
    training_config: Dict[str, Any]


class SpecialistFoundryService:
    """Specialist Foundry service for MetaQore."""

    def __init__(self):
        self.app = FastAPI(title="MetaQore Specialist Foundry")
        self.skill_registry = SkillRegistry.from_policy()
        self.chaining_policy = HierarchicalChainingPolicy.from_policy()
        
        # Load validation gate config from HMCP policy
        hmcp_config = load_hmcp_config()
        validation_gate_config = hmcp_config.get("validation_gate", {})
        self.validation_gate = ValidationGateRunner(validation_gate_config)

        self._setup_routes()

    def _setup_routes(self):
        """Set up FastAPI routes."""

        @self.app.post("/api/v1/specialists/propose")
        async def propose_specialist(proposal: SpecialistProposal):
            """Propose a new specialist for creation."""
            try:
                # Validate proposal against HMCP policies
                # TODO: Implement proper validation
                validation_result = {"approved": True, "reason": ""}

                if not validation_result["approved"]:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Proposal rejected: {validation_result['reason']}"
                    )

                # Register proposal
                specialist_id = await self._register_proposal(proposal.dict())

                # Publish event
                event = Event(
                    event_type=EventTypes.SPECIALIST_TRAINING_STARTED,
                    source="specialist-foundry",
                    data={"specialist_id": specialist_id, "proposal": proposal.dict()}
                )
                await event_bus.publish(event)

                return {"specialist_id": specialist_id, "status": "proposed"}

            except Exception as e:
                logger.error(f"Failed to propose specialist: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/v1/specialists/{specialist_id}/train")
        async def train_specialist(specialist_id: str, request: TrainingRequest):
            """Initiate training for a specialist."""
            try:
                # Validate training request
                if request.specialist_id != specialist_id:
                    raise HTTPException(status_code=400, detail="Specialist ID mismatch")

                # Check if specialist exists and is in valid state
                specialist = await self._get_specialist(specialist_id)
                if not specialist:
                    raise HTTPException(status_code=404, detail="Specialist not found")

                # Start training process
                training_job_id = await self._start_training(specialist, request.training_config)

                # Publish event
                event = Event(
                    event_type=EventTypes.SPECIALIST_TRAINING_STARTED,
                    source="specialist-foundry",
                    data={
                        "specialist_id": specialist_id,
                        "training_job_id": training_job_id,
                        "config": request.training_config
                    }
                )
                await event_bus.publish(event)

                return {"training_job_id": training_job_id, "status": "training_started"}

            except Exception as e:
                logger.error(f"Failed to start training: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/v1/specialists/{specialist_id}")
        async def get_specialist(specialist_id: str):
            """Get specialist details."""
            try:
                specialist = await self._get_specialist(specialist_id)
                if not specialist:
                    raise HTTPException(status_code=404, detail="Specialist not found")

                return specialist

            except Exception as e:
                logger.error(f"Failed to get specialist: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/v1/specialists")
        async def list_specialists():
            """List all specialists."""
            try:
                specialists = await self._list_specialists()
                return {"specialists": specialists}

            except Exception as e:
                logger.error(f"Failed to list specialists: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        async def deploy_specialist(specialist_id: str):
            """Deploy a trained specialist."""
            try:
                # Check if specialist exists and is trained
                specialist = await self._get_specialist(specialist_id)
                if not specialist:
                    raise HTTPException(status_code=404, detail="Specialist not found")

                # TODO: Implement deployment logic
                deployment_id = await self._deploy_specialist(specialist)

                # Publish event
                event = Event(
                    event_type=EventTypes.SPECIALIST_DEPLOYED,
                    source="specialist-foundry",
                    data={"specialist_id": specialist_id, "deployment_id": deployment_id}
                )
                await event_bus.publish(event)

                return {"deployment_id": deployment_id, "status": "deployed"}

            except Exception as e:
                logger.error(f"Failed to deploy specialist: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    async def _register_proposal(self, proposal: Dict[str, Any]) -> str:
        """Register a specialist proposal."""
        # TODO: Implement actual registration logic
        import uuid
        specialist_id = str(uuid.uuid4())
        
        # Store proposal data (TODO: implement actual persistence)
        self._specialists = getattr(self, '_specialists', {})
        self._specialists[specialist_id] = {
            "id": specialist_id,
            "name": proposal.get("name", f"specialist_{specialist_id}"),
            "description": proposal.get("description", ""),
            "skill_id": "clean_akkadian_dates",  # Default skill from HMCP config
            "status": "proposed",
            "capabilities": proposal.get("capabilities", []),
            "training_data_requirements": proposal.get("training_data_requirements", {}),
            "performance_targets": proposal.get("performance_targets", {}),
        }
        
        return specialist_id

    async def _start_training(self, specialist: Dict[str, Any], training_config: Dict[str, Any]) -> str:
        """Start the MOPD training process for a specialist."""
        import uuid
        
        specialist_id = specialist["id"]
        training_job_id = f"train_{uuid.uuid4().hex[:12]}"
        
        try:
            # Load HMCP config for training parameters
            hmcp_config = load_hmcp_config()
            specialist_creation_config = hmcp_config.get("specialist_creation_engine", {})
            
            # Create SpecialistModel from specialist data
            skill_id = specialist.get("skill_id", "unknown_skill")
            skill_def = self.skill_registry._skills.get(skill_id)
            
            if not skill_def:
                raise ValueError(f"Unknown skill: {skill_id}")
            
            # Determine level from chaining policy (simplified - use level 1 for now)
            level = self.chaining_policy.iter_levels().__next__()  # Get first level
            
            specialist_model = SpecialistModel(
                id=specialist_id,
                project_id="default_project",  # TODO: Get from context
                data={
                    "name": specialist.get("name", f"specialist_{specialist_id}"),
                    "description": specialist.get("description", ""),
                    "capabilities": specialist.get("capabilities", []),
                },
                created_by="specialist-foundry",
                skill_id=skill_id,
                parent_agent=skill_def.parent_agent,
                teachers=list(skill_def.allowed_teachers),
                level_key=level.key,
                level_type=level.level_type,
                parameter_count=None,  # Will be set after training
                confidence=0.0,
                metadata={
                    "requested_size_mb": skill_def.max_specialist_size_mb,
                    "training_config": training_config,
                    "job_id": training_job_id,
                }
            )
            
            # Advance to training state
            specialist_model.advance_state(SpecialistLifecycle.MOPD_TRAINING)
            
            # Create and run training loop
            training_loop = MOPDTrainingLoop(specialist_creation_config)
            outcome = training_loop.run_training(specialist_model)
            
            # Update specialist with training results
            specialist_model.confidence = outcome.metrics.get("functional_accuracy", 0.0)
            specialist_model.parameter_count = int(outcome.metrics.get("efficiency_penalty", 0) * 1000000)  # Rough estimate
            specialist_model.metadata.update({
                "training_outcome": outcome.to_dict(),
                "epochs_completed": outcome.epochs,
            })
            
            if outcome.success:
                # Advance to validation state
                specialist_model.advance_state(SpecialistLifecycle.VALIDATION_GATING)
                
                # Run validation gate
                validation_report = self.validation_gate.run(specialist_model, outcome)
                
                if validation_report.passed:
                    # Training and validation successful
                    specialist_model.advance_state(SpecialistLifecycle.ACTIVE)
                    logger.info(f"Specialist {specialist_id} training completed successfully")
                else:
                    # Validation failed
                    specialist_model.advance_state(SpecialistLifecycle.BLOCKED)
                    logger.warning(f"Specialist {specialist_id} failed validation: {validation_report.stages}")
            else:
                # Training failed
                specialist_model.advance_state(SpecialistLifecycle.BLOCKED)
                logger.error(f"Specialist {specialist_id} training failed")
            
            # Store updated specialist (TODO: implement actual persistence)
            specialist_data = specialist_model.model_dump()
            specialist_data["lifecycle_state"] = specialist_model.lifecycle_state.value
            await self._update_specialist(specialist_id, specialist_data)
            
            # Publish training completion event
            event = Event(
                event_type=EventTypes.SPECIALIST_TRAINING_COMPLETED,
                source="specialist-foundry",
                data={
                    "specialist_id": specialist_id,
                    "training_job_id": training_job_id,
                    "success": outcome.success,
                    "validation_passed": getattr(validation_report, 'passed', False) if 'validation_report' in locals() else False,
                    "final_state": specialist_model.lifecycle_state.value,
                    "metrics": outcome.metrics,
                }
            )
            await event_bus.publish(event)
            
            return training_job_id
            
        except Exception as e:
            logger.error(f"Training failed for specialist {specialist_id}: {e}")
            # Publish failure event
            event = Event(
                event_type=EventTypes.LLM_REQUEST_FAILED,
                source="specialist-foundry",
                data={
                    "specialist_id": specialist_id,
                    "training_job_id": training_job_id,
                    "error": str(e),
                    "context": "training_pipeline",
                }
            )
            await event_bus.publish(event)
            raise

    async def _update_specialist(self, specialist_id: str, specialist_data: Dict[str, Any]) -> None:
        """Update specialist data."""
        # TODO: Implement actual persistence logic
        self._specialists = getattr(self, '_specialists', {})
        self._specialists[specialist_id] = specialist_data

    async def _get_specialist(self, specialist_id: str) -> Optional[Dict[str, Any]]:
        """Get specialist by ID."""
        # TODO: Implement actual retrieval logic
        specialists = getattr(self, '_specialists', {})
        return specialists.get(specialist_id)

    async def _list_specialists(self) -> List[Dict[str, Any]]:
        """List all specialists."""
        # TODO: Implement actual listing logic
        specialists = getattr(self, '_specialists', {})
        return list(specialists.values())

    async def _deploy_specialist(self, specialist: Dict[str, Any]) -> str:
        """Deploy a specialist to the AI Gateway routing system."""
        import uuid
        
        specialist_id = specialist["id"]
        deployment_id = f"deploy_{uuid.uuid4().hex[:12]}"
        
        try:
            # Verify specialist is in ACTIVE state
            lifecycle_state = specialist.get("lifecycle_state")
            if lifecycle_state != "SpecialistLifecycle.ACTIVE":
                raise ValueError(f"Specialist {specialist_id} is not in ACTIVE state (current: {lifecycle_state})")
            
            # Get specialist model data for deployment
            specialist_model_data = specialist.get("specialist_model", {})
            skill_id = specialist_model_data.get("skill_id", specialist.get("skill_id"))
            confidence = specialist_model_data.get("confidence", 0.0)
            
            # Prepare deployment configuration
            deployment_config = {
                "deployment_id": deployment_id,
                "specialist_id": specialist_id,
                "skill_id": skill_id,
                "confidence_threshold": confidence,
                "routing_rules": {
                    "skill_match": skill_id,
                    "confidence_min": max(0.7, confidence - 0.1),  # Allow some tolerance
                    "fallback_enabled": True,
                },
                "model_metadata": {
                    "parameter_count": specialist_model_data.get("parameter_count"),
                    "training_epochs": specialist_model_data.get("metadata", {}).get("training_outcome", {}).get("epochs", 0),
                    "validation_passed": specialist_model_data.get("metadata", {}).get("training_outcome", {}).get("validation_passed", False),
                },
                "deployment_timestamp": "2026-03-08T00:00:00Z",  # TODO: Use actual timestamp
            }
            
            # TODO: Register with AI Gateway service via event bus or direct API call
            # For now, we'll simulate the registration
            
            # Update specialist state to DEPLOYED
            specialist["lifecycle_state"] = "SpecialistLifecycle.DEPLOYED"
            specialist["deployment_id"] = deployment_id
            specialist["deployment_config"] = deployment_config
            
            # Store updated specialist
            await self._update_specialist(specialist_id, specialist)
            
            # Publish deployment event
            event = Event(
                event_type=EventTypes.SPECIALIST_DEPLOYED,
                source="specialist-foundry",
                data={
                    "specialist_id": specialist_id,
                    "deployment_id": deployment_id,
                    "skill_id": skill_id,
                    "confidence": confidence,
                    "routing_config": deployment_config["routing_rules"],
                }
            )
            await event_bus.publish(event)
            
            logger.info(f"Specialist {specialist_id} deployed successfully with deployment ID {deployment_id}")
            return deployment_id
            
        except Exception as e:
            logger.error(f"Deployment failed for specialist {specialist_id}: {e}")
            # Publish failure event
            event = Event(
                event_type=EventTypes.LLM_REQUEST_FAILED,
                source="specialist-foundry",
                data={
                    "specialist_id": specialist_id,
                    "deployment_id": deployment_id,
                    "error": str(e),
                    "context": "specialist_deployment",
                }
            )
            await event_bus.publish(event)
            raise


# Global service instance
service = SpecialistFoundryService()

# Export the FastAPI app for running the service
app = service.app