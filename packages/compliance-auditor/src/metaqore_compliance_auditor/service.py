"""Compliance Auditor service implementation."""

import logging
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from metaqore_governance_core.event_bus import event_bus, Event, EventTypes, EventHandler
from metaqore_governance_core.audit import ComplianceAuditor

logger = logging.getLogger(__name__)


class ComplianceCheck(BaseModel):
    """Compliance check request."""
    framework: str  # e.g., "SOC2", "GDPR", "HIPAA"
    scope: Dict[str, Any]
    evidence_types: List[str]


class ComplianceAuditorService(EventHandler):
    """Compliance Auditor service for MetaQore."""

    def __init__(self):
        self.app = FastAPI(title="MetaQore Compliance Auditor")
        self.auditor = ComplianceAuditor()

        # Register as event handler
        event_bus.register_handler(EventTypes.AUDIT_LOG_CREATED, self)
        event_bus.register_handler(EventTypes.COMPLIANCE_VIOLATION, self)

        self._setup_routes()

    def _setup_routes(self):
        """Set up FastAPI routes."""

        @self.app.post("/api/v1/compliance/check")
        async def run_compliance_check(check: ComplianceCheck):
            """Run a compliance check against specified framework."""
            try:
                results = await self.auditor.run_compliance_check(
                    framework=check.framework,
                    scope=check.scope,
                    evidence_types=check.evidence_types
                )

                return {
                    "framework": check.framework,
                    "results": results,
                    "timestamp": "2026-01-10T00:00:00Z"
                }

            except Exception as e:
                logger.error(f"Compliance check failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/v1/compliance/frameworks")
        async def list_frameworks():
            """List supported compliance frameworks."""
            return {
                "frameworks": ["SOC2", "GDPR", "HIPAA", "ISO27001"],
                "descriptions": {
                    "SOC2": "Service Organization Control 2",
                    "GDPR": "General Data Protection Regulation",
                    "HIPAA": "Health Insurance Portability and Accountability Act",
                    "ISO27001": "Information Security Management Systems"
                }
            }

        @self.app.get("/api/v1/compliance/evidence/{evidence_id}")
        async def get_evidence(evidence_id: str):
            """Retrieve specific evidence by ID."""
            try:
                evidence = await self.auditor.get_evidence(evidence_id)
                if not evidence:
                    raise HTTPException(status_code=404, detail="Evidence not found")

                return evidence

            except Exception as e:
                logger.error(f"Failed to retrieve evidence: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/v1/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "service": "compliance-auditor"}

    async def handle_event(self, event: Event) -> None:
        """Handle incoming events for compliance monitoring."""
        try:
            if event.event_type == EventTypes.AUDIT_LOG_CREATED:
                # Process audit log for compliance
                await self._process_audit_log(event.data)

            elif event.event_type == EventTypes.COMPLIANCE_VIOLATION:
                # Handle compliance violation
                await self._handle_violation(event.data)

        except Exception as e:
            logger.error(f"Failed to handle event {event.event_type}: {e}")

    async def _process_audit_log(self, audit_data: Dict[str, Any]) -> None:
        """Process audit log entry for compliance tracking."""
        # Store evidence and check against compliance frameworks
        evidence_id = await self.auditor.collect_evidence(
            evidence_type="audit_log",
            data=audit_data,
            source="event_bus"
        )

        logger.info(f"Collected audit evidence: {evidence_id}")

    async def _handle_violation(self, violation_data: Dict[str, Any]) -> None:
        """Handle compliance violation events."""
        # Escalate violation, create remediation tasks, etc.
        logger.warning(f"Compliance violation detected: {violation_data}")

        # Could publish remediation events, create tickets, etc.


# Global service instance
service = ComplianceAuditorService()

# Export the FastAPI app for running the service
app = service.app