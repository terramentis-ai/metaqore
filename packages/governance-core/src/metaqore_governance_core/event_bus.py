"""Event bus for inter-service communication in MetaQore."""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable, Awaitable
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Base event class for the MetaQore event bus."""

    event_id: str
    event_type: str
    timestamp: datetime
    source: str
    data: Dict[str, Any]
    correlation_id: Optional[str] = None

    def __init__(
        self,
        event_type: str,
        source: str,
        data: Dict[str, Any],
        correlation_id: Optional[str] = None,
    ):
        self.event_id = str(uuid4())
        self.event_type = event_type
        self.timestamp = datetime.now(timezone.utc)
        self.source = source
        self.data = data
        self.correlation_id = correlation_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "data": self.data,
            "correlation_id": self.correlation_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Event:
        """Create event from dictionary."""
        event = cls(
            event_type=data["event_type"],
            source=data["source"],
            data=data["data"],
            correlation_id=data.get("correlation_id"),
        )
        event.event_id = data["event_id"]
        event.timestamp = datetime.fromisoformat(data["timestamp"])
        return event


class EventHandler(ABC):
    """Abstract base class for event handlers."""

    @abstractmethod
    async def handle_event(self, event: Event) -> None:
        """Handle an incoming event."""
        pass


class EventBus:
    """Central event bus for MetaQore services."""

    def __init__(self):
        self._handlers: Dict[str, List[EventHandler]] = {}
        self._middleware: List[Callable[[Event], Awaitable[Event]]] = []

    def register_handler(self, event_type: str, handler: EventHandler) -> None:
        """Register an event handler for a specific event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.info(f"Registered handler for event type: {event_type}")

    def unregister_handler(self, event_type: str, handler: EventHandler) -> None:
        """Unregister an event handler."""
        if event_type in self._handlers:
            self._handlers[event_type].remove(handler)
            if not self._handlers[event_type]:
                del self._handlers[event_type]
            logger.info(f"Unregistered handler for event type: {event_type}")

    def add_middleware(self, middleware: Callable[[Event], Awaitable[Event]]) -> None:
        """Add middleware to the event processing pipeline."""
        self._middleware.append(middleware)

    async def publish(self, event: Event) -> None:
        """Publish an event to all registered handlers."""
        # Apply middleware
        processed_event = event
        for middleware in self._middleware:
            processed_event = await middleware(processed_event)

        logger.info(f"Publishing event: {event.event_type} from {event.source}")

        # Find handlers for this event type
        handlers = self._handlers.get(event.event_type, [])

        # Publish to all handlers concurrently
        if handlers:
            tasks = [handler.handle_event(processed_event) for handler in handlers]
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            logger.warning(f"No handlers registered for event type: {event.event_type}")

    async def publish_batch(self, events: List[Event]) -> None:
        """Publish multiple events."""
        tasks = [self.publish(event) for event in events]
        await asyncio.gather(*tasks, return_exceptions=True)


# Global event bus instance
event_bus = EventBus()


# Common event types
class EventTypes:
    """Common event types used across MetaQore services."""

    # Project events
    PROJECT_CREATED = "project.created"
    PROJECT_UPDATED = "project.updated"
    PROJECT_DELETED = "project.deleted"

    # Task events
    TASK_CREATED = "task.created"
    TASK_UPDATED = "task.updated"
    TASK_COMPLETED = "task.completed"

    # Artifact events
    ARTIFACT_CREATED = "artifact.created"
    ARTIFACT_UPDATED = "artifact.updated"
    ARTIFACT_DELETED = "artifact.deleted"

    # Governance events
    CONFLICT_DETECTED = "governance.conflict_detected"
    COMPLIANCE_VIOLATION = "governance.compliance_violation"
    AUDIT_LOG_CREATED = "governance.audit_log_created"

    # Specialist events
    SPECIALIST_TRAINING_STARTED = "specialist.training_started"
    SPECIALIST_TRAINING_COMPLETED = "specialist.training_completed"
    SPECIALIST_DEPLOYED = "specialist.deployed"

    # LLM events
    LLM_REQUEST_STARTED = "llm.request_started"
    LLM_REQUEST_COMPLETED = "llm.request_completed"
    LLM_REQUEST_FAILED = "llm.request_failed"