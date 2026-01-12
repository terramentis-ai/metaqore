

# MetaQore API Reference

All inference endpoints are routed through HyPIE, MetaQore's hybrid-parallel inference engine, which dynamically selects the optimal backend (local/cloud) based on governance, compliance, and real-time performance. All endpoints, parameters, and behaviors are current as of the latest service-oriented architecture. For canonical conventions, see [.github/copilot-instructions.md](.github/copilot-instructions.md). Implementation and progress validation are logged in [session_logger.md](session_logger.md).

## Overview

MetaQore provides a comprehensive REST API for governance-only AI infrastructure. The API is organized around four main services, each running on dedicated ports with consistent versioning.

### Service Endpoints

| Service | Port | Base URL | Description |
|---------|------|----------|-------------|
| AI Gateway | 8002 | `http://localhost:8002` | LLM inference and routing |
| Specialist Foundry | 8001 | `http://localhost:8001` | Specialist training and management |
| Compliance Auditor | 8003 | `http://localhost:8003` | Compliance validation and auditing |
| AI DevOps | 8004 | `http://localhost:8004` | Infrastructure management |

### Authentication

All endpoints require authentication via API key or JWT:

```
Authorization: Bearer <api_key_or_jwt>
X-API-Key: <api_key>
```

### Response Format

All responses follow a consistent JSON structure:

```json
{
  "success": true,
  "data": {...},
  "message": "Optional message",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

Error responses:

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Error description",
    "details": {...}
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

---

## AI Gateway Service (Port 8002)

Intelligent routing service for LLM inference with specialist integration and caching.

### Generate Response

Generate a response from an LLM with optional specialist routing.

**Endpoint:** `POST /api/v1/llm/generate`

**Request Body:**
```json
{
  "prompt": "string (required)",
  "model": "string (optional, default: 'gpt-4')",
  "agent_name": "string (optional)",
  "max_tokens": "integer (optional, default: 1000)",
  "temperature": "number (optional, default: 0.7)",
  "stream": "boolean (optional, default: false)",
  "cache_ttl": "integer (optional, seconds)",
  "sensitivity": "PUBLIC|INTERNAL|SENSITIVE|CRITICAL (optional)"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "response": "Generated text response",
    "model": "gpt-4",
    "processing_time": 1.23,
    "cached": false,
    "specialist_used": "research-assistant",
    "tokens_used": 150
  }
}
```

**Streaming Response:**
```json
{
  "event": "token",
  "data": "Hello"
}
{
  "event": "complete",
  "data": {
    "model": "gpt-4",
    "total_tokens": 150
  }
}
```

### Get Available Models

List all configured LLM models.

**Endpoint:** `GET /api/v1/llm/models`

**Response:**
```json
{
  "success": true,
  "data": {
    "models": [
      {
        "name": "gpt-4",
        "provider": "openai",
        "context_window": 8192,
        "supported_features": ["chat", "completion"]
      }
    ]
  }
}
```

### Health Check

**Endpoint:** `GET /api/v1/health`

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "version": "2.1.0",
    "uptime": 3600,
    "cache_status": "connected"
  }
}
```

---

## Specialist Foundry Service (Port 8001)

Autonomous agent training and deployment service.

### List Specialists

**Endpoint:** `GET /api/v1/specialists`

**Query Parameters:**
- `status`: `DRAFT|ACTIVE|ARCHIVED` (optional)
- `skill`: Filter by skill (optional)
- `limit`: Number of results (optional, default: 50)
- `offset`: Pagination offset (optional, default: 0)

**Response:**
```json
{
  "success": true,
  "data": {
    "specialists": [
      {
        "id": "uuid",
        "name": "code-reviewer",
        "description": "Reviews code for security issues",
        "status": "ACTIVE",
        "skills": ["security", "code-analysis"],
        "performance_score": 0.95,
        "created_at": "2024-01-01T00:00:00Z"
      }
    ],
    "total": 1,
    "limit": 50,
    "offset": 0
  }
}
```

### Create Specialist Proposal

**Endpoint:** `POST /api/v1/specialists`

**Request Body:**
```json
{
  "name": "string (required)",
  "description": "string (required)",
  "skills": ["array of strings (required)"],
  "training_config": {
    "model": "string (optional)",
    "epochs": "integer (optional)",
    "dataset_size": "integer (optional)"
  },
  "validation_criteria": {
    "min_accuracy": "number (optional)",
    "max_latency": "number (optional)"
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "specialist": {
      "id": "uuid",
      "name": "code-reviewer",
      "status": "DRAFT",
      "created_at": "2024-01-01T00:00:00Z"
    }
  }
}
```

### Get Specialist

**Endpoint:** `GET /api/v1/specialists/{id}`

**Response:**
```json
{
  "success": true,
  "data": {
    "specialist": {
      "id": "uuid",
      "name": "code-reviewer",
      "description": "Reviews code for security issues",
      "status": "ACTIVE",
      "skills": ["security", "code-analysis"],
      "training_history": [...],
      "performance_metrics": {
        "accuracy": 0.95,
        "latency_ms": 150,
        "requests_served": 1000
      },
      "created_at": "2024-01-01T00:00:00Z",
      "last_trained": "2024-01-01T12:00:00Z"
    }
  }
}
```

### Train Specialist

**Endpoint:** `POST /api/v1/specialists/{id}/train`

**Request Body:**
```json
{
  "training_data": {
    "type": "supervised|unsupervised|reinforcement",
    "dataset": "path_or_url",
    "parameters": {...}
  },
  "async": "boolean (optional, default: true)"
}
```

**Response (Async):**
```json
{
  "success": true,
  "data": {
    "training_job_id": "uuid",
    "status": "queued",
    "estimated_duration": 3600
  }
}
```

### Deploy Specialist

**Endpoint:** `POST /api/v1/specialists/{id}/deploy`

**Request Body:**
```json
{
  "environment": "staging|production",
  "rollback_on_failure": "boolean (optional, default: true)"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "deployment_id": "uuid",
    "status": "deploying",
    "endpoint": "http://specialist-endpoint"
  }
}
```

### Get Training Status

**Endpoint:** `GET /api/v1/specialists/{id}/training/{job_id}`

**Response:**
```json
{
  "success": true,
  "data": {
    "job_id": "uuid",
    "status": "running|completed|failed",
    "progress": 0.75,
    "current_epoch": 15,
    "total_epochs": 20,
    "metrics": {
      "loss": 0.05,
      "accuracy": 0.95
    },
    "estimated_completion": "2024-01-01T13:00:00Z"
  }
}
```

---

## Compliance Auditor Service (Port 8003)

Multi-framework compliance validation and evidence collection.

### Run Compliance Check

**Endpoint:** `POST /api/v1/compliance/check`

**Request Body:**
```json
{
  "framework": "SOC2|GDPR|HIPAA|PCI-DSS (required)",
  "scope": {
    "organization": "string (optional)",
    "project": "string (optional)",
    "time_range": {
      "start": "ISO date (optional)",
      "end": "ISO date (optional)"
    }
  },
  "evidence_types": ["audit_logs", "data_processing", "access_control"] (optional),
  "generate_report": "boolean (optional, default: true)"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "check_id": "uuid",
    "framework": "SOC2",
    "status": "completed",
    "overall_compliance": 0.92,
    "controls": [
      {
        "id": "CC6.1",
        "name": "Data classification",
        "status": "compliant",
        "score": 1.0,
        "evidence_count": 5
      }
    ],
    "report_url": "http://auditor/reports/uuid.pdf"
  }
}
```

### Get Compliance Status

**Endpoint:** `GET /api/v1/compliance/status`

**Query Parameters:**
- `framework`: Filter by framework (optional)
- `organization`: Filter by organization (optional)

**Response:**
```json
{
  "success": true,
  "data": {
    "frameworks": [
      {
        "name": "SOC2",
        "overall_score": 0.92,
        "last_assessment": "2024-01-01T00:00:00Z",
        "next_assessment": "2024-04-01T00:00:00Z",
        "critical_findings": 0
      }
    ]
  }
}
```

### Get Audit Trail

**Endpoint:** `GET /api/v1/compliance/audit`

**Query Parameters:**
- `start_date`: ISO date (optional)
- `end_date`: ISO date (optional)
- `agent`: Filter by agent (optional)
- `action`: Filter by action (optional)
- `limit`: Number of results (optional, default: 100)

**Response:**
```json
{
  "success": true,
  "data": {
    "events": [
      {
        "id": "uuid",
        "timestamp": "2024-01-01T00:00:00Z",
        "agent": "research-assistant",
        "action": "llm_generate",
        "resource": "gpt-4",
        "sensitivity": "INTERNAL",
        "compliance_tags": ["data_processing", "access_control"],
        "evidence": {...}
      }
    ],
    "total": 1000,
    "limit": 100,
    "offset": 0
  }
}
```

### Export Audit Report

**Endpoint:** `GET /api/v1/compliance/export`

**Query Parameters:**
- `format`: `json|csv|pdf` (required)
- `start_date`: ISO date (optional)
- `end_date`: ISO date (optional)
- `framework`: Filter by framework (optional)

**Response:** File download

---

## AI DevOps Service (Port 8004)

Infrastructure management and GitOps automation.

### List Infrastructure

**Endpoint:** `GET /api/v1/infrastructure`

**Query Parameters:**
- `type`: `kubernetes|docker|vm` (optional)
- `status`: `running|stopped|failed` (optional)

**Response:**
```json
{
  "success": true,
  "data": {
    "resources": [
      {
        "id": "uuid",
        "name": "ai-gateway-prod",
        "type": "kubernetes_deployment",
        "status": "running",
        "replicas": 3,
        "cpu_usage": 0.75,
        "memory_usage": 0.60,
        "last_updated": "2024-01-01T00:00:00Z"
      }
    ]
  }
}
```

### Scale Service

**Endpoint:** `POST /api/v1/infrastructure/scale`

**Request Body:**
```json
{
  "service": "ai-gateway|specialist-foundry|compliance-auditor",
  "replicas": "integer (required)",
  "environment": "staging|production (optional)"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "scaling_operation_id": "uuid",
    "service": "ai-gateway",
    "target_replicas": 5,
    "current_replicas": 3,
    "status": "scaling",
    "estimated_completion": "2024-01-01T00:05:00Z"
  }
}
```

### Deploy Configuration

**Endpoint:** `POST /api/v1/infrastructure/deploy`

**Request Body:**
```json
{
  "service": "string (required)",
  "version": "string (required)",
  "config": {
    "environment_variables": {...},
    "resource_limits": {
      "cpu": "string",
      "memory": "string"
    }
  },
  "strategy": "rolling_update|blue_green|canary"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "deployment_id": "uuid",
    "service": "ai-gateway",
    "version": "2.1.0",
    "strategy": "rolling_update",
    "status": "deploying",
    "rollback_available": true
  }
}
```

### Get Deployment Status

**Endpoint:** `GET /api/v1/infrastructure/deployments/{id}`

**Response:**
```json
{
  "success": true,
  "data": {
    "deployment_id": "uuid",
    "service": "ai-gateway",
    "status": "completed|failed|rolling_back",
    "progress": 1.0,
    "start_time": "2024-01-01T00:00:00Z",
    "completion_time": "2024-01-01T00:05:00Z",
    "events": [
      {
        "timestamp": "2024-01-01T00:01:00Z",
        "message": "Starting deployment",
        "level": "info"
      }
    ]
  }
}
```

### Rollback Deployment

**Endpoint:** `POST /api/v1/infrastructure/deployments/{id}/rollback`

**Request Body:**
```json
{
  "target_version": "string (optional, defaults to previous)"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "rollback_id": "uuid",
    "deployment_id": "uuid",
    "target_version": "2.0.0",
    "status": "rolling_back"
  }
}
```

---

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid request data |
| `AUTHENTICATION_ERROR` | 401 | Invalid or missing credentials |
| `AUTHORIZATION_ERROR` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `CONFLICT` | 409 | Resource conflict |
| `RATE_LIMITED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Internal server error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

## Rate Limits

| Endpoint Pattern | Limit |
|------------------|-------|
| `/api/v1/llm/generate` | 120 requests/minute |
| `/api/v1/specialists/*` | 60 requests/minute |
| `/api/v1/compliance/*` | 30 requests/minute |
| `/api/v1/infrastructure/*` | 20 requests/minute |

## Webhooks

Services support webhook notifications for important events:

### Configuration

```json
{
  "url": "https://your-webhook-url.com",
  "secret": "webhook-secret",
  "events": ["specialist.trained", "compliance.violation", "deployment.completed"]
}
```

### Event Types

- `specialist.trained`: Specialist training completed
- `specialist.deployed`: Specialist deployment completed
- `compliance.check_completed`: Compliance check finished
- `compliance.violation`: Compliance violation detected
- `infrastructure.scaled`: Service scaling completed
- `infrastructure.deployment_completed`: Deployment finished

### Webhook Payload

```json
{
  "event": "specialist.trained",
  "timestamp": "2024-01-01T00:00:00Z",
  "data": {
    "specialist_id": "uuid",
    "performance_score": 0.95
  },
  "signature": "sha256=..."
}
```

---

## SDKs and Libraries

### Python SDK

```python
from metaqore import MetaQoreClient

client = MetaQoreClient(api_key="your-api-key")

# Generate response
response = client.llm.generate(
    prompt="Explain quantum computing",
    model="gpt-4"
)

# Create specialist
specialist = client.specialists.create(
    name="code-reviewer",
    skills=["security", "code-analysis"]
)
```

### JavaScript SDK

```javascript
import { MetaQore } from 'metaqore-sdk';

const client = new MetaQore({ apiKey: 'your-api-key' });

// Generate response
const response = await client.llm.generate({
  prompt: 'Explain quantum computing',
  model: 'gpt-4'
});

// List specialists
const specialists = await client.specialists.list();
```

---

*Last updated: March 8, 2026*