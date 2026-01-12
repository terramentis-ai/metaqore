

# MetaQore - Governance-Only AI Infrastructure

MetaQore is a provider-agnostic, governance-first AI platform for enterprise adoption. At its core is HyPIE—a hybrid-parallel inference engine that unifies local (e.g., llama.cpp) and cloud (e.g., OpenAI, vLLM) LLM backends under a single, policy-driven control plane. All intelligence is routed through HyPIE, which dynamically selects the optimal backend based on governance, compliance, and real-time performance.

**Canonical documentation:**
- Project conventions and architecture: [.github/copilot-instructions.md](.github/copilot-instructions.md)
- Implementation and progress validation: [session_logger.md](session_logger.md)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com)



> **[INFO: The project has migrated from a monolithic to a service-oriented architecture. Some references to the legacy monolith may remain below and will be updated. See the instructions file for canonical architecture.**

## 🎯 Core Capabilities

### Hybrid-Parallel Inference Engine (HyPIE)
- Policy-aware, adaptive routing (e.g., sensitive data to local, air-gapped Llama.cpp)
- Unified artifact/audit trail for all inference, regardless of backend
- Dynamic workload balancing and performance optimization
- Abstraction over heterogeneous hardware and providers

- **PSMP (Project State Management Protocol):** Enforces artifact lifecycle (DRAFT → ACTIVE → ARCHIVED)
- **Conflict Detection:** Prevents incompatible artifacts (dependency versions, file paths)
- **Compliance Auditing:** Immutable JSONL audit trails for GDPR/HIPAA/SOC2
- **Security Routing:** Task sensitivity classification (PUBLIC/INTERNAL/SENSITIVE/CRITICAL)
- **Specialist Management (HMCP):** Autonomous agent discovery and training
- **Multi-Tenancy:** Organization-level governance policies and isolated state
- **LLM Adapter Pattern:** Provider-agnostic integration (llama.cpp, OpenAI, vLLM, Anthropic)
- **Service Decoupling:** Independent services (Specialist Foundry, AI Gateway, Compliance Auditor, AI DevOps)

## 🏗️ Architecture Overview

**Key Components:**
- **HyPIE Router:** The intelligent dispatcher and control plane for all LLM inference, integrated into the Governance Gateway.
- **LLM Adapters:** Provider-agnostic clients for OpenAI, Anthropic, vLLM, Llama.cpp, etc., all instrumented for real-time metrics and PSMP event emission.
- **Unified Artifact Schema:** All inference results are logged as PSMP artifacts, supporting full provenance and compliance.

> **[REVIEW: Ensure all service names, ports, and architecture diagrams match the current implementation in packages/ and docker-compose.services.yml. Legacy references will be updated or removed.]**


```
External Agents → MetaQore API (:8001) → HyPIE Router (Governance Gateway) → LLM Adapters (Provider-Agnostic)
        ↓
      PSMP Engine, StateManager, Unified Artifact Logging
        ↓
      SQLite/PostgreSQL, Audit Trail, Blocking Reports
```

### Core Services

> **[REVIEW: Confirm that all listed services and ports are current. The canonical list is in .github/copilot-instructions.md.]**

- **AI Gateway** (8002): Intelligent LLM routing via HyPIE, specialist integration, caching
- **Specialist Foundry** (8001): Specialist lifecycle, training, deployment
- **Compliance Auditor** (8003): Compliance validation, evidence collection
- **AI DevOps** (8004): Infrastructure management, GitOps automation

## 🚀 Deployment

> **[REVIEW: Deployment instructions may reference legacy or outdated scripts/configs. Please verify against the latest docker-compose and service configs.]**

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Redis (for caching)

### Development Environment

1. **Clone and setup:**
  ```bash
  git clone <repository>
  cd metaqore
  python -m venv .venv
  source .venv/bin/activate  # On Windows: .venv\Scripts\activate
  pip install -r requirements.txt
  ```

2. **Start services:**
  ```bash
  docker-compose -f docker-compose.services.yml up -d
  ```

3. **Test the system:**
  ```bash
  curl http://localhost:8002/api/v1/health
  ```

### Production Deployment


```bash
# Deploy all services with monitoring
docker-compose -f docker-compose.services.yml -f docker-compose.prod.yml up -d

# Scale services as needed
docker-compose up -d --scale ai-gateway=3
```

### Configuration


```bash
# Set environment variables
export OPENAI_API_KEY="your-key"
export REDIS_PASSWORD="your-redis-password"
export JWT_SECRET_KEY="your-jwt-secret"
export ENCRYPTION_KEY="your-encryption-key"
```

## 📚 API Documentation

> **[REVIEW: API endpoints and payloads should be cross-checked with API_REFERENCE.md and the current FastAPI implementation. Outdated endpoints will be updated.]**

### AI Gateway

**Generate Intelligence Response:**
```bash
POST /api/v1/llm/generate
Content-Type: application/json
Authorization: Bearer <enterprise-token>

{
  "prompt": "Analyze market trends for Q1 2026",
  "model": "gpt-4",
  "agent_name": "market-analyst",
  "max_tokens": 2000,
  "sensitivity": "INTERNAL"
}
```

**Enterprise Response:**
```json
{
  "response": "Based on current market analysis...",
  "model": "gpt-4",
  "processing_time": 1.2,
  "cached": false,
  "specialist_used": "market-analyst",
  "compliance_validated": true
}
```

### Specialist Foundry

**Deploy Specialist:**
```bash
POST /api/v1/specialists
Content-Type: application/json
Authorization: Bearer <enterprise-token>

{
  "name": "compliance-monitor",
  "description": "Monitors regulatory compliance across enterprise systems",
  "skills": ["regulatory", "risk-analysis", "audit"],
  "training_config": {
    "model": "gpt-4",
    "performance_target": 0.95
  }
}
```

### Compliance Auditor

**Execute Compliance Assessment:**
```bash
POST /api/v1/compliance/check
Content-Type: application/json
Authorization: Bearer <enterprise-token>

{
  "framework": "SOC2",
  "scope": {"organization": "enterprise", "department": "finance"},
  "evidence_types": ["audit_logs", "access_controls", "data_processing"],
  "generate_report": true
}
```

## 🔧 Configuration

> **[REVIEW: Environment variables and configuration options should be validated against the latest service configs and .env files.]**

### Environment Variables


| Service | Variable | Default | Description |
|---------|----------|---------|-------------|
| Platform | `EVENT_BUS_URL` | `redis://localhost:6379` | Event bus connection |
| AI Gateway | `REDIS_CACHE_URL` | `redis://localhost:6379/1` | Cache backend |
| Specialist Foundry | `TRAINING_TIMEOUT` | `3600` | Max training duration (seconds) |
| Compliance Auditor | `AUDIT_RETENTION_DAYS` | `2555` | Audit trail retention (days) |

### Intelligence Providers


Configure AI provider credentials:

```bash

# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# vLLM
VLLM_BASE_URL=http://localhost:8000
```

## 🧪 Testing

> **[REVIEW: Test suite structure and commands should be validated against the current tests/unit, tests/integration, and any new test directories.]**


Run the test suite:

```bash

# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v
```

## 📊 Monitoring

> **[REVIEW: Monitoring, metrics, and logging sections should be updated to reflect any new observability tooling or endpoints.]**

### Health Checks


All services expose health endpoints:
- `GET /api/v1/health` - Service health status

### Metrics


Prometheus metrics available at `/metrics` on each service.

### Logging


Structured JSON logging with configurable levels:
- `DEBUG`: Detailed debugging information
- `INFO`: General operational messages
- `WARNING`: Warning conditions
- `ERROR`: Error conditions

## 🔒 Security

> **[REVIEW: Security and compliance features should be cross-checked with the latest implementation in packages/governance-core and compliance-auditor.]**

- **Information Classification:** PUBLIC/INTERNAL/SENSITIVE/CRITICAL sensitivity levels
- **Provider Isolation:** Sensitive workloads routed to compliant providers
- **Audit Trails:** Immutable compliance evidence with cryptographic verification
- **Access Control:** JWT-based authentication, role-based permissions
- **Data Encryption:** End-to-end encryption for sensitive operations
- **Regulatory Compliance:** SOC2, GDPR, HIPAA validation built-in

## 📈 Performance

> **[REVIEW: Performance benchmarks and scaling instructions should be validated against current infrastructure and load test results.]**

### Benchmarks

- **Throughput:** 1000+ RPS with Redis caching
- **Latency:** <100ms for cached responses
- **Concurrent Workloads:** 10,000+ simultaneous connections
- **Uptime:** 99.9% availability with automated failover

### Scaling


Horizontal scaling supported across all services:
```bash
docker-compose up -d --scale ai-gateway=5 --scale specialist-foundry=3
```

### Resource Optimization

- **Intelligent Caching:** Redis-backed response caching with TTL management
- **Load Balancing:** Automatic distribution across service instances
- **Resource Monitoring:** Real-time metrics and automated scaling triggers

## 🤝 Contributing

> **[REVIEW: Contribution workflow should reference the canonical delegator (instructions file) and validator (session_logger.md). All major changes must be logged for traceability.]**


1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Commit changes: `git commit -m 'feat: your message'`
4. Push to branch: `git push origin feat/your-feature`
5. Open a Pull Request
6. Reference [.github/copilot-instructions.md](.github/copilot-instructions.md) for conventions and [session_logger.md](session_logger.md) for progress logging.

## 📄 License

This enterprise software is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Partnerships

MetaQore leverages industry-leading open-source technologies:

- **FastAPI**: High-performance web framework for enterprise APIs
- **Pydantic**: Enterprise-grade data validation and serialization
- **Redis**: High-performance caching and event streaming infrastructure
- **Docker**: Enterprise containerization and orchestration

## 📞 Support

> **[REVIEW: Support and documentation links should be validated for accuracy and completeness.]**


- **Documentation:** [API Reference](API_REFERENCE.md) | [Deployment Guide](DEPLOYMENT_GUIDE.md)
- **Support:** Technical support and implementation services
- **Professional Services:** Custom integration, training, and optimization
- **Security Audits:** Security assessments and compliance validation

---

**MetaQore** - Enabling enterprise AI adoption with governance you can trust.