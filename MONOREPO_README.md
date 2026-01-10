# MetaQore Monorepo

This monorepo contains the MetaQore platform components organized as independent services with shared governance.

## Structure

```
metaqore/
├── packages/
│   ├── governance-core/          # Shared governance components
│   │   ├── src/metaqore_governance_core/
│   │   │   ├── models.py         # Pydantic models
│   │   │   ├── psmp.py           # PSMP state machine
│   │   │   ├── state_manager.py  # State orchestration
│   │   │   ├── audit.py          # Compliance auditing
│   │   │   ├── security.py       # Security routing
│   │   │   ├── event_bus.py      # Inter-service communication
│   │   │   └── hmcp/             # HMCP components
│   │   └── pyproject.toml
│   │
│   ├── specialist-foundry/       # HMCP specialist management
│   │   ├── src/metaqore_specialist_foundry/
│   │   │   └── service.py        # FastAPI service
│   │   └── pyproject.toml
│   │
│   ├── ai-gateway/               # LLM proxy and routing
│   │   ├── src/metaqore_ai_gateway/
│   │   │   └── service.py        # FastAPI service
│   │   └── pyproject.toml
│   │
│   ├── compliance-auditor/       # Compliance monitoring
│   │   ├── src/metaqore_compliance_auditor/
│   │   │   └── service.py        # FastAPI service
│   │   └── pyproject.toml
│   │
│   └── ai-devops/                # Infrastructure management
│       ├── src/metaqore_ai_devops/
│       │   └── service.py        # FastAPI service
│       └── pyproject.toml
│
├── pyproject.toml                # Monorepo configuration
├── nx.json                       # Nx workspace config
├── docker-compose.services.yml   # Service orchestration
└── README.md
```

## Services

### Governance Core
Shared library containing:
- PSMP (Project State Management Protocol)
- HMCP (Hierarchical Multi-agent Control Protocol)
- State management and persistence
- Audit trail and compliance
- Security routing
- Event bus for inter-service communication

### Specialist Foundry
- Autonomous agent discovery and training
- HMCP specialist management
- Training pipeline APIs
- Validation gates

### AI Gateway
- Provider-agnostic LLM proxy
- Intelligent routing and load balancing
- Policy enforcement
- Caching and optimization

### Compliance Auditor
- Evidence collection and validation
- Framework compliance (SOC2, GDPR, HIPAA)
- Audit trail analysis
- Blocking reports and remediation

### AI DevOps
- Infrastructure as Code
- Kubernetes deployments
- GitOps workflows
- Monitoring and scaling

## Development

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Nx CLI (optional, for monorepo management)

### Setup
```bash
# Install dependencies for all packages
pip install -e packages/governance-core
pip install -e packages/specialist-foundry
pip install -e packages/ai-gateway
pip install -e packages/compliance-auditor
pip install -e packages/ai-devops

# Or use Nx
npx nx run-many --target=install --all
```

### Running Services
```bash
# Start all services
docker-compose -f docker-compose.services.yml up

# Or run individually
cd packages/specialist-foundry && python -m metaqore_specialist_foundry.service
cd packages/ai-gateway && python -m metaqore_ai_gateway.service
# etc.
```

### Testing
```bash
# Run tests for all packages
npx nx run-many --target=test --all

# Run specific service tests
npx nx run specialist-foundry:test
```

## Architecture

Services communicate via an event-driven architecture using Redis-backed event bus. Each service is independently deployable and scalable.

- **Event Bus**: Async communication between services
- **Shared State**: Governance-core provides consistent state management
- **API Gateway**: Routes external requests to appropriate services
- **Service Mesh**: Handles service discovery and load balancing

## Deployment

Services can be deployed independently or as a complete stack using Docker Compose. Each service exposes health checks and metrics endpoints.