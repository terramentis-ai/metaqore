# MetaQore Production Deployment Guide

## Overview

This guide covers deploying MetaQore to production environments using Docker Compose and Kubernetes. MetaQore consists of four microservices with shared governance components and Redis-based event communication.

## Prerequisites

### System Requirements

- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2
- **CPU**: 4+ cores (8+ recommended)
- **RAM**: 16GB+ (32GB+ recommended)
- **Storage**: 100GB+ SSD
- **Network**: Stable internet connection

### Software Requirements

- Docker Engine 20.10+
- Docker Compose 2.0+
- Git
- Python 3.11+ (for local development)
- kubectl (for Kubernetes deployment)
- Helm 3.x (optional, for advanced deployments)

### External Dependencies

- **Redis**: For caching and event bus
- **PostgreSQL**: For state persistence (optional, defaults to SQLite)
- **SSL Certificate**: For HTTPS termination

## Quick Start Deployment

### 1. Clone Repository

```bash
git clone <repository-url>
cd metaqore
```

### 2. Environment Setup

Create environment file:

```bash
cp .env.example .env
# Edit .env with your production values
```

Required environment variables:

```bash
# API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
VLLM_BASE_URL=http://your-vllm-endpoint

# Security
JWT_SECRET_KEY=your-256-bit-secret
ENCRYPTION_KEY=your-32-byte-key
REDIS_PASSWORD=secure-redis-password

# Database (optional)
DATABASE_URL=postgresql://user:pass@host:5432/metaqore

# Monitoring
GRAFANA_PASSWORD=secure-grafana-password
```

### 3. SSL Certificate Setup

For HTTPS, place certificates in `nginx/ssl/`:

```bash
mkdir -p nginx/ssl
# Copy your SSL certificate and key
cp your-domain.crt nginx/ssl/
cp your-domain.key nginx/ssl/
```

### 4. Deploy Services

```bash
# Development deployment
docker-compose -f docker-compose.services.yml up -d

# Production deployment
docker-compose -f docker-compose.services.yml -f docker-compose.prod.yml up -d
```

### 5. Verify Deployment

```bash
# Check service health
curl https://your-domain/api/v1/health

# Check all services
docker-compose ps

# View logs
docker-compose logs -f ai-gateway
```

## Detailed Service Configuration

### AI Gateway Service

**Port**: 8002
**Purpose**: Intelligent LLM routing with specialist integration

**Scaling Configuration**:
```yaml
deploy:
  replicas: 3  # Scale based on load
  resources:
    limits:
      cpus: '4.0'
      memory: 8G
```

**Environment Variables**:
```bash
LOG_LEVEL=INFO
CACHE_TTL=3600
MAX_CONCURRENT_REQUESTS=100
REDIS_CACHE_URL=redis://redis:6379/1
SPECIALIST_FOUNDRY_URL=http://specialist-foundry:8000
```

### Specialist Foundry Service

**Port**: 8001
**Purpose**: Autonomous agent training and deployment

**Resource Requirements**:
```yaml
resources:
  limits:
    cpus: '2.0'
    memory: 4G
  reservations:
    cpus: '1.0'
    memory: 2G
```

### Compliance Auditor Service

**Port**: 8003
**Purpose**: Multi-framework compliance validation

**Configuration**:
```bash
AUDIT_RETENTION_DAYS=2555  # 7 years
COMPLIANCE_CHECK_INTERVAL=86400  # Daily
```

### AI DevOps Service

**Port**: 8004
**Purpose**: Infrastructure management and GitOps

**Kubernetes Integration**:
```bash
AUTO_SCALE_ENABLED=true
KUBECONFIG=/path/to/kubeconfig
```

## Kubernetes Deployment

### 1. Prerequisites

```bash
# Install kubectl and helm
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl && sudo mv kubectl /usr/local/bin/

# Install Helm
curl https://get.helm.sh/helm-v3.12.0-linux-amd64.tar.gz | tar -xz
sudo mv linux-amd64/helm /usr/local/bin/
```

### 2. Create Namespace

```bash
kubectl create namespace metaqore
```

### 3. Deploy Dependencies

```bash
# Add Helm repositories
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Deploy PostgreSQL
helm install metaqore-postgres bitnami/postgresql \
  --namespace metaqore \
  --set auth.postgresPassword=metaqore123 \
  --set auth.database=metaqore

# Deploy Redis
helm install metaqore-redis bitnami/redis \
  --namespace metaqore \
  --set auth.password=redis123 \
  --set replica.replicaCount=1

# Deploy Prometheus Stack
helm install metaqore-monitoring prometheus-community/kube-prometheus-stack \
  --namespace metaqore \
  --set grafana.adminPassword=admin123
```

### 4. Deploy MetaQore Services

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Or use Helm
helm install metaqore ./helm/metaqore --namespace metaqore
```

### 5. Configure Ingress

```yaml
# k8s/ingress.yml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: metaqore-ingress
  namespace: metaqore
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - your-domain.com
    secretName: metaqore-tls
  rules:
  - host: your-domain.com
    http:
      paths:
      - path: /api/v1/llm
        pathType: Prefix
        backend:
          service:
            name: ai-gateway
            port:
              number: 8000
      - path: /api/v1/specialists
        pathType: Prefix
        backend:
          service:
            name: specialist-foundry
            port:
              number: 8000
      - path: /api/v1/compliance
        pathType: Prefix
        backend:
          service:
            name: compliance-auditor
            port:
              number: 8000
      - path: /api/v1/infrastructure
        pathType: Prefix
        backend:
          service:
            name: ai-devops
            port:
              number: 8000
```

## Monitoring and Observability

### Prometheus Metrics

Each service exposes metrics at `/metrics`:

```bash
# Query AI Gateway metrics
curl http://ai-gateway:8000/metrics

# Example metrics
metaqore_requests_total{service="ai-gateway",method="POST",endpoint="/api/v1/llm/generate"} 1500
metaqore_request_duration_seconds{service="ai-gateway",quantile="0.95"} 1.2
metaqore_cache_hit_ratio{service="ai-gateway"} 0.85
```

### Grafana Dashboards

Access Grafana at `http://your-domain:3000`

Pre-configured dashboards:
- **MetaQore Overview**: System-wide metrics
- **AI Gateway Performance**: Request latency, throughput, cache metrics
- **Specialist Training**: Training progress, model performance
- **Compliance Status**: Audit results, violation trends
- **Infrastructure Health**: Resource usage, scaling events

### Logging

All services use structured JSON logging:

```json
{
  "timestamp": "2024-01-01T00:00:00Z",
  "level": "INFO",
  "service": "ai-gateway",
  "request_id": "uuid",
  "message": "LLM request processed",
  "duration_ms": 1250,
  "model": "gpt-4",
  "cached": false
}
```

### Centralized Logging

```bash
# Using ELK Stack
helm install metaqore-logging elastic/eck-operator --namespace metaqore

# Or using Loki
helm install metaqore-loki grafana/loki-stack --namespace metaqore
```

## Security Configuration

### Network Security

```yaml
# k8s/network-policy.yml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: metaqore-network-policy
  namespace: metaqore
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
```

### Secret Management

```yaml
# k8s/secrets.yml
apiVersion: v1
kind: Secret
metadata:
  name: metaqore-secrets
  namespace: metaqore
type: Opaque
data:
  openai-api-key: <base64-encoded>
  anthropic-api-key: <base64-encoded>
  jwt-secret: <base64-encoded>
  encryption-key: <base64-encoded>
```

### RBAC Configuration

```yaml
# k8s/rbac.yml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: metaqore-service-role
  namespace: metaqore
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
```

## Backup and Recovery

### Database Backup

```bash
# PostgreSQL backup
kubectl exec -n metaqore metaqore-postgres-0 -- pg_dump -U metaqore metaqore > backup.sql

# Automated backup job
kubectl apply -f k8s/backup-job.yml
```

### Redis Backup

```bash
# Redis backup
kubectl exec -n metaqore metaqore-redis-master-0 -- redis-cli --rdb backup.rdb

# Automated backup
kubectl apply -f k8s/redis-backup.yml
```

### Disaster Recovery

```bash
# Restore from backup
kubectl exec -n metaqore metaqore-postgres-0 -- psql -U metaqore -d metaqore < backup.sql

# Failover procedure
kubectl scale deployment metaqore-ai-gateway --replicas=0
kubectl scale deployment metaqore-ai-gateway --replicas=3
```

## Performance Tuning

### AI Gateway Optimization

```yaml
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-gateway-hpa
  namespace: metaqore
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-gateway
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Cache Configuration

```bash
# Redis cluster for high availability
helm install metaqore-redis bitnami/redis-cluster \
  --namespace metaqore \
  --set cluster.nodes=6 \
  --set cluster.replicas=1
```

### Load Balancing

```yaml
# NGINX configuration
upstream ai_gateway {
    least_conn;
    server ai-gateway-1:8000;
    server ai-gateway-2:8000;
    server ai-gateway-3:8000;
}

server {
    listen 80;
    server_name your-domain.com;

    location /api/v1/llm/ {
        proxy_pass http://ai_gateway;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Troubleshooting

### Common Issues

**Service Unhealthy**
```bash
# Check pod status
kubectl get pods -n metaqore

# View logs
kubectl logs -f deployment/ai-gateway -n metaqore

# Check events
kubectl get events -n metaqore --sort-by=.metadata.creationTimestamp
```

**High Latency**
```bash
# Check resource usage
kubectl top pods -n metaqore

# Analyze metrics
curl http://prometheus:9090/graph

# Scale services
kubectl scale deployment ai-gateway --replicas=5
```

**Database Connection Issues**
```bash
# Test database connectivity
kubectl exec -it metaqore-postgres-0 -n metaqore -- psql -U metaqore -d metaqore

# Check connection pool
kubectl exec -it deployment/ai-gateway -n metaqore -- netstat -tlnp
```

### Debug Commands

```bash
# Port forwarding for local testing
kubectl port-forward -n metaqore svc/ai-gateway 8002:8000

# Execute commands in pods
kubectl exec -it deployment/ai-gateway -n metaqore -- /bin/bash

# View configuration
kubectl get configmap -n metaqore -o yaml

# Check secrets
kubectl get secrets -n metaqore
```

## Maintenance Procedures

### Rolling Updates

```bash
# Update service image
kubectl set image deployment/ai-gateway ai-gateway=metaqore/ai-gateway:v2.1.1

# Monitor rollout
kubectl rollout status deployment/ai-gateway

# Rollback if needed
kubectl rollout undo deployment/ai-gateway
```

### Certificate Renewal

```bash
# Using cert-manager
kubectl get certificate -n metaqore

# Manual renewal
kubectl delete secret metaqore-tls
kubectl create secret tls metaqore-tls --cert=fullchain.pem --key=privkey.pem
```

### Log Rotation

```yaml
# k8s/log-rotation.yml
apiVersion: v1
kind: ConfigMap
metadata:
  name: logrotate-config
  namespace: metaqore
data:
  logrotate.conf: |
    /var/log/metaqore/*.log {
        daily
        rotate 30
        compress
        missingok
        notifempty
    }
```

## Support and Resources

### Documentation
- [API Reference](API_REFERENCE.md)
- [Architecture Guide](docs/ARCHITECTURE.md)
- [Troubleshooting Guide](docs/TROUBLESHOOTING.md)

### Community
- GitHub Issues: Bug reports and feature requests
- Discussions: General questions and community support
- Slack: Real-time community chat

### Enterprise Support
- 24/7 technical support
- On-site deployment assistance
- Custom integration services
- Training and certification programs

---

*Last updated: March 8, 2026*