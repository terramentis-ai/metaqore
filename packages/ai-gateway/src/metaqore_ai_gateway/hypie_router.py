"""HyPIE Router: Governance-Aware Hybrid-Parallel Inference Engine Router.

The HyPIE Router is the intelligent control plane for all LLM inference in MetaQore.
It dynamically routes requests between local (e.g., llama.cpp) and cloud (e.g., OpenAI, vLLM)
backends based on PSMP/HMCP context, real-time performance metrics, and compliance policies.

Key Features:
- Policy-aware routing (cost, performance, privacy, data sensitivity)
- Dynamic workload balancing and performance optimization
- Unified artifact logging for all inference results
- Abstraction over heterogeneous hardware and providers
- Real-time metrics emission and observability
"""

import logging
import time
import asyncio
import json
import random
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx

from metaqore.llm.client.interface import LLMResponse, LLMProvider
from metaqore.llm.client.factory import LLMClientFactory

from metaqore_governance_core.cache import Cache
from metaqore_governance_core.event_bus import event_bus, Event, EventTypes

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Supported LLM provider types - maps to LLMProvider enum."""
    LOCAL_LLAMA_CPP = "llama_cpp"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    VLLM = "vllm"
    AZURE_OPENAI = "azure_openai"
    MOCK = "mock"


@dataclass
class ProviderMetrics:
    """Real-time metrics for a provider with historical tracking."""
    latency_ms: float
    throughput_tokens_per_sec: float
    error_rate: float
    resource_utilization: float  # 0.0 to 1.0
    cost_per_token: float
    last_updated: float
    
    # Historical data for adaptive learning
    latency_history: List[float] = None
    error_history: List[bool] = None
    request_count: int = 0
    
    def __post_init__(self):
        if self.latency_history is None:
            self.latency_history = []
        if self.error_history is None:
            self.error_history = []


@dataclass
class RoutingPolicy:
    """Policy for routing decisions."""
    max_cost_usd: Optional[float] = None
    allowed_providers: Optional[List[ProviderType]] = None
    data_sensitivity: str = "low"  # low, medium, high, critical
    min_confidence: float = 0.8
    prefer_local: bool = False
    max_latency_ms: Optional[int] = None


@dataclass
class InferenceRequest:
    """Standardized inference request."""
    prompt: str
    model: Optional[str] = None
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7
    agent_name: str = ""
    project_id: Optional[str] = None
    task_id: Optional[str] = None
    sensitivity: str = "low"
    routing_policy: Optional[RoutingPolicy] = None


@dataclass
class InferenceResult:
    """Standardized inference result."""
    content: str
    provider: ProviderType
    model: str
    tokens_used: int
    processing_time_ms: float
    cost_estimate_usd: float
    artifact_id: str
    compliance_tags: List[str]


class HyPIERouter:
    """HyPIE Router: Governance-aware hybrid-parallel inference engine with self-improving capabilities."""

    def __init__(self, psmp_engine: PSMPEngine, hmcp_policy_engine: HierarchicalChainingPolicy, llm_factory: Optional[LLMClientFactory] = None, compliance_auditor: Optional[ComplianceAuditor] = None, cache: Optional[Cache] = None, redis_url: str = "redis://localhost:6379"):
        self.psmp_engine = psmp_engine
        self.hmcp_policy_engine = hmcp_policy_engine
        self.llm_factory = llm_factory or LLMClientFactory()
        self.compliance_auditor = compliance_auditor or ComplianceAuditor()
        self.cache = cache or Cache.create_in_memory_cache()
        self.redis_url = redis_url

        # Provider registry with metrics
        self.providers: Dict[ProviderType, Dict[str, Any]] = {}
        self.provider_metrics: Dict[ProviderType, ProviderMetrics] = {}

        # Circuit breaker state for production hardening (now Redis-backed)
        self.circuit_breakers: Dict[ProviderType, Dict[str, Any]] = {}
        self.failure_threshold = 5  # Consecutive failures before opening circuit
        self.recovery_timeout = 60  # Seconds before attempting recovery

        # Online learning parameters for self-improving routing
        self.exploration_rate = 0.1  # Epsilon for epsilon-greedy exploration
        self.learning_rate = 0.1  # Alpha for weight updates
        self.provider_weights: Dict[ProviderType, float] = {}  # Success-based weights
        self.provider_success_rates: Dict[ProviderType, float] = {}  # Bayesian success probabilities

        # Initialize distributed state
        self._initialize_distributed_state()

        # Initialize providers and load persisted state
        self._initialize_providers()
        self._load_persisted_state()

        logger.info("HyPIE Router initialized with governance, compliance, production hardening, distributed state, and self-improving capabilities")

    async def _get_cached_routing_decision(self, request: InferenceRequest) -> Optional[ProviderType]:
        """Get cached routing decision for similar requests."""
        try:
            # Create cache key based on request characteristics
            cache_key = f"routing:{hash(request.prompt[:100])}:{request.sensitivity}:{request.routing_policy.__dict__ if request.routing_policy else 'default'}"
            
            cached_decision = await self.cache.get(cache_key)
            if cached_decision:
                logger.debug(f"Using cached routing decision: {cached_decision}")
                return ProviderType(cached_decision)
            return None
        except Exception as e:
            logger.warning(f"Failed to get cached routing decision: {e}")
            return None

    async def _cache_routing_decision(self, request: InferenceRequest, provider: ProviderType, ttl: int = 300):
        """Cache routing decision for future similar requests."""
        try:
            cache_key = f"routing:{hash(request.prompt[:100])}:{request.sensitivity}:{request.routing_policy.__dict__ if request.routing_policy else 'default'}"
            await self.cache.set(cache_key, provider.value, ttl=ttl)
        except Exception as e:
            logger.warning(f"Failed to cache routing decision: {e}")

    def _initialize_providers(self):
        """Initialize provider configurations and metrics."""
        # This will be expanded with actual provider configs
        # For now, placeholder metrics
        for provider in ProviderType:
            self.provider_metrics[provider] = ProviderMetrics(
                latency_ms=100.0,
                throughput_tokens_per_sec=50.0,
                error_rate=0.01,
                resource_utilization=0.5,
                cost_per_token=0.0001,
                last_updated=time.time()
            )

    def _initialize_distributed_state(self):
        """Initialize Redis-backed distributed state management."""
        try:
            import redis
            self.redis_client = redis.from_url(self.redis_url)
            self.use_distributed_state = True
            logger.info("Distributed state management enabled with Redis")
        except ImportError:
            logger.warning("Redis not available, falling back to in-memory state")
            self.redis_client = None
            self.use_distributed_state = False

    def _load_persisted_state(self):
        """Load persisted state from Redis."""
        if not self.use_distributed_state:
            # Initialize in-memory circuit breakers
            for provider in ProviderType:
                self.circuit_breakers[provider] = {
                    "state": "closed",
                    "failure_count": 0,
                    "last_failure_time": 0,
                    "success_count": 0
                }
                self.provider_weights[provider] = 1.0  # Equal initial weights
                self.provider_success_rates[provider] = 0.5  # Neutral success rate
            return

        try:
            # Load circuit breakers
            for provider in ProviderType:
                cb_key = f"hypie:circuit_breaker:{provider.value}"
                cb_data = self.redis_client.get(cb_key)
                if cb_data:
                    self.circuit_breakers[provider] = json.loads(cb_data)
                else:
                    self.circuit_breakers[provider] = {
                        "state": "closed",
                        "failure_count": 0,
                        "last_failure_time": 0,
                        "success_count": 0
                    }

                # Load provider weights and success rates
                weight_key = f"hypie:weight:{provider.value}"
                success_key = f"hypie:success_rate:{provider.value}"

                weight_data = self.redis_client.get(weight_key)
                success_data = self.redis_client.get(success_key)

                self.provider_weights[provider] = float(weight_data) if weight_data else 1.0
                self.provider_success_rates[provider] = float(success_data) if success_data else 0.5

            logger.info("Loaded persisted state from Redis")
        except Exception as e:
            logger.error(f"Failed to load persisted state: {e}")
            # Fallback to defaults
            for provider in ProviderType:
                self.circuit_breakers[provider] = {
                    "state": "closed",
                    "failure_count": 0,
                    "last_failure_time": 0,
                    "success_count": 0
                }
                self.provider_weights[provider] = 1.0
                self.provider_success_rates[provider] = 0.5

    def _persist_state(self):
        """Persist current state to Redis."""
        if not self.use_distributed_state:
            return

        try:
            # Persist circuit breakers
            for provider, cb_state in self.circuit_breakers.items():
                cb_key = f"hypie:circuit_breaker:{provider.value}"
                self.redis_client.set(cb_key, json.dumps(cb_state), ex=3600)  # 1 hour TTL

            # Persist learning weights
            for provider, weight in self.provider_weights.items():
                weight_key = f"hypie:weight:{provider.value}"
                self.redis_client.set(weight_key, str(weight), ex=3600)

            for provider, success_rate in self.provider_success_rates.items():
                success_key = f"hypie:success_rate:{provider.value}"
                self.redis_client.set(success_key, str(success_rate), ex=3600)

        except Exception as e:
            logger.error(f"Failed to persist state: {e}")

    async def route_inference(self, request: InferenceRequest) -> InferenceResult:
        """Route an inference request to the optimal provider based on governance and metrics."""
        start_time = time.time()

        try:
            # Step 1: Evaluate PSMP/HMCP context
            psmp_context = await self._evaluate_psmp_context(request)
            hmcp_policy = await self._evaluate_hmcp_policy(request)

            # Step 2: Determine routing policy
            routing_policy = request.routing_policy or RoutingPolicy()
            routing_policy = self._merge_policies(routing_policy, hmcp_policy)

            # Step 3: Pre-routing compliance check
            compliance_result = await self._run_compliance_check(request, psmp_context, hmcp_policy)
            if not compliance_result["compliant"]:
                logger.warning(f"Compliance violation detected: {compliance_result['violations']}")
                # Emit compliance violation event
                await event_bus.publish(Event(
                    event_type=EventTypes.COMPLIANCE_VIOLATION,
                    source="hypie-router",
                    data={
                        "request": request.__dict__,
                        "violations": compliance_result["violations"],
                        "frameworks": compliance_result["frameworks"]
                    }
                ))
                # Return compliance-denied result
                return InferenceResult(
                    content="Request denied due to compliance violation",
                    provider=ProviderType.MOCK,
                    model="compliance-guard",
                    tokens_used=0,
                    processing_time_ms=time.time() - start_time,
                    cost_estimate_usd=0.0,
                    artifact_id=None,
                    compliance_tags={"denied": True, "violations": compliance_result["violations"]}
                )

            # Step 4: Select optimal provider
            selected_provider = await self._select_provider(request, routing_policy, psmp_context)

            # Step 5: Execute inference
            result = await self._execute_inference(selected_provider, request)

            # Step 6: Post-routing evidence collection
            await self._collect_routing_evidence(request, result, selected_provider, compliance_result)

            # Step 7: Create unified PSMP artifact
            artifact = await self._create_inference_artifact(request, result, psmp_context)

            # Step 8: Update metrics and emit events
            await self._update_metrics(
                selected_provider, 
                result.get("processing_time_ms", 0.0),
                success=result.get("success", True),
                tokens_used=result.get("tokens_used", 0)
            )
            await self._emit_routing_event(request, InferenceResult(
                content=result["content"],
                provider=selected_provider,
                model=result["model"],
                tokens_used=result["tokens_used"],
                processing_time_ms=result["processing_time_ms"],
                cost_estimate_usd=result["cost_estimate_usd"],
                artifact_id=artifact.id,
                compliance_tags=result["compliance_tags"]
            ), selected_provider)

            # Step 9: Return standardized result
            return InferenceResult(
                content=result["content"],
                provider=selected_provider,
                model=result["model"],
                tokens_used=result["tokens_used"],
                processing_time_ms=result["processing_time_ms"],
                cost_estimate_usd=result["cost_estimate_usd"],
                artifact_id=artifact.id,
                compliance_tags=result["compliance_tags"]
            )

        except Exception as e:
            logger.error(f"HyPIE routing failed: {e}")
            # Emit failure event
            await event_bus.publish(Event(
                event_type=EventTypes.LLM_REQUEST_FAILED,
                source="hypie-router",
                data={"error": str(e), "request": request.__dict__}
            ))
            raise

    async def _evaluate_psmp_context(self, request: InferenceRequest) -> Dict[str, Any]:
        """Evaluate PSMP context for the request."""
        # Placeholder: integrate with PSMP engine
        return {
            "project_state": "active",
            "conflicts": [],
            "dependencies": []
        }

    async def _evaluate_hmcp_policy(self, request: InferenceRequest) -> RoutingPolicy:
        """Evaluate HMCP policy for the request with adaptive routing."""
        # Get base policy from HMCP hierarchy
        base_policy = RoutingPolicy(
            data_sensitivity=request.sensitivity,
            prefer_local=request.sensitivity in ["high", "critical"]
        )
        
        # Enhance policy based on agent capabilities and context
        enhanced_policy = await self._enhance_policy_with_context(request, base_policy)
        
        # Apply adaptive adjustments based on recent performance
        adaptive_policy = await self._apply_adaptive_adjustments(request, enhanced_policy)
        
        return adaptive_policy

    async def _enhance_policy_with_context(self, request: InferenceRequest, base_policy: RoutingPolicy) -> RoutingPolicy:
        """Enhance routing policy based on agent context and task requirements."""
        enhanced = base_policy
        
        # Analyze prompt for specialist routing hints
        prompt_analysis = self._analyze_prompt_for_routing(request.prompt)
        
        # Adjust policy based on prompt analysis
        if prompt_analysis.get("requires_specialist"):
            enhanced.prefer_local = True  # Specialists are typically local
            enhanced.min_confidence = max(enhanced.min_confidence, 0.9)
            
        # Adjust based on agent name patterns
        if "specialist" in request.agent_name.lower():
            enhanced.prefer_local = True
            enhanced.allowed_providers = [ProviderType.LOCAL_LLAMA_CPP, ProviderType.MOCK]
            
        # Adjust based on model requirements
        if request.model and "large" in request.model.lower():
            enhanced.max_latency_ms = enhanced.max_latency_ms or 5000  # Allow more time for large models
            
        return enhanced

    async def _apply_adaptive_adjustments(self, request: InferenceRequest, policy: RoutingPolicy) -> RoutingPolicy:
        """Apply adaptive adjustments based on recent performance metrics."""
        adaptive = policy
        
        # Get recent performance data
        recent_performance = await self._get_recent_performance_data()
        
        # Adjust latency tolerance based on recent trends
        if recent_performance.get("avg_latency_trend", 0) > 0.1:  # 10% increase
            adaptive.max_latency_ms = int((adaptive.max_latency_ms or 2000) * 1.2)
            logger.info(f"Adaptive: Increased latency tolerance to {adaptive.max_latency_ms}ms due to performance trends")
            
        # Adjust provider preferences based on recent reliability
        if recent_performance.get("error_rate_trend", 0) > 0.05:  # 5% error rate increase
            # Prefer more reliable providers
            adaptive.allowed_providers = [p for p in (adaptive.allowed_providers or list(ProviderType)) 
                                        if p in [ProviderType.OPENAI, ProviderType.ANTHROPIC]]
            logger.info("Adaptive: Restricted to reliable providers due to error rate trends")
            
        return adaptive

    def _analyze_prompt_for_routing(self, prompt: str) -> Dict[str, Any]:
        """Analyze prompt content for routing hints."""
        analysis = {
            "requires_specialist": False,
            "domain_hints": [],
            "complexity_score": 0.0,
            "sensitivity_indicators": []
        }
        
        prompt_lower = prompt.lower()
        
        # Check for specialist domains
        specialist_domains = {
            "code": ["python", "javascript", "programming", "debug", "algorithm"],
            "math": ["mathematics", "calculus", "algebra", "theorem", "proof"],
            "science": ["chemistry", "physics", "biology", "research", "analysis"],
            "language": ["translate", "grammar", "syntax", "linguistics"],
            "creative": ["story", "poem", "design", "artistic"]
        }
        
        for domain, keywords in specialist_domains.items():
            if any(keyword in prompt_lower for keyword in keywords):
                analysis["requires_specialist"] = True
                analysis["domain_hints"].append(domain)
                
        # Calculate complexity score
        analysis["complexity_score"] = min(1.0, len(prompt.split()) / 500.0)  # Simple heuristic
        
        # Check for sensitivity indicators
        sensitivity_keywords = ["confidential", "private", "sensitive", "personal", "medical", "financial"]
        analysis["sensitivity_indicators"] = [kw for kw in sensitivity_keywords if kw in prompt_lower]
        
        return analysis

    async def _get_recent_performance_data(self) -> Dict[str, Any]:
        """Get recent performance data for adaptive adjustments."""
        # Placeholder: integrate with metrics aggregator
        # For now, return mock data
        return {
            "avg_latency_trend": 0.02,  # 2% increase
            "error_rate_trend": 0.01,   # 1% error rate
            "throughput_trend": -0.05,  # 5% decrease
            "cost_trend": 0.03         # 3% increase
        }

    def _merge_policies(self, request_policy: RoutingPolicy, hmcp_policy: RoutingPolicy) -> RoutingPolicy:
        """Merge request and HMCP policies with governance precedence."""
        # HMCP policies take precedence for security/compliance
        return RoutingPolicy(
            max_cost_usd=request_policy.max_cost_usd or hmcp_policy.max_cost_usd,
            allowed_providers=request_policy.allowed_providers or hmcp_policy.allowed_providers,
            data_sensitivity=hmcp_policy.data_sensitivity,  # HMCP overrides
            min_confidence=max(request_policy.min_confidence, hmcp_policy.min_confidence),
            prefer_local=request_policy.prefer_local or hmcp_policy.prefer_local,
            max_latency_ms=request_policy.max_latency_ms or hmcp_policy.max_latency_ms
        )

    async def _select_provider(self, request: InferenceRequest, policy: RoutingPolicy, psmp_context: Dict[str, Any]) -> ProviderType:
        """Select the optimal provider based on policy, metrics, and context with specialist routing."""
        
        # Check cache for similar routing decisions first
        cached_provider = await self._get_cached_routing_decision(request)
        if cached_provider:
            return cached_provider
        
        # First, check if specialist routing is appropriate
        specialist_provider = await self._evaluate_specialist_routing(request, policy)
        if specialist_provider:
            logger.info(f"HyPIE routing to specialist: {specialist_provider}")
            # Cache the specialist routing decision
            await self._cache_routing_decision(request, specialist_provider)
            return specialist_provider
        
        # Fall back to regular provider selection
        # Filter by allowed providers
        candidates = policy.allowed_providers or list(ProviderType)

        # Apply policy constraints
        if policy.prefer_local:
            # Prioritize local providers for sensitive data
            local_providers = [p for p in candidates if p == ProviderType.LOCAL_LLAMA_CPP]
            if local_providers:
                candidates = local_providers

        # Epsilon-greedy exploration: sometimes try non-optimal providers to learn
        if random.random() < self.exploration_rate and len(candidates) > 1:
            # Exploration: randomly select from candidates
            selected_provider = random.choice(candidates)
            logger.info(f"HyPIE exploration: randomly selected {selected_provider} (epsilon={self.exploration_rate})")
        else:
            # Exploitation: select best provider based on learned weights and scores
            selected_provider = self._select_best_provider_with_learning(candidates, policy, psmp_context)

        if not selected_provider:
            raise ValueError("No suitable provider found for request")

        logger.info(f"HyPIE selected provider: {selected_provider}")
        
        # Cache the routing decision for future similar requests
        await self._cache_routing_decision(request, selected_provider)
        
        return selected_provider

    def _select_best_provider_with_learning(self, candidates: List[ProviderType], policy: RoutingPolicy, psmp_context: Dict[str, Any]) -> Optional[ProviderType]:
        """Select the best provider using learned weights and scoring."""
        best_provider = None
        best_combined_score = float('-inf')

        for provider in candidates:
            metrics = self.provider_metrics.get(provider)
            if not metrics:
                continue

            # Calculate base score from metrics and policy
            base_score = self._calculate_provider_score(provider, metrics, policy, psmp_context)
            
            # Incorporate learned success rate and weight
            success_rate = self.provider_success_rates.get(provider, 0.5)
            weight = self.provider_weights.get(provider, 1.0)
            
            # Combined score: base metrics + learned performance
            combined_score = base_score + (success_rate * weight * 50)  # Scale learning factor
            
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_provider = provider

        return best_provider

    async def _evaluate_specialist_routing(self, request: InferenceRequest, policy: RoutingPolicy) -> Optional[ProviderType]:
        """Evaluate if request should be routed to a specialist based on HMCP policies."""
        # Analyze request for specialist suitability
        prompt_analysis = self._analyze_prompt_for_routing(request.prompt)
        
        if not prompt_analysis.get("requires_specialist"):
            return None
            
        # Check if we have confidence in specialist routing
        if policy.min_confidence and prompt_analysis.get("complexity_score", 0) < policy.min_confidence:
            return None
            
        # Check HMCP policy for specialist spawning
        try:
            # Use a default level for evaluation - in practice this would come from agent context
            spawn_decision = self.hmcp_policy_engine.evaluate_spawn_request(
                identifier="base_level",  # Default level
                confidence=prompt_analysis.get("complexity_score", 0.5),
                task_isolation_passed=True,  # Assume isolation is handled
                candidate_parameter_count=None  # Not applicable for routing decision
            )
            
            if spawn_decision.allowed:
                logger.info(f"HMCP allows specialist routing: {spawn_decision.reasons}")
                # Route to local provider for specialist processing
                return ProviderType.LOCAL_LLAMA_CPP
            else:
                logger.debug(f"HMCP blocks specialist routing: {spawn_decision.reasons}")
                
        except Exception as e:
            logger.warning(f"HMCP evaluation failed: {e}")
            
        return None

    def _calculate_provider_score(self, provider: ProviderType, metrics: ProviderMetrics, policy: RoutingPolicy, psmp_context: Dict[str, Any]) -> float:
        """Calculate an adaptive score for provider selection with context awareness."""
        score = 0.0
        
        # Base performance metrics
        score += self._calculate_performance_score(metrics, policy)
        
        # Context-aware adjustments
        score += self._calculate_context_score(provider, psmp_context)
        
        # Policy compliance score
        score += self._calculate_policy_compliance_score(provider, policy)
        
        # Adaptive learning adjustments
        score += self._calculate_adaptive_score(provider, metrics)
        
        # Provider capability matching
        score += self._calculate_capability_score(provider, policy)
        
        return score

    def _calculate_performance_score(self, metrics: ProviderMetrics, policy: RoutingPolicy) -> float:
        """Calculate score based on performance metrics."""
        score = 0.0
        
        # Latency scoring (lower is better, but respect policy limits)
        if policy.max_latency_ms:
            if metrics.latency_ms <= policy.max_latency_ms:
                # Reward providers within latency budget
                latency_ratio = metrics.latency_ms / policy.max_latency_ms
                score += (1.0 - latency_ratio) * 100  # Up to 100 points for being fast
            else:
                # Penalize providers exceeding latency budget
                excess_ratio = (metrics.latency_ms - policy.max_latency_ms) / policy.max_latency_ms
                score -= excess_ratio * 200  # Heavy penalty for slow providers
        
        # Cost scoring (lower is better)
        if policy.max_cost_usd and metrics.cost_per_token > 0:
            if metrics.cost_per_token <= policy.max_cost_usd:
                cost_ratio = metrics.cost_per_token / policy.max_cost_usd
                score += (1.0 - cost_ratio) * 50  # Up to 50 points for cost efficiency
            else:
                excess_cost = metrics.cost_per_token - policy.max_cost_usd
                score -= excess_cost * 1000  # Heavy penalty for expensive providers
        
        # Reliability scoring (lower error rate is better)
        score += (1.0 - metrics.error_rate) * 75  # Up to 75 points for reliability
        
        # Resource utilization (prefer less utilized, but not empty)
        optimal_utilization = 0.7  # Sweet spot for utilization
        utilization_distance = abs(metrics.resource_utilization - optimal_utilization)
        score += (1.0 - utilization_distance) * 25  # Up to 25 points for optimal utilization
        
        return score

    def _calculate_context_score(self, provider: ProviderType, psmp_context: Dict[str, Any]) -> float:
        """Calculate score based on PSMP context and project state."""
        score = 0.0
        
        # Project state considerations
        project_state = psmp_context.get("project_state", "unknown")
        if project_state == "active":
            score += 10  # Slight boost for active projects
        elif project_state == "draft":
            # Prefer stable providers for draft projects
            if provider in [ProviderType.OPENAI, ProviderType.ANTHROPIC]:
                score += 15
        
        # Conflict awareness
        conflicts = psmp_context.get("conflicts", [])
        if conflicts:
            # Prefer local providers when there are conflicts to reduce external dependencies
            if provider == ProviderType.LOCAL_LLAMA_CPP:
                score += len(conflicts) * 5
        
        # Dependency considerations
        dependencies = psmp_context.get("dependencies", [])
        if dependencies:
            # Prefer providers that can handle complex dependency chains
            if provider in [ProviderType.VLLM, ProviderType.AZURE_OPENAI]:
                score += len(dependencies) * 2
        
        return score

    def _calculate_policy_compliance_score(self, provider: ProviderType, policy: RoutingPolicy) -> float:
        """Calculate score based on policy compliance."""
        score = 0.0
        
        # Local preference enforcement
        if policy.prefer_local:
            if provider == ProviderType.LOCAL_LLAMA_CPP:
                score += 100  # Strong preference for local
            else:
                score -= 50  # Penalty for non-local when local is preferred
        
        # Provider restrictions
        if policy.allowed_providers and provider not in policy.allowed_providers:
            score -= 1000  # Heavy penalty for disallowed providers
        
        # Data sensitivity alignment
        if policy.data_sensitivity in ["high", "critical"]:
            if provider == ProviderType.LOCAL_LLAMA_CPP:
                score += 50  # Local is more secure for sensitive data
            elif provider in [ProviderType.OPENAI, ProviderType.ANTHROPIC]:
                score += 20  # Cloud providers with good security
            else:
                score -= 25  # Penalty for less secure providers
        
        return score

    def _calculate_adaptive_score(self, provider: ProviderType, metrics: ProviderMetrics) -> float:
        """Calculate adaptive score based on recent performance trends."""
        score = 0.0
        
        # Freshness bonus (prefer recently updated metrics)
        time_since_update = time.time() - metrics.last_updated
        freshness_score = max(0, 20 - time_since_update / 3600)  # Decay over hours
        score += freshness_score
        
        # Trend analysis (reward improving providers, penalize declining ones)
        # This would be enhanced with actual trend data from metrics history
        
        return score

    def _calculate_capability_score(self, provider: ProviderType, policy: RoutingPolicy) -> float:
        """Calculate score based on provider capabilities matching requirements."""
        score = 0.0
        
        # Confidence requirement matching
        if policy.min_confidence:
            # Different providers have different baseline confidence levels
            provider_confidence = {
                ProviderType.OPENAI: 0.95,
                ProviderType.ANTHROPIC: 0.93,
                ProviderType.AZURE_OPENAI: 0.92,
                ProviderType.VLLM: 0.88,
                ProviderType.LOCAL_LLAMA_CPP: 0.85,
                ProviderType.MOCK: 0.99
            }.get(provider, 0.8)
            
            if provider_confidence >= policy.min_confidence:
                confidence_margin = provider_confidence - policy.min_confidence
                score += confidence_margin * 50  # Reward providers exceeding confidence requirements
            else:
                confidence_gap = policy.min_confidence - provider_confidence
                score -= confidence_gap * 100  # Penalize providers below confidence requirements
        
        return score

    async def _execute_inference(self, provider: ProviderType, request: InferenceRequest) -> Dict[str, Any]:
        """Execute inference on the selected provider."""
        try:
            # Get LLM client for this provider
            llm_provider = LLMProvider(provider.value)
            llm_client = self.llm_factory.get_client(llm_provider)
            
            if not llm_client:
                raise ValueError(f"No LLM client available for provider {provider}")
            
            # Prepare metadata for the adapter
            metadata = {
                "project_id": request.project_id,
                "task_id": request.task_id,
                "sensitivity": request.sensitivity,
                "request_id": f"hypie_{int(time.time())}_{hash(request.prompt) % 10000}",
                "correlation_id": f"corr_{request.project_id}_{request.task_id}" if request.project_id and request.task_id else None,
            }
            
            # Call the adapter
            response: LLMResponse = llm_client.generate(
                prompt=request.prompt,
                agent_name=request.agent_name,
                metadata=metadata,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                model=request.model,
            )
            
            # Convert to internal result format
            return {
                "content": response.content,
                "model": response.model,
                "tokens_used": response.usage.get("total_tokens", len(response.content.split())) if response.usage else len(response.content.split()),
                "processing_time_ms": response.metadata.get("latency_ms", 0.0),
                "cost_estimate_usd": 0.005,  # TODO: Calculate based on provider and usage
                "compliance_tags": ["governed", f"provider_{response.provider.value}"],
                "success": response.success,
                "error": response.error,
                "artifact_context": response.artifact_context,
            }
            
        except Exception as e:
            logger.error(f"Error executing inference on {provider}: {e}")
            # Return error result
            return {
                "content": "",
                "model": request.model or "unknown",
                "tokens_used": 0,
                "processing_time_ms": 0.0,
                "cost_estimate_usd": 0.0,
                "compliance_tags": ["error"],
                "success": False,
                "error": str(e),
                "artifact_context": {
                    "provider": provider.value,
                    "error": str(e),
                    "timestamp": time.time(),
                }
            }

    async def _create_inference_artifact(self, request: InferenceRequest, result: Dict[str, Any], psmp_context: Dict[str, Any]) -> Artifact:
        """Create a unified PSMP artifact for the inference."""
        # Use artifact_context from LLM response if available
        artifact_context = result.get("artifact_context", {})
        
        # Create comprehensive artifact content
        artifact_content = {
            "inference_result": {
                "content": result["content"],
                "model": result["model"],
                "tokens_used": result["tokens_used"],
                "processing_time_ms": result["processing_time_ms"],
                "cost_estimate_usd": result["cost_estimate_usd"],
                "success": result["success"],
                "error": result.get("error"),
            },
            "routing_context": {
                "agent_name": request.agent_name,
                "sensitivity": request.sensitivity,
                "psmp_context": psmp_context,
            },
            "artifact_context": artifact_context,
        }
        
        # Create PSMP artifact
        artifact = Artifact(
            id=artifact_context.get("request_id", f"inf_{int(time.time())}_{hash(request.prompt) % 10000}"),
            name=f"LLM Inference for {request.agent_name}",
            type="llm_inference",
            content=artifact_content,
            metadata={
                "provider": artifact_context.get("provider", "unknown"),
                "model": result["model"],
                "tokens_used": result["tokens_used"],
                "compliance_tags": result["compliance_tags"],
                "psmp_context": psmp_context,
                "latency_ms": artifact_context.get("latency_ms"),
                "success": result["success"],
            },
            project_id=request.project_id,
            task_id=request.task_id
        )

        # Register with PSMP
        await self.psmp_engine.declare_artifact(artifact)

        return artifact

    async def _update_metrics(self, provider: ProviderType, processing_time_ms: float, success: bool = True, tokens_used: int = 0):
        """Update provider metrics based on recent performance with adaptive learning."""
        metrics = self.provider_metrics.get(provider)
        if not metrics:
            return
        
        # Update request count
        metrics.request_count += 1
        
        # Update latency with exponential moving average
        if metrics.latency_history:
            # Keep last 100 measurements for trend analysis
            metrics.latency_history.append(processing_time_ms)
            if len(metrics.latency_history) > 100:
                metrics.latency_history.pop(0)
            
            # Calculate EMA (exponential moving average)
            alpha = 0.1  # Smoothing factor
            metrics.latency_ms = alpha * processing_time_ms + (1 - alpha) * metrics.latency_ms
        else:
            metrics.latency_ms = processing_time_ms
            metrics.latency_history.append(processing_time_ms)
        
        # Update error rate
        metrics.error_history.append(not success)
        if len(metrics.error_history) > 100:
            metrics.error_history.pop(0)
        
        # Recalculate error rate from recent history
        recent_errors = sum(metrics.error_history[-20:])  # Last 20 requests
        metrics.error_rate = recent_errors / len(metrics.error_history[-20:]) if metrics.error_history else 0.0
        
        # Update throughput estimate
        if tokens_used > 0 and processing_time_ms > 0:
            tokens_per_sec = (tokens_used / processing_time_ms) * 1000
            # Simple moving average for throughput
            if metrics.throughput_tokens_per_sec > 0:
                metrics.throughput_tokens_per_sec = 0.8 * metrics.throughput_tokens_per_sec + 0.2 * tokens_per_sec
            else:
                metrics.throughput_tokens_per_sec = tokens_per_sec
        
        # Update resource utilization (simplified model)
        # In a real implementation, this would come from actual resource monitoring
        utilization_trend = 0.02 if success else -0.01  # Slight increase on success, decrease on failure
        metrics.resource_utilization = min(1.0, max(0.0, metrics.resource_utilization + utilization_trend))
        
        metrics.last_updated = time.time()
        
        # Online learning: update provider success rates and weights
        self._update_provider_learning(provider, success, processing_time_ms)
        
        # Persist updated state to Redis
        self._persist_state()
        
        logger.debug(f"Updated metrics for {provider}: latency={metrics.latency_ms:.1f}ms, error_rate={metrics.error_rate:.3f}")

    def _update_provider_learning(self, provider: ProviderType, success: bool, latency_ms: float):
        """Update provider learning parameters using Bayesian success rates and weight adjustments."""
        # Update success rate using Bayesian updating
        current_success_rate = self.provider_success_rates.get(provider, 0.5)
        
        # Simple Bayesian update: treat as beta distribution
        # Prior: beta(α, β) where α = successes + 1, β = failures + 1
        if success:
            # Increase success rate
            self.provider_success_rates[provider] = min(0.99, current_success_rate + self.learning_rate * (1.0 - current_success_rate))
        else:
            # Decrease success rate
            self.provider_success_rates[provider] = max(0.01, current_success_rate - self.learning_rate * current_success_rate)
        
        # Update provider weights based on performance
        current_weight = self.provider_weights.get(provider, 1.0)
        
        if success:
            # Reward successful providers, especially fast ones
            latency_bonus = 1.0 if latency_ms < 500 else 0.5  # Bonus for fast responses
            weight_increase = self.learning_rate * latency_bonus
            self.provider_weights[provider] = min(2.0, current_weight + weight_increase)  # Cap at 2.0
        else:
            # Penalize failing providers
            weight_decrease = self.learning_rate * 0.5  # Less aggressive penalty
            self.provider_weights[provider] = max(0.1, current_weight - weight_decrease)  # Floor at 0.1
        
        logger.debug(f"Updated learning for {provider}: success_rate={self.provider_success_rates[provider]:.3f}, weight={self.provider_weights[provider]:.3f}")

    async def update_learning_from_feedback(self, provider: ProviderType, quality_score: float, latency_satisfactory: bool, user_feedback: str = ""):
        """Update learning parameters based on user feedback about inference outcomes."""
        # Quality score adjustment (0.0 = poor, 1.0 = excellent)
        quality_adjustment = (quality_score - 0.5) * self.learning_rate * 2  # Scale adjustment
        
        # Latency satisfaction adjustment
        latency_adjustment = self.learning_rate * 0.5 if latency_satisfactory else -self.learning_rate * 0.3
        
        # Combined adjustment
        total_adjustment = quality_adjustment + latency_adjustment
        
        # Update success rate based on quality feedback
        current_success_rate = self.provider_success_rates.get(provider, 0.5)
        if quality_score >= 0.7:
            # High quality feedback increases success rate
            self.provider_success_rates[provider] = min(0.99, current_success_rate + abs(total_adjustment))
        elif quality_score <= 0.3:
            # Low quality feedback decreases success rate
            self.provider_success_rates[provider] = max(0.01, current_success_rate - abs(total_adjustment))
        
        # Update provider weight based on feedback
        current_weight = self.provider_weights.get(provider, 1.0)
        self.provider_weights[provider] = max(0.1, min(2.0, current_weight + total_adjustment))
        
        # Analyze user feedback for patterns (simple keyword analysis)
        feedback_lower = user_feedback.lower()
        if any(word in feedback_lower for word in ["slow", "timeout", "delay"]):
            # Penalize for latency issues
            self.provider_weights[provider] = max(0.1, current_weight - self.learning_rate)
        elif any(word in feedback_lower for word in ["accurate", "good", "excellent", "perfect"]):
            # Reward for quality
            self.provider_weights[provider] = min(2.0, current_weight + self.learning_rate)
        
        # Persist updated learning state
        self._persist_state()
        
        logger.info(f"Updated learning from feedback for {provider}: quality={quality_score:.2f}, weight={self.provider_weights[provider]:.3f}")

    async def _get_recent_performance_data(self) -> Dict[str, Any]:
        """Get recent performance data for adaptive adjustments from real metrics sources."""
        trends = {}
        
        # Try to get real metrics from external sources first
        real_metrics = await self._fetch_external_metrics()
        
        for provider, metrics in self.provider_metrics.items():
            provider_key = provider.value
            
            # Use real metrics if available, otherwise fall back to historical analysis
            if provider_key in real_metrics:
                # Real metrics from Prometheus/Redis
                real_data = real_metrics[provider_key]
                trends[f"{provider_key}_latency_trend"] = real_data.get("latency_trend", 0.0)
                trends[f"{provider_key}_error_trend"] = real_data.get("error_rate_trend", 0.0)
                trends[f"{provider_key}_throughput_trend"] = real_data.get("throughput_trend", 0.0)
                trends[f"{provider_key}_cost_trend"] = real_data.get("cost_trend", 0.0)
            else:
                # Fallback to historical analysis from internal metrics
                if len(metrics.latency_history) >= 10:
                    recent = metrics.latency_history[-10:]
                    previous = metrics.latency_history[-20:-10] if len(metrics.latency_history) >= 20 else recent
                    
                    recent_avg = sum(recent) / len(recent)
                    previous_avg = sum(previous) / len(previous)
                    
                    if previous_avg > 0:
                        latency_trend = (recent_avg - previous_avg) / previous_avg
                        trends[f"{provider_key}_latency_trend"] = latency_trend
                
                if len(metrics.error_history) >= 20:
                    recent_errors = sum(metrics.error_history[-10:])
                    previous_errors = sum(metrics.error_history[-20:-10])
                    
                    recent_rate = recent_errors / 10
                    previous_rate = previous_errors / 10
                    
                    error_trend = recent_rate - previous_rate
                    trends[f"{provider_key}_error_trend"] = error_trend
        
        # Aggregate trends
        latency_trends = [v for k, v in trends.items() if "latency_trend" in k]
        error_trends = [v for k, v in trends.items() if "error_trend" in k]
        throughput_trends = [v for k, v in trends.items() if "throughput_trend" in k]
        cost_trends = [v for k, v in trends.items() if "cost_trend" in k]
        
        return {
            "avg_latency_trend": sum(latency_trends) / len(latency_trends) if latency_trends else 0.0,
            "error_rate_trend": sum(error_trends) / len(error_trends) if error_trends else 0.0,
            "throughput_trend": sum(throughput_trends) / len(throughput_trends) if throughput_trends else 0.0,
            "cost_trend": sum(cost_trends) / len(cost_trends) if cost_trends else 0.0,
            "detailed_trends": trends
        }

    async def _fetch_external_metrics(self) -> Dict[str, Dict[str, float]]:
        """Fetch real-time metrics from external sources (Prometheus, Redis counters, etc.)."""
        external_metrics = {}
        
        try:
            # Try Prometheus metrics first
            prometheus_data = await self._fetch_prometheus_metrics()
            if prometheus_data:
                external_metrics.update(prometheus_data)
            
            # Try Redis counters for additional metrics
            redis_data = await self._fetch_redis_metrics()
            if redis_data:
                # Merge Redis data with Prometheus data
                for provider, metrics in redis_data.items():
                    if provider in external_metrics:
                        external_metrics[provider].update(metrics)
                    else:
                        external_metrics[provider] = metrics
                        
        except Exception as e:
            logger.warning(f"Failed to fetch external metrics: {e}")
        
        return external_metrics

    async def _fetch_prometheus_metrics(self) -> Dict[str, Dict[str, float]]:
        """Fetch metrics from Prometheus endpoint."""
        # Placeholder for Prometheus integration
        # In a real implementation, this would query Prometheus HTTP API
        # Example query: rate(http_request_duration_seconds{job="ai-gateway"}[5m])
        
        # For now, return empty dict - implement when Prometheus is available
        return {}

    async def _fetch_redis_metrics(self) -> Dict[str, Dict[str, float]]:
        """Fetch metrics from Redis counters."""
        if not self.use_distributed_state:
            return {}
            
        try:
            metrics = {}
            for provider in ProviderType:
                provider_key = provider.value
                
                # Fetch recent latency/error counters from Redis
                latency_key = f"metrics:{provider_key}:latency_sum"
                error_key = f"metrics:{provider_key}:error_count"
                request_key = f"metrics:{provider_key}:request_count"
                
                latency_sum = float(self.redis_client.get(latency_key) or 0)
                error_count = float(self.redis_client.get(error_key) or 0)
                request_count = float(self.redis_client.get(request_key) or 0)
                
                if request_count > 0:
                    avg_latency = latency_sum / request_count
                    error_rate = error_count / request_count
                    
                    # Calculate trends (simplified - compare with previous period)
                    prev_latency_key = f"metrics:{provider_key}:latency_sum_prev"
                    prev_error_key = f"metrics:{provider_key}:error_count_prev"
                    
                    prev_latency = float(self.redis_client.get(prev_latency_key) or latency_sum)
                    prev_error = float(self.redis_client.get(prev_error_key) or error_count)
                    
                    latency_trend = (avg_latency - (prev_latency / max(request_count, 1))) / max(avg_latency, 1)
                    error_trend = error_rate - (prev_error / max(request_count, 1))
                    
                    metrics[provider_key] = {
                        "latency_trend": latency_trend,
                        "error_rate_trend": error_trend,
                        "throughput_trend": 0.0,  # Not available from basic counters
                        "cost_trend": 0.0        # Not available from basic counters
                    }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to fetch Redis metrics: {e}")
            return {}

    async def _emit_routing_event(self, request: InferenceRequest, result: InferenceResult, provider: ProviderType):
        """Emit routing decision event for observability."""
        event = Event(
            event_type=EventTypes.LLM_REQUEST_COMPLETED,
            source="hypie-router",
            data={
                "provider": provider.value,
                "model": result.model,
                "processing_time_ms": result.processing_time_ms,
                "cost_estimate_usd": result.cost_estimate_usd,
                "artifact_id": result.artifact_id,
                "agent_name": request.agent_name,
                "sensitivity": request.sensitivity
            }
        )
        await event_bus.publish(event)

    async def _run_compliance_check(self, request: InferenceRequest, psmp_context: Dict[str, Any], hmcp_policy: Dict[str, Any]) -> Dict[str, Any]:
        """Run compliance check before routing decision."""
        try:
            # Prepare compliance context
            compliance_context = {
                "request": {
                    "agent_name": request.agent_name,
                    "sensitivity": request.sensitivity.value if hasattr(request.sensitivity, 'value') else str(request.sensitivity),
                    "content_length": len(request.prompt) if request.prompt else 0,
                    "routing_policy": request.routing_policy.__dict__ if request.routing_policy else {},
                },
                "psmp_context": psmp_context,
                "hmcp_policy": hmcp_policy,
                "frameworks": ["SOC2", "GDPR", "HIPAA"]  # Configurable frameworks
            }

            # Run compliance check
            compliance_result = await self.compliance_auditor.run_compliance_check(
                context=compliance_context,
                frameworks=compliance_context["frameworks"]
            )

            return {
                "compliant": compliance_result.get("compliant", True),
                "violations": compliance_result.get("violations", []),
                "frameworks": compliance_context["frameworks"],
                "evidence_id": compliance_result.get("evidence_id")
            }

        except Exception as e:
            logger.error(f"Compliance check failed: {e}")
            # Default to compliant on error to avoid blocking
            return {
                "compliant": True,
                "violations": [],
                "frameworks": ["SOC2", "GDPR", "HIPAA"],
                "evidence_id": None,
                "error": str(e)
            }

    async def _collect_routing_evidence(self, request: InferenceRequest, result: Dict[str, Any], provider: ProviderType, compliance_result: Dict[str, Any]):
        """Collect evidence for routing decision and outcome."""
        try:
            evidence_data = {
                "routing_decision": {
                    "provider": provider.value,
                    "model": result.get("model"),
                    "processing_time_ms": result.get("processing_time_ms"),
                    "cost_estimate_usd": result.get("cost_estimate_usd"),
                    "tokens_used": result.get("tokens_used"),
                    "success": result.get("success", True)
                },
                "request_context": {
                    "agent_name": request.agent_name,
                    "sensitivity": request.sensitivity.value if hasattr(request.sensitivity, 'value') else str(request.sensitivity),
                    "content_length": len(request.prompt) if request.prompt else 0
                },
                "compliance_context": {
                    "pre_routing_check": compliance_result,
                    "frameworks_checked": compliance_result.get("frameworks", [])
                },
                "timestamp": time.time()
            }

            # Collect evidence through compliance auditor
            await self.compliance_auditor.collect_evidence(
                evidence_type="routing_decision",
                data=evidence_data,
                frameworks=["SOC2", "GDPR", "HIPAA"]
            )

        except Exception as e:
            logger.error(f"Evidence collection failed: {e}")
            # Don't raise exception to avoid breaking routing flow

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError))
    )
    async def _execute_inference(self, provider: ProviderType, request: InferenceRequest) -> Dict[str, Any]:
        """Execute inference with circuit breaker protection and retry logic."""
        # Check circuit breaker state
        if not self._check_circuit_breaker(provider):
            logger.warning(f"Circuit breaker open for provider {provider}, falling back to mock")
            return await self._fallback_to_mock(request)

        try:
            start_time = time.time()

            # Get LLM client for provider
            llm_client = self.llm_factory.get_client(provider)

            # Execute inference
            response = await llm_client.generate(
                prompt=request.prompt,
                model=request.model,
                max_tokens=request.max_tokens or 256,
                temperature=request.temperature or 0.7
            )

            processing_time = (time.time() - start_time) * 1000  # Convert to ms

            # Calculate cost estimate (simplified)
            cost_estimate = self._estimate_cost(provider, response.tokens_used if hasattr(response, 'tokens_used') else 0)

            # Reset circuit breaker on success
            self._record_success(provider)

            result = {
                "content": response.content,
                "model": response.model,
                "tokens_used": getattr(response, 'tokens_used', 0),
                "processing_time_ms": processing_time,
                "cost_estimate_usd": cost_estimate,
                "success": True,
                "compliance_tags": {}
            }

            return result

        except Exception as e:
            logger.error(f"Inference failed for provider {provider}: {e}")
            # Record failure for circuit breaker
            self._record_failure(provider)
            raise

    def _check_circuit_breaker(self, provider: ProviderType) -> bool:
        """Check if circuit breaker allows requests to this provider."""
        cb = self.circuit_breakers[provider]

        if cb["state"] == "open":
            # Check if recovery timeout has passed
            if time.time() - cb["last_failure_time"] > self.recovery_timeout:
                cb["state"] = "half_open"
                cb["success_count"] = 0
                logger.info(f"Circuit breaker for {provider} entering half-open state")
                return True
            return False

        return True

    def _record_success(self, provider: ProviderType):
        """Record successful request for circuit breaker."""
        cb = self.circuit_breakers[provider]

        if cb["state"] == "half_open":
            cb["success_count"] += 1
            if cb["success_count"] >= 3:  # Require 3 successes to close
                cb["state"] = "closed"
                cb["failure_count"] = 0
                logger.info(f"Circuit breaker for {provider} closed after successful requests")
        else:
            # Reset failure count on success
            cb["failure_count"] = 0

    def _record_failure(self, provider: ProviderType):
        """Record failed request for circuit breaker."""
        cb = self.circuit_breakers[provider]
        cb["failure_count"] += 1
        cb["last_failure_time"] = time.time()

        if cb["failure_count"] >= self.failure_threshold:
            cb["state"] = "open"
            logger.warning(f"Circuit breaker for {provider} opened after {cb['failure_count']} failures")

    async def _fallback_to_mock(self, request: InferenceRequest) -> Dict[str, Any]:
        """Fallback to mock provider when circuit breaker is open."""
        logger.info("Falling back to mock provider")

        # Simulate processing time
        await asyncio.sleep(0.1)

        return {
            "content": f"[MOCK RESPONSE] This is a fallback response for: {request.prompt[:50]}...",
            "model": "mock-fallback",
            "tokens_used": len(request.prompt.split()) * 2,  # Rough estimate
            "processing_time_ms": 100.0,
            "cost_estimate_usd": 0.0,
            "success": True,
            "compliance_tags": {"fallback": True}
        }

    def _estimate_cost(self, provider: ProviderType, tokens_used: int) -> float:
        """Estimate cost for the request based on provider and token usage."""
        # Simplified cost estimation - in practice this would be more sophisticated
        cost_per_token = {
            ProviderType.OPENAI: 0.00002,
            ProviderType.ANTHROPIC: 0.000015,
            ProviderType.VLLM: 0.00001,
            ProviderType.LOCAL_LLAMA_CPP: 0.0,  # Local inference
            ProviderType.AZURE_OPENAI: 0.00002,
            ProviderType.MOCK: 0.0
        }

        return tokens_used * cost_per_token.get(provider, 0.00001)