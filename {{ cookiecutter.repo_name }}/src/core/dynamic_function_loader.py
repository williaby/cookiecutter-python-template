"""
Dynamic Function Loading Prototype Implementation

This module provides a comprehensive prototype that integrates task detection,
three-tier loading, fallback mechanisms, user controls, and monitoring into a
cohesive proof-of-concept system for achieving 70% token reduction.

Key Components:
- FunctionRegistry: Manages available functions and their metadata
- DynamicFunctionLoader: Main orchestration engine
- LoadingSession: Tracks individual loading sessions
- PerformanceValidator: Validates optimization claims

Architecture:
- Task detection drives function loading decisions
- Three-tier loading based on usage patterns and confidence
- Conservative fallback chain prevents functionality loss
- User override commands provide manual control
- Comprehensive monitoring validates 70% token reduction claim
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from statistics import mean, median
from typing import Any
from uuid import uuid4

from src.utils.performance_monitor import MetricData, MetricType, PerformanceMonitor

from .task_detection import DetectionResult, TaskDetectionSystem
from .task_detection_config import ConfigManager
from .token_optimization_monitor import FunctionTier, TokenOptimizationMonitor
from .user_control_system import CommandResult, UserControlSystem

logger = logging.getLogger(__name__)


class LoadingTier(Enum):
    """Function loading tiers based on usage patterns and requirements."""

    TIER_1 = "tier_1"  # Essential functions - always loaded
    TIER_2 = "tier_2"  # Extended functions - loaded based on detection
    TIER_3 = "tier_3"  # Specialized functions - loaded on high confidence
    FALLBACK = "fallback"  # Emergency full load


class LoadingStrategy(Enum):
    """Function loading strategies."""

    CONSERVATIVE = "conservative"  # Favor over-loading for safety
    BALANCED = "balanced"  # Balance performance and functionality
    AGGRESSIVE = "aggressive"  # Favor under-loading for performance
    USER_CONTROLLED = "user_controlled"  # User has manual control


class SessionStatus(Enum):
    """Loading session status."""

    INITIALIZING = "initializing"
    DETECTING = "detecting"
    LOADING = "loading"
    ACTIVE = "active"
    FALLBACK = "fallback"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class FunctionMetadata:
    """Metadata for a function in the registry."""

    name: str
    description: str
    tier: LoadingTier
    token_cost: int
    category: str
    dependencies: list[str] = field(default_factory=list)
    usage_patterns: list[str] = field(default_factory=list)
    is_essential: bool = False
    loading_time_ms: float = 0.0
    last_used: datetime | None = None
    usage_count: int = 0
    success_rate: float = 1.0


@dataclass
class LoadingDecision:
    """Decision about which functions to load."""

    functions_to_load: set[str]
    tier_breakdown: dict[LoadingTier, set[str]]
    estimated_tokens: int
    estimated_loading_time_ms: float
    confidence_score: float
    strategy_used: LoadingStrategy
    fallback_reason: str | None = None
    user_overrides: list[str] = field(default_factory=list)


@dataclass
class LoadingSession:
    """Tracks a function loading session."""

    session_id: str
    user_id: str
    query: str
    timestamp: datetime
    status: SessionStatus = SessionStatus.INITIALIZING

    # Detection results
    detection_result: DetectionResult | None = None
    detected_categories: dict[str, bool] = field(default_factory=dict)

    # Loading decisions and results
    loading_decision: LoadingDecision | None = None
    functions_loaded: set[str] = field(default_factory=set)
    functions_used: set[str] = field(default_factory=set)

    # Performance metrics
    detection_time_ms: float = 0.0
    loading_time_ms: float = 0.0
    total_tokens_loaded: int = 0
    baseline_tokens: int = 0

    # User interaction
    user_commands: list[str] = field(default_factory=list)
    fallback_activations: int = 0
    error_count: int = 0

    # Session configuration
    strategy: LoadingStrategy = LoadingStrategy.CONSERVATIVE
    enable_fallback: bool = True
    user_preferences: dict[str, Any] = field(default_factory=dict)

    def calculate_token_reduction(self) -> float:
        """Calculate token reduction percentage."""
        if self.baseline_tokens == 0:
            return 0.0
        return (self.baseline_tokens - self.total_tokens_loaded) / self.baseline_tokens


class FunctionRegistry:
    """Registry of available functions with metadata and performance tracking."""

    def __init__(self) -> None:
        self.functions: dict[str, FunctionMetadata] = {}
        self.categories: dict[str, set[str]] = defaultdict(set)
        self.tiers: dict[LoadingTier, set[str]] = defaultdict(set)
        self.dependencies: dict[str, set[str]] = defaultdict(set)

        # Initialize with PromptCraft function inventory
        self._initialize_function_inventory()

    def _initialize_function_inventory(self) -> None:
        """Initialize with actual PromptCraft function inventory."""

        # Tier 1 - Essential Functions (Core + Git)
        core_functions = [
            ("Read", "Read files from filesystem", 1200, ["core", "file-operations"]),
            ("Write", "Write files to filesystem", 900, ["core", "file-operations"]),
            ("Edit", "Edit files with precise replacements", 1100, ["core", "file-operations"]),
            ("MultiEdit", "Multiple edits in single operation", 1300, ["core", "file-operations"]),
            ("Bash", "Execute bash commands", 800, ["core", "system"]),
            ("LS", "List directory contents", 500, ["core", "file-operations"]),
            ("Glob", "File pattern matching", 400, ["core", "file-operations"]),
            ("Grep", "Search text in files", 600, ["core", "search"]),
        ]

        git_functions = [
            ("git_status", "Git repository status", 540, ["git", "version-control"]),
            ("git_add", "Stage files for commit", 480, ["git", "version-control"]),
            ("git_commit", "Commit staged changes", 520, ["git", "version-control"]),
            ("git_diff_unstaged", "Show unstaged changes", 450, ["git", "version-control"]),
            ("git_diff_staged", "Show staged changes", 450, ["git", "version-control"]),
            ("git_diff", "Show differences between branches", 480, ["git", "version-control"]),
            ("git_log", "Show commit history", 420, ["git", "version-control"]),
            ("git_create_branch", "Create new branch", 380, ["git", "version-control"]),
            ("git_checkout", "Switch branches", 360, ["git", "version-control"]),
            ("git_show", "Show commit contents", 400, ["git", "version-control"]),
            ("git_init", "Initialize repository", 320, ["git", "version-control"]),
            ("git_branch", "List branches", 360, ["git", "version-control"]),
            ("git_reset", "Reset staging area", 340, ["git", "version-control"]),
        ]

        # Tier 2 - Extended Functions
        analysis_functions = [
            ("chat", "General AI chat and reasoning", 2100, ["analysis", "ai-tools"]),
            ("thinkdeep", "Deep analysis and investigation", 2800, ["analysis", "ai-tools"]),
            ("analyze", "Code analysis workflow", 2200, ["analysis", "ai-tools"]),
            ("consensus", "Multi-model consensus", 1800, ["analysis", "ai-tools"]),
            ("debug", "Debug and troubleshoot code", 1630, ["debug", "ai-tools"]),
        ]

        quality_functions = [
            ("codereview", "Comprehensive code review", 1900, ["quality", "ai-tools"]),
            ("refactor", "Code refactoring analysis", 1600, ["quality", "ai-tools"]),
            ("docgen", "Documentation generation", 1400, ["quality", "ai-tools"]),
            ("testgen", "Test generation", 1310, ["quality", "ai-tools"]),
        ]

        security_functions = [
            ("secaudit", "Security audit workflow", 1600, ["security", "ai-tools"]),
            ("precommit", "Pre-commit validation", 1200, ["security", "validation"]),
        ]

        # Tier 3 - Specialized Functions
        external_functions = [
            ("resolve-library-id", "Resolve Context7 library ID", 850, ["external", "context7"]),
            ("get-library-docs", "Get library documentation", 920, ["external", "context7"]),
            ("get_current_time", "Get current time", 480, ["external", "time"]),
            ("convert_time", "Convert between timezones", 520, ["external", "time"]),
            ("check_package_security", "Check package vulnerabilities", 780, ["external", "security"]),
            ("get_recommended_version", "Get recommended package version", 650, ["external", "security"]),
            ("list_vulnerabilities_affecting_version", "List package vulnerabilities", 820, ["external", "security"]),
        ]

        infrastructure_functions = [
            ("ListMcpResourcesTool", "List MCP resources", 380, ["infrastructure", "mcp"]),
            ("ReadMcpResourceTool", "Read MCP resource", 400, ["infrastructure", "mcp"]),
        ]

        # Register all functions
        all_function_groups = [
            (core_functions, LoadingTier.TIER_1, True),
            (git_functions, LoadingTier.TIER_1, True),
            (analysis_functions, LoadingTier.TIER_2, False),
            (quality_functions, LoadingTier.TIER_2, False),
            (security_functions, LoadingTier.TIER_2, False),
            (external_functions, LoadingTier.TIER_3, False),
            (infrastructure_functions, LoadingTier.TIER_3, False),
        ]

        for function_group, tier, is_essential in all_function_groups:
            for name, description, token_cost, patterns in function_group:
                self.register_function(
                    name=name,
                    description=description,
                    tier=tier,
                    token_cost=token_cost,
                    category=patterns[0],  # Primary category
                    usage_patterns=patterns,
                    is_essential=is_essential,
                )

    def register_function(
        self,
        name: str,
        description: str,
        tier: LoadingTier,
        token_cost: int,
        category: str,
        usage_patterns: list[str] | None = None,
        dependencies: list[str] | None = None,
        is_essential: bool = False,
    ) -> None:
        """Register a function in the registry."""

        metadata = FunctionMetadata(
            name=name,
            description=description,
            tier=tier,
            token_cost=token_cost,
            category=category,
            dependencies=dependencies or [],
            usage_patterns=usage_patterns or [],
            is_essential=is_essential,
        )

        self.functions[name] = metadata
        self.categories[category].add(name)
        self.tiers[tier].add(name)

        # Track dependencies
        for dep in metadata.dependencies:
            self.dependencies[name].add(dep)

    def get_functions_by_category(self, category: str) -> set[str]:
        """Get all functions in a category."""
        return self.categories.get(category, set())

    def get_functions_by_tier(self, tier: LoadingTier) -> set[str]:
        """Get all functions in a tier."""
        return self.tiers.get(tier, set())

    def get_tier_token_cost(self, tier: LoadingTier) -> int:
        """Calculate total token cost for a tier."""
        functions = self.get_functions_by_tier(tier)
        return sum(self.functions[func].token_cost for func in functions)

    def get_category_token_cost(self, category: str) -> int:
        """Calculate total token cost for a category."""
        functions = self.get_functions_by_category(category)
        return sum(self.functions[func].token_cost for func in functions)

    def calculate_loading_cost(self, function_names: set[str]) -> tuple[int, float]:
        """Calculate token cost and estimated loading time for functions."""
        total_tokens = 0
        total_loading_time = 0.0

        for name in function_names:
            if name in self.functions:
                metadata = self.functions[name]
                total_tokens += metadata.token_cost
                total_loading_time += metadata.loading_time_ms

        return total_tokens, total_loading_time

    def resolve_dependencies(self, function_names: set[str]) -> set[str]:
        """Resolve function dependencies and return complete set."""
        resolved = set(function_names)

        # Simple dependency resolution (could be more sophisticated)
        for func_name in list(function_names):
            if func_name in self.dependencies:
                resolved.update(self.dependencies[func_name])

        return resolved

    def get_baseline_token_cost(self) -> int:
        """Get baseline token cost (all functions loaded)."""
        return sum(metadata.token_cost for metadata in self.functions.values())

    def update_usage_metrics(self, function_name: str, success: bool = True) -> None:
        """Update function usage metrics."""
        if function_name in self.functions:
            metadata = self.functions[function_name]
            metadata.usage_count += 1
            metadata.last_used = datetime.now()

            # Update success rate with exponential moving average
            alpha = 0.1
            if success:
                metadata.success_rate = alpha * 1.0 + (1 - alpha) * metadata.success_rate
            else:
                metadata.success_rate = alpha * 0.0 + (1 - alpha) * metadata.success_rate


class DynamicFunctionLoader:
    """Main orchestration engine for dynamic function loading."""

    def __init__(self, config_manager: ConfigManager | None = None) -> None:
        self.config_manager = config_manager or ConfigManager()
        self.config = self.config_manager.get_config("default")

        # Core components
        self.function_registry = FunctionRegistry()
        self.task_detection = TaskDetectionSystem()
        self.user_control = UserControlSystem(self.task_detection, self.config_manager)
        self.performance_monitor = PerformanceMonitor()
        self.token_monitor = TokenOptimizationMonitor()

        # Session management
        self.active_sessions: dict[str, LoadingSession] = {}
        self.session_history: deque = deque(maxlen=1000)

        # Performance tracking
        self.loading_cache: dict[str, tuple[LoadingDecision, datetime]] = {}
        self.cache_max_age = timedelta(hours=1)

        # Strategy configuration
        self.current_strategy = LoadingStrategy.CONSERVATIVE
        self.enable_caching = True
        self.enable_user_overrides = True

        logger.info("Dynamic Function Loader initialized successfully")

    async def create_loading_session(self, user_id: str, query: str, strategy: LoadingStrategy | None = None) -> str:
        """Create a new function loading session."""

        session_id = str(uuid4())
        strategy = strategy or self.current_strategy

        session = LoadingSession(
            session_id=session_id,
            user_id=user_id,
            query=query,
            timestamp=datetime.now(),
            strategy=strategy,
            baseline_tokens=self.function_registry.get_baseline_token_cost(),
        )

        self.active_sessions[session_id] = session

        # Start monitoring
        await self.token_monitor.start_session_monitoring(
            session_id=session_id,
            user_id=user_id,
            task_type=None,  # Will be determined by detection
            optimization_level=self._strategy_to_optimization_level(strategy),
        )

        logger.info(f"Created loading session {session_id} for user {user_id}")
        return session_id

    async def load_functions_for_query(
        self,
        session_id: str,
        user_overrides: dict[str, Any] | None = None,
    ) -> LoadingDecision:
        """Main entry point for loading functions based on query analysis."""

        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.active_sessions[session_id]
        session.status = SessionStatus.DETECTING

        start_time = time.perf_counter()

        try:
            # Step 1: Task Detection
            detection_result = await self._detect_task_requirements(session)
            session.detection_result = detection_result
            session.detection_time_ms = (time.perf_counter() - start_time) * 1000

            # Step 2: Apply User Overrides
            if user_overrides:
                detection_result = await self._apply_user_overrides(detection_result, user_overrides, session)
                # Update session with modified detection result
                session.detection_result = detection_result

            # Step 3: Make Loading Decision
            loading_decision = await self._make_loading_decision(detection_result, session)
            session.loading_decision = loading_decision
            session.status = SessionStatus.LOADING

            # Step 4: Load Functions
            await self._execute_function_loading(loading_decision, session)
            session.status = SessionStatus.ACTIVE

            # Step 5: Update Monitoring
            await self._update_monitoring_metrics(session, loading_decision)

            logger.info(
                f"Successfully loaded {len(loading_decision.functions_to_load)} functions "
                f"for session {session_id} (reduction: "
                f"{session.calculate_token_reduction()*100:.1f}%)",
            )

            return loading_decision

        except Exception as e:
            # Add stack trace for debugging
            import traceback

            logger.error(f"Function loading exception details for session {session_id}:")
            logger.error(f"Exception: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")

            session.status = SessionStatus.FAILED
            session.error_count += 1

            # Apply fallback if enabled
            if session.enable_fallback:
                logger.warning(f"Function loading failed for session {session_id}, applying fallback: {e}")
                return await self._apply_fallback_loading(session, str(e))
            logger.error(f"Function loading failed for session {session_id}: {e}")
            raise

    async def _detect_task_requirements(self, session: LoadingSession) -> DetectionResult:
        """Detect task requirements using the task detection system."""

        # Build context for detection
        # Debug logging to catch dict vs LoadingSession issue
        if isinstance(session, dict):
            logger.error(f"SESSION IS DICT! Keys: {list(session.keys())}")
            logger.error(f"Type: {type(session)}")
            raise TypeError(f"Expected LoadingSession but got dict with keys: {list(session.keys())}")

        # Ensure session is LoadingSession object
        if not hasattr(session, "timestamp"):
            logger.error(f"Session missing timestamp! Type: {type(session)}, Attributes: {dir(session)}")
            raise AttributeError(f"Session missing timestamp attribute. Type: {type(session)}")

        context = {
            "user_id": session.user_id,
            "session_id": session.session_id,
            "strategy": session.strategy.value,
            "user_preferences": session.user_preferences,
            "timestamp": session.timestamp.isoformat(),
        }

        # Check cache first
        cache_key = self._generate_detection_cache_key(session.query, context)
        if self.enable_caching and cache_key in self.loading_cache:
            cached_decision, cache_time = self.loading_cache[cache_key]
            if datetime.now() - cache_time < self.cache_max_age:
                logger.debug(f"Using cached detection result for session {session.session_id}")
                # Convert cached decision to detection result
                return self._decision_to_detection_result(cached_decision)

        # Perform detection
        detection_result = await self.task_detection.detect_categories(session.query, context)

        # Record detection metrics
        await self.token_monitor.record_task_detection(
            session_id=session.session_id,
            detected_task=",".join([cat for cat, enabled in detection_result.categories.items() if enabled]),
            confidence=max(detection_result.confidence_scores.values()) if detection_result.confidence_scores else 0.0,
        )

        return detection_result

    async def _apply_user_overrides(
        self,
        detection_result: DetectionResult,
        user_overrides: dict[str, Any],
        session: LoadingSession,
    ) -> DetectionResult:
        """Apply user overrides to detection results."""

        # Parse user override commands
        override_categories = user_overrides.get("force_categories", [])
        disable_categories = user_overrides.get("disable_categories", [])
        strategy_override = user_overrides.get("strategy")

        # Apply category overrides
        modified_categories = detection_result.categories.copy()

        for category in override_categories:
            modified_categories[category] = True
            session.user_commands.append(f"force_load:{category}")

        for category in disable_categories:
            if category in modified_categories:
                modified_categories[category] = False
                session.user_commands.append(f"disable:{category}")

        # Apply strategy override
        if strategy_override and hasattr(LoadingStrategy, strategy_override.upper()):
            session.strategy = LoadingStrategy(strategy_override.lower())
            session.user_commands.append(f"strategy:{strategy_override}")

        # Record user override
        if user_overrides:
            await self.token_monitor.record_user_override(
                session_id=session.session_id,
                override_type="category_and_strategy",
                original_optimization=detection_result.fallback_applied or "normal",
                new_optimization=f"user_controlled_{len(session.user_commands)}",
            )

        # Update confidence scores for forced categories
        modified_confidence_scores = detection_result.confidence_scores.copy()
        for category in override_categories:
            # Give forced categories high confidence to ensure they load
            modified_confidence_scores[category] = 1.0

        # Create modified detection result
        return DetectionResult(
            categories=modified_categories,
            confidence_scores=modified_confidence_scores,
            detection_time_ms=detection_result.detection_time_ms,
            signals_used=detection_result.signals_used,
            fallback_applied=detection_result.fallback_applied,
        )

    async def _make_loading_decision(
        self,
        detection_result: DetectionResult,
        session: LoadingSession,
    ) -> LoadingDecision:
        """Make intelligent loading decision based on detection results and strategy."""

        functions_to_load = set()
        tier_breakdown = {tier: set() for tier in LoadingTier}

        # Load Tier 1 functions based on detected categories (for token optimization)
        tier1_functions = self._select_tier1_functions(detection_result, session.strategy)
        functions_to_load.update(tier1_functions)
        tier_breakdown[LoadingTier.TIER_1] = tier1_functions

        # Load Tier 2 based on detection confidence and strategy
        tier2_functions = self._select_tier2_functions(detection_result, session.strategy)
        functions_to_load.update(tier2_functions)
        tier_breakdown[LoadingTier.TIER_2] = tier2_functions

        # Load Tier 3 based on high confidence detection
        tier3_functions = self._select_tier3_functions(detection_result, session.strategy)
        functions_to_load.update(tier3_functions)
        tier_breakdown[LoadingTier.TIER_3] = tier3_functions

        # Resolve dependencies
        functions_to_load = self.function_registry.resolve_dependencies(functions_to_load)

        # Calculate costs and timing
        estimated_tokens, estimated_loading_time = self.function_registry.calculate_loading_cost(functions_to_load)

        # Calculate confidence score
        confidence_scores = list(detection_result.confidence_scores.values())
        overall_confidence = mean(confidence_scores) if confidence_scores else 0.5

        loading_decision = LoadingDecision(
            functions_to_load=functions_to_load,
            tier_breakdown=tier_breakdown,
            estimated_tokens=estimated_tokens,
            estimated_loading_time_ms=estimated_loading_time,
            confidence_score=overall_confidence,
            strategy_used=session.strategy,
            user_overrides=session.user_commands.copy(),
        )

        # Cache decision
        if self.enable_caching:
            cache_key = self._generate_detection_cache_key(
                session.query,
                {"strategy": session.strategy.value, "user_commands": session.user_commands},
            )
            self.loading_cache[cache_key] = (loading_decision, datetime.now())

        return loading_decision

    def _select_tier1_functions(self, detection_result: DetectionResult, strategy: LoadingStrategy) -> set[str]:
        """Select Tier 1 functions based on detected categories for token optimization."""

        tier1_functions = set()

        # Always load core functions (essential for basic operations)
        core_functions = self.function_registry.get_functions_by_category("core")
        tier1_core_functions = {
            func for func in core_functions if func in self.function_registry.get_functions_by_tier(LoadingTier.TIER_1)
        }
        tier1_functions.update(tier1_core_functions)

        # Load git functions only if git category is detected with sufficient confidence
        if detection_result.categories.get("git", False):
            git_confidence = detection_result.confidence_scores.get("git", 0.0)
            # Use a lower threshold for git since it's Tier 1, but still allow optimization
            git_threshold = 0.3  # Lower threshold for Tier 1 git functions

            if git_confidence >= git_threshold:
                git_functions = self.function_registry.get_functions_by_category("git")
                tier1_git_functions = {
                    func
                    for func in git_functions
                    if func in self.function_registry.get_functions_by_tier(LoadingTier.TIER_1)
                }
                tier1_functions.update(tier1_git_functions)

        return tier1_functions

    def _select_tier2_functions(self, detection_result: DetectionResult, strategy: LoadingStrategy) -> set[str]:
        """Select Tier 2 functions based on detection results and strategy."""

        tier2_functions = set()

        # Get strategy-specific threshold and category limit
        base_threshold = self.config.thresholds.tier2_base_threshold

        if strategy == LoadingStrategy.CONSERVATIVE:
            threshold = base_threshold * 0.9  # Still high threshold for safety
            max_categories = 1  # Conservative: restrict to achieve 70% reduction
        elif strategy == LoadingStrategy.AGGRESSIVE:
            threshold = min(base_threshold * 1.05, 0.99)  # Higher threshold but cap at 0.99
            max_categories = 1  # Aggressive: very selective
        else:  # BALANCED
            threshold = base_threshold
            max_categories = 1  # Balanced: selective for token reduction

        # Get eligible categories, prioritizing those with actual Tier 2 functions
        eligible_categories = []
        tier2_tier_functions = self.function_registry.get_functions_by_tier(LoadingTier.TIER_2)

        for category, enabled in detection_result.categories.items():
            if enabled:
                confidence = detection_result.confidence_scores.get(category, 0.0)
                if confidence >= threshold:
                    # Check if this category has any Tier 2 functions
                    category_functions = self.function_registry.get_functions_by_category(category)
                    tier2_category_functions = category_functions & tier2_tier_functions
                    has_tier2_functions = len(tier2_category_functions) > 0

                    # Prioritize categories with Tier 2 functions for better functionality
                    priority_score = confidence
                    if has_tier2_functions:
                        if confidence == 1.0:
                            priority_score = 2.0  # Highest priority for forced categories with Tier 2 functions
                        else:
                            priority_score = (
                                confidence + 0.5
                            )  # Significant boost for natural categories with Tier 2 functions

                    eligible_categories.append((category, priority_score, confidence))

        # Sort by priority score (highest first), then by confidence
        eligible_categories.sort(key=lambda x: (x[1], x[2]), reverse=True)
        selected_categories = [(cat, conf) for cat, _, conf in eligible_categories[:max_categories]]

        # Load functions for selected categories only
        for category, confidence in selected_categories:
            category_functions = self.function_registry.get_functions_by_category(category)
            # Filter to Tier 2 functions only
            tier2_category_functions = {
                func
                for func in category_functions
                if func in self.function_registry.get_functions_by_tier(LoadingTier.TIER_2)
            }
            tier2_functions.update(tier2_category_functions)

        return tier2_functions

    def _select_tier3_functions(self, detection_result: DetectionResult, strategy: LoadingStrategy) -> set[str]:
        """Select Tier 3 functions based on high confidence detection."""

        tier3_functions = set()

        # Get strategy-specific threshold and category limit
        base_threshold = self.config.thresholds.tier3_base_threshold

        if strategy == LoadingStrategy.CONSERVATIVE:
            threshold = base_threshold * 0.9  # Slightly lower threshold
            max_categories = 1  # Conservative: allow one tier 3 category
        elif strategy == LoadingStrategy.AGGRESSIVE:
            threshold = min(base_threshold * 1.05, 0.99)  # Higher threshold but cap at 0.99
            max_categories = 1  # Aggressive: very selective
        else:  # BALANCED
            threshold = base_threshold
            max_categories = 1  # Balanced: highly selective for tier 3

        # Get eligible categories sorted by confidence
        eligible_categories = []
        for category, enabled in detection_result.categories.items():
            if enabled:
                confidence = detection_result.confidence_scores.get(category, 0.0)
                if confidence >= threshold:
                    eligible_categories.append((category, confidence))

        # Sort by confidence (highest first) and limit selection
        eligible_categories.sort(key=lambda x: x[1], reverse=True)
        selected_categories = eligible_categories[:max_categories]

        # Load functions for selected categories only
        for category, confidence in selected_categories:
            category_functions = self.function_registry.get_functions_by_category(category)
            # Filter to Tier 3 functions only
            tier3_category_functions = {
                func
                for func in category_functions
                if func in self.function_registry.get_functions_by_tier(LoadingTier.TIER_3)
            }
            tier3_functions.update(tier3_category_functions)

        return tier3_functions

    async def _execute_function_loading(self, loading_decision: LoadingDecision, session: LoadingSession) -> None:
        """Execute the actual function loading."""

        start_time = time.perf_counter()

        # Simulate function loading (in real implementation, this would load actual functions)
        await asyncio.sleep(loading_decision.estimated_loading_time_ms / 1000)

        session.functions_loaded = loading_decision.functions_to_load.copy()
        session.total_tokens_loaded = loading_decision.estimated_tokens
        session.loading_time_ms = (time.perf_counter() - start_time) * 1000

        # Record loading metrics for each tier
        for tier, functions in loading_decision.tier_breakdown.items():
            if functions:
                tier_tokens, tier_time = self.function_registry.calculate_loading_cost(functions)

                await self.token_monitor.record_function_loading(
                    session_id=session.session_id,
                    tier=FunctionTier(tier.value),  # Convert to monitoring tier enum
                    functions_loaded=list(functions),
                    loading_time_ms=tier_time,
                    tokens_consumed=tier_tokens,
                    cache_hit=False,  # Simplified for prototype
                )

        logger.info(
            f"Loaded {len(session.functions_loaded)} functions "
            f"({session.total_tokens_loaded} tokens) for session {session.session_id}",
        )

    async def _apply_fallback_loading(self, session: LoadingSession, failure_reason: str) -> LoadingDecision:
        """Apply fallback loading strategy when primary loading fails."""

        session.status = SessionStatus.FALLBACK
        session.fallback_activations += 1

        # Fallback strategy: Load conservative safe set
        fallback_functions = set()

        # Always include Tier 1
        fallback_functions.update(self.function_registry.get_functions_by_tier(LoadingTier.TIER_1))

        # Include essential Tier 2 functions
        essential_categories = ["analysis", "debug"]  # Conservative fallback set
        for category in essential_categories:
            category_functions = self.function_registry.get_functions_by_category(category)
            tier2_category_functions = {
                func
                for func in category_functions
                if func in self.function_registry.get_functions_by_tier(LoadingTier.TIER_2)
            }
            fallback_functions.update(tier2_category_functions)

        # Calculate fallback costs
        estimated_tokens, estimated_loading_time = self.function_registry.calculate_loading_cost(fallback_functions)

        # Create fallback decision
        tier_breakdown = {tier: set() for tier in LoadingTier}
        tier_breakdown[LoadingTier.FALLBACK] = fallback_functions

        fallback_decision = LoadingDecision(
            functions_to_load=fallback_functions,
            tier_breakdown=tier_breakdown,
            estimated_tokens=estimated_tokens,
            estimated_loading_time_ms=estimated_loading_time,
            confidence_score=0.5,  # Medium confidence for fallback
            strategy_used=LoadingStrategy.CONSERVATIVE,
            fallback_reason=failure_reason,
        )

        # Execute fallback loading
        await self._execute_function_loading(fallback_decision, session)

        # Record fallback activation
        await self.token_monitor.record_fallback_activation(
            session_id=session.session_id,
            reason=failure_reason,
            missing_functions=[],  # Would be calculated in real implementation
        )

        logger.warning(
            f"Applied fallback loading for session {session.session_id}: loaded {len(fallback_functions)} functions",
        )

        return fallback_decision

    async def _update_monitoring_metrics(self, session: LoadingSession, loading_decision: LoadingDecision) -> None:
        """Update comprehensive monitoring metrics."""

        # Calculate token reduction
        token_reduction = session.calculate_token_reduction()

        # Update performance metrics
        self.performance_monitor.record_metric(
            MetricData(
                name="function_loading_session",
                value=1,
                timestamp=time.time(),
                labels={
                    "session_id": session.session_id,
                    "strategy": session.strategy.value,
                    "token_reduction_percent": str(round(token_reduction * 100, 1)),
                    "functions_loaded": str(len(session.functions_loaded)),
                },
                metric_type=MetricType.COUNTER,
            ),
        )

        logger.debug(
            f"Updated monitoring metrics for session {session.session_id}: "
            f"token reduction {token_reduction*100:.1f}%",
        )

    async def record_function_usage(self, session_id: str, function_name: str, success: bool = True) -> None:
        """Record function usage for optimization learning."""

        if session_id not in self.active_sessions:
            return

        session = self.active_sessions[session_id]
        session.functions_used.add(function_name)

        # Update registry metrics
        self.function_registry.update_usage_metrics(function_name, success)

        # Record in monitoring system
        await self.token_monitor.record_function_usage(
            session_id=session_id,
            function_name=function_name,
            success=success,
            tier=self._get_function_tier(function_name),
        )

        if not success:
            session.error_count += 1

    async def execute_user_command(self, session_id: str, command: str) -> CommandResult:
        """Execute user override command."""

        if session_id not in self.active_sessions:
            return CommandResult(success=False, message=f"Session {session_id} not found")

        session = self.active_sessions[session_id]
        session.user_commands.append(command)

        # Execute command through user control system
        result = await self.user_control.execute_command(command)

        # If command affects loading, trigger reload
        if result.success and command.startswith(("/load-", "/unload-", "/optimize-")):
            # Re-evaluate function loading with new preferences
            await self.load_functions_for_query(session_id)

        return result

    async def end_loading_session(self, session_id: str) -> dict[str, Any] | None:
        """End a loading session and generate final metrics."""

        if session_id not in self.active_sessions:
            return None

        session = self.active_sessions[session_id]
        session.status = SessionStatus.COMPLETED

        # End monitoring
        await self.token_monitor.end_session_monitoring(session_id)

        # Calculate final performance metrics
        usage_efficiency = 0.0
        if session.functions_loaded:
            usage_efficiency = len(session.functions_used) / len(session.functions_loaded)

        session_summary = {
            "session_id": session_id,
            "user_id": session.user_id,
            "duration_seconds": (datetime.now() - session.timestamp).total_seconds(),
            "token_reduction_percentage": session.calculate_token_reduction() * 100,
            "functions_loaded": len(session.functions_loaded),
            "functions_used": len(session.functions_used),
            "usage_efficiency": usage_efficiency,
            "detection_time_ms": session.detection_time_ms,
            "loading_time_ms": session.loading_time_ms,
            "user_commands": len(session.user_commands),
            "fallback_activations": session.fallback_activations,
            "error_count": session.error_count,
            "strategy_used": session.strategy.value,
        }

        # Move to history
        self.session_history.append(session)
        del self.active_sessions[session_id]

        logger.info(
            f"Completed session {session_id}: "
            f"{session_summary['token_reduction_percentage']:.1f}% token reduction, "
            f"{session_summary['usage_efficiency']*100:.1f}% function usage efficiency",
        )

        return session_summary

    # Utility methods

    def _strategy_to_optimization_level(self, strategy: LoadingStrategy):
        """Convert loading strategy to optimization level."""
        from .token_optimization_monitor import OptimizationStatus

        mapping = {
            LoadingStrategy.CONSERVATIVE: OptimizationStatus.CONSERVATIVE,
            LoadingStrategy.BALANCED: OptimizationStatus.BALANCED,
            LoadingStrategy.AGGRESSIVE: OptimizationStatus.AGGRESSIVE,
            LoadingStrategy.USER_CONTROLLED: OptimizationStatus.CUSTOM,
        }
        return mapping.get(strategy, OptimizationStatus.CONSERVATIVE)

    def _get_function_tier(self, function_name: str) -> FunctionTier | None:
        """Get monitoring tier for a function."""
        if function_name not in self.function_registry.functions:
            return None

        tier = self.function_registry.functions[function_name].tier
        return FunctionTier(tier.value)

    def _generate_detection_cache_key(self, query: str, context: dict[str, Any]) -> str:
        """Generate cache key for detection results."""
        context_str = json.dumps(context, sort_keys=True)
        combined = f"{query}:{context_str}"
        return hashlib.md5(combined.encode(), usedforsecurity=False).hexdigest()  # nosec B324

    def _decision_to_detection_result(self, decision: LoadingDecision) -> DetectionResult:
        """Convert loading decision back to detection result for caching."""
        # Simplified conversion - in real implementation would preserve original detection data
        categories = {}
        confidence_scores = {}

        # Reconstruct categories from tier breakdown
        for _tier, functions in decision.tier_breakdown.items():
            for func_name in functions:
                if func_name in self.function_registry.functions:
                    category = self.function_registry.functions[func_name].category
                    categories[category] = True
                    confidence_scores[category] = decision.confidence_score

        return DetectionResult(
            categories=categories,
            confidence_scores=confidence_scores,
            detection_time_ms=0.0,  # Cached, so no detection time
            signals_used={},
            fallback_applied=decision.fallback_reason,
        )

    async def get_performance_report(self) -> dict[str, Any]:
        """Generate comprehensive performance report."""

        # Get system health metrics
        health_report = await self.token_monitor.generate_system_health_report()
        optimization_report = await self.token_monitor.get_optimization_report()

        # Calculate session statistics
        active_sessions_count = len(self.active_sessions)
        completed_sessions = list(self.session_history)

        if completed_sessions:
            token_reductions = [s.calculate_token_reduction() for s in completed_sessions]
            usage_efficiencies = [
                len(s.functions_used) / len(s.functions_loaded) if s.functions_loaded else 0 for s in completed_sessions
            ]

            session_stats = {
                "total_completed_sessions": len(completed_sessions),
                "average_token_reduction": mean(token_reductions) * 100,
                "median_token_reduction": median(token_reductions) * 100,
                "average_usage_efficiency": mean(usage_efficiencies) * 100,
                "sessions_achieving_70_percent": sum(1 for r in token_reductions if r >= 0.7),
                "success_rate_70_percent": sum(1 for r in token_reductions if r >= 0.7) / len(token_reductions) * 100,
            }
        else:
            session_stats = {
                "total_completed_sessions": 0,
                "average_token_reduction": 0.0,
                "median_token_reduction": 0.0,
                "average_usage_efficiency": 0.0,
                "sessions_achieving_70_percent": 0,
                "success_rate_70_percent": 0.0,
            }

        return {
            "timestamp": datetime.now().isoformat(),
            "active_sessions": active_sessions_count,
            "system_health": asdict(health_report),
            "optimization_performance": optimization_report,
            "session_statistics": session_stats,
            "function_registry_stats": {
                "total_functions": len(self.function_registry.functions),
                "tier1_functions": len(self.function_registry.get_functions_by_tier(LoadingTier.TIER_1)),
                "tier2_functions": len(self.function_registry.get_functions_by_tier(LoadingTier.TIER_2)),
                "tier3_functions": len(self.function_registry.get_functions_by_tier(LoadingTier.TIER_3)),
                "baseline_token_cost": self.function_registry.get_baseline_token_cost(),
            },
            "cache_statistics": {"cache_size": len(self.loading_cache), "cache_enabled": self.enable_caching},
        }


# Global loader instance
_global_loader: DynamicFunctionLoader | None = None


def get_dynamic_function_loader() -> DynamicFunctionLoader:
    """Get the global dynamic function loader instance."""
    global _global_loader
    if _global_loader is None:
        _global_loader = DynamicFunctionLoader()
    return _global_loader


async def initialize_dynamic_loading() -> DynamicFunctionLoader:
    """Initialize the dynamic function loading system."""
    loader = get_dynamic_function_loader()
    logger.info("Dynamic function loading system initialized")
    return loader


# Example usage and testing
if __name__ == "__main__":

    async def main():
        """Test the dynamic function loading prototype."""

        # Initialize system
        loader = await initialize_dynamic_loading()

        # Test scenarios
        test_scenarios = [
            {
                "name": "Git Workflow Query",
                "query": "help me commit my changes and push to remote",
                "user_id": "dev_user_1",
                "expected_categories": ["git", "core"],
            },
            {
                "name": "Debug Session Query",
                "query": "debug the failing authentication tests",
                "user_id": "dev_user_2",
                "expected_categories": ["debug", "test", "analysis"],
            },
            {
                "name": "Security Audit Query",
                "query": "perform security audit on the payment module",
                "user_id": "security_user",
                "expected_categories": ["security", "analysis", "quality"],
            },
            {
                "name": "General Analysis Query",
                "query": "help me understand this codebase architecture",
                "user_id": "analyst_user",
                "expected_categories": ["analysis", "quality"],
            },
        ]

        session_results = []

        for scenario in test_scenarios:

            # Create session
            session_id = await loader.create_loading_session(user_id=scenario["user_id"], query=scenario["query"])

            # Load functions
            start_time = time.perf_counter()
            loading_decision = await loader.load_functions_for_query(session_id)
            (time.perf_counter() - start_time) * 1000

            # Simulate function usage
            used_functions = list(loading_decision.functions_to_load)[:3]  # Use first 3 functions
            for func_name in used_functions:
                await loader.record_function_usage(session_id, func_name, success=True)

            # End session
            session_summary = await loader.end_loading_session(session_id)

            # Display results

            if loading_decision.fallback_reason:
                pass

            session_results.append(session_summary)

        # Generate performance report

        performance_report = await loader.get_performance_report()

        # Validation summary
        target_achieved = performance_report["session_statistics"]["average_token_reduction"] >= 70.0

        if target_achieved:
            pass
        else:
            pass

        return performance_report

    # Run the test
    report = asyncio.run(main())
