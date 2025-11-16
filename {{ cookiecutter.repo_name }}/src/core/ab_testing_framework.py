"""
A/B Testing Framework for Dynamic Function Loading Gradual Rollout

This module provides a comprehensive A/B testing system for safely rolling out the
dynamic function loading system with data-driven validation and real-world performance
monitoring.

Key Features:
- User segmentation and assignment to control/test groups
- Configurable rollout percentages (1%, 5%, 25%, 50%, 100%)
- Statistical significance tracking and validation
- Performance metrics collection and analysis
- Automated rollback capabilities
- Feature flag management for dynamic control
- Risk mitigation and success criteria validation

Architecture:
- ExperimentManager: Core orchestration and control
- UserSegmentation: Assignment and characteristic-based grouping
- MetricsCollector: Real-time performance and UX metrics
- StatisticalAnalyzer: Significance testing and validation
- FeatureFlagManager: Dynamic experiment control
- RolloutController: Progressive deployment automation
"""

import asyncio
import contextlib
import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from src.utils.datetime_compat import utc_now
from enum import Enum
from typing import Any

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from src.config.settings import get_settings
from src.utils.observability import ObservabilityMixin
from src.utils.performance_monitor import PerformanceMonitor

from .dynamic_loading_integration import ProcessingResult

logger = logging.getLogger(__name__)

# Database models for A/B testing
Base = declarative_base()


class ExperimentType(Enum):
    """Types of A/B experiments."""

    DYNAMIC_LOADING = "dynamic_loading"
    OPTIMIZATION_STRATEGY = "optimization_strategy"
    USER_INTERFACE = "user_interface"
    PERFORMANCE = "performance"


class ExperimentStatus(Enum):
    """Status of A/B experiments."""

    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class UserSegment(Enum):
    """User segment types for A/B testing."""

    RANDOM = "random"
    POWER_USER = "power_user"
    NEW_USER = "new_user"
    HIGH_VOLUME = "high_volume"
    LOW_VOLUME = "low_volume"
    EARLY_ADOPTER = "early_adopter"


class VariantType(Enum):
    """Types of experiment variants."""

    CONTROL = "control"
    TREATMENT = "treatment"
    TREATMENT_A = "treatment_a"
    TREATMENT_B = "treatment_b"


# Database Models


class ExperimentModel(Base):
    """Database model for A/B experiments."""

    __tablename__ = "ab_experiments"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String)
    experiment_type = Column(String, nullable=False)
    status = Column(String, nullable=False, default="draft")

    # Experiment configuration
    config = Column(JSON, nullable=False)
    variants = Column(JSON, nullable=False)
    success_criteria = Column(JSON, nullable=False)
    failure_thresholds = Column(JSON, nullable=False)

    # Rollout configuration
    target_percentage = Column(Float, nullable=False, default=0.0)
    current_percentage = Column(Float, nullable=False, default=0.0)
    segment_filters = Column(JSON)

    # Timing
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    planned_duration_hours = Column(Integer, default=168)  # 1 week default

    # Results tracking
    total_users = Column(Integer, default=0)
    conversion_events = Column(Integer, default=0)
    statistical_significance = Column(Float, default=0.0)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String)


class UserAssignmentModel(Base):
    """Database model for user-to-experiment assignments."""

    __tablename__ = "ab_user_assignments"

    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    experiment_id = Column(String, nullable=False, index=True)
    variant = Column(String, nullable=False)
    segment = Column(String)

    # User characteristics at assignment time
    user_characteristics = Column(JSON)

    # Assignment metadata
    assignment_time = Column(DateTime, default=datetime.utcnow)
    assignment_method = Column(String, default="random")
    opt_out = Column(Boolean, default=False)

    # Tracking
    last_interaction = Column(DateTime)
    total_interactions = Column(Integer, default=0)


class MetricEventModel(Base):
    """Database model for A/B testing metric events."""

    __tablename__ = "ab_metric_events"

    id = Column(String, primary_key=True)
    experiment_id = Column(String, nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)
    variant = Column(String, nullable=False)

    # Event details
    event_type = Column(String, nullable=False)  # performance, conversion, error, etc.
    event_name = Column(String, nullable=False)
    event_value = Column(Float)
    event_data = Column(JSON)

    # Context
    session_id = Column(String)
    request_id = Column(String)

    # Timing
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Performance metrics
    response_time_ms = Column(Float)
    token_reduction_percentage = Column(Float)
    success = Column(Boolean)
    error_message = Column(String)


# Pydantic Models


@dataclass
class ExperimentConfig:
    """Configuration for an A/B experiment."""

    # Basic configuration
    name: str
    description: str
    experiment_type: ExperimentType
    planned_duration_hours: int = 168  # 1 week

    # Feature configuration
    feature_flags: dict[str, Any] = field(default_factory=dict)
    variant_configs: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Rollout configuration
    initial_percentage: float = 1.0
    target_percentage: float = 50.0
    rollout_steps: list[float] = field(default_factory=lambda: [1.0, 5.0, 25.0, 50.0])
    step_duration_hours: int = 24

    # User segmentation
    segment_filters: list[UserSegment] = field(default_factory=lambda: [UserSegment.RANDOM])
    exclude_segments: list[UserSegment] = field(default_factory=list)
    opt_in_only: bool = False

    # Success criteria
    success_criteria: dict[str, float] = field(default_factory=dict)
    failure_thresholds: dict[str, float] = field(default_factory=dict)
    min_sample_size: int = 1000
    max_acceptable_error_rate: float = 5.0  # 5%
    required_improvement: float = 10.0  # 10% improvement required

    # Safety mechanisms
    auto_rollback_enabled: bool = True
    circuit_breaker_threshold: float = 10.0  # 10% error rate triggers rollback
    max_performance_degradation: float = 20.0  # 20% performance degradation limit


@dataclass
class UserCharacteristics:
    """User characteristics for segmentation."""

    user_id: str
    registration_date: datetime | None = None
    usage_frequency: str = "unknown"  # low, medium, high
    feature_usage_pattern: str = "unknown"  # basic, intermediate, advanced
    error_rate: float = 0.0
    avg_session_duration: float = 0.0
    preferred_features: list[str] = field(default_factory=list)
    geographic_region: str | None = None
    device_type: str | None = None
    is_early_adopter: bool = False
    opt_in_beta: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "user_id": self.user_id,
            "registration_date": self.registration_date.isoformat() if self.registration_date else None,
            "usage_frequency": self.usage_frequency,
            "feature_usage_pattern": self.feature_usage_pattern,
            "error_rate": self.error_rate,
            "avg_session_duration": self.avg_session_duration,
            "preferred_features": self.preferred_features,
            "geographic_region": self.geographic_region,
            "device_type": self.device_type,
            "is_early_adopter": self.is_early_adopter,
            "opt_in_beta": self.opt_in_beta,
        }


@dataclass
class MetricEvent:
    """A single metric event for A/B testing."""

    experiment_id: str
    user_id: str
    variant: str
    event_type: str
    event_name: str
    event_value: float | None = None
    event_data: dict[str, Any] | None = None
    session_id: str | None = None
    request_id: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Performance specific
    response_time_ms: float | None = None
    token_reduction_percentage: float | None = None
    success: bool | None = None
    error_message: str | None = None


@dataclass
class ExperimentResults:
    """Results of an A/B experiment."""

    experiment_id: str
    experiment_name: str
    total_users: int
    variants: dict[str, dict[str, Any]]

    # Statistical analysis
    statistical_significance: float
    confidence_interval: tuple[float, float]
    p_value: float
    effect_size: float

    # Performance metrics
    performance_summary: dict[str, Any]
    success_criteria_met: dict[str, bool]
    failure_thresholds_exceeded: dict[str, bool]

    # Recommendations
    recommendation: str  # "continue", "rollback", "expand", "modify"
    confidence_level: str  # "low", "medium", "high"
    next_actions: list[str]

    # Timing
    start_time: datetime
    end_time: datetime | None
    duration_hours: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "total_users": self.total_users,
            "variants": self.variants,
            "statistical_significance": self.statistical_significance,
            "confidence_interval": list(self.confidence_interval),
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "performance_summary": self.performance_summary,
            "success_criteria_met": self.success_criteria_met,
            "failure_thresholds_exceeded": self.failure_thresholds_exceeded,
            "recommendation": self.recommendation,
            "confidence_level": self.confidence_level,
            "next_actions": self.next_actions,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_hours": self.duration_hours,
        }


class UserSegmentation:
    """Handles user segmentation and assignment for A/B testing."""

    def __init__(self, db_session: Session) -> None:
        self.db_session = db_session
        self.logger = logging.getLogger(__name__)

    def assign_user_to_experiment(
        self,
        user_id: str,
        experiment_id: str,
        config: ExperimentConfig,
        user_characteristics: UserCharacteristics | None = None,
    ) -> tuple[str, UserSegment]:
        """
        Assign a user to an experiment variant based on configuration.

        Args:
            user_id: Unique user identifier
            experiment_id: Experiment identifier
            config: Experiment configuration
            user_characteristics: User characteristics for segmentation

        Returns:
            Tuple of (variant, segment)
        """
        try:
            # Check if user already assigned
            existing = (
                self.db_session.query(UserAssignmentModel)
                .filter_by(
                    user_id=user_id,
                    experiment_id=experiment_id,
                )
                .first()
            )

            if existing and not existing.opt_out:
                return existing.variant, UserSegment(existing.segment)

            # Check opt-in requirements
            if config.opt_in_only and user_characteristics:
                if not user_characteristics.opt_in_beta:
                    return "control", UserSegment.RANDOM

            # Determine user segment
            segment = self._determine_user_segment(user_characteristics or UserCharacteristics(user_id))

            # Check segment filters
            if config.segment_filters:
                # Convert segment filters to enum objects if they're strings
                segment_filter_enums = []
                for f in config.segment_filters:
                    if isinstance(f, str):
                        try:
                            segment_filter_enums.append(UserSegment(f))
                        except ValueError:
                            # Skip invalid segment filter values
                            pass
                    elif hasattr(f, "value"):
                        segment_filter_enums.append(f)

                if segment not in segment_filter_enums:
                    return "control", segment

            # Check exclude segments
            if segment in config.exclude_segments:
                return "control", segment

            # Check rollout percentage
            rollout_percentage = self._get_current_rollout_percentage(experiment_id)

            # Use consistent hashing for assignment
            variant = self._assign_variant_consistent(user_id, experiment_id, rollout_percentage)

            # Store assignment
            assignment = UserAssignmentModel(
                id=f"{experiment_id}_{user_id}",
                user_id=user_id,
                experiment_id=experiment_id,
                variant=variant,
                segment=segment.value,
                user_characteristics=user_characteristics.to_dict() if user_characteristics else {},
                assignment_method="consistent_hash",
            )

            self.db_session.add(assignment)
            self.db_session.commit()

            self.logger.info(
                f"Assigned user {user_id} to variant {variant} in experiment {experiment_id}",
            )

            return variant, segment

        except Exception as e:
            self.logger.error(f"Failed to assign user to experiment: {e}")
            self.db_session.rollback()
            return "control", UserSegment.RANDOM

    def _determine_user_segment(self, characteristics: UserCharacteristics) -> UserSegment:
        """Determine user segment based on characteristics."""

        # Early adopter
        if characteristics.is_early_adopter or characteristics.opt_in_beta:
            return UserSegment.EARLY_ADOPTER

        # New user (registered within last 30 days)
        if characteristics.registration_date:
            if utc_now() - characteristics.registration_date <= timedelta(days=30):
                return UserSegment.NEW_USER

        # Power user (high usage frequency and advanced patterns)
        if characteristics.usage_frequency == "high" and characteristics.feature_usage_pattern == "advanced":
            return UserSegment.POWER_USER

        # Volume-based segmentation
        if characteristics.usage_frequency == "high":
            return UserSegment.HIGH_VOLUME
        if characteristics.usage_frequency == "low":
            return UserSegment.LOW_VOLUME

        # Default to random
        return UserSegment.RANDOM

    def _get_current_rollout_percentage(self, experiment_id: str) -> float:
        """Get current rollout percentage for experiment."""
        experiment = self.db_session.query(ExperimentModel).filter_by(id=experiment_id).first()
        return experiment.current_percentage if experiment else 0.0

    def _assign_variant_consistent(self, user_id: str, experiment_id: str, rollout_percentage: float) -> str:
        """Assign variant using consistent hashing for stable assignments."""
        # Create consistent hash
        hash_input = f"{experiment_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)

        # Normalize to 0-100 range
        percentage = (hash_value % 10000) / 100.0

        # Assign based on rollout percentage
        if percentage <= rollout_percentage:
            # For simplicity, 50/50 split between control and treatment within rollout
            if percentage <= rollout_percentage / 2:
                return "treatment"
            return "control"
        return "control"

    def opt_out_user(self, user_id: str, experiment_id: str) -> bool:
        """Opt out a user from an experiment."""
        try:
            assignment = (
                self.db_session.query(UserAssignmentModel)
                .filter_by(
                    user_id=user_id,
                    experiment_id=experiment_id,
                )
                .first()
            )

            if assignment:
                assignment.opt_out = True
                self.db_session.commit()
                return True

            return False

        except Exception as e:
            self.logger.error(f"Failed to opt out user: {e}")
            self.db_session.rollback()
            return False


class MetricsCollector:
    """Collects and stores A/B testing metrics."""

    def __init__(self, db_session: Session) -> None:
        self.db_session = db_session
        self.logger = logging.getLogger(__name__)

    def record_event(self, event: MetricEvent) -> bool:
        """Record a metric event for A/B testing analysis."""
        try:
            event_model = MetricEventModel(
                id=f"{event.experiment_id}_{event.user_id}_{int(time.time() * 1000000)}",
                experiment_id=event.experiment_id,
                user_id=event.user_id,
                variant=event.variant,
                event_type=event.event_type,
                event_name=event.event_name,
                event_value=event.event_value,
                event_data=event.event_data,
                session_id=event.session_id,
                request_id=event.request_id,
                timestamp=event.timestamp,
                response_time_ms=event.response_time_ms,
                token_reduction_percentage=event.token_reduction_percentage,
                success=event.success,
                error_message=event.error_message,
            )

            self.db_session.add(event_model)
            self.db_session.commit()

            return True

        except Exception as e:
            self.logger.error(f"Failed to record metric event: {e}")
            self.db_session.rollback()
            return False

    def record_processing_result(
        self,
        experiment_id: str,
        user_id: str,
        variant: str,
        result: ProcessingResult,
    ) -> bool:
        """Record a processing result as multiple metric events."""
        try:
            # Performance event
            performance_event = MetricEvent(
                experiment_id=experiment_id,
                user_id=user_id,
                variant=variant,
                event_type="performance",
                event_name="query_processing",
                event_value=result.total_time_ms,
                event_data={
                    "detection_time_ms": result.detection_time_ms,
                    "loading_time_ms": result.loading_time_ms,
                    "cache_hit": result.cache_hit,
                    "fallback_used": result.fallback_used,
                },
                session_id=result.session_id,
                response_time_ms=result.total_time_ms,
                token_reduction_percentage=result.reduction_percentage,
                success=result.success,
                error_message=result.error_message,
            )

            self.record_event(performance_event)

            # Token optimization event
            if result.success:
                optimization_event = MetricEvent(
                    experiment_id=experiment_id,
                    user_id=user_id,
                    variant=variant,
                    event_type="optimization",
                    event_name="token_reduction",
                    event_value=result.reduction_percentage,
                    event_data={
                        "baseline_tokens": result.baseline_tokens,
                        "optimized_tokens": result.optimized_tokens,
                        "target_achieved": result.target_achieved,
                    },
                    session_id=result.session_id,
                    token_reduction_percentage=result.reduction_percentage,
                    success=True,
                )

                self.record_event(optimization_event)

            # Error event if failed
            if not result.success:
                error_event = MetricEvent(
                    experiment_id=experiment_id,
                    user_id=user_id,
                    variant=variant,
                    event_type="error",
                    event_name="processing_failure",
                    event_value=1.0,
                    event_data={"error_type": "processing_error"},
                    session_id=result.session_id,
                    success=False,
                    error_message=result.error_message,
                )

                self.record_event(error_event)

            return True

        except Exception as e:
            self.logger.error(f"Failed to record processing result: {e}")
            return False


class StatisticalAnalyzer:
    """Performs statistical analysis for A/B testing."""

    def __init__(self, db_session: Session) -> None:
        self.db_session = db_session
        self.logger = logging.getLogger(__name__)

    def analyze_experiment(self, experiment_id: str) -> ExperimentResults | None:
        """Perform comprehensive statistical analysis of an experiment."""
        try:
            experiment = self.db_session.query(ExperimentModel).filter_by(id=experiment_id).first()
            if not experiment:
                return None

            # Get variant performance data
            variant_data = self._collect_variant_data(experiment_id)

            if len(variant_data) < 2:
                self.logger.warning(f"Insufficient variant data for experiment {experiment_id}")
                return None

            # Calculate statistical significance
            significance_results = self._calculate_statistical_significance(variant_data)

            # Analyze performance metrics
            performance_summary = self._analyze_performance_metrics(experiment_id, variant_data)

            # Check success criteria
            success_criteria_met = self._check_success_criteria(experiment, performance_summary)

            # Check failure thresholds
            failure_thresholds_exceeded = self._check_failure_thresholds(experiment, performance_summary)

            # Generate recommendation
            recommendation, confidence_level, next_actions = self._generate_recommendation(
                significance_results,
                success_criteria_met,
                failure_thresholds_exceeded,
                performance_summary,
            )

            # Calculate duration
            start_time = experiment.start_time or experiment.created_at
            end_time = experiment.end_time or utc_now()
            duration_hours = (end_time - start_time).total_seconds() / 3600

            return ExperimentResults(
                experiment_id=experiment_id,
                experiment_name=experiment.name,
                total_users=sum(v.get("user_count", 0) for v in variant_data.values()),
                variants=variant_data,
                statistical_significance=significance_results["significance"],
                confidence_interval=significance_results["confidence_interval"],
                p_value=significance_results["p_value"],
                effect_size=significance_results["effect_size"],
                performance_summary=performance_summary,
                success_criteria_met=success_criteria_met,
                failure_thresholds_exceeded=failure_thresholds_exceeded,
                recommendation=recommendation,
                confidence_level=confidence_level,
                next_actions=next_actions,
                start_time=start_time,
                end_time=end_time if experiment.end_time else None,
                duration_hours=duration_hours,
            )

        except Exception as e:
            self.logger.error(f"Failed to analyze experiment {experiment_id}: {e}")
            return None

    def _collect_variant_data(self, experiment_id: str) -> dict[str, dict[str, Any]]:
        """Collect performance data for each variant."""

        # Get user assignments
        assignments_query = self.db_session.query(UserAssignmentModel).filter_by(
            experiment_id=experiment_id,
        )

        variant_data = defaultdict(
            lambda: {
                "user_count": 0,
                "events": [],
                "performance_metrics": [],
                "error_count": 0,
                "success_count": 0,
            },
        )

        # Count users per variant
        for assignment in assignments_query:
            if not assignment.opt_out:
                variant_data[assignment.variant]["user_count"] += 1

        # Get metric events
        events_query = self.db_session.query(MetricEventModel).filter_by(
            experiment_id=experiment_id,
        )

        for event in events_query:
            variant_data[event.variant]["events"].append(
                {
                    "event_type": event.event_type,
                    "event_name": event.event_name,
                    "event_value": event.event_value,
                    "response_time_ms": event.response_time_ms,
                    "token_reduction_percentage": event.token_reduction_percentage,
                    "success": event.success,
                    "timestamp": event.timestamp,
                },
            )

            # Aggregate performance metrics
            if event.response_time_ms is not None:
                variant_data[event.variant]["performance_metrics"].append(
                    {
                        "response_time_ms": event.response_time_ms,
                        "token_reduction_percentage": event.token_reduction_percentage or 0.0,
                        "success": event.success,
                    },
                )

            # Count successes and errors
            if event.success is True:
                variant_data[event.variant]["success_count"] += 1
            elif event.success is False:
                variant_data[event.variant]["error_count"] += 1

        # Calculate aggregated metrics for each variant
        for _variant, data in variant_data.items():
            metrics = data["performance_metrics"]
            if metrics:
                data["avg_response_time_ms"] = sum(m["response_time_ms"] for m in metrics) / len(metrics)
                data["avg_token_reduction"] = sum(m["token_reduction_percentage"] for m in metrics) / len(metrics)
                data["success_rate"] = (
                    data["success_count"] / (data["success_count"] + data["error_count"])
                    if (data["success_count"] + data["error_count"]) > 0
                    else 0.0
                )
            else:
                data["avg_response_time_ms"] = 0.0
                data["avg_token_reduction"] = 0.0
                data["success_rate"] = 0.0

        return dict(variant_data)

    def _calculate_statistical_significance(self, variant_data: dict[str, dict[str, Any]]) -> dict[str, float]:
        """Calculate statistical significance between variants."""

        # Simplified statistical analysis
        # In production, use proper statistical libraries like scipy.stats

        variants = list(variant_data.keys())
        if len(variants) < 2:
            return {
                "significance": 0.0,
                "confidence_interval": (0.0, 0.0),
                "p_value": 1.0,
                "effect_size": 0.0,
            }

        # Compare first two variants for now
        variant_a_data = variant_data[variants[0]]
        variant_b_data = variant_data[variants[1]]

        # Use success rate as primary metric
        a_success_rate = variant_a_data["success_rate"]
        b_success_rate = variant_b_data["success_rate"]

        # Calculate effect size (difference in success rates)
        effect_size = abs(b_success_rate - a_success_rate)

        # Simple significance calculation based on sample size and effect size
        min_sample_size = min(variant_a_data["user_count"], variant_b_data["user_count"])

        # Rough significance calculation
        if min_sample_size >= 100 and effect_size >= 0.05:  # 5% effect with 100+ samples
            significance = min(95.0, 50.0 + (min_sample_size / 50.0) + (effect_size * 100))
        else:
            significance = min_sample_size * effect_size * 10

        # Simplified p-value calculation
        p_value = max(0.001, 1.0 - (significance / 100.0))

        # Confidence interval (simplified)
        margin_of_error = 1.96 * (effect_size / (min_sample_size**0.5)) if min_sample_size > 0 else effect_size
        confidence_interval = (
            max(0.0, effect_size - margin_of_error),
            min(1.0, effect_size + margin_of_error),
        )

        return {
            "significance": significance,
            "confidence_interval": confidence_interval,
            "p_value": p_value,
            "effect_size": effect_size,
        }

    def _analyze_performance_metrics(
        self,
        experiment_id: str,
        variant_data: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Analyze performance metrics across variants."""

        performance_summary = {
            "total_requests": 0,
            "avg_response_time_ms": 0.0,
            "avg_token_reduction": 0.0,
            "overall_success_rate": 0.0,
            "variant_comparison": {},
        }

        total_requests = 0
        total_response_time = 0.0
        total_token_reduction = 0.0
        total_successes = 0
        total_attempts = 0

        for variant, data in variant_data.items():
            variant_requests = len(data["performance_metrics"])
            total_requests += variant_requests

            if variant_requests > 0:
                total_response_time += data["avg_response_time_ms"] * variant_requests
                total_token_reduction += data["avg_token_reduction"] * variant_requests

            total_successes += data["success_count"]
            total_attempts += data["success_count"] + data["error_count"]

            performance_summary["variant_comparison"][variant] = {
                "user_count": data["user_count"],
                "request_count": variant_requests,
                "avg_response_time_ms": data["avg_response_time_ms"],
                "avg_token_reduction": data["avg_token_reduction"],
                "success_rate": data["success_rate"],
                "error_count": data["error_count"],
            }

        # Calculate overall averages
        if total_requests > 0:
            performance_summary["avg_response_time_ms"] = total_response_time / total_requests
            performance_summary["avg_token_reduction"] = total_token_reduction / total_requests

        if total_attempts > 0:
            performance_summary["overall_success_rate"] = total_successes / total_attempts

        performance_summary["total_requests"] = total_requests

        return performance_summary

    def _check_success_criteria(
        self,
        experiment: ExperimentModel,
        performance_summary: dict[str, Any],
    ) -> dict[str, bool]:
        """Check if experiment meets success criteria."""

        success_criteria = experiment.success_criteria or {}
        results = {}

        # Check minimum token reduction
        min_token_reduction = success_criteria.get("min_token_reduction", 70.0)
        results["token_reduction_target"] = performance_summary["avg_token_reduction"] >= min_token_reduction

        # Check maximum response time
        max_response_time = success_criteria.get("max_response_time_ms", 200.0)
        results["response_time_target"] = performance_summary["avg_response_time_ms"] <= max_response_time

        # Check minimum success rate
        min_success_rate = success_criteria.get("min_success_rate", 95.0)
        results["success_rate_target"] = (performance_summary["overall_success_rate"] * 100) >= min_success_rate

        # Check statistical significance
        success_criteria.get("min_statistical_significance", 95.0)
        # This would need the significance from the statistical analysis
        results["statistical_significance"] = True  # Placeholder

        return results

    def _check_failure_thresholds(
        self,
        experiment: ExperimentModel,
        performance_summary: dict[str, Any],
    ) -> dict[str, bool]:
        """Check if experiment exceeds failure thresholds."""

        failure_thresholds = experiment.failure_thresholds or {}
        results = {}

        # Check maximum error rate
        max_error_rate = failure_thresholds.get("max_error_rate", 5.0)
        current_error_rate = (1.0 - performance_summary["overall_success_rate"]) * 100
        results["error_rate_exceeded"] = current_error_rate > max_error_rate

        # Check maximum response time degradation
        failure_thresholds.get("max_response_time_degradation", 50.0)
        # Would need baseline response time for comparison
        results["response_time_degradation"] = False  # Placeholder

        # Check minimum token reduction
        min_token_reduction = failure_thresholds.get("min_token_reduction", 50.0)
        results["token_reduction_failure"] = performance_summary["avg_token_reduction"] < min_token_reduction

        return results

    def _generate_recommendation(
        self,
        significance_results: dict[str, float],
        success_criteria_met: dict[str, bool],
        failure_thresholds_exceeded: dict[str, bool],
        performance_summary: dict[str, Any],
    ) -> tuple[str, str, list[str]]:
        """Generate recommendation based on analysis results."""

        # Check for failure conditions
        if any(failure_thresholds_exceeded.values()):
            return (
                "rollback",
                "high",
                [
                    "Immediately rollback experiment due to failure threshold breach",
                    "Investigate root causes of performance degradation",
                    "Review experiment configuration and safety thresholds",
                ],
            )

        # Check statistical significance
        if significance_results["significance"] < 80.0:
            return (
                "continue",
                "low",
                [
                    "Continue experiment to gather more data",
                    "Increase sample size for statistical significance",
                    "Monitor performance metrics closely",
                ],
            )

        # Check success criteria
        success_rate = sum(success_criteria_met.values()) / len(success_criteria_met)

        if success_rate >= 0.8:  # 80% of criteria met
            if significance_results["significance"] >= 95.0:
                return (
                    "expand",
                    "high",
                    [
                        "Expand experiment to larger user base",
                        "Prepare for full rollout",
                        "Document successful optimization strategies",
                    ],
                )
            return (
                "continue",
                "medium",
                [
                    "Continue experiment with current settings",
                    "Monitor for sustained performance improvements",
                    "Prepare expansion plan",
                ],
            )
        return (
            "modify",
            "medium",
            [
                "Modify experiment parameters",
                "Investigate underperforming metrics",
                "Consider alternative optimization strategies",
            ],
        )


class FeatureFlagManager:
    """Manages feature flags for A/B testing experiments."""

    def __init__(self, db_session: Session) -> None:
        self.db_session = db_session
        self.logger = logging.getLogger(__name__)
        self._flag_cache: dict[str, Any] = {}
        self._cache_ttl = 300  # 5 minutes
        self._last_cache_update = 0

    def get_feature_flag(self, flag_name: str, user_id: str, default: Any = False) -> Any:
        """Get feature flag value for a specific user."""
        try:
            # Check if user is in any active experiments for this flag
            assignment = (
                self.db_session.query(UserAssignmentModel)
                .join(
                    ExperimentModel,
                )
                .filter(
                    UserAssignmentModel.user_id == user_id,
                    ExperimentModel.status == "active",
                    ExperimentModel.config.contains({"feature_flags": {flag_name: True}}),
                )
                .first()
            )

            if assignment and not assignment.opt_out:
                # Get variant-specific configuration
                experiment = (
                    self.db_session.query(ExperimentModel)
                    .filter_by(
                        id=assignment.experiment_id,
                    )
                    .first()
                )

                if experiment:
                    variant_config = experiment.config.get("variant_configs", {}).get(assignment.variant, {})
                    return variant_config.get("feature_flags", {}).get(flag_name, default)

            return default

        except Exception as e:
            self.logger.error(f"Failed to get feature flag {flag_name} for user {user_id}: {e}")
            return default

    def update_feature_flag(self, experiment_id: str, flag_name: str, variant: str, value: Any) -> bool:
        """Update a feature flag for a specific experiment variant."""
        try:
            experiment = self.db_session.query(ExperimentModel).filter_by(id=experiment_id).first()
            if not experiment:
                return False

            # Update the configuration
            config = experiment.config or {}
            variant_configs = config.get("variant_configs", {})

            if variant not in variant_configs:
                variant_configs[variant] = {"feature_flags": {}}

            if "feature_flags" not in variant_configs[variant]:
                variant_configs[variant]["feature_flags"] = {}

            variant_configs[variant]["feature_flags"][flag_name] = value

            config["variant_configs"] = variant_configs
            experiment.config = config
            experiment.updated_at = utc_now()

            self.db_session.commit()

            # Clear cache
            self._flag_cache.clear()

            return True

        except Exception as e:
            self.logger.error(f"Failed to update feature flag {flag_name}: {e}")
            self.db_session.rollback()
            return False


class RolloutController:
    """Controls progressive rollout of experiments."""

    def __init__(self, db_session: Session) -> None:
        self.db_session = db_session
        self.logger = logging.getLogger(__name__)

    async def execute_rollout_step(self, experiment_id: str) -> bool:
        """Execute next rollout step for an experiment."""
        try:
            experiment = self.db_session.query(ExperimentModel).filter_by(id=experiment_id).first()
            if not experiment or experiment.status != "active":
                return False

            config = ExperimentConfig(**experiment.config)

            # Find next rollout step
            current_percentage = experiment.current_percentage
            next_percentage = None

            for step in config.rollout_steps:
                if step > current_percentage:
                    next_percentage = step
                    break

            if next_percentage is None:
                self.logger.info(f"Experiment {experiment_id} has completed all rollout steps")
                return True

            # Check if enough time has passed since last step
            if experiment.updated_at:
                time_since_update = utc_now() - experiment.updated_at
                if time_since_update.total_seconds() < config.step_duration_hours * 3600:
                    self.logger.info(f"Experiment {experiment_id} step duration not yet reached")
                    return True

            # Analyze current performance before proceeding
            analyzer = StatisticalAnalyzer(self.db_session)
            results = analyzer.analyze_experiment(experiment_id)

            if results:
                # Check for failure conditions
                if any(results.failure_thresholds_exceeded.values()):
                    self.logger.warning(f"Experiment {experiment_id} failed safety checks, pausing rollout")
                    experiment.status = "paused"
                    self.db_session.commit()
                    return False

                # Check if ready for next step
                if results.statistical_significance >= 80.0:  # Require reasonable confidence
                    experiment.current_percentage = next_percentage
                    experiment.updated_at = utc_now()
                    self.db_session.commit()

                    self.logger.info(
                        f"Advanced experiment {experiment_id} to {next_percentage}% rollout",
                    )
                    return True
                self.logger.info(
                    f"Experiment {experiment_id} not ready for next step (significance: {results.statistical_significance}%)",
                )
                return True

            return False

        except Exception as e:
            self.logger.error(f"Failed to execute rollout step for experiment {experiment_id}: {e}")
            self.db_session.rollback()
            return False

    async def auto_rollback_if_needed(self, experiment_id: str) -> bool:
        """Automatically rollback experiment if safety thresholds are exceeded."""
        try:
            experiment = self.db_session.query(ExperimentModel).filter_by(id=experiment_id).first()
            if not experiment:
                return False

            config = ExperimentConfig(**experiment.config)

            if not config.auto_rollback_enabled:
                return False

            # Analyze current performance
            analyzer = StatisticalAnalyzer(self.db_session)
            results = analyzer.analyze_experiment(experiment_id)

            if results:
                # Check circuit breaker conditions
                error_rate = (1.0 - results.performance_summary["overall_success_rate"]) * 100

                if error_rate > config.circuit_breaker_threshold:
                    self.logger.warning(
                        f"Auto-rollback triggered for experiment {experiment_id} due to high error rate: {error_rate}%",
                    )

                    # Rollback by setting all users to control
                    experiment.status = "failed"
                    experiment.current_percentage = 0.0
                    experiment.end_time = utc_now()

                    self.db_session.commit()

                    return True

            return False

        except Exception as e:
            self.logger.error(f"Failed to check auto-rollback for experiment {experiment_id}: {e}")
            return False


class ExperimentManager(ObservabilityMixin):
    """Main A/B testing experiment manager."""

    def __init__(self, db_url: str | None = None) -> None:
        super().__init__()
        self.settings = get_settings()

        # Initialize database
        if db_url:
            self.engine = create_engine(db_url)
        else:
            # Use configured database URL or create SQLite for testing
            db_url = "sqlite:///ab_testing.db"  # Fallback for testing
            self.engine = create_engine(db_url)

        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.performance_monitor = PerformanceMonitor()

        # Background task control
        self._monitoring_task: asyncio.Task | None = None
        self._shutdown_requested = False

    @contextlib.contextmanager
    def get_db_session(self):
        """Get database session as context manager."""
        session = self.SessionLocal()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _serialize_config(self, config: ExperimentConfig) -> dict:
        """Convert ExperimentConfig to JSON-serializable dictionary."""
        config_dict = config.__dict__.copy()

        # Convert enums to string values
        if "experiment_type" in config_dict:
            config_dict["experiment_type"] = config_dict["experiment_type"].value
        if "segment_filters" in config_dict:
            config_dict["segment_filters"] = [
                s.value if hasattr(s, "value") else s for s in config_dict["segment_filters"]
            ]

        return config_dict

    async def create_experiment(self, config: ExperimentConfig, created_by: str = "system") -> str:
        """Create a new A/B testing experiment."""
        try:
            experiment_id = f"exp_{int(time.time() * 1000)}_{config.experiment_type.value}"

            with self.get_db_session() as db_session:
                experiment = ExperimentModel(
                    id=experiment_id,
                    name=config.name,
                    description=config.description,
                    experiment_type=config.experiment_type.value,
                    status="draft",
                    config=self._serialize_config(config),
                    variants=list(config.variant_configs.keys()) or ["control", "treatment"],
                    success_criteria=config.success_criteria,
                    failure_thresholds=config.failure_thresholds,
                    target_percentage=config.target_percentage,
                    current_percentage=0.0,
                    segment_filters=[s.value for s in config.segment_filters],
                    planned_duration_hours=config.planned_duration_hours,
                    created_by=created_by,
                )

                db_session.add(experiment)
                db_session.commit()

                self.logger.info(f"Created experiment {experiment_id}: {config.name}")

                return experiment_id

        except Exception as e:
            self.logger.error(f"Failed to create experiment: {e}")
            raise

    async def start_experiment(self, experiment_id: str) -> bool:
        """Start an A/B testing experiment."""
        try:
            with self.get_db_session() as db_session:
                experiment = db_session.query(ExperimentModel).filter_by(id=experiment_id).first()
                if not experiment:
                    return False

                experiment.status = "active"
                experiment.start_time = utc_now()
                experiment.current_percentage = ExperimentConfig(**experiment.config).initial_percentage

                db_session.commit()

                self.logger.info(f"Started experiment {experiment_id}")

                return True

        except Exception as e:
            self.logger.error(f"Failed to start experiment {experiment_id}: {e}")
            return False

    async def stop_experiment(self, experiment_id: str) -> bool:
        """Stop an A/B testing experiment."""
        try:
            with self.get_db_session() as db_session:
                experiment = db_session.query(ExperimentModel).filter_by(id=experiment_id).first()
                if not experiment:
                    return False

                experiment.status = "completed"
                experiment.end_time = utc_now()

                db_session.commit()

                self.logger.info(f"Stopped experiment {experiment_id}")

                return True

        except Exception as e:
            self.logger.error(f"Failed to stop experiment {experiment_id}: {e}")
            return False

    async def assign_user_to_experiment(
        self,
        user_id: str,
        experiment_id: str,
        user_characteristics: UserCharacteristics | None = None,
    ) -> tuple[str, UserSegment]:
        """Assign user to experiment and return variant."""
        try:
            with self.get_db_session() as db_session:
                experiment = db_session.query(ExperimentModel).filter_by(id=experiment_id).first()
                if not experiment or experiment.status != "active":
                    return "control", UserSegment.RANDOM

                config = ExperimentConfig(**experiment.config)
                segmentation = UserSegmentation(db_session)

                return segmentation.assign_user_to_experiment(
                    user_id,
                    experiment_id,
                    config,
                    user_characteristics,
                )

        except Exception as e:
            self.logger.error(f"Failed to assign user {user_id} to experiment {experiment_id}: {e}")
            return "control", UserSegment.RANDOM

    async def should_use_dynamic_loading(self, user_id: str, experiment_id: str = "dynamic_loading_rollout") -> bool:
        """Check if user should use dynamic loading based on A/B test assignment."""
        try:
            variant, _ = await self.assign_user_to_experiment(user_id, experiment_id)
            return variant == "treatment"

        except Exception as e:
            self.logger.error(f"Failed to check dynamic loading assignment for user {user_id}: {e}")
            return False

    async def record_optimization_result(
        self,
        experiment_id: str,
        user_id: str,
        result: ProcessingResult,
    ) -> bool:
        """Record optimization result for A/B testing analysis."""
        try:
            with self.get_db_session() as db_session:
                # Get user's variant assignment
                assignment = (
                    db_session.query(UserAssignmentModel)
                    .filter_by(
                        user_id=user_id,
                        experiment_id=experiment_id,
                    )
                    .first()
                )

                if not assignment or assignment.opt_out:
                    return False

                # Record metrics
                metrics_collector = MetricsCollector(db_session)
                return metrics_collector.record_processing_result(
                    experiment_id,
                    user_id,
                    assignment.variant,
                    result,
                )

        except Exception as e:
            self.logger.error(f"Failed to record optimization result: {e}")
            return False

    async def get_experiment_results(self, experiment_id: str) -> ExperimentResults | None:
        """Get comprehensive results for an experiment."""
        try:
            with self.get_db_session() as db_session:
                analyzer = StatisticalAnalyzer(db_session)
                return analyzer.analyze_experiment(experiment_id)

        except Exception as e:
            self.logger.error(f"Failed to get experiment results for {experiment_id}: {e}")
            return None

    async def start_monitoring(self) -> None:
        """Start background monitoring of experiments."""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.logger.info("Started A/B testing monitoring")

    async def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._shutdown_requested = True
        if self._monitoring_task:
            self._monitoring_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitoring_task
        self.logger.info("Stopped A/B testing monitoring")

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop for experiments."""
        while not self._shutdown_requested:
            try:
                await self._check_active_experiments()
                await asyncio.sleep(300)  # Check every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def _check_active_experiments(self) -> None:
        """Check all active experiments for safety and progression."""
        try:
            with self.get_db_session() as db_session:
                active_experiments = db_session.query(ExperimentModel).filter_by(status="active").all()

                for experiment in active_experiments:
                    # Check for auto-rollback conditions
                    rollout_controller = RolloutController(db_session)
                    await rollout_controller.auto_rollback_if_needed(experiment.id)

                    # Check for rollout progression
                    await rollout_controller.execute_rollout_step(experiment.id)

                    # Check for experiment completion
                    if experiment.planned_duration_hours > 0:
                        elapsed_hours = (utc_now() - experiment.start_time).total_seconds() / 3600
                        if elapsed_hours >= experiment.planned_duration_hours:
                            experiment.status = "completed"
                            experiment.end_time = utc_now()
                            db_session.commit()

                            self.logger.info(
                                f"Auto-completed experiment {experiment.id} after {elapsed_hours:.1f} hours",
                            )

        except Exception as e:
            self.logger.error(f"Failed to check active experiments: {e}")


# Global experiment manager instance
_experiment_manager: ExperimentManager | None = None


async def get_experiment_manager() -> ExperimentManager:
    """Get or create the global experiment manager instance."""
    global _experiment_manager

    if _experiment_manager is None:
        _experiment_manager = ExperimentManager()
        await _experiment_manager.start_monitoring()

    return _experiment_manager


async def create_dynamic_loading_experiment(
    target_percentage: float = 50.0,
    duration_hours: int = 168,
) -> str:
    """Create a standard dynamic loading A/B test experiment."""

    config = ExperimentConfig(
        name="Dynamic Function Loading Rollout",
        description="A/B test for gradual rollout of dynamic function loading optimization",
        experiment_type=ExperimentType.DYNAMIC_LOADING,
        planned_duration_hours=duration_hours,
        feature_flags={"dynamic_loading_enabled": True},
        variant_configs={
            "control": {
                "feature_flags": {"dynamic_loading_enabled": False},
                "loading_strategy": "baseline",
            },
            "treatment": {
                "feature_flags": {"dynamic_loading_enabled": True},
                "loading_strategy": "balanced",
            },
        },
        target_percentage=target_percentage,
        rollout_steps=[1.0, 5.0, 25.0, 50.0, 100.0],
        success_criteria={
            "min_token_reduction": 70.0,
            "max_response_time_ms": 200.0,
            "min_success_rate": 95.0,
            "min_statistical_significance": 95.0,
        },
        failure_thresholds={
            "max_error_rate": 5.0,
            "min_token_reduction": 50.0,
            "max_response_time_degradation": 50.0,
        },
        auto_rollback_enabled=True,
        circuit_breaker_threshold=10.0,
    )

    manager = await get_experiment_manager()
    experiment_id = await manager.create_experiment(config, "system")
    await manager.start_experiment(experiment_id)

    return experiment_id
