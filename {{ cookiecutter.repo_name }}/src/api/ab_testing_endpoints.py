"""
FastAPI Endpoints for A/B Testing Framework

This module provides RESTful API endpoints for managing A/B testing experiments,
monitoring performance, and accessing real-time dashboard data.

Key Endpoints:
- Experiment management (create, start, stop, configure)
- User assignment and feature flag resolution
- Metrics collection and analysis
- Dashboard data and visualizations
- Real-time monitoring and alerting
- Statistical analysis and reporting

Security Features:
- Rate limiting for API protection
- Input validation and sanitization
- Audit logging for all operations
- Authentication and authorization
"""

import logging
import time
from datetime import datetime
from src.utils.datetime_compat import utc_now
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.core.ab_testing_framework import (
    ExperimentConfig,
    ExperimentManager,
    ExperimentType,
    UserCharacteristics,
    UserSegment,
    create_dynamic_loading_experiment,
    get_experiment_manager,
)
from src.monitoring.ab_testing_dashboard import get_dashboard_instance
from src.security.audit_logging import audit_logger_instance
from src.security.rate_limiting import RateLimits, rate_limit

logger = logging.getLogger(__name__)

# Create router for A/B testing endpoints
router = APIRouter(prefix="/api/v1/ab-testing", tags=["ab-testing"])


# Request/Response Models


class CreateExperimentRequest(BaseModel):
    """Request model for creating A/B experiments."""

    name: str = Field(..., min_length=1, max_length=200, description="Experiment name")
    description: str = Field(..., min_length=1, max_length=1000, description="Experiment description")
    experiment_type: str = Field(default="dynamic_loading", description="Type of experiment")

    # Duration and rollout
    planned_duration_hours: int = Field(default=168, ge=1, le=8760, description="Planned duration in hours")
    target_percentage: float = Field(default=50.0, ge=0.1, le=100.0, description="Target rollout percentage")
    rollout_steps: list[float] = Field(default=[1.0, 5.0, 25.0, 50.0], description="Rollout percentage steps")
    step_duration_hours: int = Field(default=24, ge=1, le=168, description="Duration per rollout step")

    # Feature configuration
    feature_flags: dict[str, Any] = Field(default_factory=dict, description="Feature flags for experiment")
    variant_configs: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Variant-specific configurations",
    )

    # Success criteria
    success_criteria: dict[str, float] = Field(default_factory=dict, description="Success criteria thresholds")
    failure_thresholds: dict[str, float] = Field(default_factory=dict, description="Failure threshold values")

    # User segmentation
    segment_filters: list[str] = Field(default=["random"], description="User segment filters")
    exclude_segments: list[str] = Field(default_factory=list, description="Segments to exclude")
    opt_in_only: bool = Field(default=False, description="Require explicit opt-in")

    # Safety settings
    auto_rollback_enabled: bool = Field(default=True, description="Enable automatic rollback")
    circuit_breaker_threshold: float = Field(
        default=10.0,
        ge=0.0,
        le=100.0,
        description="Error rate threshold for circuit breaker",
    )

    @field_validator("experiment_type")
    @classmethod
    def validate_experiment_type(cls, v):
        valid_types = ["dynamic_loading", "optimization_strategy", "user_interface", "performance"]
        if v not in valid_types:
            raise ValueError(f"Experiment type must be one of: {valid_types}")
        return v

    @field_validator("rollout_steps")
    @classmethod
    def validate_rollout_steps(cls, v):
        if not v or len(v) > 10:
            raise ValueError("Rollout steps must be provided and contain at most 10 steps")

        for step in v:
            if not 0.1 <= step <= 100.0:
                raise ValueError("Each rollout step must be between 0.1 and 100.0")

        # Ensure steps are in ascending order
        if sorted(v) != v:
            raise ValueError("Rollout steps must be in ascending order")

        return v


class UserAssignmentRequest(BaseModel):
    """Request model for user assignment to experiments."""

    user_id: str = Field(..., min_length=1, max_length=100, description="User identifier")
    experiment_id: str = Field(..., min_length=1, max_length=100, description="Experiment identifier")

    # Optional user characteristics
    usage_frequency: str | None = Field(default=None, description="User usage frequency")
    feature_usage_pattern: str | None = Field(default=None, description="User feature usage pattern")
    is_early_adopter: bool = Field(default=False, description="Whether user is an early adopter")
    opt_in_beta: bool = Field(default=False, description="Whether user opted into beta features")

    @field_validator("usage_frequency")
    @classmethod
    def validate_usage_frequency(cls, v):
        if v is not None and v not in ["low", "medium", "high"]:
            raise ValueError("Usage frequency must be 'low', 'medium', or 'high'")
        return v

    @field_validator("feature_usage_pattern")
    @classmethod
    def validate_feature_pattern(cls, v):
        if v is not None and v not in ["basic", "intermediate", "advanced"]:
            raise ValueError("Feature usage pattern must be 'basic', 'intermediate', or 'advanced'")
        return v


class MetricEventRequest(BaseModel):
    """Request model for recording metric events."""

    experiment_id: str = Field(..., min_length=1, max_length=100, description="Experiment identifier")
    user_id: str = Field(..., min_length=1, max_length=100, description="User identifier")
    event_type: str = Field(..., description="Type of metric event")
    event_name: str = Field(..., description="Name of the metric event")

    # Event data
    event_value: float | None = Field(default=None, description="Numeric value for the event")
    event_data: dict[str, Any] | None = Field(default=None, description="Additional event data")
    session_id: str | None = Field(default=None, description="Session identifier")

    # Performance metrics
    response_time_ms: float | None = Field(default=None, ge=0.0, description="Response time in milliseconds")
    token_reduction_percentage: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Token reduction percentage",
    )
    success: bool | None = Field(default=None, description="Whether operation was successful")
    error_message: str | None = Field(default=None, max_length=500, description="Error message if operation failed")

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, v):
        valid_types = ["performance", "conversion", "error", "engagement", "business"]
        if v not in valid_types:
            raise ValueError(f"Event type must be one of: {valid_types}")
        return v


class ExperimentResponse(BaseModel):
    """Response model for experiment data."""

    id: str
    name: str
    description: str
    experiment_type: str
    status: str

    # Configuration
    target_percentage: float
    current_percentage: float
    planned_duration_hours: int

    # Results
    total_users: int = 0
    statistical_significance: float = 0.0

    # Timing
    created_at: str
    start_time: str | None = None
    end_time: str | None = None

    # Health
    active_alerts: int = 0
    risk_level: str = "unknown"


class UserAssignmentResponse(BaseModel):
    """Response model for user assignment."""

    user_id: str
    experiment_id: str
    variant: str
    segment: str
    assignment_time: str
    success: bool
    message: str


class DashboardResponse(BaseModel):
    """Response model for dashboard data."""

    experiment_id: str
    experiment_name: str
    status: str

    # Key metrics
    total_users: int
    statistical_significance: float
    success_rate: float
    error_rate: float
    avg_response_time_ms: float
    avg_token_reduction: float

    # Risk assessment
    risk_level: str
    confidence_level: str
    active_alerts: int

    # Recommendations
    recommendations: list[str]

    # Last updated
    last_updated: str


class ExperimentResultsResponse(BaseModel):
    """Response model for experiment results."""

    experiment_id: str
    experiment_name: str
    total_users: int

    # Statistical analysis
    statistical_significance: float
    confidence_interval: list[float]
    p_value: float
    effect_size: float

    # Performance summary
    performance_summary: dict[str, Any]

    # Success criteria
    success_criteria_met: dict[str, bool]
    failure_thresholds_exceeded: dict[str, bool]

    # Recommendation
    recommendation: str
    confidence_level: str
    next_actions: list[str]


# Dependency for getting experiment manager
async def get_experiment_manager_dependency() -> ExperimentManager:
    """Dependency to get the experiment manager instance."""
    try:
        return await get_experiment_manager()
    except Exception as e:
        logger.error(f"Failed to get experiment manager: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="A/B testing system unavailable",
        )


# Experiment Management Endpoints


@router.post("/experiments", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
@rate_limit(RateLimits.API_DEFAULT)
async def create_experiment(
    request: Request,
    experiment_request: CreateExperimentRequest,
    manager: ExperimentManager = Depends(get_experiment_manager_dependency),
) -> ExperimentResponse:
    """
    Create a new A/B testing experiment.

    This endpoint allows creation of experiments with comprehensive configuration
    including rollout strategy, success criteria, and safety thresholds.
    """
    start_time = time.perf_counter()

    try:
        # Convert request to experiment config
        config = ExperimentConfig(
            name=experiment_request.name,
            description=experiment_request.description,
            experiment_type=ExperimentType(experiment_request.experiment_type),
            planned_duration_hours=experiment_request.planned_duration_hours,
            feature_flags=experiment_request.feature_flags,
            variant_configs=experiment_request.variant_configs,
            target_percentage=experiment_request.target_percentage,
            rollout_steps=experiment_request.rollout_steps,
            step_duration_hours=experiment_request.step_duration_hours,
            segment_filters=[UserSegment(s) for s in experiment_request.segment_filters],
            exclude_segments=[UserSegment(s) for s in experiment_request.exclude_segments],
            opt_in_only=experiment_request.opt_in_only,
            success_criteria=experiment_request.success_criteria,
            failure_thresholds=experiment_request.failure_thresholds,
            auto_rollback_enabled=experiment_request.auto_rollback_enabled,
            circuit_breaker_threshold=experiment_request.circuit_breaker_threshold,
        )

        # Create experiment
        experiment_id = await manager.create_experiment(config, created_by="api")

        # Get experiment details
        with manager.get_db_session() as db_session:
            from src.core.ab_testing_framework import ExperimentModel

            experiment = db_session.query(ExperimentModel).filter_by(id=experiment_id).first()

            if not experiment:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to create experiment",
                )

        # Log API usage
        processing_time = (time.perf_counter() - start_time) * 1000
        audit_logger_instance.log_api_event(
            request=request,
            response_status=status.HTTP_201_CREATED,
            processing_time=processing_time / 1000,
            additional_data={
                "experiment_id": experiment_id,
                "experiment_name": experiment_request.name,
                "experiment_type": experiment_request.experiment_type,
                "target_percentage": experiment_request.target_percentage,
            },
        )

        return ExperimentResponse(
            id=experiment.id,
            name=experiment.name,
            description=experiment.description,
            experiment_type=experiment.experiment_type,
            status=experiment.status,
            target_percentage=experiment.target_percentage,
            current_percentage=experiment.current_percentage,
            planned_duration_hours=experiment.planned_duration_hours,
            total_users=experiment.total_users,
            statistical_significance=experiment.statistical_significance,
            created_at=experiment.created_at.isoformat(),
            start_time=experiment.start_time.isoformat() if experiment.start_time else None,
            end_time=experiment.end_time.isoformat() if experiment.end_time else None,
            risk_level="low",
        )

    except ValueError as e:
        logger.warning(f"Invalid experiment configuration: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create experiment: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create experiment")


@router.post("/experiments/{experiment_id}/start")
@rate_limit(RateLimits.API_DEFAULT)
async def start_experiment(
    request: Request,
    experiment_id: str,
    manager: ExperimentManager = Depends(get_experiment_manager_dependency),
) -> JSONResponse:
    """Start an A/B testing experiment."""
    try:
        success = await manager.start_experiment(experiment_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experiment not found or cannot be started",
            )

        # Log API usage
        audit_logger_instance.log_api_event(
            request=request,
            response_status=status.HTTP_200_OK,
            processing_time=0.1,
            additional_data={"experiment_id": experiment_id, "action": "start"},
        )

        return JSONResponse(
            content={"success": True, "message": f"Experiment {experiment_id} started successfully"},
            status_code=status.HTTP_200_OK,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start experiment {experiment_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to start experiment")


@router.post("/experiments/{experiment_id}/stop")
@rate_limit(RateLimits.API_DEFAULT)
async def stop_experiment(
    request: Request,
    experiment_id: str,
    manager: ExperimentManager = Depends(get_experiment_manager_dependency),
) -> JSONResponse:
    """Stop an A/B testing experiment."""
    try:
        success = await manager.stop_experiment(experiment_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experiment not found or cannot be stopped",
            )

        # Log API usage
        audit_logger_instance.log_api_event(
            request=request,
            response_status=status.HTTP_200_OK,
            processing_time=0.1,
            additional_data={"experiment_id": experiment_id, "action": "stop"},
        )

        return JSONResponse(
            content={"success": True, "message": f"Experiment {experiment_id} stopped successfully"},
            status_code=status.HTTP_200_OK,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop experiment {experiment_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to stop experiment")


@router.get("/experiments", response_model=list[ExperimentResponse])
@rate_limit(RateLimits.API_DEFAULT)
async def list_experiments(
    request: Request,
    status_filter: str | None = None,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    manager: ExperimentManager = Depends(get_experiment_manager_dependency),
) -> list[ExperimentResponse]:
    """List all A/B testing experiments with optional filtering."""
    try:
        with manager.get_db_session() as db_session:
            from src.core.ab_testing_framework import ExperimentModel

            query = db_session.query(ExperimentModel)

            if status_filter:
                query = query.filter(ExperimentModel.status == status_filter)

            experiments = query.offset(offset).limit(limit).all()

            results = []
            for experiment in experiments:
                results.append(
                    ExperimentResponse(
                        id=experiment.id,
                        name=experiment.name,
                        description=experiment.description,
                        experiment_type=experiment.experiment_type,
                        status=experiment.status,
                        target_percentage=experiment.target_percentage,
                        current_percentage=experiment.current_percentage,
                        planned_duration_hours=experiment.planned_duration_hours,
                        total_users=experiment.total_users,
                        statistical_significance=experiment.statistical_significance,
                        created_at=experiment.created_at.isoformat(),
                        start_time=experiment.start_time.isoformat() if experiment.start_time else None,
                        end_time=experiment.end_time.isoformat() if experiment.end_time else None,
                        risk_level="unknown",
                    ),
                )

            return results

    except Exception as e:
        logger.error(f"Failed to list experiments: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list experiments")


@router.get("/experiments/{experiment_id}", response_model=ExperimentResponse)
@rate_limit(RateLimits.API_DEFAULT)
async def get_experiment(
    request: Request,
    experiment_id: str,
    manager: ExperimentManager = Depends(get_experiment_manager_dependency),
) -> ExperimentResponse:
    """Get details of a specific experiment."""
    try:
        with manager.get_db_session() as db_session:
            from src.core.ab_testing_framework import ExperimentModel

            experiment = db_session.query(ExperimentModel).filter_by(id=experiment_id).first()

            if not experiment:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found")

            return ExperimentResponse(
                id=experiment.id,
                name=experiment.name,
                description=experiment.description,
                experiment_type=experiment.experiment_type,
                status=experiment.status,
                target_percentage=experiment.target_percentage,
                current_percentage=experiment.current_percentage,
                planned_duration_hours=experiment.planned_duration_hours,
                total_users=experiment.total_users,
                statistical_significance=experiment.statistical_significance,
                created_at=experiment.created_at.isoformat(),
                start_time=experiment.start_time.isoformat() if experiment.start_time else None,
                end_time=experiment.end_time.isoformat() if experiment.end_time else None,
                risk_level="unknown",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get experiment {experiment_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get experiment")


# User Assignment Endpoints


@router.post("/assign-user", response_model=UserAssignmentResponse)
@rate_limit(RateLimits.API_DEFAULT)
async def assign_user_to_experiment(
    request: Request,
    assignment_request: UserAssignmentRequest,
    manager: ExperimentManager = Depends(get_experiment_manager_dependency),
) -> UserAssignmentResponse:
    """Assign a user to an experiment variant."""
    try:
        # Create user characteristics
        user_characteristics = UserCharacteristics(
            user_id=assignment_request.user_id,
            usage_frequency=assignment_request.usage_frequency or "unknown",
            feature_usage_pattern=assignment_request.feature_usage_pattern or "unknown",
            is_early_adopter=assignment_request.is_early_adopter,
            opt_in_beta=assignment_request.opt_in_beta,
        )

        # Assign user to experiment
        variant, segment = await manager.assign_user_to_experiment(
            assignment_request.user_id,
            assignment_request.experiment_id,
            user_characteristics,
        )

        # Log API usage
        audit_logger_instance.log_api_event(
            request=request,
            response_status=status.HTTP_200_OK,
            processing_time=0.1,
            additional_data={
                "user_id": assignment_request.user_id,
                "experiment_id": assignment_request.experiment_id,
                "variant": variant,
                "segment": segment.value,
            },
        )

        return UserAssignmentResponse(
            user_id=assignment_request.user_id,
            experiment_id=assignment_request.experiment_id,
            variant=variant,
            segment=segment.value,
            assignment_time=utc_now().isoformat(),
            success=True,
            message="User assigned successfully",
        )

    except Exception as e:
        logger.error(f"Failed to assign user to experiment: {e}")
        return UserAssignmentResponse(
            user_id=assignment_request.user_id,
            experiment_id=assignment_request.experiment_id,
            variant="control",
            segment="random",
            assignment_time=utc_now().isoformat(),
            success=False,
            message=f"Assignment failed: {e!s}",
        )


@router.get("/check-dynamic-loading/{user_id}")
@rate_limit(RateLimits.API_DEFAULT)
async def check_dynamic_loading_assignment(
    request: Request,
    user_id: str,
    experiment_id: str = "dynamic_loading_rollout",
    manager: ExperimentManager = Depends(get_experiment_manager_dependency),
) -> JSONResponse:
    """Check if a user should use dynamic loading based on A/B test assignment."""
    try:
        should_use = await manager.should_use_dynamic_loading(user_id, experiment_id)

        return JSONResponse(
            content={
                "user_id": user_id,
                "experiment_id": experiment_id,
                "use_dynamic_loading": should_use,
                "timestamp": utc_now().isoformat(),
            },
        )

    except Exception as e:
        logger.error(f"Failed to check dynamic loading assignment: {e}")
        return JSONResponse(
            content={
                "user_id": user_id,
                "experiment_id": experiment_id,
                "use_dynamic_loading": False,
                "error": str(e),
                "timestamp": utc_now().isoformat(),
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# Metrics Collection Endpoints


@router.post("/metrics/record-event")
@rate_limit(RateLimits.API_DEFAULT)
async def record_metric_event(
    request: Request,
    metric_request: MetricEventRequest,
    manager: ExperimentManager = Depends(get_experiment_manager_dependency),
) -> JSONResponse:
    """Record a metric event for A/B testing analysis."""
    try:
        from src.core.ab_testing_framework import MetricEvent

        # Create metric event
        event = MetricEvent(
            experiment_id=metric_request.experiment_id,
            user_id=metric_request.user_id,
            variant="unknown",  # Will be resolved by the system
            event_type=metric_request.event_type,
            event_name=metric_request.event_name,
            event_value=metric_request.event_value,
            event_data=metric_request.event_data,
            session_id=metric_request.session_id,
            response_time_ms=metric_request.response_time_ms,
            token_reduction_percentage=metric_request.token_reduction_percentage,
            success=metric_request.success,
            error_message=metric_request.error_message,
        )

        # Record through metrics collector
        with manager.get_db_session() as db_session:
            from src.core.ab_testing_framework import MetricsCollector

            collector = MetricsCollector(db_session)
            success = collector.record_event(event)

        if success:
            return JSONResponse(
                content={"success": True, "message": "Metric event recorded successfully"},
            )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to record metric event")

    except Exception as e:
        logger.error(f"Failed to record metric event: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to record metric event")


# Dashboard and Monitoring Endpoints


@router.get("/dashboard/{experiment_id}", response_class=HTMLResponse)
@rate_limit(RateLimits.API_DEFAULT)
async def get_experiment_dashboard(
    request: Request,
    experiment_id: str,
) -> HTMLResponse:
    """Get HTML dashboard for an experiment."""
    try:
        dashboard = await get_dashboard_instance()
        html_content = await dashboard.generate_dashboard_html(experiment_id)

        return HTMLResponse(content=html_content)

    except Exception as e:
        logger.error(f"Failed to generate dashboard for experiment {experiment_id}: {e}")
        error_html = f"""
        <html>
        <head><title>Dashboard Error</title></head>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h1>Dashboard Error</h1>
            <p>Failed to load dashboard for experiment {experiment_id}</p>
            <p>Error: {e!s}</p>
            <button onclick="window.location.reload()">Retry</button>
        </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


@router.get("/dashboard-data/{experiment_id}", response_model=DashboardResponse)
@rate_limit(RateLimits.API_DEFAULT)
async def get_dashboard_data(
    request: Request,
    experiment_id: str,
) -> DashboardResponse:
    """Get dashboard data as JSON for an experiment."""
    try:
        dashboard = await get_dashboard_instance()
        dashboard_data = await dashboard.get_dashboard_data(experiment_id)

        if not dashboard_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experiment not found or no data available",
            )

        return DashboardResponse(
            experiment_id=dashboard_data["experiment_id"],
            experiment_name=dashboard_data["experiment_name"],
            status=dashboard_data["status"],
            total_users=dashboard_data["total_users"],
            statistical_significance=dashboard_data["statistical_significance"],
            success_rate=dashboard_data["success_rate"],
            error_rate=dashboard_data["error_rate"],
            avg_response_time_ms=dashboard_data["avg_response_time_ms"],
            avg_token_reduction=dashboard_data["avg_token_reduction"],
            risk_level=dashboard_data["risk_level"],
            confidence_level=dashboard_data["confidence_level"],
            active_alerts=len(dashboard_data["active_alerts"]),
            recommendations=dashboard_data["recommendations"],
            last_updated=dashboard_data["last_updated"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dashboard data for experiment {experiment_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get dashboard data")


@router.get("/experiments/{experiment_id}/results", response_model=ExperimentResultsResponse)
@rate_limit(RateLimits.API_DEFAULT)
async def get_experiment_results(
    request: Request,
    experiment_id: str,
    manager: ExperimentManager = Depends(get_experiment_manager_dependency),
) -> ExperimentResultsResponse:
    """Get comprehensive results and analysis for an experiment."""
    try:
        results = await manager.get_experiment_results(experiment_id)

        if not results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experiment not found or no results available",
            )

        return ExperimentResultsResponse(
            experiment_id=results.experiment_id,
            experiment_name=results.experiment_name,
            total_users=results.total_users,
            statistical_significance=results.statistical_significance,
            confidence_interval=list(results.confidence_interval),
            p_value=results.p_value,
            effect_size=results.effect_size,
            performance_summary=results.performance_summary,
            success_criteria_met=results.success_criteria_met,
            failure_thresholds_exceeded=results.failure_thresholds_exceeded,
            recommendation=results.recommendation,
            confidence_level=results.confidence_level,
            next_actions=results.next_actions,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get experiment results for {experiment_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get experiment results",
        )


@router.get("/overview", response_model=list[ExperimentResponse])
@rate_limit(RateLimits.API_DEFAULT)
async def get_experiments_overview(
    request: Request,
) -> list[ExperimentResponse]:
    """Get overview of all experiments for main dashboard."""
    try:
        dashboard = await get_dashboard_instance()
        summaries = await dashboard.get_experiment_summary()

        results = []
        for summary in summaries:
            results.append(
                ExperimentResponse(
                    id=summary["id"],
                    name=summary["name"],
                    status=summary["status"],
                    experiment_type="unknown",  # Would need to be added to summary
                    description="",  # Would need to be added to summary
                    target_percentage=summary.get("target_percentage", 0.0),
                    current_percentage=summary.get("current_percentage", 0.0),
                    planned_duration_hours=0,  # Would need to be added to summary
                    total_users=summary.get("total_users", 0),
                    statistical_significance=summary.get("statistical_significance", 0.0),
                    created_at=summary["created_at"],
                    start_time=summary.get("start_time"),
                    end_time=summary.get("end_time"),
                    active_alerts=summary.get("active_alerts", 0),
                    risk_level=summary.get("risk_level", "unknown"),
                ),
            )

        return results

    except Exception as e:
        logger.error(f"Failed to get experiments overview: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get experiments overview",
        )


# Quick Setup Endpoints


@router.post("/quick-setup/dynamic-loading")
@rate_limit(RateLimits.API_DEFAULT)
async def quick_setup_dynamic_loading_experiment(
    request: Request,
    target_percentage: float = Query(default=50.0, ge=0.1, le=100.0),
    duration_hours: int = Query(default=168, ge=1, le=8760),
    manager: ExperimentManager = Depends(get_experiment_manager_dependency),
) -> JSONResponse:
    """Quickly set up a standard dynamic loading A/B test experiment."""
    try:
        experiment_id = await create_dynamic_loading_experiment(
            target_percentage=target_percentage,
            duration_hours=duration_hours,
        )

        # Log API usage
        audit_logger_instance.log_api_event(
            request=request,
            response_status=status.HTTP_201_CREATED,
            processing_time=0.5,
            additional_data={
                "experiment_id": experiment_id,
                "experiment_type": "dynamic_loading",
                "target_percentage": target_percentage,
                "duration_hours": duration_hours,
            },
        )

        return JSONResponse(
            content={
                "success": True,
                "experiment_id": experiment_id,
                "message": "Dynamic loading experiment created and started successfully",
                "dashboard_url": f"/api/v1/ab-testing/dashboard/{experiment_id}",
            },
            status_code=status.HTTP_201_CREATED,
        )

    except Exception as e:
        logger.error(f"Failed to create dynamic loading experiment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create dynamic loading experiment",
        )


# Health and Status Endpoints


@router.get("/health")
@rate_limit(RateLimits.HEALTH_CHECK)
async def ab_testing_health_check(request: Request) -> JSONResponse:
    """Health check endpoint for A/B testing system."""
    try:
        manager = await get_experiment_manager()

        # Quick health check
        with manager.get_db_session() as db_session:
            # Test database connectivity
            from sqlalchemy import text

            db_session.execute(text("SELECT 1"))

        return JSONResponse(
            content={
                "status": "healthy",
                "service": "ab-testing",
                "timestamp": utc_now().isoformat(),
                "database_connected": True,
            },
        )

    except Exception as e:
        logger.error(f"A/B testing health check failed: {e}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "service": "ab-testing",
                "timestamp": utc_now().isoformat(),
                "error": str(e),
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
