"""
FastAPI Endpoints for Dynamic Function Loading Integration

This module provides RESTful API endpoints that demonstrate the dynamic function
loading system integrated with the existing PromptCraft FastAPI application.
These endpoints showcase real-world usage patterns and provide a practical
interface for testing and validation.

Key Endpoints:
- Query processing with optimization
- Real-time performance monitoring
- User command execution
- System health and metrics
- Interactive demonstration interface

The endpoints follow PromptCraft's security standards with rate limiting,
input validation, and comprehensive error handling.
"""

import logging
import time
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.core.comprehensive_prototype_demo import ComprehensivePrototypeDemo
from src.core.dynamic_function_loader import LoadingStrategy
from src.core.dynamic_loading_integration import DynamicLoadingIntegration, IntegrationMode, get_integration_instance
from src.security.audit_logging import audit_logger_instance
from src.security.rate_limiting import RateLimits, rate_limit

logger = logging.getLogger(__name__)

# Create router for dynamic loading endpoints
router = APIRouter(prefix="/api/v1/dynamic-loading", tags=["dynamic-loading"])


# Request/Response Models


class QueryOptimizationRequest(BaseModel):
    """Request model for query optimization."""

    query: str = Field(..., min_length=1, max_length=2000, description="Query to optimize")
    user_id: str = Field(default="api_user", max_length=100, description="User identifier")
    strategy: str = Field(default="balanced", description="Loading strategy")
    user_commands: list[str] | None = Field(default=None, description="Optional user commands")

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v):
        valid_strategies = ["conservative", "balanced", "aggressive"]
        if v.lower() not in valid_strategies:
            raise ValueError(f"Strategy must be one of: {valid_strategies}")
        return v.lower()

    @field_validator("user_commands")
    @classmethod
    def validate_commands(cls, v):
        if v is not None:
            if len(v) > 10:
                raise ValueError("Maximum 10 user commands allowed")
            for cmd in v:
                if not cmd.startswith("/"):
                    raise ValueError("User commands must start with '/'")
        return v


class QueryOptimizationResponse(BaseModel):
    """Response model for query optimization."""

    success: bool
    session_id: str
    processing_time_ms: float
    optimization_results: dict[str, Any]
    user_commands_executed: int
    error_message: str | None = None


class SystemStatusResponse(BaseModel):
    """Response model for system status."""

    integration_health: str
    mode: str
    metrics: dict[str, Any]
    components: dict[str, bool]
    features: dict[str, bool]
    uptime_hours: float


class PerformanceReportResponse(BaseModel):
    """Response model for performance reports."""

    report_timestamp: str
    integration_metrics: dict[str, Any]
    performance_summary: dict[str, Any]
    timing_analysis: dict[str, Any]
    system_health: dict[str, Any]


class UserCommandRequest(BaseModel):
    """Request model for user command execution."""

    command: str = Field(..., min_length=1, max_length=200, description="Command to execute")
    session_id: str | None = Field(default=None, description="Optional session ID")
    user_id: str = Field(default="api_user", max_length=100, description="User identifier")

    @field_validator("command")
    @classmethod
    def validate_command(cls, v):
        if not v.startswith("/"):
            raise ValueError("Commands must start with '/'")
        return v


class UserCommandResponse(BaseModel):
    """Response model for user command execution."""

    success: bool
    message: str
    command: str
    data: dict[str, Any] | None = None
    suggestions: list[str] | None = None


class DemoRunRequest(BaseModel):
    """Request model for running demonstrations."""

    scenario_types: list[str] | None = Field(default=None, description="Specific scenario types to run")
    export_results: bool = Field(default=False, description="Whether to include detailed results")

    @field_validator("scenario_types")
    @classmethod
    def validate_scenario_types(cls, v):
        if v is not None:
            valid_types = [
                "basic_optimization",
                "complex_workflow",
                "user_interaction",
                "performance_stress",
                "real_world_use_case",
                "edge_case_handling",
            ]
            for scenario_type in v:
                if scenario_type not in valid_types:
                    raise ValueError(f"Invalid scenario type: {scenario_type}. Valid types: {valid_types}")
        return v


# Dependency for getting integration instance
async def get_integration_dependency() -> DynamicLoadingIntegration:
    """Dependency to get the dynamic loading integration instance."""
    try:
        return await get_integration_instance(mode=IntegrationMode.PRODUCTION)
    except Exception as e:
        logger.error(f"Failed to get integration instance: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dynamic loading system unavailable",
        )


# Main API Endpoints


@router.post("/optimize-query", response_model=QueryOptimizationResponse)
@rate_limit(RateLimits.API_DEFAULT)
async def optimize_query(
    request: Request,
    query_request: QueryOptimizationRequest,
    integration: DynamicLoadingIntegration = Depends(get_integration_dependency),
) -> QueryOptimizationResponse:
    """
    Optimize a query using the dynamic function loading system.

    This endpoint demonstrates the core functionality by processing a user query
    and applying intelligent function loading optimizations.
    """
    start_time = time.perf_counter()

    try:
        # Convert strategy string to enum
        strategy_map = {
            "conservative": LoadingStrategy.CONSERVATIVE,
            "balanced": LoadingStrategy.BALANCED,
            "aggressive": LoadingStrategy.AGGRESSIVE,
        }
        strategy = strategy_map[query_request.strategy]

        # Process the query through the integrated system
        result = await integration.process_query(
            query=query_request.query,
            user_id=query_request.user_id,
            strategy=strategy,
            user_commands=query_request.user_commands,
        )

        # Log API usage
        processing_time = (time.perf_counter() - start_time) * 1000
        audit_logger_instance.log_api_event(
            request=request,
            response_status=status.HTTP_200_OK,
            processing_time=processing_time / 1000,  # Convert to seconds
            additional_data={
                "query_length": len(query_request.query),
                "strategy": query_request.strategy,
                "token_reduction": result.reduction_percentage,
                "user_commands": len(query_request.user_commands or []),
            },
        )

        return QueryOptimizationResponse(
            success=result.success,
            session_id=result.session_id,
            processing_time_ms=result.total_time_ms,
            optimization_results={
                "baseline_tokens": result.baseline_tokens,
                "optimized_tokens": result.optimized_tokens,
                "reduction_percentage": result.reduction_percentage,
                "target_achieved": result.target_achieved,
                "detection_result": {
                    "categories": result.detection_result.categories,
                    "confidence": result.detection_result.confidence,
                    "reasoning": result.detection_result.reasoning,
                },
                "performance_metrics": {
                    "detection_time_ms": result.detection_time_ms,
                    "loading_time_ms": result.loading_time_ms,
                    "cache_hit": result.cache_hit,
                    "fallback_used": result.fallback_used,
                },
            },
            user_commands_executed=len(result.user_commands),
            error_message=result.error_message,
        )

    except Exception as e:
        logger.error(f"Query optimization failed: {e}", exc_info=True)

        # Log API error
        processing_time = (time.perf_counter() - start_time) * 1000
        audit_logger_instance.log_api_event(
            request=request,
            response_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            processing_time=processing_time / 1000,
            additional_data={"error": str(e)},
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query optimization failed: {e!s}",
        )


@router.get("/status", response_model=SystemStatusResponse)
@rate_limit(RateLimits.HEALTH_CHECK)
async def get_system_status(
    request: Request,
    integration: DynamicLoadingIntegration = Depends(get_integration_dependency),
) -> SystemStatusResponse:
    """
    Get comprehensive system status and health information.

    This endpoint provides real-time status of the dynamic loading system
    including component health, metrics, and performance indicators.
    """
    try:
        status_data = await integration.get_system_status()

        return SystemStatusResponse(
            integration_health=status_data["integration_health"],
            mode=status_data["mode"],
            metrics=status_data["metrics"],
            components=status_data["components"],
            features=status_data["features"],
            uptime_hours=status_data["metrics"]["uptime_hours"],
        )

    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system status",
        )


@router.get("/performance-report", response_model=PerformanceReportResponse)
@rate_limit(RateLimits.API_DEFAULT)
async def get_performance_report(
    request: Request,
    integration: DynamicLoadingIntegration = Depends(get_integration_dependency),
) -> PerformanceReportResponse:
    """
    Get comprehensive performance report and metrics.

    This endpoint provides detailed performance analytics including
    optimization results, timing analysis, and system health metrics.
    """
    try:
        report = await integration.get_performance_report()
        integration_report = report["integration_report"]

        return PerformanceReportResponse(
            report_timestamp=report["timestamp"],
            integration_metrics=integration_report["integration_metrics"],
            performance_summary=integration_report["performance_summary"],
            timing_analysis=integration_report["timing_analysis"],
            system_health=integration_report["system_health"],
        )

    except Exception as e:
        logger.error(f"Failed to get performance report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve performance report",
        )


@router.post("/user-command", response_model=UserCommandResponse)
@rate_limit(RateLimits.API_DEFAULT)
async def execute_user_command(
    request: Request,
    command_request: UserCommandRequest,
    integration: DynamicLoadingIntegration = Depends(get_integration_dependency),
) -> UserCommandResponse:
    """
    Execute a user command through the dynamic loading system.

    This endpoint allows users to manually control function loading,
    set optimization preferences, and interact with the system.
    """
    try:
        result = await integration.execute_user_command(
            command=command_request.command,
            session_id=command_request.session_id,
            user_id=command_request.user_id,
        )

        # Log command execution
        audit_logger_instance.log_api_event(
            request=request,
            response_status=status.HTTP_200_OK,
            processing_time=0.1,  # Command execution is typically fast
            additional_data={
                "command": command_request.command,
                "success": result.success,
                "session_id": command_request.session_id,
            },
        )

        return UserCommandResponse(
            success=result.success,
            message=result.message,
            command=command_request.command,  # Get command from request instead of result
            data=result.data,
            suggestions=result.suggestions,
        )

    except Exception as e:
        logger.error(f"User command execution failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Command execution failed: {e!s}",
        )


@router.post("/demo/run")
@rate_limit(RateLimits.API_SLOW)
async def run_comprehensive_demo(
    request: Request,
    demo_request: DemoRunRequest,
    integration: DynamicLoadingIntegration = Depends(get_integration_dependency),
) -> JSONResponse:
    """
    Run comprehensive demonstration scenarios.

    This endpoint executes the full prototype demonstration system,
    validating performance, functionality, and production readiness.
    """
    try:
        # Initialize demo system
        demo = ComprehensivePrototypeDemo(mode=IntegrationMode.DEMO)

        if not await demo.initialize():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to initialize demonstration system",
            )

        # Run comprehensive demonstration
        results = await demo.run_comprehensive_demo()

        # Log demo execution
        audit_logger_instance.log_api_event(
            request=request,
            response_status=status.HTTP_200_OK,
            processing_time=results["demo_metadata"]["total_time_seconds"],
            additional_data={
                "scenarios_tested": results["demo_metadata"]["total_scenarios"],
                "success_rate": results["performance_summary"]["success_rate"],
                "avg_reduction": results["performance_summary"]["token_optimization"]["average_reduction"],
                "readiness_level": results["production_readiness"]["readiness_level"],
            },
        )

        # Filter results based on export_results flag
        if demo_request.export_results:
            response_data = results
        else:
            # Return summary only
            response_data = {
                "demo_summary": {
                    "total_scenarios": results["demo_metadata"]["total_scenarios"],
                    "total_time_seconds": results["demo_metadata"]["total_time_seconds"],
                    "success_rate": results["performance_summary"]["success_rate"],
                    "average_reduction": results["performance_summary"]["token_optimization"]["average_reduction"],
                    "target_achievement_rate": results["performance_summary"]["token_optimization"][
                        "target_achievement_rate"
                    ],
                    "average_processing_time": results["performance_summary"]["performance_metrics"][
                        "average_processing_time_ms"
                    ],
                    "readiness_level": results["production_readiness"]["readiness_level"],
                    "overall_score": results["production_readiness"]["overall_score"],
                },
                "key_achievements": results["production_readiness"]["key_achievements"],
                "deployment_recommendation": results["production_readiness"]["deployment_recommendation"],
            }

        return JSONResponse(
            content=response_data,
            status_code=status.HTTP_200_OK,
        )

    except Exception as e:
        logger.error(f"Demo execution failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Demo execution failed: {e!s}",
        )


@router.get("/metrics/live")
@rate_limit(RateLimits.HEALTH_CHECK)
async def get_live_metrics(
    request: Request,
    integration: DynamicLoadingIntegration = Depends(get_integration_dependency),
) -> JSONResponse:
    """
    Get live metrics for real-time monitoring.

    This endpoint provides current system metrics suitable for
    monitoring dashboards and real-time status displays.
    """
    try:
        status_data = await integration.get_system_status()
        metrics = status_data["metrics"]

        live_metrics = {
            "timestamp": datetime.now().isoformat(),
            "health_status": status_data["integration_health"],
            "performance": {
                "queries_processed": metrics["total_queries_processed"],
                "success_rate": metrics["success_rate"],
                "average_reduction": metrics["average_reduction_percentage"],
                "target_achievement_rate": metrics["target_achievement_rate"],
                "average_processing_time_ms": metrics["average_total_time_ms"],
                "cache_hit_rate": metrics["cache_hit_rate"],
            },
            "system": {
                "uptime_hours": metrics["uptime_hours"],
                "error_count": metrics["error_count"],
                "warning_count": metrics["warning_count"],
                "active_sessions": status_data["active_sessions"],
            },
            "optimization": {
                "successful_optimizations": metrics["successful_optimizations"],
                "fallback_activations": metrics["fallback_activations"],
                "user_commands_executed": metrics["user_commands_executed"],
                "user_command_success_rate": metrics["user_command_success_rate"],
            },
        }

        return JSONResponse(content=live_metrics)

    except Exception as e:
        logger.error(f"Failed to get live metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve live metrics",
        )


@router.get("/function-registry/stats")
@rate_limit(RateLimits.API_DEFAULT)
async def get_function_registry_stats(
    request: Request,
    integration: DynamicLoadingIntegration = Depends(get_integration_dependency),
) -> JSONResponse:
    """
    Get function registry statistics and information.

    This endpoint provides detailed information about the function registry
    including tier breakdown, category distribution, and optimization potential.
    """
    try:
        if not integration.function_loader:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Function loader not available",
            )

        registry = integration.function_loader.function_registry

        # Get tier breakdown
        tier_stats = {}
        for i, tier in enumerate(registry.tiers, 1):
            tier_functions = registry.get_functions_by_tier(tier)
            tier_tokens = registry.get_tier_token_cost(tier)

            tier_stats[f"tier_{i}"] = {
                "name": tier.value,  # tier is LoadingTier enum, needs .value
                "function_count": len(tier_functions),
                "token_cost": tier_tokens,
                "sample_functions": list(tier_functions)[:5],  # First 5 functions
            }

        # Get category breakdown
        category_stats = {}
        for category in registry.categories:
            category_functions = registry.get_functions_by_category(category)
            category_tokens, _ = registry.calculate_loading_cost(category_functions)

            category_stats[category] = {
                "function_count": len(category_functions),
                "token_cost": category_tokens,
                "sample_functions": list(category_functions)[:3],  # First 3 functions
            }

        stats = {
            "registry_summary": {
                "total_functions": len(registry.functions),
                "total_categories": len(registry.categories),
                "total_tiers": len(registry.tiers),
                "baseline_token_cost": registry.get_baseline_token_cost(),
                "last_updated": datetime.now().isoformat(),
            },
            "tier_breakdown": tier_stats,
            "category_breakdown": category_stats,
            "optimization_potential": {
                "max_possible_reduction": (
                    100.0
                    - (tier_stats.get("tier_1", {}).get("token_cost", 0) / registry.get_baseline_token_cost() * 100)
                    if registry.get_baseline_token_cost() > 0
                    else 0
                ),
                "typical_reduction_range": "60-85%",
                "aggressive_reduction_potential": "80-90%",
            },
        }

        return JSONResponse(content=stats)

    except Exception as e:
        logger.error(f"Failed to get function registry stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve function registry statistics",
        )


# Health check endpoint specifically for dynamic loading
@router.get("/health")
@rate_limit(RateLimits.HEALTH_CHECK)
async def dynamic_loading_health_check(request: Request) -> JSONResponse:
    """
    Health check endpoint specifically for the dynamic loading system.

    This provides a quick health status check suitable for load balancers
    and monitoring systems.
    """
    try:
        integration = await get_integration_instance(mode=IntegrationMode.PRODUCTION)
        status_data = await integration.get_system_status()

        health_status = status_data["integration_health"]

        if health_status in ["healthy", "degraded"]:
            return JSONResponse(
                content={
                    "status": health_status,
                    "service": "dynamic-function-loading",
                    "uptime_hours": status_data["metrics"]["uptime_hours"],
                    "queries_processed": status_data["metrics"]["total_queries_processed"],
                    "success_rate": status_data["metrics"]["success_rate"],
                    "timestamp": datetime.now().isoformat(),
                },
                status_code=status.HTTP_200_OK,
            )
        return JSONResponse(
            content={
                "status": health_status,
                "service": "dynamic-function-loading",
                "error": "System not healthy",
                "timestamp": datetime.now().isoformat(),
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    except Exception as e:
        logger.error(f"Dynamic loading health check failed: {e}")
        return JSONResponse(
            content={
                "status": "failed",
                "service": "dynamic-function-loading",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
