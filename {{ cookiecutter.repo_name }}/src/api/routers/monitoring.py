"""API endpoints for performance monitoring and metrics.

This module provides REST endpoints for accessing performance metrics,
SLA compliance status, and system health information.
"""

import time
from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel

from src.utils.performance_monitor import (
    get_performance_monitor,
    get_resource_monitor,
    get_sla_monitor,
)

# Initialize router
router = APIRouter(
    prefix="/api/v1/monitoring",
    tags=["monitoring"],
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"},
    },
)


class MetricsResponse(BaseModel):
    """Response model for metrics endpoint."""

    counters: dict[str, int]
    gauges: dict[str, float]
    histograms: dict[str, dict[str, float]]
    timers: dict[str, dict[str, float]]
    timestamp: float


class SLAComplianceResponse(BaseModel):
    """Response model for SLA compliance endpoint."""

    compliance_status: dict[str, dict[str, Any]]
    overall_compliant: bool
    timestamp: float


class SLAViolationResponse(BaseModel):
    """Response model for SLA violations endpoint."""

    violations: list[dict[str, Any]]
    total_violations: int
    timestamp: float


class HealthResponse(BaseModel):
    """Response model for health endpoint."""

    status: str
    uptime: float
    metrics_count: int
    sla_compliant: bool
    timestamp: float


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Get system metrics",
    description="Retrieve current system performance metrics including counters, gauges, histograms, and timers",
)
async def get_metrics() -> MetricsResponse:
    """Get current system metrics.

    Returns:
        Current system metrics.
    """
    monitor = get_performance_monitor()
    metrics = monitor.get_all_metrics()

    return MetricsResponse(
        counters=metrics.get("counters", {}),
        gauges=metrics.get("gauges", {}),
        histograms=metrics.get("histograms", {}),
        timers=metrics.get("timers", {}),
        timestamp=time.time(),
    )


@router.get(
    "/sla/compliance",
    response_model=SLAComplianceResponse,
    summary="Get SLA compliance status",
    description="Check current SLA compliance status against defined targets",
)
async def get_sla_compliance() -> SLAComplianceResponse:
    """Get SLA compliance status.

    Returns:
        SLA compliance status.
    """
    monitor = get_performance_monitor()
    sla_monitor = get_sla_monitor()

    metrics = monitor.get_all_metrics()
    compliance_status = sla_monitor.check_sla_compliance(metrics)

    # Check overall compliance
    overall_compliant = all(status.get("compliant", False) for status in compliance_status.values())

    return SLAComplianceResponse(
        compliance_status=compliance_status,
        overall_compliant=overall_compliant,
        timestamp=time.time(),
    )


@router.get(
    "/sla/violations",
    response_model=SLAViolationResponse,
    summary="Get SLA violations",
    description="Retrieve SLA violations, optionally filtered by time range",
)
async def get_sla_violations(
    since: float | None = Query(None, description="Unix timestamp to filter violations since"),
) -> SLAViolationResponse:
    """Get SLA violations.

    Args:
        since: Optional timestamp to filter violations.

    Returns:
        SLA violations.
    """
    sla_monitor = get_sla_monitor()
    violations = sla_monitor.get_violations(since)

    return SLAViolationResponse(
        violations=violations,
        total_violations=len(violations),
        timestamp=time.time(),
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Get system health status",
    description="Get overall system health including metrics and SLA compliance",
)
async def get_health() -> HealthResponse:
    """Get system health status.

    Returns:
        System health status.
    """
    monitor = get_performance_monitor()
    sla_monitor = get_sla_monitor()

    # Get metrics
    metrics = monitor.get_all_metrics()

    # Check SLA compliance
    compliance_status = sla_monitor.check_sla_compliance(metrics)
    sla_compliant = all(status.get("compliant", False) for status in compliance_status.values())

    # Calculate metrics count
    metrics_count = (
        len(metrics.get("counters", {}))
        + len(metrics.get("gauges", {}))
        + len(metrics.get("histograms", {}))
        + len(metrics.get("timers", {}))
    )

    return HealthResponse(
        status="healthy" if sla_compliant else "degraded",
        uptime=0.0,  # Placeholder - would track actual uptime
        metrics_count=metrics_count,
        sla_compliant=sla_compliant,
        timestamp=time.time(),
    )


@router.post(
    "/metrics/reset",
    summary="Reset all metrics",
    description="Reset all performance metrics to their initial state",
)
async def reset_metrics() -> dict[str, str]:
    """Reset all performance metrics.

    Returns:
        Reset confirmation.
    """
    monitor = get_performance_monitor()
    monitor.reset_metrics()

    return {"message": "All metrics have been reset", "timestamp": str(time.time())}


@router.post("/sla/violations/clear", summary="Clear SLA violations", description="Clear all recorded SLA violations")
async def clear_sla_violations() -> dict[str, str]:
    """Clear all SLA violations.

    Returns:
        Clear confirmation.
    """
    sla_monitor = get_sla_monitor()
    sla_monitor.clear_violations()

    return {"message": "All SLA violations have been cleared", "timestamp": str(time.time())}


@router.get(
    "/system/resources",
    summary="Get system resource usage",
    description="Get current system resource usage including memory and CPU",
)
async def get_system_resources() -> dict[str, Any]:
    """Get system resource usage.

    Returns:
        System resource usage.
    """
    monitor = get_performance_monitor()

    # Get resource metrics
    gauges = monitor.get_all_metrics().get("gauges", {})

    return {
        "memory_usage_mb": gauges.get("memory_usage_mb", 0.0),
        "cpu_usage_percent": gauges.get("cpu_usage_percent", 0.0),
        "timestamp": time.time(),
    }


@router.post(
    "/system/resources/start",
    summary="Start resource monitoring",
    description="Start continuous system resource monitoring",
)
async def start_resource_monitoring() -> dict[str, str]:
    """Start system resource monitoring.

    Returns:
        Start confirmation.
    """
    resource_monitor = get_resource_monitor()
    await resource_monitor.start_monitoring()

    return {"message": "Resource monitoring started", "timestamp": str(time.time())}


@router.post(
    "/system/resources/stop",
    summary="Stop resource monitoring",
    description="Stop continuous system resource monitoring",
)
async def stop_resource_monitoring() -> dict[str, str]:
    """Stop system resource monitoring.

    Returns:
        Stop confirmation.
    """
    resource_monitor = get_resource_monitor()
    await resource_monitor.stop_monitoring()

    return {"message": "Resource monitoring stopped", "timestamp": str(time.time())}
