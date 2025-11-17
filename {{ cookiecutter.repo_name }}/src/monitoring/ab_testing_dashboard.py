"""
A/B Testing Metrics Dashboard

This module provides real-time monitoring and visualization for A/B testing experiments,
with comprehensive metrics collection, alerting, and decision-making support.

Features:
- Real-time experiment monitoring
- Statistical significance tracking
- Performance metrics visualization
- Automated alerting and notifications
- Decision support analytics
- Risk assessment and safety monitoring
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from jinja2 import Template

from ..core.ab_testing_framework import (
    ExperimentManager, get_experiment_manager, ExperimentResults,
    MetricEvent, UserCharacteristics
)
from ..utils.observability import ObservabilityMixin
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MetricType(Enum):
    """Types of metrics tracked."""
    PERFORMANCE = "performance"
    CONVERSION = "conversion"
    ERROR = "error"
    ENGAGEMENT = "engagement"
    BUSINESS = "business"


@dataclass
class Alert:
    """Alert for experiment monitoring."""

    id: str
    experiment_id: str
    level: AlertLevel
    title: str
    message: str
    metric_type: MetricType
    current_value: float
    threshold_value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "level": self.level.value,
            "title": self.title,
            "message": self.message,
            "metric_type": self.metric_type.value,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged
        }


@dataclass
class DashboardMetrics:
    """Complete dashboard metrics for an experiment."""

    experiment_id: str
    experiment_name: str
    status: str

    # Overview metrics
    total_users: int
    active_users_24h: int
    conversion_rate: float
    statistical_significance: float

    # Performance metrics
    avg_response_time_ms: float
    avg_token_reduction: float
    success_rate: float
    error_rate: float

    # Variant comparison
    variants: Dict[str, Dict[str, Any]]

    # Time series data
    performance_timeline: List[Dict[str, Any]]
    conversion_timeline: List[Dict[str, Any]]
    error_timeline: List[Dict[str, Any]]

    # Alerts and recommendations
    active_alerts: List[Alert]
    recommendations: List[str]

    # Risk assessment
    risk_level: str  # low, medium, high, critical
    confidence_level: str  # low, medium, high

    # Metadata
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "status": self.status,
            "total_users": self.total_users,
            "active_users_24h": self.active_users_24h,
            "conversion_rate": self.conversion_rate,
            "statistical_significance": self.statistical_significance,
            "avg_response_time_ms": self.avg_response_time_ms,
            "avg_token_reduction": self.avg_token_reduction,
            "success_rate": self.success_rate,
            "error_rate": self.error_rate,
            "variants": self.variants,
            "performance_timeline": self.performance_timeline,
            "conversion_timeline": self.conversion_timeline,
            "error_timeline": self.error_timeline,
            "active_alerts": [alert.to_dict() for alert in self.active_alerts],
            "recommendations": self.recommendations,
            "risk_level": self.risk_level,
            "confidence_level": self.confidence_level,
            "last_updated": self.last_updated.isoformat()
        }


class MetricsCollector(ObservabilityMixin):
    """Collects and aggregates metrics for dashboard display."""

    def __init__(self, experiment_manager: ExperimentManager):
        super().__init__()
        self.experiment_manager = experiment_manager
        self.logger = logging.getLogger(__name__)

    async def collect_experiment_metrics(self, experiment_id: str) -> Optional[DashboardMetrics]:
        """Collect comprehensive metrics for an experiment."""
        try:
            with self.experiment_manager.get_db_session() as db_session:
                from ..core.ab_testing_framework import ExperimentModel, UserAssignmentModel, MetricEventModel

                # Get experiment details
                experiment = db_session.query(ExperimentModel).filter_by(id=experiment_id).first()
                if not experiment:
                    return None

                # Get experiment results
                results = await self.experiment_manager.get_experiment_results(experiment_id)
                if not results:
                    return None

                # Collect overview metrics
                total_users = results.total_users

                # Get active users in last 24h
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                active_users_24h = db_session.query(UserAssignmentModel).filter(
                    UserAssignmentModel.experiment_id == experiment_id,
                    UserAssignmentModel.last_interaction >= cutoff_time
                ).count()

                # Calculate conversion rate (success rate as proxy)
                conversion_rate = results.performance_summary.get("overall_success_rate", 0.0) * 100

                # Collect performance metrics
                perf_summary = results.performance_summary
                avg_response_time = perf_summary.get("avg_response_time_ms", 0.0)
                avg_token_reduction = perf_summary.get("avg_token_reduction", 0.0)
                success_rate = perf_summary.get("overall_success_rate", 0.0) * 100
                error_rate = (1.0 - perf_summary.get("overall_success_rate", 1.0)) * 100

                # Collect time series data
                performance_timeline = await self._collect_performance_timeline(experiment_id, db_session)
                conversion_timeline = await self._collect_conversion_timeline(experiment_id, db_session)
                error_timeline = await self._collect_error_timeline(experiment_id, db_session)

                # Generate alerts
                active_alerts = await self._generate_alerts(experiment_id, results)

                # Generate recommendations
                recommendations = self._generate_recommendations(results)

                # Assess risk level
                risk_level = self._assess_risk_level(results, error_rate)
                confidence_level = self._assess_confidence_level(results)

                return DashboardMetrics(
                    experiment_id=experiment_id,
                    experiment_name=experiment.name,
                    status=experiment.status,
                    total_users=total_users,
                    active_users_24h=active_users_24h,
                    conversion_rate=conversion_rate,
                    statistical_significance=results.statistical_significance,
                    avg_response_time_ms=avg_response_time,
                    avg_token_reduction=avg_token_reduction,
                    success_rate=success_rate,
                    error_rate=error_rate,
                    variants=results.variants,
                    performance_timeline=performance_timeline,
                    conversion_timeline=conversion_timeline,
                    error_timeline=error_timeline,
                    active_alerts=active_alerts,
                    recommendations=recommendations,
                    risk_level=risk_level,
                    confidence_level=confidence_level
                )

        except Exception as e:
            self.logger.error(f"Failed to collect metrics for experiment {experiment_id}: {e}")
            return None

    async def _collect_performance_timeline(self, experiment_id: str, db_session) -> List[Dict[str, Any]]:
        """Collect performance metrics over time."""
        try:
            from ..core.ab_testing_framework import MetricEventModel

            # Get performance events from last 7 days
            cutoff_time = datetime.utcnow() - timedelta(days=7)

            events = db_session.query(MetricEventModel).filter(
                MetricEventModel.experiment_id == experiment_id,
                MetricEventModel.event_type == "performance",
                MetricEventModel.timestamp >= cutoff_time
            ).order_by(MetricEventModel.timestamp).all()

            # Group by hour and calculate averages
            timeline = {}
            for event in events:
                hour_key = event.timestamp.replace(minute=0, second=0, microsecond=0)
                if hour_key not in timeline:
                    timeline[hour_key] = {
                        "timestamp": hour_key.isoformat(),
                        "response_times": [],
                        "token_reductions": [],
                        "success_count": 0,
                        "total_count": 0
                    }

                timeline[hour_key]["total_count"] += 1

                if event.response_time_ms:
                    timeline[hour_key]["response_times"].append(event.response_time_ms)

                if event.token_reduction_percentage:
                    timeline[hour_key]["token_reductions"].append(event.token_reduction_percentage)

                if event.success:
                    timeline[hour_key]["success_count"] += 1

            # Calculate averages
            result = []
            for hour_data in timeline.values():
                avg_response_time = (
                    sum(hour_data["response_times"]) / len(hour_data["response_times"])
                    if hour_data["response_times"] else 0.0
                )
                avg_token_reduction = (
                    sum(hour_data["token_reductions"]) / len(hour_data["token_reductions"])
                    if hour_data["token_reductions"] else 0.0
                )
                success_rate = (
                    hour_data["success_count"] / hour_data["total_count"]
                    if hour_data["total_count"] > 0 else 0.0
                )

                result.append({
                    "timestamp": hour_data["timestamp"],
                    "avg_response_time_ms": avg_response_time,
                    "avg_token_reduction": avg_token_reduction,
                    "success_rate": success_rate * 100,
                    "total_requests": hour_data["total_count"]
                })

            return sorted(result, key=lambda x: x["timestamp"])

        except Exception as e:
            self.logger.error(f"Failed to collect performance timeline: {e}")
            return []

    async def _collect_conversion_timeline(self, experiment_id: str, db_session) -> List[Dict[str, Any]]:
        """Collect conversion metrics over time."""
        try:
            from ..core.ab_testing_framework import MetricEventModel

            # Get optimization events (successful token reductions are conversions)
            cutoff_time = datetime.utcnow() - timedelta(days=7)

            events = db_session.query(MetricEventModel).filter(
                MetricEventModel.experiment_id == experiment_id,
                MetricEventModel.event_type == "optimization",
                MetricEventModel.timestamp >= cutoff_time
            ).order_by(MetricEventModel.timestamp).all()

            # Group by hour and calculate conversion rates
            timeline = {}
            for event in events:
                hour_key = event.timestamp.replace(minute=0, second=0, microsecond=0)
                if hour_key not in timeline:
                    timeline[hour_key] = {
                        "timestamp": hour_key.isoformat(),
                        "conversions": 0,
                        "total_attempts": 0
                    }

                timeline[hour_key]["total_attempts"] += 1

                # Consider successful token reduction above 70% as conversion
                if event.token_reduction_percentage and event.token_reduction_percentage >= 70.0:
                    timeline[hour_key]["conversions"] += 1

            # Calculate conversion rates
            result = []
            for hour_data in timeline.values():
                conversion_rate = (
                    hour_data["conversions"] / hour_data["total_attempts"]
                    if hour_data["total_attempts"] > 0 else 0.0
                )

                result.append({
                    "timestamp": hour_data["timestamp"],
                    "conversion_rate": conversion_rate * 100,
                    "conversions": hour_data["conversions"],
                    "total_attempts": hour_data["total_attempts"]
                })

            return sorted(result, key=lambda x: x["timestamp"])

        except Exception as e:
            self.logger.error(f"Failed to collect conversion timeline: {e}")
            return []

    async def _collect_error_timeline(self, experiment_id: str, db_session) -> List[Dict[str, Any]]:
        """Collect error metrics over time."""
        try:
            from ..core.ab_testing_framework import MetricEventModel

            # Get all events from last 7 days
            cutoff_time = datetime.utcnow() - timedelta(days=7)

            events = db_session.query(MetricEventModel).filter(
                MetricEventModel.experiment_id == experiment_id,
                MetricEventModel.timestamp >= cutoff_time
            ).order_by(MetricEventModel.timestamp).all()

            # Group by hour and calculate error rates
            timeline = {}
            for event in events:
                hour_key = event.timestamp.replace(minute=0, second=0, microsecond=0)
                if hour_key not in timeline:
                    timeline[hour_key] = {
                        "timestamp": hour_key.isoformat(),
                        "errors": 0,
                        "total_events": 0
                    }

                timeline[hour_key]["total_events"] += 1

                if event.event_type == "error" or event.success is False:
                    timeline[hour_key]["errors"] += 1

            # Calculate error rates
            result = []
            for hour_data in timeline.values():
                error_rate = (
                    hour_data["errors"] / hour_data["total_events"]
                    if hour_data["total_events"] > 0 else 0.0
                )

                result.append({
                    "timestamp": hour_data["timestamp"],
                    "error_rate": error_rate * 100,
                    "error_count": hour_data["errors"],
                    "total_events": hour_data["total_events"]
                })

            return sorted(result, key=lambda x: x["timestamp"])

        except Exception as e:
            self.logger.error(f"Failed to collect error timeline: {e}")
            return []

    async def _generate_alerts(self, experiment_id: str, results: ExperimentResults) -> List[Alert]:
        """Generate alerts based on experiment results."""
        alerts = []

        try:
            # High error rate alert
            error_rate = (1.0 - results.performance_summary.get("overall_success_rate", 1.0)) * 100
            if error_rate > 10.0:
                alerts.append(Alert(
                    id=f"{experiment_id}_high_error_rate",
                    experiment_id=experiment_id,
                    level=AlertLevel.CRITICAL if error_rate > 20.0 else AlertLevel.WARNING,
                    title="High Error Rate Detected",
                    message=f"Error rate is {error_rate:.1f}%, exceeding safe thresholds",
                    metric_type=MetricType.ERROR,
                    current_value=error_rate,
                    threshold_value=10.0
                ))

            # Poor performance alert
            avg_response_time = results.performance_summary.get("avg_response_time_ms", 0.0)
            if avg_response_time > 500.0:
                alerts.append(Alert(
                    id=f"{experiment_id}_slow_response",
                    experiment_id=experiment_id,
                    level=AlertLevel.WARNING,
                    title="Slow Response Times",
                    message=f"Average response time is {avg_response_time:.1f}ms, impacting user experience",
                    metric_type=MetricType.PERFORMANCE,
                    current_value=avg_response_time,
                    threshold_value=500.0
                ))

            # Low token reduction alert
            avg_token_reduction = results.performance_summary.get("avg_token_reduction", 0.0)
            if avg_token_reduction < 50.0:
                alerts.append(Alert(
                    id=f"{experiment_id}_low_optimization",
                    experiment_id=experiment_id,
                    level=AlertLevel.WARNING,
                    title="Low Optimization Performance",
                    message=f"Token reduction is only {avg_token_reduction:.1f}%, below target",
                    metric_type=MetricType.PERFORMANCE,
                    current_value=avg_token_reduction,
                    threshold_value=70.0
                ))

            # Statistical significance alert
            if results.statistical_significance < 80.0 and results.total_users > 500:
                alerts.append(Alert(
                    id=f"{experiment_id}_low_significance",
                    experiment_id=experiment_id,
                    level=AlertLevel.INFO,
                    title="Low Statistical Significance",
                    message=f"Statistical significance is {results.statistical_significance:.1f}% with {results.total_users} users",
                    metric_type=MetricType.CONVERSION,
                    current_value=results.statistical_significance,
                    threshold_value=95.0
                ))

            # Check failure thresholds
            for threshold_name, exceeded in results.failure_thresholds_exceeded.items():
                if exceeded:
                    alerts.append(Alert(
                        id=f"{experiment_id}_threshold_{threshold_name}",
                        experiment_id=experiment_id,
                        level=AlertLevel.CRITICAL,
                        title="Failure Threshold Exceeded",
                        message=f"Experiment has exceeded failure threshold: {threshold_name}",
                        metric_type=MetricType.ERROR,
                        current_value=1.0,
                        threshold_value=0.0
                    ))

        except Exception as e:
            self.logger.error(f"Failed to generate alerts: {e}")

        return alerts

    def _generate_recommendations(self, results: ExperimentResults) -> List[str]:
        """Generate actionable recommendations based on results."""
        recommendations = []

        try:
            # Recommendation based on overall performance
            if results.recommendation == "expand":
                recommendations.append("✅ Experiment is performing well. Consider expanding to larger user base.")
                recommendations.append("📊 Document successful optimization strategies for future reference.")
            elif results.recommendation == "rollback":
                recommendations.append("⚠️ Immediate rollback recommended due to performance issues.")
                recommendations.append("🔍 Investigate root causes before attempting future rollouts.")
            elif results.recommendation == "continue":
                recommendations.append("⏳ Continue current experiment to gather more data.")
                recommendations.append("📈 Monitor key metrics closely for trend changes.")
            elif results.recommendation == "modify":
                recommendations.append("🔧 Consider modifying experiment parameters.")
                recommendations.append("💡 Review alternative optimization strategies.")

            # Specific metric-based recommendations
            error_rate = (1.0 - results.performance_summary.get("overall_success_rate", 1.0)) * 100
            if error_rate > 5.0:
                recommendations.append(f"🚨 Error rate ({error_rate:.1f}%) requires immediate attention.")

            avg_response_time = results.performance_summary.get("avg_response_time_ms", 0.0)
            if avg_response_time > 300.0:
                recommendations.append(f"⏱️ Response time ({avg_response_time:.1f}ms) needs optimization.")

            avg_token_reduction = results.performance_summary.get("avg_token_reduction", 0.0)
            if avg_token_reduction < 70.0:
                recommendations.append(f"🎯 Token reduction ({avg_token_reduction:.1f}%) below target - review strategy.")

            # Statistical significance recommendations
            if results.statistical_significance < 95.0:
                if results.total_users < 1000:
                    recommendations.append("👥 Increase sample size to improve statistical significance.")
                else:
                    recommendations.append("📊 Consider longer experiment duration for better significance.")

            # Success criteria recommendations
            success_rate = sum(results.success_criteria_met.values()) / len(results.success_criteria_met) if results.success_criteria_met else 0.0
            if success_rate < 0.8:
                recommendations.append("🎯 Review success criteria - some targets may not be achievable.")

        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")

        return recommendations

    def _assess_risk_level(self, results: ExperimentResults, error_rate: float) -> str:
        """Assess risk level based on experiment performance."""
        try:
            # Critical risk conditions
            if any(results.failure_thresholds_exceeded.values()):
                return "critical"

            if error_rate > 15.0:
                return "critical"

            # High risk conditions
            if error_rate > 8.0:
                return "high"

            avg_response_time = results.performance_summary.get("avg_response_time_ms", 0.0)
            if avg_response_time > 800.0:
                return "high"

            # Medium risk conditions
            if error_rate > 3.0:
                return "medium"

            avg_token_reduction = results.performance_summary.get("avg_token_reduction", 0.0)
            if avg_token_reduction < 50.0:
                return "medium"

            # Low risk (good performance)
            return "low"

        except Exception as e:
            self.logger.error(f"Failed to assess risk level: {e}")
            return "medium"

    def _assess_confidence_level(self, results: ExperimentResults) -> str:
        """Assess confidence level in experiment results."""
        try:
            # High confidence conditions
            if (results.statistical_significance >= 95.0 and
                results.total_users >= 1000 and
                results.duration_hours >= 48):
                return "high"

            # Medium confidence conditions
            if (results.statistical_significance >= 80.0 and
                results.total_users >= 500):
                return "medium"

            # Low confidence (insufficient data)
            return "low"

        except Exception as e:
            self.logger.error(f"Failed to assess confidence level: {e}")
            return "low"


class DashboardVisualizer:
    """Creates visualizations for the A/B testing dashboard."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_performance_chart(self, timeline_data: List[Dict[str, Any]]) -> str:
        """Create performance timeline chart."""
        try:
            if not timeline_data:
                return self._create_empty_chart("No performance data available")

            df = pd.DataFrame(timeline_data)

            # Create subplot with secondary y-axis
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Response Time & Token Reduction', 'Success Rate'),
                vertical_spacing=0.15
            )

            # Response time and token reduction
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['avg_response_time_ms'],
                    name='Response Time (ms)',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['avg_token_reduction'],
                    name='Token Reduction (%)',
                    line=dict(color='green'),
                    yaxis='y2'
                ),
                row=1, col=1
            )

            # Success rate
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['success_rate'],
                    name='Success Rate (%)',
                    line=dict(color='orange'),
                    fill='tonexty'
                ),
                row=2, col=1
            )

            fig.update_layout(
                title="Performance Metrics Over Time",
                height=600,
                showlegend=True
            )

            return fig.to_html(include_plotlyjs='cdn')

        except Exception as e:
            self.logger.error(f"Failed to create performance chart: {e}")
            return self._create_empty_chart("Error creating performance chart")

    def create_variant_comparison_chart(self, variants: Dict[str, Dict[str, Any]]) -> str:
        """Create variant comparison chart."""
        try:
            if not variants:
                return self._create_empty_chart("No variant data available")

            variant_names = list(variants.keys())
            metrics = ['avg_response_time_ms', 'avg_token_reduction', 'success_rate']

            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=('Response Time (ms)', 'Token Reduction (%)', 'Success Rate (%)'),
                horizontal_spacing=0.1
            )

            colors = ['blue', 'red', 'green', 'orange']

            for i, metric in enumerate(metrics):
                values = [variants[variant].get(metric, 0) for variant in variant_names]

                fig.add_trace(
                    go.Bar(
                        x=variant_names,
                        y=values,
                        name=metric.replace('_', ' ').title(),
                        marker_color=colors[i % len(colors)],
                        showlegend=False
                    ),
                    row=1, col=i+1
                )

            fig.update_layout(
                title="Variant Performance Comparison",
                height=400
            )

            return fig.to_html(include_plotlyjs='cdn')

        except Exception as e:
            self.logger.error(f"Failed to create variant comparison chart: {e}")
            return self._create_empty_chart("Error creating variant comparison chart")

    def create_conversion_funnel(self, metrics: DashboardMetrics) -> str:
        """Create conversion funnel visualization."""
        try:
            # Create funnel data
            stages = ['Total Users', 'Active Users', 'Successful Queries', 'Target Conversions']

            total_users = metrics.total_users
            active_users = metrics.active_users_24h
            successful_queries = int(total_users * metrics.success_rate / 100)
            target_conversions = int(successful_queries * metrics.conversion_rate / 100)

            values = [total_users, active_users, successful_queries, target_conversions]

            fig = go.Figure(go.Funnel(
                y=stages,
                x=values,
                textinfo="value+percent initial",
                marker=dict(color=["blue", "lightblue", "green", "lightgreen"])
            ))

            fig.update_layout(
                title="User Conversion Funnel",
                height=400
            )

            return fig.to_html(include_plotlyjs='cdn')

        except Exception as e:
            self.logger.error(f"Failed to create conversion funnel: {e}")
            return self._create_empty_chart("Error creating conversion funnel")

    def create_statistical_significance_gauge(self, significance: float) -> str:
        """Create statistical significance gauge chart."""
        try:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=significance,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Statistical Significance (%)"},
                delta={'reference': 95.0},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 80], 'color': "lightgray"},
                        {'range': [80, 95], 'color': "yellow"},
                        {'range': [95, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 95
                    }
                }
            ))

            fig.update_layout(height=300)

            return fig.to_html(include_plotlyjs='cdn')

        except Exception as e:
            self.logger.error(f"Failed to create significance gauge: {e}")
            return self._create_empty_chart("Error creating significance gauge")

    def _create_empty_chart(self, message: str) -> str:
        """Create empty chart with message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            height=300
        )
        return fig.to_html(include_plotlyjs='cdn')


class ABTestingDashboard(ObservabilityMixin):
    """Main A/B testing dashboard class."""

    def __init__(self, experiment_manager: ExperimentManager):
        super().__init__()
        self.experiment_manager = experiment_manager
        self.metrics_collector = MetricsCollector(experiment_manager)
        self.visualizer = DashboardVisualizer()
        self.logger = logging.getLogger(__name__)

        # Dashboard templates
        self.dashboard_template = self._load_dashboard_template()

    async def generate_dashboard_html(self, experiment_id: str) -> str:
        """Generate complete HTML dashboard for an experiment."""
        try:
            # Collect metrics
            metrics = await self.metrics_collector.collect_experiment_metrics(experiment_id)
            if not metrics:
                return self._generate_error_dashboard("Experiment not found or no data available")

            # Generate visualizations
            performance_chart = self.visualizer.create_performance_chart(metrics.performance_timeline)
            variant_comparison = self.visualizer.create_variant_comparison_chart(metrics.variants)
            conversion_funnel = self.visualizer.create_conversion_funnel(metrics)
            significance_gauge = self.visualizer.create_statistical_significance_gauge(metrics.statistical_significance)

            # Prepare template data
            template_data = {
                'experiment': metrics.to_dict(),
                'performance_chart': performance_chart,
                'variant_comparison': variant_comparison,
                'conversion_funnel': conversion_funnel,
                'significance_gauge': significance_gauge,
                'last_updated': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
            }

            # Render dashboard
            return self.dashboard_template.render(**template_data)

        except Exception as e:
            self.logger.error(f"Failed to generate dashboard for experiment {experiment_id}: {e}")
            return self._generate_error_dashboard(f"Error generating dashboard: {str(e)}")

    async def get_dashboard_data(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get dashboard data as JSON for API endpoints."""
        try:
            metrics = await self.metrics_collector.collect_experiment_metrics(experiment_id)
            return metrics.to_dict() if metrics else None

        except Exception as e:
            self.logger.error(f"Failed to get dashboard data for experiment {experiment_id}: {e}")
            return None

    async def get_experiment_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all experiments for overview dashboard."""
        try:
            with self.experiment_manager.get_db_session() as db_session:
                from ..core.ab_testing_framework import ExperimentModel

                experiments = db_session.query(ExperimentModel).all()
                summaries = []

                for experiment in experiments:
                    metrics = await self.metrics_collector.collect_experiment_metrics(experiment.id)

                    summary = {
                        'id': experiment.id,
                        'name': experiment.name,
                        'status': experiment.status,
                        'created_at': experiment.created_at.isoformat(),
                        'start_time': experiment.start_time.isoformat() if experiment.start_time else None,
                        'end_time': experiment.end_time.isoformat() if experiment.end_time else None,
                        'current_percentage': experiment.current_percentage,
                        'target_percentage': experiment.target_percentage
                    }

                    if metrics:
                        summary.update({
                            'total_users': metrics.total_users,
                            'statistical_significance': metrics.statistical_significance,
                            'success_rate': metrics.success_rate,
                            'error_rate': metrics.error_rate,
                            'risk_level': metrics.risk_level,
                            'active_alerts': len(metrics.active_alerts)
                        })

                    summaries.append(summary)

                return summaries

        except Exception as e:
            self.logger.error(f"Failed to get experiment summary: {e}")
            return []

    def _load_dashboard_template(self) -> Template:
        """Load dashboard HTML template."""
        template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>A/B Testing Dashboard - {{ experiment.experiment_name }}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .metric-card { transition: transform 0.2s; }
        .metric-card:hover { transform: translateY(-2px); }
        .alert-badge { position: absolute; top: -5px; right: -5px; }
        .risk-low { color: #28a745; }
        .risk-medium { color: #ffc107; }
        .risk-high { color: #fd7e14; }
        .risk-critical { color: #dc3545; }
    </style>
</head>
<body class="bg-light">
    <nav class="navbar navbar-dark bg-dark">
        <div class="container-fluid">
            <span class="navbar-brand mb-0 h1">
                <i class="fas fa-chart-line me-2"></i>A/B Testing Dashboard
            </span>
            <span class="navbar-text">
                Last Updated: {{ last_updated }}
            </span>
        </div>
    </nav>

    <div class="container-fluid mt-4">
        <!-- Experiment Header -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-body">
                        <h2 class="card-title">{{ experiment.experiment_name }}</h2>
                        <div class="row">
                            <div class="col-md-3">
                                <span class="badge bg-{{ 'success' if experiment.status == 'active' else 'secondary' }} fs-6">
                                    {{ experiment.status.title() }}
                                </span>
                            </div>
                            <div class="col-md-3">
                                <span class="risk-{{ experiment.risk_level }}">
                                    <i class="fas fa-shield-alt me-1"></i>Risk: {{ experiment.risk_level.title() }}
                                </span>
                            </div>
                            <div class="col-md-3">
                                <i class="fas fa-users me-1"></i>{{ experiment.total_users }} Total Users
                            </div>
                            <div class="col-md-3">
                                <i class="fas fa-certificate me-1"></i>{{ "%.1f"|format(experiment.statistical_significance) }}% Significance
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Key Metrics -->
        <div class="row mb-4">
            <div class="col-lg-3 col-md-6 mb-3">
                <div class="card metric-card h-100">
                    <div class="card-body text-center">
                        <h5 class="card-title">Success Rate</h5>
                        <h2 class="text-success">{{ "%.1f"|format(experiment.success_rate) }}%</h2>
                        <small class="text-muted">Target: ≥95%</small>
                    </div>
                </div>
            </div>
            <div class="col-lg-3 col-md-6 mb-3">
                <div class="card metric-card h-100">
                    <div class="card-body text-center">
                        <h5 class="card-title">Token Reduction</h5>
                        <h2 class="text-primary">{{ "%.1f"|format(experiment.avg_token_reduction) }}%</h2>
                        <small class="text-muted">Target: ≥70%</small>
                    </div>
                </div>
            </div>
            <div class="col-lg-3 col-md-6 mb-3">
                <div class="card metric-card h-100">
                    <div class="card-body text-center">
                        <h5 class="card-title">Response Time</h5>
                        <h2 class="text-info">{{ "%.1f"|format(experiment.avg_response_time_ms) }}ms</h2>
                        <small class="text-muted">Target: ≤200ms</small>
                    </div>
                </div>
            </div>
            <div class="col-lg-3 col-md-6 mb-3">
                <div class="card metric-card h-100 position-relative">
                    <div class="card-body text-center">
                        <h5 class="card-title">Error Rate</h5>
                        <h2 class="text-{{ 'danger' if experiment.error_rate > 5 else 'success' }}">
                            {{ "%.1f"|format(experiment.error_rate) }}%
                        </h2>
                        <small class="text-muted">Target: ≤5%</small>
                        {% if experiment.active_alerts %}
                        <span class="badge bg-danger alert-badge">{{ experiment.active_alerts|length }}</span>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>

        <!-- Charts Row -->
        <div class="row mb-4">
            <div class="col-lg-8 mb-3">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0"><i class="fas fa-chart-line me-2"></i>Performance Timeline</h5>
                    </div>
                    <div class="card-body">
                        {{ performance_chart|safe }}
                    </div>
                </div>
            </div>
            <div class="col-lg-4 mb-3">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0"><i class="fas fa-tachometer-alt me-2"></i>Statistical Significance</h5>
                    </div>
                    <div class="card-body">
                        {{ significance_gauge|safe }}
                    </div>
                </div>
            </div>
        </div>

        <!-- Variant Comparison and Conversion Funnel -->
        <div class="row mb-4">
            <div class="col-lg-8 mb-3">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0"><i class="fas fa-balance-scale me-2"></i>Variant Comparison</h5>
                    </div>
                    <div class="card-body">
                        {{ variant_comparison|safe }}
                    </div>
                </div>
            </div>
            <div class="col-lg-4 mb-3">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0"><i class="fas fa-funnel-dollar me-2"></i>Conversion Funnel</h5>
                    </div>
                    <div class="card-body">
                        {{ conversion_funnel|safe }}
                    </div>
                </div>
            </div>
        </div>

        <!-- Alerts and Recommendations -->
        <div class="row">
            <div class="col-lg-6 mb-3">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">
                            <i class="fas fa-exclamation-triangle me-2"></i>Active Alerts
                            {% if experiment.active_alerts %}
                            <span class="badge bg-danger ms-2">{{ experiment.active_alerts|length }}</span>
                            {% endif %}
                        </h5>
                    </div>
                    <div class="card-body">
                        {% if experiment.active_alerts %}
                        {% for alert in experiment.active_alerts %}
                        <div class="alert alert-{{ 'danger' if alert.level == 'critical' else 'warning' }} alert-dismissible fade show">
                            <strong>{{ alert.title }}</strong><br>
                            {{ alert.message }}
                            <small class="d-block mt-1">
                                Current: {{ "%.1f"|format(alert.current_value) }} | Threshold: {{ "%.1f"|format(alert.threshold_value) }}
                            </small>
                        </div>
                        {% endfor %}
                        {% else %}
                        <p class="text-success mb-0">
                            <i class="fas fa-check-circle me-2"></i>No active alerts
                        </p>
                        {% endif %}
                    </div>
                </div>
            </div>
            <div class="col-lg-6 mb-3">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0"><i class="fas fa-lightbulb me-2"></i>Recommendations</h5>
                    </div>
                    <div class="card-body">
                        {% if experiment.recommendations %}
                        <ul class="list-unstyled mb-0">
                        {% for recommendation in experiment.recommendations %}
                        <li class="mb-2">{{ recommendation }}</li>
                        {% endfor %}
                        </ul>
                        {% else %}
                        <p class="text-muted mb-0">No specific recommendations at this time.</p>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Auto-refresh every 5 minutes
        setTimeout(() => {
            window.location.reload();
        }, 300000);
    </script>
</body>
</html>
        """
        return Template(template_str)

    def _generate_error_dashboard(self, error_message: str) -> str:
        """Generate error dashboard HTML."""
        return f"""
        <html>
        <head><title>Dashboard Error</title></head>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h1>Dashboard Error</h1>
            <p>{error_message}</p>
            <button onclick="window.location.reload()">Retry</button>
        </body>
        </html>
        """


# Global dashboard instance
_dashboard_instance: Optional[ABTestingDashboard] = None


async def get_dashboard_instance() -> ABTestingDashboard:
    """Get or create the global dashboard instance."""
    global _dashboard_instance

    if _dashboard_instance is None:
        experiment_manager = await get_experiment_manager()
        _dashboard_instance = ABTestingDashboard(experiment_manager)

    return _dashboard_instance
