"""
Comprehensive Metrics Collection Framework

This module provides a comprehensive metrics collection system for the token optimization
monitoring. It handles data aggregation, historical analysis, trend detection, and
statistical validation of the 70% token reduction goal.
"""

import asyncio
import json
import logging
import sqlite3
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats

from src.core.token_optimization_monitor import (
    TokenOptimizationMonitor,
    TokenUsageMetrics,
    FunctionTier,
    OptimizationStatus
)
from src.utils.observability import create_structured_logger

logger = logging.getLogger(__name__)


class MetricAggregationType(Enum):
    """Types of metric aggregations."""

    SUM = "sum"
    AVERAGE = "average"
    MEDIAN = "median"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    PERCENTILE_95 = "p95"
    PERCENTILE_99 = "p99"
    STANDARD_DEVIATION = "stddev"


class TimeWindow(Enum):
    """Time windows for metric aggregation."""

    MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    HOUR = "1h"
    SIX_HOURS = "6h"
    DAY = "1d"
    WEEK = "1w"
    MONTH = "1M"


@dataclass
class MetricPoint:
    """Individual metric data point."""

    timestamp: datetime
    metric_name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedMetric:
    """Aggregated metric over a time window."""

    metric_name: str
    time_window: TimeWindow
    window_start: datetime
    window_end: datetime
    aggregation_type: MetricAggregationType
    value: float
    sample_count: int
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class TrendAnalysis:
    """Trend analysis result for a metric."""

    metric_name: str
    time_period: timedelta
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_strength: float  # 0.0 to 1.0
    slope: float
    r_squared: float
    statistical_significance: bool
    confidence_interval: Tuple[float, float]
    predicted_values: List[float]


@dataclass
class ValidationResult:
    """Statistical validation result for optimization claims."""

    claim: str
    validated: bool
    confidence_level: float
    p_value: float
    effect_size: float
    sample_size: int
    statistical_power: float
    evidence_strength: str  # "weak", "moderate", "strong", "very_strong"
    details: Dict[str, Any]


class MetricsDatabase:
    """SQLite database for persistent metrics storage."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path("metrics.db")
        self.logger = create_structured_logger("metrics_database")
        self._initialize_database()

    def _initialize_database(self):
        """Initialize SQLite database schema."""

        with sqlite3.connect(self.db_path) as conn:
            # Metric points table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metric_points (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    labels TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Aggregated metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS aggregated_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    time_window TEXT NOT NULL,
                    window_start TEXT NOT NULL,
                    window_end TEXT NOT NULL,
                    aggregation_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    sample_count INTEGER NOT NULL,
                    labels TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Validation results table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS validation_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    claim TEXT NOT NULL,
                    validated BOOLEAN NOT NULL,
                    confidence_level REAL NOT NULL,
                    p_value REAL,
                    effect_size REAL,
                    sample_size INTEGER,
                    statistical_power REAL,
                    evidence_strength TEXT,
                    details TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metric_points_timestamp ON metric_points(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metric_points_name ON metric_points(metric_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_aggregated_window ON aggregated_metrics(window_start, window_end)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_validation_created ON validation_results(created_at)")

    async def store_metric_point(self, point: MetricPoint):
        """Store a metric point in the database."""

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO metric_points
                    (timestamp, metric_name, value, labels, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    point.timestamp.isoformat(),
                    point.metric_name,
                    point.value,
                    json.dumps(point.labels),
                    json.dumps(point.metadata)
                ))
        except Exception as e:
            self.logger.error(f"Failed to store metric point: {e}")

    async def store_aggregated_metric(self, metric: AggregatedMetric):
        """Store an aggregated metric in the database."""

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO aggregated_metrics
                    (metric_name, time_window, window_start, window_end,
                     aggregation_type, value, sample_count, labels)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metric.metric_name,
                    metric.time_window.value,
                    metric.window_start.isoformat(),
                    metric.window_end.isoformat(),
                    metric.aggregation_type.value,
                    metric.value,
                    metric.sample_count,
                    json.dumps(metric.labels)
                ))
        except Exception as e:
            self.logger.error(f"Failed to store aggregated metric: {e}")

    async def store_validation_result(self, result: ValidationResult):
        """Store a validation result in the database."""

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO validation_results
                    (claim, validated, confidence_level, p_value, effect_size,
                     sample_size, statistical_power, evidence_strength, details)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.claim,
                    result.validated,
                    result.confidence_level,
                    result.p_value,
                    result.effect_size,
                    result.sample_size,
                    result.statistical_power,
                    result.evidence_strength,
                    json.dumps(result.details)
                ))
        except Exception as e:
            self.logger.error(f"Failed to store validation result: {e}")

    async def get_metric_points(self, metric_name: str, start_time: datetime,
                              end_time: datetime, labels: Optional[Dict[str, str]] = None) -> List[MetricPoint]:
        """Retrieve metric points from the database."""

        query = """
            SELECT timestamp, metric_name, value, labels, metadata
            FROM metric_points
            WHERE metric_name = ? AND timestamp BETWEEN ? AND ?
        """
        params = [metric_name, start_time.isoformat(), end_time.isoformat()]

        if labels:
            # Simple label filtering (in production, use proper JSON queries)
            query += " AND labels LIKE ?"
            params.append(f"%{list(labels.items())[0][0]}%")

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(query, params)
                points = []

                for row in cursor.fetchall():
                    points.append(MetricPoint(
                        timestamp=datetime.fromisoformat(row[0]),
                        metric_name=row[1],
                        value=row[2],
                        labels=json.loads(row[3] or "{}"),
                        metadata=json.loads(row[4] or "{}")
                    ))

                return points
        except Exception as e:
            self.logger.error(f"Failed to retrieve metric points: {e}")
            return []


class MetricsAggregator:
    """Aggregate metrics over different time windows."""

    def __init__(self, database: MetricsDatabase):
        self.database = database
        self.logger = create_structured_logger("metrics_aggregator")

    async def aggregate_metrics(self, metric_name: str, time_window: TimeWindow,
                              start_time: datetime, end_time: datetime,
                              aggregation_types: List[MetricAggregationType],
                              labels: Optional[Dict[str, str]] = None) -> List[AggregatedMetric]:
        """Aggregate metrics over specified time windows."""

        # Get raw metric points
        points = await self.database.get_metric_points(metric_name, start_time, end_time, labels)

        if not points:
            return []

        # Group points by time windows
        window_duration = self._get_window_duration(time_window)
        windows = self._group_points_by_windows(points, window_duration, start_time, end_time)

        aggregated_metrics = []

        for window_start, window_end, window_points in windows:
            if not window_points:
                continue

            values = [point.value for point in window_points]

            for agg_type in aggregation_types:
                agg_value = self._calculate_aggregation(values, agg_type)

                aggregated_metric = AggregatedMetric(
                    metric_name=metric_name,
                    time_window=time_window,
                    window_start=window_start,
                    window_end=window_end,
                    aggregation_type=agg_type,
                    value=agg_value,
                    sample_count=len(values),
                    labels=labels or {}
                )

                aggregated_metrics.append(aggregated_metric)
                await self.database.store_aggregated_metric(aggregated_metric)

        return aggregated_metrics

    def _get_window_duration(self, time_window: TimeWindow) -> timedelta:
        """Get timedelta for a time window."""

        durations = {
            TimeWindow.MINUTE: timedelta(minutes=1),
            TimeWindow.FIVE_MINUTES: timedelta(minutes=5),
            TimeWindow.FIFTEEN_MINUTES: timedelta(minutes=15),
            TimeWindow.HOUR: timedelta(hours=1),
            TimeWindow.SIX_HOURS: timedelta(hours=6),
            TimeWindow.DAY: timedelta(days=1),
            TimeWindow.WEEK: timedelta(weeks=1),
            TimeWindow.MONTH: timedelta(days=30)
        }

        return durations[time_window]

    def _group_points_by_windows(self, points: List[MetricPoint], window_duration: timedelta,
                                start_time: datetime, end_time: datetime) -> List[Tuple[datetime, datetime, List[MetricPoint]]]:
        """Group metric points into time windows."""

        windows = []
        current_window_start = start_time

        while current_window_start < end_time:
            current_window_end = min(current_window_start + window_duration, end_time)

            window_points = [
                point for point in points
                if current_window_start <= point.timestamp < current_window_end
            ]

            windows.append((current_window_start, current_window_end, window_points))
            current_window_start = current_window_end

        return windows

    def _calculate_aggregation(self, values: List[float], agg_type: MetricAggregationType) -> float:
        """Calculate aggregated value based on aggregation type."""

        if not values:
            return 0.0

        if agg_type == MetricAggregationType.SUM:
            return sum(values)
        elif agg_type == MetricAggregationType.AVERAGE:
            return statistics.mean(values)
        elif agg_type == MetricAggregationType.MEDIAN:
            return statistics.median(values)
        elif agg_type == MetricAggregationType.COUNT:
            return float(len(values))
        elif agg_type == MetricAggregationType.MIN:
            return min(values)
        elif agg_type == MetricAggregationType.MAX:
            return max(values)
        elif agg_type == MetricAggregationType.PERCENTILE_95:
            return np.percentile(values, 95)
        elif agg_type == MetricAggregationType.PERCENTILE_99:
            return np.percentile(values, 99)
        elif agg_type == MetricAggregationType.STANDARD_DEVIATION:
            return statistics.stdev(values) if len(values) > 1 else 0.0
        else:
            return 0.0


class TrendAnalyzer:
    """Analyze trends in metrics over time."""

    def __init__(self, database: MetricsDatabase):
        self.database = database
        self.logger = create_structured_logger("trend_analyzer")

    async def analyze_trend(self, metric_name: str, time_period: timedelta,
                          end_time: Optional[datetime] = None) -> TrendAnalysis:
        """Analyze trend for a metric over a specified time period."""

        if end_time is None:
            end_time = datetime.now()
        start_time = end_time - time_period

        # Get metric points
        points = await self.database.get_metric_points(metric_name, start_time, end_time)

        if len(points) < 5:  # Need minimum points for trend analysis
            return TrendAnalysis(
                metric_name=metric_name,
                time_period=time_period,
                trend_direction="insufficient_data",
                trend_strength=0.0,
                slope=0.0,
                r_squared=0.0,
                statistical_significance=False,
                confidence_interval=(0.0, 0.0),
                predicted_values=[]
            )

        # Prepare data for linear regression
        timestamps = [(point.timestamp - start_time).total_seconds() for point in points]
        values = [point.value for point in points]

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(timestamps, values)
        r_squared = r_value ** 2

        # Determine trend direction and strength
        trend_direction = "stable"
        if abs(slope) > 0.001:  # Threshold for considering a trend significant
            trend_direction = "increasing" if slope > 0 else "decreasing"

        trend_strength = min(abs(r_squared), 1.0)
        statistical_significance = p_value < 0.05

        # Calculate confidence interval for slope
        degrees_freedom = len(points) - 2
        t_value = stats.t.ppf(0.975, degrees_freedom)  # 95% confidence
        margin_error = t_value * std_err
        confidence_interval = (slope - margin_error, slope + margin_error)

        # Generate predicted values
        future_timestamps = [max(timestamps) + i * 3600 for i in range(1, 25)]  # Next 24 hours
        predicted_values = [slope * t + intercept for t in future_timestamps]

        return TrendAnalysis(
            metric_name=metric_name,
            time_period=time_period,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            slope=slope,
            r_squared=r_squared,
            statistical_significance=statistical_significance,
            confidence_interval=confidence_interval,
            predicted_values=predicted_values
        )

    async def detect_anomalies(self, metric_name: str, time_period: timedelta,
                             sensitivity: float = 2.0) -> List[MetricPoint]:
        """Detect anomalies in metric values using statistical methods."""

        end_time = datetime.now()
        start_time = end_time - time_period

        points = await self.database.get_metric_points(metric_name, start_time, end_time)

        if len(points) < 10:
            return []

        values = [point.value for point in points]
        mean_value = statistics.mean(values)
        std_dev = statistics.stdev(values)

        # Detect outliers using z-score
        anomalies = []
        for point in points:
            z_score = abs(point.value - mean_value) / std_dev if std_dev > 0 else 0
            if z_score > sensitivity:
                anomalies.append(point)

        return anomalies


class StatisticalValidator:
    """Perform statistical validation of optimization claims."""

    def __init__(self, database: MetricsDatabase):
        self.database = database
        self.logger = create_structured_logger("statistical_validator")

    async def validate_token_reduction_claim(self, target_reduction: float = 0.70,
                                           time_period: timedelta = timedelta(days=7)) -> ValidationResult:
        """Validate the token reduction claim with statistical rigor."""

        end_time = datetime.now()
        start_time = end_time - time_period

        # Get token reduction metrics
        points = await self.database.get_metric_points(
            "token_reduction_percentage", start_time, end_time
        )

        if len(points) < 10:
            return ValidationResult(
                claim=f"Token reduction >= {target_reduction * 100}%",
                validated=False,
                confidence_level=0.0,
                p_value=1.0,
                effect_size=0.0,
                sample_size=len(points),
                statistical_power=0.0,
                evidence_strength="insufficient_data",
                details={"reason": "Insufficient sample size for statistical validation"}
            )

        # Convert to proportions for analysis
        reductions = [point.value / 100.0 for point in points]

        # One-sample t-test against target
        t_statistic, p_value = stats.ttest_1samp(reductions, target_reduction)

        # Calculate effect size (Cohen's d)
        sample_mean = statistics.mean(reductions)
        sample_std = statistics.stdev(reductions)
        effect_size = (sample_mean - target_reduction) / sample_std if sample_std > 0 else 0.0

        # Calculate statistical power (simplified)
        power_analysis = self._calculate_power(len(points), effect_size, 0.05)

        # Determine validation status
        validated = (sample_mean >= target_reduction) and (p_value < 0.05) and (power_analysis > 0.8)

        # Confidence level
        confidence_level = max(0.0, min(1.0, 1.0 - p_value))

        # Evidence strength
        evidence_strength = self._determine_evidence_strength(effect_size, p_value, power_analysis)

        result = ValidationResult(
            claim=f"Token reduction >= {target_reduction * 100}%",
            validated=validated,
            confidence_level=confidence_level,
            p_value=p_value,
            effect_size=effect_size,
            sample_size=len(points),
            statistical_power=power_analysis,
            evidence_strength=evidence_strength,
            details={
                "sample_mean": sample_mean * 100,
                "sample_std": sample_std * 100,
                "target_reduction": target_reduction * 100,
                "t_statistic": t_statistic,
                "degrees_freedom": len(points) - 1,
                "analysis_period_days": time_period.days
            }
        )

        await self.database.store_validation_result(result)
        return result

    async def validate_performance_claims(self) -> List[ValidationResult]:
        """Validate multiple performance claims."""

        claims_to_validate = [
            ("token_reduction_70_percent", "token_reduction_percentage", 70.0, ">="),
            ("loading_latency_under_200ms", "function_loading_latency_ms", 200.0, "<="),
            ("success_rate_above_95", "success_rate_percentage", 95.0, ">="),
            ("task_accuracy_above_80", "task_detection_accuracy", 80.0, ">=")
        ]

        results = []

        for claim_name, metric_name, target_value, comparison in claims_to_validate:
            result = await self._validate_single_claim(
                claim_name, metric_name, target_value, comparison
            )
            results.append(result)

        return results

    async def _validate_single_claim(self, claim_name: str, metric_name: str,
                                   target_value: float, comparison: str) -> ValidationResult:
        """Validate a single performance claim."""

        time_period = timedelta(days=7)
        end_time = datetime.now()
        start_time = end_time - time_period

        points = await self.database.get_metric_points(metric_name, start_time, end_time)

        if len(points) < 5:
            return ValidationResult(
                claim=f"{claim_name}: {metric_name} {comparison} {target_value}",
                validated=False,
                confidence_level=0.0,
                p_value=1.0,
                effect_size=0.0,
                sample_size=len(points),
                statistical_power=0.0,
                evidence_strength="insufficient_data",
                details={"reason": "Insufficient sample size"}
            )

        values = [point.value for point in points]
        sample_mean = statistics.mean(values)
        sample_std = statistics.stdev(values) if len(values) > 1 else 0.0

        # Determine if claim is validated
        if comparison == ">=":
            validated = sample_mean >= target_value
            alternative = "greater"
        elif comparison == "<=":
            validated = sample_mean <= target_value
            alternative = "less"
        else:
            validated = False
            alternative = "two-sided"

        # Statistical test
        if sample_std > 0:
            t_statistic, p_value = stats.ttest_1samp(values, target_value)
            if alternative != "two-sided":
                p_value = p_value / 2  # One-tailed test
        else:
            t_statistic = 0.0
            p_value = 1.0 if not validated else 0.0

        # Effect size
        effect_size = abs(sample_mean - target_value) / sample_std if sample_std > 0 else 0.0

        # Statistical power
        power = self._calculate_power(len(values), effect_size, 0.05)

        # Confidence level
        confidence_level = max(0.0, min(1.0, 1.0 - p_value)) if validated else 0.0

        # Evidence strength
        evidence_strength = self._determine_evidence_strength(effect_size, p_value, power)

        result = ValidationResult(
            claim=f"{claim_name}: {metric_name} {comparison} {target_value}",
            validated=validated and p_value < 0.05,
            confidence_level=confidence_level,
            p_value=p_value,
            effect_size=effect_size,
            sample_size=len(values),
            statistical_power=power,
            evidence_strength=evidence_strength,
            details={
                "sample_mean": sample_mean,
                "sample_std": sample_std,
                "target_value": target_value,
                "comparison": comparison,
                "t_statistic": t_statistic
            }
        )

        await self.database.store_validation_result(result)
        return result

    def _calculate_power(self, sample_size: int, effect_size: float, alpha: float = 0.05) -> float:
        """Calculate statistical power (simplified calculation)."""

        if sample_size < 2 or effect_size == 0:
            return 0.0

        # Simplified power calculation using normal approximation
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = effect_size * (sample_size ** 0.5) - z_alpha
        power = stats.norm.cdf(z_beta)

        return max(0.0, min(1.0, power))

    def _determine_evidence_strength(self, effect_size: float, p_value: float, power: float) -> str:
        """Determine evidence strength based on statistical measures."""

        # Cohen's conventions for effect size
        if effect_size < 0.2:
            effect_category = "small"
        elif effect_size < 0.5:
            effect_category = "medium"
        elif effect_size < 0.8:
            effect_category = "large"
        else:
            effect_category = "very_large"

        # Overall strength assessment
        if p_value > 0.05:
            return "weak"
        elif p_value > 0.01:
            if power > 0.8 and effect_size > 0.5:
                return "moderate"
            else:
                return "weak"
        elif p_value > 0.001:
            if power > 0.8 and effect_size > 0.3:
                return "strong"
            else:
                return "moderate"
        else:
            if power > 0.8 and effect_size > 0.5:
                return "very_strong"
            else:
                return "strong"


class MetricsCollector:
    """Main metrics collection orchestrator."""

    def __init__(self, monitor: TokenOptimizationMonitor, db_path: Optional[Path] = None):
        self.monitor = monitor
        self.database = MetricsDatabase(db_path)
        self.aggregator = MetricsAggregator(self.database)
        self.trend_analyzer = TrendAnalyzer(self.database)
        self.validator = StatisticalValidator(self.database)
        self.logger = create_structured_logger("metrics_collector")

        # Collection settings
        self.collection_interval = 30.0  # seconds
        self.aggregation_interval = 300.0  # 5 minutes

        # Background tasks
        self._collection_task: Optional[asyncio.Task] = None
        self._aggregation_task: Optional[asyncio.Task] = None

        # Metrics buffer
        self.metrics_buffer: deque = deque(maxlen=1000)

    async def start_collection(self):
        """Start metrics collection."""

        if self._collection_task and not self._collection_task.done():
            return

        self._collection_task = asyncio.create_task(self._collection_loop())
        self._aggregation_task = asyncio.create_task(self._aggregation_loop())

        self.logger.info("Started metrics collection")

    async def stop_collection(self):
        """Stop metrics collection."""

        if self._collection_task and not self._collection_task.done():
            self._collection_task.cancel()

        if self._aggregation_task and not self._aggregation_task.done():
            self._aggregation_task.cancel()

        # Wait for tasks to complete
        for task in [self._collection_task, self._aggregation_task]:
            if task:
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self.logger.info("Stopped metrics collection")

    async def _collection_loop(self):
        """Main metrics collection loop."""

        while True:
            try:
                await self._collect_current_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(self.collection_interval)

    async def _aggregation_loop(self):
        """Main metrics aggregation loop."""

        while True:
            try:
                await self._perform_aggregations()
                await asyncio.sleep(self.aggregation_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in metrics aggregation loop: {e}")
                await asyncio.sleep(self.aggregation_interval)

    async def _collect_current_metrics(self):
        """Collect current metrics from the monitor."""

        current_time = datetime.now()

        # System health metrics
        health_report = await self.monitor.generate_system_health_report()

        metrics_to_collect = [
            ("token_reduction_percentage", health_report.average_token_reduction_percentage),
            ("loading_latency_ms", health_report.average_loading_latency_ms),
            ("loading_latency_p95_ms", health_report.p95_loading_latency_ms),
            ("loading_latency_p99_ms", health_report.p99_loading_latency_ms),
            ("success_rate_percentage", health_report.overall_success_rate * 100),
            ("task_detection_accuracy", health_report.task_detection_accuracy_rate * 100),
            ("fallback_activation_rate", health_report.fallback_activation_rate * 100),
            ("concurrent_sessions", health_report.concurrent_sessions_handled),
            ("total_sessions", health_report.total_sessions)
        ]

        for metric_name, value in metrics_to_collect:
            point = MetricPoint(
                timestamp=current_time,
                metric_name=metric_name,
                value=value,
                labels={"source": "system_health"},
                metadata={"collection_interval": self.collection_interval}
            )

            self.metrics_buffer.append(point)
            await self.database.store_metric_point(point)

        # Function tier metrics
        for tier, tier_metrics in self.monitor.function_metrics.items():
            tier_label = {"tier": tier.value, "source": "function_tier"}

            tier_metrics_to_collect = [
                (f"functions_loaded_{tier.value}", tier_metrics.functions_loaded),
                (f"loading_time_ms_{tier.value}", tier_metrics.loading_time_ms),
                (f"cache_hits_{tier.value}", tier_metrics.cache_hits),
                (f"cache_misses_{tier.value}", tier_metrics.cache_misses),
                (f"tokens_consumed_{tier.value}", tier_metrics.tokens_consumed),
                (f"usage_frequency_{tier.value}", tier_metrics.usage_frequency * 100)
            ]

            for metric_name, value in tier_metrics_to_collect:
                point = MetricPoint(
                    timestamp=current_time,
                    metric_name=metric_name,
                    value=value,
                    labels=tier_label,
                    metadata={"tier": tier.value}
                )

                self.metrics_buffer.append(point)
                await self.database.store_metric_point(point)

        # Validation metrics
        validation_point = MetricPoint(
            timestamp=current_time,
            metric_name="optimization_validation_confidence",
            value=self.monitor.validation_confidence * 100,
            labels={"source": "validation"},
            metadata={"validated": self.monitor.optimization_validated}
        )

        self.metrics_buffer.append(validation_point)
        await self.database.store_metric_point(validation_point)

    async def _perform_aggregations(self):
        """Perform metric aggregations for different time windows."""

        current_time = datetime.now()

        # Aggregation configurations
        aggregation_configs = [
            (TimeWindow.FIVE_MINUTES, timedelta(hours=1)),
            (TimeWindow.FIFTEEN_MINUTES, timedelta(hours=6)),
            (TimeWindow.HOUR, timedelta(days=1)),
            (TimeWindow.SIX_HOURS, timedelta(days=7))
        ]

        # Metrics to aggregate
        metrics_to_aggregate = [
            "token_reduction_percentage",
            "loading_latency_ms",
            "success_rate_percentage",
            "task_detection_accuracy"
        ]

        # Aggregation types to compute
        agg_types = [
            MetricAggregationType.AVERAGE,
            MetricAggregationType.MEDIAN,
            MetricAggregationType.PERCENTILE_95,
            MetricAggregationType.PERCENTILE_99,
            MetricAggregationType.MIN,
            MetricAggregationType.MAX
        ]

        for time_window, lookback_period in aggregation_configs:
            start_time = current_time - lookback_period

            for metric_name in metrics_to_aggregate:
                try:
                    await self.aggregator.aggregate_metrics(
                        metric_name=metric_name,
                        time_window=time_window,
                        start_time=start_time,
                        end_time=current_time,
                        aggregation_types=agg_types
                    )
                except Exception as e:
                    self.logger.error(f"Failed to aggregate {metric_name} for {time_window}: {e}")

    async def generate_comprehensive_report(self, time_period: timedelta = timedelta(days=7)) -> Dict[str, Any]:
        """Generate comprehensive metrics report."""

        current_time = datetime.now()

        # Perform validation
        validation_results = await self.validator.validate_performance_claims()
        token_reduction_validation = await self.validator.validate_token_reduction_claim()

        # Perform trend analysis
        trend_analyses = {}
        for metric in ["token_reduction_percentage", "loading_latency_ms", "success_rate_percentage"]:
            trend = await self.trend_analyzer.analyze_trend(metric, time_period)
            trend_analyses[metric] = asdict(trend)

        # Detect anomalies
        anomalies = {}
        for metric in ["token_reduction_percentage", "loading_latency_ms"]:
            metric_anomalies = await self.trend_analyzer.detect_anomalies(metric, time_period)
            anomalies[metric] = len(metric_anomalies)

        # Generate export data
        export_data = await self.monitor.export_metrics(include_raw_data=False)

        report = {
            "report_timestamp": current_time.isoformat(),
            "analysis_period": {
                "duration_days": time_period.days,
                "start_time": (current_time - time_period).isoformat(),
                "end_time": current_time.isoformat()
            },
            "validation_results": {
                "token_reduction_claim": asdict(token_reduction_validation),
                "performance_claims": [asdict(result) for result in validation_results],
                "overall_validation_status": all(result.validated for result in validation_results + [token_reduction_validation])
            },
            "trend_analysis": trend_analyses,
            "anomaly_detection": anomalies,
            "system_metrics": export_data,
            "statistical_summary": {
                "sample_sizes": {
                    metric: len(await self.database.get_metric_points(
                        metric, current_time - time_period, current_time
                    ))
                    for metric in ["token_reduction_percentage", "loading_latency_ms", "success_rate_percentage"]
                },
                "confidence_intervals": {
                    result.claim.split(":")[0]: result.confidence_level
                    for result in validation_results + [token_reduction_validation]
                }
            }
        }

        return report

    async def export_for_external_analysis(self, format: str = "json") -> Dict[str, Any]:
        """Export metrics data for external analysis tools."""

        report = await self.generate_comprehensive_report()

        if format.lower() == "prometheus":
            # Convert to Prometheus format would go here
            pass
        elif format.lower() == "grafana":
            # Convert to Grafana dashboard format would go here
            pass

        return {
            "export_format": format,
            "export_timestamp": datetime.now().isoformat(),
            "data": report
        }


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _global_collector
    if _global_collector is None:
        from src.core.token_optimization_monitor import get_token_optimization_monitor
        monitor = get_token_optimization_monitor()
        _global_collector = MetricsCollector(monitor)
    return _global_collector


async def initialize_metrics_collection() -> MetricsCollector:
    """Initialize the metrics collection system."""
    collector = get_metrics_collector()
    await collector.start_collection()
    logger.info("Metrics collection system initialized and started")
    return collector
