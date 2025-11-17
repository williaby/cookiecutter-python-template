"""
Real-Time Performance Dashboard for Token Optimization Monitoring

This module provides a comprehensive real-time dashboard for monitoring token optimization,
function loading performance, and user experience metrics. It integrates with Grafana,
Prometheus, and provides standalone web-based dashboards.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.core.token_optimization_monitor import (
    TokenOptimizationMonitor,
    get_token_optimization_monitor,
    FunctionTier,
    OptimizationStatus
)
from src.utils.performance_monitor import get_performance_monitor
from src.utils.observability import create_structured_logger

logger = logging.getLogger(__name__)


class MetricsExporter:
    """Export metrics in Prometheus format."""

    def __init__(self, monitor: TokenOptimizationMonitor):
        self.monitor = monitor
        self.performance_monitor = get_performance_monitor()
        self.logger = create_structured_logger("metrics_exporter")

    async def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""

        metrics_lines = []
        timestamp = int(time.time() * 1000)

        # Token optimization metrics
        health_report = await self.monitor.generate_system_health_report()

        metrics_lines.extend([
            "# HELP token_reduction_percentage Token reduction achieved by optimization",
            "# TYPE token_reduction_percentage gauge",
            f"token_reduction_percentage{{type=\"average\"}} {health_report.average_token_reduction_percentage} {timestamp}",
            f"token_reduction_percentage{{type=\"median\"}} {health_report.median_token_reduction_percentage} {timestamp}",
            "",
            "# HELP function_loading_latency_ms Function loading latency by tier",
            "# TYPE function_loading_latency_ms histogram",
            f"function_loading_latency_ms{{type=\"average\"}} {health_report.average_loading_latency_ms} {timestamp}",
            f"function_loading_latency_ms{{type=\"p95\"}} {health_report.p95_loading_latency_ms} {timestamp}",
            f"function_loading_latency_ms{{type=\"p99\"}} {health_report.p99_loading_latency_ms} {timestamp}",
            "",
            "# HELP system_success_rate Overall system success rate",
            "# TYPE system_success_rate gauge",
            f"system_success_rate {health_report.overall_success_rate} {timestamp}",
            "",
            "# HELP task_detection_accuracy Task detection accuracy rate",
            "# TYPE task_detection_accuracy gauge",
            f"task_detection_accuracy {health_report.task_detection_accuracy_rate} {timestamp}",
            "",
            "# HELP concurrent_sessions Current number of active sessions",
            "# TYPE concurrent_sessions gauge",
            f"concurrent_sessions {health_report.concurrent_sessions_handled} {timestamp}",
            "",
            "# HELP optimization_validation_confidence Confidence in optimization validation",
            "# TYPE optimization_validation_confidence gauge",
            f"optimization_validation_confidence {self.monitor.validation_confidence} {timestamp}",
        ])

        # Function tier metrics
        for tier, tier_metrics in self.monitor.function_metrics.items():
            tier_name = tier.value
            metrics_lines.extend([
                f"# HELP functions_loaded_total Total functions loaded by tier",
                f"# TYPE functions_loaded_total counter",
                f"functions_loaded_total{{tier=\"{tier_name}\"}} {tier_metrics.functions_loaded} {timestamp}",
                "",
                f"cache_hits_total{{tier=\"{tier_name}\"}} {tier_metrics.cache_hits} {timestamp}",
                f"cache_misses_total{{tier=\"{tier_name}\"}} {tier_metrics.cache_misses} {timestamp}",
                f"tokens_consumed_total{{tier=\"{tier_name}\"}} {tier_metrics.tokens_consumed} {timestamp}",
                f"usage_frequency{{tier=\"{tier_name}\"}} {tier_metrics.usage_frequency} {timestamp}",
                ""
            ])

        # Performance monitor metrics
        perf_metrics = self.performance_monitor.get_all_metrics()

        for counter_name, value in perf_metrics.get("counters", {}).items():
            metrics_lines.extend([
                f"# HELP {counter_name} Counter metric",
                f"# TYPE {counter_name} counter",
                f"{counter_name} {value} {timestamp}",
                ""
            ])

        for gauge_name, value in perf_metrics.get("gauges", {}).items():
            metrics_lines.extend([
                f"# HELP {gauge_name} Gauge metric",
                f"# TYPE {gauge_name} gauge",
                f"{gauge_name} {value} {timestamp}",
                ""
            ])

        return "\n".join(metrics_lines)

    async def export_json_metrics(self) -> Dict[str, Any]:
        """Export metrics in JSON format."""

        return await self.monitor.export_metrics(format="json", include_raw_data=False)


class RealTimeDashboard:
    """Real-time web dashboard for monitoring."""

    def __init__(self, monitor: TokenOptimizationMonitor):
        self.monitor = monitor
        self.metrics_exporter = MetricsExporter(monitor)
        self.connected_clients: List[WebSocket] = []
        self.logger = create_structured_logger("dashboard")

        # Dashboard update interval
        self.update_interval_seconds = 5.0
        self._update_task: Optional[asyncio.Task] = None

    async def start_real_time_updates(self):
        """Start real-time dashboard updates."""
        if self._update_task and not self._update_task.done():
            return

        self._update_task = asyncio.create_task(self._update_loop())
        self.logger.info("Started real-time dashboard updates")

    async def stop_real_time_updates(self):
        """Stop real-time dashboard updates."""
        if self._update_task and not self._update_task.done():
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Stopped real-time dashboard updates")

    async def _update_loop(self):
        """Main update loop for dashboard."""
        while True:
            try:
                # Generate current metrics
                dashboard_data = await self._generate_dashboard_data()

                # Send to all connected clients
                if self.connected_clients:
                    await self._broadcast_to_clients(dashboard_data)

                await asyncio.sleep(self.update_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in dashboard update loop: {e}")
                await asyncio.sleep(self.update_interval_seconds)

    async def _generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate comprehensive dashboard data."""

        health_report = await self.monitor.generate_system_health_report()

        # Token optimization chart data
        token_reduction_history = []
        loading_latency_history = []

        for health_metric in list(self.monitor.system_health_history)[-20:]:  # Last 20 data points
            token_reduction_history.append({
                "timestamp": health_metric.timestamp.isoformat(),
                "average": health_metric.average_token_reduction_percentage,
                "median": health_metric.median_token_reduction_percentage
            })

            loading_latency_history.append({
                "timestamp": health_metric.timestamp.isoformat(),
                "average": health_metric.average_loading_latency_ms,
                "p95": health_metric.p95_loading_latency_ms,
                "p99": health_metric.p99_loading_latency_ms
            })

        # Function tier performance
        tier_performance = {}
        for tier, metrics in self.monitor.function_metrics.items():
            cache_hit_rate = 0.0
            if (metrics.cache_hits + metrics.cache_misses) > 0:
                cache_hit_rate = metrics.cache_hits / (metrics.cache_hits + metrics.cache_misses)

            tier_performance[tier.value] = {
                "functions_loaded": metrics.functions_loaded,
                "loading_time_ms": metrics.loading_time_ms,
                "cache_hit_rate": cache_hit_rate,
                "tokens_consumed": metrics.tokens_consumed,
                "usage_frequency": metrics.usage_frequency
            }

        # Active sessions summary
        active_sessions_summary = []
        for session_id in list(self.monitor.active_sessions)[:10]:  # Show first 10
            if session_id in self.monitor.session_metrics:
                session = self.monitor.session_metrics[session_id]
                token_reduction = 0.0
                if session.baseline_tokens_loaded > 0:
                    token_reduction = (1.0 - session.optimized_tokens_loaded / session.baseline_tokens_loaded) * 100

                active_sessions_summary.append({
                    "session_id": session_id[:8] + "...",  # Truncated for display
                    "user_id": session.user_id,
                    "task_type": session.task_type or "unknown",
                    "optimization_level": session.optimization_level.value,
                    "token_reduction_percentage": round(token_reduction, 1),
                    "functions_loaded": session.optimized_functions_loaded,
                    "functions_used": len(session.functions_actually_used),
                    "duration_minutes": round((datetime.now() - session.timestamp).total_seconds() / 60, 1)
                })

        # Validation status with detailed breakdown
        validation_details = {
            "overall_validated": self.monitor.optimization_validated,
            "confidence_percentage": round(self.monitor.validation_confidence * 100, 1),
            "target_reduction_percentage": self.monitor.token_reduction_target * 100,
            "current_average_reduction": round(health_report.average_token_reduction_percentage, 1),
            "criteria_status": {
                "token_reduction_target": health_report.average_token_reduction_percentage >= 70.0,
                "sample_size_adequate": health_report.total_sessions >= 10,
                "task_accuracy_acceptable": health_report.task_detection_accuracy_rate >= 0.80,
                "success_rate_acceptable": health_report.overall_success_rate >= 0.95,
                "latency_acceptable": health_report.p95_loading_latency_ms <= 200.0
            }
        }

        return {
            "timestamp": datetime.now().isoformat(),
            "system_health": {
                "total_sessions": health_report.total_sessions,
                "concurrent_sessions": health_report.concurrent_sessions_handled,
                "average_token_reduction": round(health_report.average_token_reduction_percentage, 1),
                "overall_success_rate": round(health_report.overall_success_rate * 100, 1),
                "task_detection_accuracy": round(health_report.task_detection_accuracy_rate * 100, 1),
                "average_loading_latency": round(health_report.average_loading_latency_ms, 1),
                "p95_loading_latency": round(health_report.p95_loading_latency_ms, 1)
            },
            "token_reduction_history": token_reduction_history,
            "loading_latency_history": loading_latency_history,
            "tier_performance": tier_performance,
            "active_sessions": active_sessions_summary,
            "validation_status": validation_details,
            "alerts": await self._generate_alerts(health_report)
        }

    async def _generate_alerts(self, health_report) -> List[Dict[str, Any]]:
        """Generate current alerts based on thresholds."""

        alerts = []

        # Token reduction alerts
        if health_report.average_token_reduction_percentage < self.monitor.min_acceptable_reduction * 100:
            alerts.append({
                "level": "warning",
                "title": "Token Reduction Below Minimum",
                "message": f"Average token reduction ({health_report.average_token_reduction_percentage:.1f}%) is below minimum threshold ({self.monitor.min_acceptable_reduction * 100}%)",
                "timestamp": datetime.now().isoformat()
            })

        # Performance alerts
        if health_report.p95_loading_latency_ms > self.monitor.max_acceptable_latency_ms:
            alerts.append({
                "level": "error",
                "title": "High Loading Latency",
                "message": f"P95 loading latency ({health_report.p95_loading_latency_ms:.1f}ms) exceeds threshold ({self.monitor.max_acceptable_latency_ms}ms)",
                "timestamp": datetime.now().isoformat()
            })

        # Success rate alerts
        if health_report.overall_success_rate < 0.95:
            alerts.append({
                "level": "warning",
                "title": "Low Success Rate",
                "message": f"Overall success rate ({health_report.overall_success_rate * 100:.1f}%) is below 95%",
                "timestamp": datetime.now().isoformat()
            })

        # Task detection alerts
        if health_report.task_detection_accuracy_rate < 0.80:
            alerts.append({
                "level": "warning",
                "title": "Low Task Detection Accuracy",
                "message": f"Task detection accuracy ({health_report.task_detection_accuracy_rate * 100:.1f}%) is below 80%",
                "timestamp": datetime.now().isoformat()
            })

        # Fallback activation alerts
        if health_report.fallback_activation_rate > 0.10:  # More than 10% fallback rate
            alerts.append({
                "level": "warning",
                "title": "High Fallback Activation Rate",
                "message": f"Fallback activation rate ({health_report.fallback_activation_rate * 100:.1f}%) suggests optimization issues",
                "timestamp": datetime.now().isoformat()
            })

        return alerts

    async def _broadcast_to_clients(self, data: Dict[str, Any]):
        """Broadcast data to all connected WebSocket clients."""

        if not self.connected_clients:
            return

        message = json.dumps(data)
        disconnected_clients = []

        for client in self.connected_clients:
            try:
                await client.send_text(message)
            except Exception:
                disconnected_clients.append(client)

        # Remove disconnected clients
        for client in disconnected_clients:
            self.connected_clients.remove(client)

    async def add_client(self, websocket: WebSocket):
        """Add a new WebSocket client."""
        await websocket.accept()
        self.connected_clients.append(websocket)

        # Send initial data
        dashboard_data = await self._generate_dashboard_data()
        await websocket.send_text(json.dumps(dashboard_data))

        self.logger.info(f"Dashboard client connected. Total clients: {len(self.connected_clients)}")

    async def remove_client(self, websocket: WebSocket):
        """Remove a WebSocket client."""
        if websocket in self.connected_clients:
            self.connected_clients.remove(websocket)

        self.logger.info(f"Dashboard client disconnected. Total clients: {len(self.connected_clients)}")

    def get_dashboard_html(self) -> str:
        """Generate HTML for the dashboard."""

        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Token Optimization Performance Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        .header {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #2563eb;
        }
        .metric-label {
            color: #6b7280;
            margin-top: 5px;
        }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .alerts {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .alert {
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 10px;
        }
        .alert.warning {
            background: #fef3c7;
            border-left: 4px solid #f59e0b;
        }
        .alert.error {
            background: #fee2e2;
            border-left: 4px solid #ef4444;
        }
        .alert.success {
            background: #d1fae5;
            border-left: 4px solid #10b981;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-success { background: #10b981; }
        .status-warning { background: #f59e0b; }
        .status-error { background: #ef4444; }
        .table {
            width: 100%;
            border-collapse: collapse;
        }
        .table th, .table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e5e7eb;
        }
        .table th {
            background: #f9fafb;
            font-weight: 600;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Token Optimization Performance Dashboard</h1>
        <p>Real-time monitoring of dynamic function loading optimization and validation</p>
        <div id="connection-status">
            <span class="status-indicator status-warning"></span>
            Connecting...
        </div>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-value" id="token-reduction">--</div>
            <div class="metric-label">Average Token Reduction %</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="success-rate">--</div>
            <div class="metric-label">Overall Success Rate %</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="loading-latency">--</div>
            <div class="metric-label">Average Loading Latency (ms)</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="active-sessions">--</div>
            <div class="metric-label">Active Sessions</div>
        </div>
    </div>

    <div class="chart-container">
        <h3>Token Reduction Over Time</h3>
        <canvas id="token-reduction-chart" width="400" height="200"></canvas>
    </div>

    <div class="chart-container">
        <h3>Loading Latency Performance</h3>
        <canvas id="latency-chart" width="400" height="200"></canvas>
    </div>

    <div class="alerts">
        <h3>Current Alerts</h3>
        <div id="alerts-container">
            <p>No alerts at this time.</p>
        </div>
    </div>

    <div class="metric-card">
        <h3>Validation Status</h3>
        <div id="validation-status"></div>
    </div>

    <div class="metric-card">
        <h3>Active Sessions</h3>
        <table class="table">
            <thead>
                <tr>
                    <th>Session</th>
                    <th>User</th>
                    <th>Task Type</th>
                    <th>Token Reduction %</th>
                    <th>Functions Used</th>
                    <th>Duration (min)</th>
                </tr>
            </thead>
            <tbody id="sessions-table">
                <tr><td colspan="6">Loading...</td></tr>
            </tbody>
        </table>
    </div>

    <script>
        let socket;
        let tokenReductionChart;
        let latencyChart;

        function initializeWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/dashboard`;

            socket = new WebSocket(wsUrl);

            socket.onopen = function() {
                document.getElementById('connection-status').innerHTML =
                    '<span class="status-indicator status-success"></span>Connected';
            };

            socket.onmessage = function(event) {
                const data = JSON.parse(event.data);
                updateDashboard(data);
            };

            socket.onclose = function() {
                document.getElementById('connection-status').innerHTML =
                    '<span class="status-indicator status-error"></span>Disconnected';
                setTimeout(initializeWebSocket, 5000);
            };

            socket.onerror = function() {
                document.getElementById('connection-status').innerHTML =
                    '<span class="status-indicator status-error"></span>Connection Error';
            };
        }

        function initializeCharts() {
            // Token Reduction Chart
            const tokenCtx = document.getElementById('token-reduction-chart').getContext('2d');
            tokenReductionChart = new Chart(tokenCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Average Reduction %',
                        data: [],
                        borderColor: '#2563eb',
                        backgroundColor: 'rgba(37, 99, 235, 0.1)',
                        tension: 0.1
                    }, {
                        label: 'Target (70%)',
                        data: [],
                        borderColor: '#10b981',
                        borderDash: [5, 5],
                        fill: false
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            });

            // Latency Chart
            const latencyCtx = document.getElementById('latency-chart').getContext('2d');
            latencyChart = new Chart(latencyCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Average Latency (ms)',
                        data: [],
                        borderColor: '#8b5cf6',
                        backgroundColor: 'rgba(139, 92, 246, 0.1)',
                        tension: 0.1
                    }, {
                        label: 'P95 Latency (ms)',
                        data: [],
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245, 158, 11, 0.1)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
        }

        function updateDashboard(data) {
            // Update metric cards
            document.getElementById('token-reduction').textContent =
                data.system_health.average_token_reduction + '%';
            document.getElementById('success-rate').textContent =
                data.system_health.overall_success_rate + '%';
            document.getElementById('loading-latency').textContent =
                data.system_health.average_loading_latency + ' ms';
            document.getElementById('active-sessions').textContent =
                data.system_health.concurrent_sessions;

            // Update charts
            updateTokenReductionChart(data.token_reduction_history);
            updateLatencyChart(data.loading_latency_history);

            // Update alerts
            updateAlerts(data.alerts);

            // Update validation status
            updateValidationStatus(data.validation_status);

            // Update sessions table
            updateSessionsTable(data.active_sessions);
        }

        function updateTokenReductionChart(history) {
            const labels = history.map(h => new Date(h.timestamp).toLocaleTimeString());
            const averageData = history.map(h => h.average);
            const targetData = new Array(history.length).fill(70);

            tokenReductionChart.data.labels = labels;
            tokenReductionChart.data.datasets[0].data = averageData;
            tokenReductionChart.data.datasets[1].data = targetData;
            tokenReductionChart.update();
        }

        function updateLatencyChart(history) {
            const labels = history.map(h => new Date(h.timestamp).toLocaleTimeString());
            const averageData = history.map(h => h.average);
            const p95Data = history.map(h => h.p95);

            latencyChart.data.labels = labels;
            latencyChart.data.datasets[0].data = averageData;
            latencyChart.data.datasets[1].data = p95Data;
            latencyChart.update();
        }

        function updateAlerts(alerts) {
            const container = document.getElementById('alerts-container');

            if (alerts.length === 0) {
                container.innerHTML = '<p>No alerts at this time.</p>';
                return;
            }

            const alertsHtml = alerts.map(alert => `
                <div class="alert ${alert.level}">
                    <strong>${alert.title}</strong><br>
                    ${alert.message}
                    <small style="float: right;">${new Date(alert.timestamp).toLocaleTimeString()}</small>
                </div>
            `).join('');

            container.innerHTML = alertsHtml;
        }

        function updateValidationStatus(status) {
            const container = document.getElementById('validation-status');
            const statusClass = status.overall_validated ? 'success' : 'warning';

            let criteriaHtml = '';
            for (const [key, passed] of Object.entries(status.criteria_status)) {
                const indicator = passed ? 'status-success' : 'status-error';
                const label = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                criteriaHtml += `
                    <div style="margin: 5px 0;">
                        <span class="status-indicator ${indicator}"></span>
                        ${label}
                    </div>
                `;
            }

            container.innerHTML = `
                <div class="alert ${statusClass}">
                    <strong>Optimization Validated: ${status.overall_validated ? 'YES' : 'NO'}</strong><br>
                    Confidence: ${status.confidence_percentage}%<br>
                    Current Average Reduction: ${status.current_average_reduction}%
                    (Target: ${status.target_reduction_percentage}%)
                </div>
                <div style="margin-top: 15px;">
                    <strong>Validation Criteria:</strong>
                    ${criteriaHtml}
                </div>
            `;
        }

        function updateSessionsTable(sessions) {
            const tbody = document.getElementById('sessions-table');

            if (sessions.length === 0) {
                tbody.innerHTML = '<tr><td colspan="6">No active sessions</td></tr>';
                return;
            }

            const rowsHtml = sessions.map(session => `
                <tr>
                    <td>${session.session_id}</td>
                    <td>${session.user_id}</td>
                    <td>${session.task_type}</td>
                    <td>${session.token_reduction_percentage}%</td>
                    <td>${session.functions_used}/${session.functions_loaded}</td>
                    <td>${session.duration_minutes}</td>
                </tr>
            `).join('');

            tbody.innerHTML = rowsHtml;
        }

        // Initialize everything
        document.addEventListener('DOMContentLoaded', function() {
            initializeCharts();
            initializeWebSocket();
        });
    </script>
</body>
</html>
        """


class AlertManager:
    """Manage alerts and notifications for performance issues."""

    def __init__(self, monitor: TokenOptimizationMonitor):
        self.monitor = monitor
        self.logger = create_structured_logger("alert_manager")

        # Alert thresholds
        self.thresholds = {
            "token_reduction_min": 50.0,  # Minimum 50% reduction
            "token_reduction_target": 70.0,  # Target 70% reduction
            "latency_max_ms": 200.0,  # Maximum loading latency
            "success_rate_min": 0.95,  # Minimum 95% success rate
            "task_accuracy_min": 0.80,  # Minimum 80% task detection accuracy
            "fallback_rate_max": 0.10,  # Maximum 10% fallback rate
        }

        # Alert state tracking
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_history: List[Dict[str, Any]] = []

        # Notification channels
        self.notification_channels = []

    async def check_alerts(self) -> List[Dict[str, Any]]:
        """Check current metrics against thresholds and generate alerts."""

        health_report = await self.monitor.generate_system_health_report()
        current_alerts = []

        # Token reduction alerts
        if health_report.average_token_reduction_percentage < self.thresholds["token_reduction_min"]:
            alert = {
                "id": "low_token_reduction",
                "level": "critical",
                "title": "Token Reduction Below Minimum Threshold",
                "message": f"Average token reduction ({health_report.average_token_reduction_percentage:.1f}%) is below minimum threshold ({self.thresholds['token_reduction_min']}%)",
                "metric_value": health_report.average_token_reduction_percentage,
                "threshold_value": self.thresholds["token_reduction_min"],
                "timestamp": datetime.now().isoformat()
            }
            current_alerts.append(alert)

        elif health_report.average_token_reduction_percentage < self.thresholds["token_reduction_target"]:
            alert = {
                "id": "token_reduction_below_target",
                "level": "warning",
                "title": "Token Reduction Below Target",
                "message": f"Average token reduction ({health_report.average_token_reduction_percentage:.1f}%) is below target ({self.thresholds['token_reduction_target']}%)",
                "metric_value": health_report.average_token_reduction_percentage,
                "threshold_value": self.thresholds["token_reduction_target"],
                "timestamp": datetime.now().isoformat()
            }
            current_alerts.append(alert)

        # Latency alerts
        if health_report.p95_loading_latency_ms > self.thresholds["latency_max_ms"]:
            alert = {
                "id": "high_loading_latency",
                "level": "error",
                "title": "High Function Loading Latency",
                "message": f"P95 loading latency ({health_report.p95_loading_latency_ms:.1f}ms) exceeds threshold ({self.thresholds['latency_max_ms']}ms)",
                "metric_value": health_report.p95_loading_latency_ms,
                "threshold_value": self.thresholds["latency_max_ms"],
                "timestamp": datetime.now().isoformat()
            }
            current_alerts.append(alert)

        # Success rate alerts
        if health_report.overall_success_rate < self.thresholds["success_rate_min"]:
            alert = {
                "id": "low_success_rate",
                "level": "warning",
                "title": "Low Overall Success Rate",
                "message": f"Overall success rate ({health_report.overall_success_rate * 100:.1f}%) is below threshold ({self.thresholds['success_rate_min'] * 100}%)",
                "metric_value": health_report.overall_success_rate * 100,
                "threshold_value": self.thresholds["success_rate_min"] * 100,
                "timestamp": datetime.now().isoformat()
            }
            current_alerts.append(alert)

        # Task detection accuracy alerts
        if health_report.task_detection_accuracy_rate < self.thresholds["task_accuracy_min"]:
            alert = {
                "id": "low_task_accuracy",
                "level": "warning",
                "title": "Low Task Detection Accuracy",
                "message": f"Task detection accuracy ({health_report.task_detection_accuracy_rate * 100:.1f}%) is below threshold ({self.thresholds['task_accuracy_min'] * 100}%)",
                "metric_value": health_report.task_detection_accuracy_rate * 100,
                "threshold_value": self.thresholds["task_accuracy_min"] * 100,
                "timestamp": datetime.now().isoformat()
            }
            current_alerts.append(alert)

        # Fallback rate alerts
        if health_report.fallback_activation_rate > self.thresholds["fallback_rate_max"]:
            alert = {
                "id": "high_fallback_rate",
                "level": "warning",
                "title": "High Fallback Activation Rate",
                "message": f"Fallback activation rate ({health_report.fallback_activation_rate * 100:.1f}%) exceeds threshold ({self.thresholds['fallback_rate_max'] * 100}%)",
                "metric_value": health_report.fallback_activation_rate * 100,
                "threshold_value": self.thresholds["fallback_rate_max"] * 100,
                "timestamp": datetime.now().isoformat()
            }
            current_alerts.append(alert)

        # Update active alerts and send notifications
        await self._update_active_alerts(current_alerts)

        return current_alerts

    async def _update_active_alerts(self, current_alerts: List[Dict[str, Any]]):
        """Update active alerts and send notifications for new alerts."""

        current_alert_ids = {alert["id"] for alert in current_alerts}
        previous_alert_ids = set(self.active_alerts.keys())

        # New alerts
        new_alert_ids = current_alert_ids - previous_alert_ids
        for alert in current_alerts:
            if alert["id"] in new_alert_ids:
                await self._send_notification(alert)
                self.logger.warning(
                    f"New alert triggered: {alert['title']}",
                    alert_id=alert["id"],
                    level=alert["level"],
                    metric_value=alert["metric_value"],
                    threshold_value=alert["threshold_value"]
                )

        # Resolved alerts
        resolved_alert_ids = previous_alert_ids - current_alert_ids
        for alert_id in resolved_alert_ids:
            resolved_alert = self.active_alerts[alert_id].copy()
            resolved_alert["resolved_at"] = datetime.now().isoformat()
            self.alert_history.append(resolved_alert)

            self.logger.info(f"Alert resolved: {resolved_alert['title']}", alert_id=alert_id)

        # Update active alerts
        self.active_alerts = {alert["id"]: alert for alert in current_alerts}

    async def _send_notification(self, alert: Dict[str, Any]):
        """Send alert notification through configured channels."""

        # Add to alert history
        self.alert_history.append(alert.copy())

        # Send through notification channels
        for channel in self.notification_channels:
            try:
                await channel.send_alert(alert)
            except Exception as e:
                self.logger.error(f"Failed to send alert through channel {channel}: {e}")

    def add_notification_channel(self, channel):
        """Add a notification channel for alerts."""
        self.notification_channels.append(channel)
        self.logger.info(f"Added notification channel: {type(channel).__name__}")

    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert summary for the specified time period."""

        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent_alerts = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert["timestamp"]) >= cutoff_time
        ]

        alert_counts = {}
        for alert in recent_alerts:
            level = alert["level"]
            alert_counts[level] = alert_counts.get(level, 0) + 1

        return {
            "time_period_hours": hours,
            "total_alerts": len(recent_alerts),
            "alert_counts_by_level": alert_counts,
            "active_alerts": len(self.active_alerts),
            "recent_alerts": recent_alerts[-10:]  # Last 10 alerts
        }


# FastAPI integration for dashboard
def create_dashboard_app(monitor: TokenOptimizationMonitor) -> FastAPI:
    """Create FastAPI app with dashboard endpoints."""

    app = FastAPI(title="Token Optimization Dashboard")
    dashboard = RealTimeDashboard(monitor)
    metrics_exporter = MetricsExporter(monitor)
    alert_manager = AlertManager(monitor)

    @app.on_event("startup")
    async def startup_event():
        await dashboard.start_real_time_updates()

    @app.on_event("shutdown")
    async def shutdown_event():
        await dashboard.stop_real_time_updates()

    @app.get("/", response_class=HTMLResponse)
    async def get_dashboard():
        return dashboard.get_dashboard_html()

    @app.get("/metrics")
    async def get_prometheus_metrics():
        return await metrics_exporter.export_prometheus_metrics()

    @app.get("/api/metrics")
    async def get_json_metrics():
        return await metrics_exporter.export_json_metrics()

    @app.get("/api/health")
    async def get_health_report():
        return await monitor.generate_system_health_report()

    @app.get("/api/optimization-report")
    async def get_optimization_report(user_id: Optional[str] = None):
        return await monitor.get_optimization_report(user_id)

    @app.get("/api/alerts")
    async def get_current_alerts():
        return await alert_manager.check_alerts()

    @app.get("/api/alerts/summary")
    async def get_alert_summary(hours: int = 24):
        return alert_manager.get_alert_summary(hours)

    @app.websocket("/ws/dashboard")
    async def websocket_endpoint(websocket: WebSocket):
        await dashboard.add_client(websocket)
        try:
            while True:
                # Keep connection alive
                await websocket.receive_text()
        except WebSocketDisconnect:
            await dashboard.remove_client(websocket)

    return app


# Global dashboard instance
_dashboard_app: Optional[FastAPI] = None


def get_dashboard_app() -> FastAPI:
    """Get the global dashboard FastAPI app."""
    global _dashboard_app
    if _dashboard_app is None:
        monitor = get_token_optimization_monitor()
        _dashboard_app = create_dashboard_app(monitor)
    return _dashboard_app
