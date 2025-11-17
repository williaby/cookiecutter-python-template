"""
Integration Utilities for External Monitoring Systems

This module provides utilities for integrating the token optimization monitoring
system with external platforms like Prometheus, Grafana, DataDog, CloudWatch,
and other monitoring infrastructure.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import aiohttp
import requests

from src.core.token_optimization_monitor import TokenOptimizationMonitor
from src.monitoring.metrics_collector import MetricsCollector
from src.utils.observability import create_structured_logger

logger = logging.getLogger(__name__)


class ExternalIntegration(ABC):
    """Abstract base class for external monitoring integrations."""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = create_structured_logger(f"integration_{name}")
        self.enabled = config.get("enabled", True)

    @abstractmethod
    async def send_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Send metrics to the external system."""
        pass

    @abstractmethod
    async def send_alert(self, alert: Dict[str, Any]) -> bool:
        """Send alert to the external system."""
        pass

    @abstractmethod
    async def validate_configuration(self) -> bool:
        """Validate the configuration for this integration."""
        pass


class PrometheusIntegration(ExternalIntegration):
    """Integration with Prometheus monitoring system."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("prometheus", config)
        self.pushgateway_url = config.get("pushgateway_url", "http://localhost:9091")
        self.job_name = config.get("job_name", "token_optimization")
        self.instance = config.get("instance", "default")

    async def send_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Send metrics to Prometheus via Pushgateway."""

        if not self.enabled:
            return True

        try:
            prometheus_metrics = self._convert_to_prometheus_format(metrics)

            url = f"{self.pushgateway_url}/metrics/job/{self.job_name}/instance/{self.instance}"

            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=prometheus_metrics) as response:
                    if response.status == 200:
                        self.logger.info("Successfully sent metrics to Prometheus")
                        return True
                    else:
                        self.logger.error(f"Failed to send metrics to Prometheus: {response.status}")
                        return False

        except Exception as e:
            self.logger.error(f"Error sending metrics to Prometheus: {e}")
            return False

    async def send_alert(self, alert: Dict[str, Any]) -> bool:
        """Send alert to Prometheus (via AlertManager)."""

        if not self.enabled:
            return True

        # Prometheus alerts are typically handled by AlertManager
        # This would require additional configuration
        self.logger.info(f"Alert would be sent to AlertManager: {alert['title']}")
        return True

    async def validate_configuration(self) -> bool:
        """Validate Prometheus configuration."""

        try:
            url = f"{self.pushgateway_url}/api/v1/metrics"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return response.status == 200
        except Exception as e:
            self.logger.error(f"Prometheus configuration validation failed: {e}")
            return False

    def _convert_to_prometheus_format(self, metrics: Dict[str, Any]) -> str:
        """Convert metrics to Prometheus exposition format."""

        lines = []
        timestamp = int(datetime.now().timestamp() * 1000)

        # System health metrics
        system_health = metrics.get("system_health", {})
        for metric_name, value in system_health.items():
            prometheus_name = f"token_optimization_{metric_name}"
            lines.append(f"# TYPE {prometheus_name} gauge")
            lines.append(f"{prometheus_name} {value} {timestamp}")

        # Validation status
        validation = metrics.get("validation_status", {})
        if validation:
            lines.append("# TYPE token_optimization_validated gauge")
            lines.append(f"token_optimization_validated {int(validation.get('optimization_validated', False))} {timestamp}")
            lines.append("# TYPE token_optimization_confidence gauge")
            lines.append(f"token_optimization_confidence {validation.get('validation_confidence', 0)} {timestamp}")

        # Function tier metrics
        tier_performance = metrics.get("tier_performance", {})
        for tier, tier_metrics in tier_performance.items():
            for metric_name, value in tier_metrics.items():
                prometheus_name = f"token_optimization_tier_{metric_name}"
                lines.append(f"# TYPE {prometheus_name} gauge")
                lines.append(f'{prometheus_name}{{tier="{tier}"}} {value} {timestamp}')

        return "\n".join(lines)


class GrafanaIntegration(ExternalIntegration):
    """Integration with Grafana for dashboard creation and management."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("grafana", config)
        self.api_url = config.get("api_url", "http://localhost:3000/api")
        self.api_key = config.get("api_key")
        self.org_id = config.get("org_id", 1)

    async def send_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Grafana doesn't typically receive metrics directly."""
        return True

    async def send_alert(self, alert: Dict[str, Any]) -> bool:
        """Send alert notification to Grafana."""

        if not self.enabled or not self.api_key:
            return True

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            # Create annotation for the alert
            annotation = {
                "time": int(datetime.now().timestamp() * 1000),
                "text": f"{alert['title']}: {alert['message']}",
                "tags": ["alert", alert["level"]],
                "title": alert["title"]
            }

            url = f"{self.api_url}/annotations"

            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=annotation) as response:
                    if response.status == 200:
                        self.logger.info("Successfully sent alert annotation to Grafana")
                        return True
                    else:
                        self.logger.error(f"Failed to send alert to Grafana: {response.status}")
                        return False

        except Exception as e:
            self.logger.error(f"Error sending alert to Grafana: {e}")
            return False

    async def validate_configuration(self) -> bool:
        """Validate Grafana configuration."""

        if not self.api_key:
            return False

        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            url = f"{self.api_url}/org"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    return response.status == 200
        except Exception as e:
            self.logger.error(f"Grafana configuration validation failed: {e}")
            return False

    async def create_dashboard(self, metrics: Dict[str, Any]) -> Optional[str]:
        """Create or update Grafana dashboard."""

        if not self.enabled or not self.api_key:
            return None

        dashboard_config = self._generate_dashboard_config(metrics)

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            url = f"{self.api_url}/dashboards/db"

            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=dashboard_config) as response:
                    if response.status == 200:
                        result = await response.json()
                        dashboard_url = result.get("url")
                        self.logger.info(f"Successfully created/updated Grafana dashboard: {dashboard_url}")
                        return dashboard_url
                    else:
                        self.logger.error(f"Failed to create Grafana dashboard: {response.status}")
                        return None

        except Exception as e:
            self.logger.error(f"Error creating Grafana dashboard: {e}")
            return None

    def _generate_dashboard_config(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Grafana dashboard configuration."""

        return {
            "dashboard": {
                "id": None,
                "title": "Token Optimization Performance",
                "tags": ["token-optimization", "performance"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "Token Reduction Over Time",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "token_optimization_average_token_reduction",
                                "legendFormat": "Average Token Reduction %"
                            }
                        ],
                        "yAxes": [{"label": "Percentage", "min": 0, "max": 100}],
                        "gridPos": {"h": 9, "w": 12, "x": 0, "y": 0}
                    },
                    {
                        "id": 2,
                        "title": "Loading Latency",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "token_optimization_average_loading_latency",
                                "legendFormat": "Average Latency (ms)"
                            },
                            {
                                "expr": "token_optimization_p95_loading_latency",
                                "legendFormat": "P95 Latency (ms)"
                            }
                        ],
                        "yAxes": [{"label": "Milliseconds", "min": 0}],
                        "gridPos": {"h": 9, "w": 12, "x": 12, "y": 0}
                    },
                    {
                        "id": 3,
                        "title": "Success Rate",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "token_optimization_overall_success_rate",
                                "legendFormat": "Success Rate %"
                            }
                        ],
                        "gridPos": {"h": 9, "w": 6, "x": 0, "y": 9}
                    },
                    {
                        "id": 4,
                        "title": "Validation Status",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "token_optimization_validated",
                                "legendFormat": "Optimization Validated"
                            }
                        ],
                        "gridPos": {"h": 9, "w": 6, "x": 6, "y": 9}
                    }
                ],
                "time": {"from": "now-1h", "to": "now"},
                "refresh": "30s"
            },
            "overwrite": True
        }


class DataDogIntegration(ExternalIntegration):
    """Integration with DataDog monitoring platform."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("datadog", config)
        self.api_key = config.get("api_key")
        self.app_key = config.get("app_key")
        self.site = config.get("site", "datadoghq.com")
        self.api_url = f"https://api.{self.site}"

    async def send_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Send metrics to DataDog."""

        if not self.enabled or not self.api_key:
            return True

        try:
            datadog_metrics = self._convert_to_datadog_format(metrics)

            headers = {
                "DD-API-KEY": self.api_key,
                "Content-Type": "application/json"
            }

            url = f"{self.api_url}/api/v1/series"

            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json={"series": datadog_metrics}) as response:
                    if response.status == 202:
                        self.logger.info("Successfully sent metrics to DataDog")
                        return True
                    else:
                        self.logger.error(f"Failed to send metrics to DataDog: {response.status}")
                        return False

        except Exception as e:
            self.logger.error(f"Error sending metrics to DataDog: {e}")
            return False

    async def send_alert(self, alert: Dict[str, Any]) -> bool:
        """Send alert event to DataDog."""

        if not self.enabled or not self.api_key:
            return True

        try:
            headers = {
                "DD-API-KEY": self.api_key,
                "Content-Type": "application/json"
            }

            event = {
                "title": alert["title"],
                "text": alert["message"],
                "alert_type": alert["level"],
                "source_type_name": "token_optimization",
                "tags": ["token-optimization", f"level:{alert['level']}"]
            }

            url = f"{self.api_url}/api/v1/events"

            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=event) as response:
                    if response.status == 202:
                        self.logger.info("Successfully sent alert to DataDog")
                        return True
                    else:
                        self.logger.error(f"Failed to send alert to DataDog: {response.status}")
                        return False

        except Exception as e:
            self.logger.error(f"Error sending alert to DataDog: {e}")
            return False

    async def validate_configuration(self) -> bool:
        """Validate DataDog configuration."""

        if not self.api_key:
            return False

        try:
            headers = {"DD-API-KEY": self.api_key}
            url = f"{self.api_url}/api/v1/validate"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    return response.status == 200
        except Exception as e:
            self.logger.error(f"DataDog configuration validation failed: {e}")
            return False

    def _convert_to_datadog_format(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert metrics to DataDog format."""

        series = []
        timestamp = int(datetime.now().timestamp())

        # System health metrics
        system_health = metrics.get("system_health", {})
        for metric_name, value in system_health.items():
            series.append({
                "metric": f"token_optimization.{metric_name}",
                "points": [[timestamp, value]],
                "tags": ["source:token_optimization", "type:system_health"],
                "type": "gauge"
            })

        # Validation metrics
        validation = metrics.get("validation_status", {})
        if validation:
            series.append({
                "metric": "token_optimization.validated",
                "points": [[timestamp, int(validation.get("optimization_validated", False))]],
                "tags": ["source:token_optimization", "type:validation"],
                "type": "gauge"
            })

            series.append({
                "metric": "token_optimization.confidence",
                "points": [[timestamp, validation.get("validation_confidence", 0)]],
                "tags": ["source:token_optimization", "type:validation"],
                "type": "gauge"
            })

        # Function tier metrics
        tier_performance = metrics.get("tier_performance", {})
        for tier, tier_metrics in tier_performance.items():
            for metric_name, value in tier_metrics.items():
                series.append({
                    "metric": f"token_optimization.tier.{metric_name}",
                    "points": [[timestamp, value]],
                    "tags": ["source:token_optimization", f"tier:{tier}", "type:function_tier"],
                    "type": "gauge"
                })

        return series


class CloudWatchIntegration(ExternalIntegration):
    """Integration with AWS CloudWatch."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("cloudwatch", config)
        self.region = config.get("region", "us-east-1")
        self.namespace = config.get("namespace", "TokenOptimization")
        self.aws_access_key = config.get("aws_access_key")
        self.aws_secret_key = config.get("aws_secret_key")

    async def send_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Send metrics to CloudWatch."""

        if not self.enabled:
            return True

        try:
            # In a real implementation, you would use boto3
            # For now, we'll simulate the CloudWatch integration
            self.logger.info("Would send metrics to CloudWatch")
            return True
        except Exception as e:
            self.logger.error(f"Error sending metrics to CloudWatch: {e}")
            return False

    async def send_alert(self, alert: Dict[str, Any]) -> bool:
        """Send alert to CloudWatch (via SNS)."""

        if not self.enabled:
            return True

        try:
            # In a real implementation, you would use boto3 SNS
            self.logger.info(f"Would send alert to CloudWatch/SNS: {alert['title']}")
            return True
        except Exception as e:
            self.logger.error(f"Error sending alert to CloudWatch: {e}")
            return False

    async def validate_configuration(self) -> bool:
        """Validate CloudWatch configuration."""

        # In a real implementation, you would validate AWS credentials
        return self.aws_access_key is not None and self.aws_secret_key is not None


class SlackIntegration(ExternalIntegration):
    """Integration with Slack for alert notifications."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("slack", config)
        self.webhook_url = config.get("webhook_url")
        self.channel = config.get("channel", "#alerts")
        self.username = config.get("username", "Token Optimization Monitor")

    async def send_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Slack doesn't typically receive metrics."""
        return True

    async def send_alert(self, alert: Dict[str, Any]) -> bool:
        """Send alert notification to Slack."""

        if not self.enabled or not self.webhook_url:
            return True

        try:
            # Format alert for Slack
            color = {
                "error": "danger",
                "warning": "warning",
                "info": "good"
            }.get(alert["level"], "warning")

            payload = {
                "channel": self.channel,
                "username": self.username,
                "icon_emoji": ":warning:",
                "attachments": [
                    {
                        "color": color,
                        "title": alert["title"],
                        "text": alert["message"],
                        "fields": [
                            {
                                "title": "Level",
                                "value": alert["level"].upper(),
                                "short": True
                            },
                            {
                                "title": "Timestamp",
                                "value": alert["timestamp"],
                                "short": True
                            }
                        ],
                        "footer": "Token Optimization Monitor",
                        "ts": int(datetime.now().timestamp())
                    }
                ]
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status == 200:
                        self.logger.info("Successfully sent alert to Slack")
                        return True
                    else:
                        self.logger.error(f"Failed to send alert to Slack: {response.status}")
                        return False

        except Exception as e:
            self.logger.error(f"Error sending alert to Slack: {e}")
            return False

    async def validate_configuration(self) -> bool:
        """Validate Slack configuration."""

        if not self.webhook_url:
            return False

        try:
            test_payload = {
                "text": "Token Optimization Monitor - Configuration Test",
                "channel": self.channel,
                "username": self.username
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=test_payload) as response:
                    return response.status == 200
        except Exception as e:
            self.logger.error(f"Slack configuration validation failed: {e}")
            return False


class IntegrationManager:
    """Manager for all external integrations."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.integrations: Dict[str, ExternalIntegration] = {}
        self.logger = create_structured_logger("integration_manager")

        self._initialize_integrations()

    def _initialize_integrations(self):
        """Initialize all configured integrations."""

        integration_classes = {
            "prometheus": PrometheusIntegration,
            "grafana": GrafanaIntegration,
            "datadog": DataDogIntegration,
            "cloudwatch": CloudWatchIntegration,
            "slack": SlackIntegration
        }

        for integration_name, integration_config in self.config.items():
            if integration_name in integration_classes:
                try:
                    integration_class = integration_classes[integration_name]
                    integration = integration_class(integration_config)
                    self.integrations[integration_name] = integration
                    self.logger.info(f"Initialized {integration_name} integration")
                except Exception as e:
                    self.logger.error(f"Failed to initialize {integration_name} integration: {e}")

    async def validate_all_configurations(self) -> Dict[str, bool]:
        """Validate all integration configurations."""

        validation_results = {}

        for name, integration in self.integrations.items():
            try:
                is_valid = await integration.validate_configuration()
                validation_results[name] = is_valid

                if is_valid:
                    self.logger.info(f"{name} integration configuration is valid")
                else:
                    self.logger.warning(f"{name} integration configuration is invalid")

            except Exception as e:
                self.logger.error(f"Error validating {name} integration: {e}")
                validation_results[name] = False

        return validation_results

    async def send_metrics_to_all(self, metrics: Dict[str, Any]) -> Dict[str, bool]:
        """Send metrics to all configured integrations."""

        results = {}

        for name, integration in self.integrations.items():
            try:
                success = await integration.send_metrics(metrics)
                results[name] = success

                if not success:
                    self.logger.warning(f"Failed to send metrics to {name}")

            except Exception as e:
                self.logger.error(f"Error sending metrics to {name}: {e}")
                results[name] = False

        return results

    async def send_alert_to_all(self, alert: Dict[str, Any]) -> Dict[str, bool]:
        """Send alert to all configured integrations."""

        results = {}

        for name, integration in self.integrations.items():
            try:
                success = await integration.send_alert(alert)
                results[name] = success

                if not success:
                    self.logger.warning(f"Failed to send alert to {name}")

            except Exception as e:
                self.logger.error(f"Error sending alert to {name}: {e}")
                results[name] = False

        return results

    def get_integration(self, name: str) -> Optional[ExternalIntegration]:
        """Get a specific integration by name."""
        return self.integrations.get(name)

    def list_integrations(self) -> List[str]:
        """List all configured integration names."""
        return list(self.integrations.keys())

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all integrations."""

        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_healthy": True,
            "integrations": {}
        }

        validation_results = await self.validate_all_configurations()

        for name, is_valid in validation_results.items():
            integration_health = {
                "enabled": self.integrations[name].enabled,
                "configuration_valid": is_valid,
                "healthy": self.integrations[name].enabled and is_valid
            }

            health_status["integrations"][name] = integration_health

            if not integration_health["healthy"]:
                health_status["overall_healthy"] = False

        return health_status


def load_integration_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load integration configuration from file."""

    default_config = {
        "prometheus": {
            "enabled": False,
            "pushgateway_url": "http://localhost:9091",
            "job_name": "token_optimization",
            "instance": "default"
        },
        "grafana": {
            "enabled": False,
            "api_url": "http://localhost:3000/api",
            "api_key": None,
            "org_id": 1
        },
        "datadog": {
            "enabled": False,
            "api_key": None,
            "app_key": None,
            "site": "datadoghq.com"
        },
        "cloudwatch": {
            "enabled": False,
            "region": "us-east-1",
            "namespace": "TokenOptimization",
            "aws_access_key": None,
            "aws_secret_key": None
        },
        "slack": {
            "enabled": False,
            "webhook_url": None,
            "channel": "#alerts",
            "username": "Token Optimization Monitor"
        }
    }

    if config_path:
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                # Merge with defaults
                for integration_name, integration_config in file_config.items():
                    if integration_name in default_config:
                        default_config[integration_name].update(integration_config)
        except Exception as e:
            logger.error(f"Failed to load integration config from {config_path}: {e}")

    return default_config


# Global integration manager
_integration_manager: Optional[IntegrationManager] = None


def get_integration_manager(config_path: Optional[str] = None) -> IntegrationManager:
    """Get the global integration manager instance."""
    global _integration_manager
    if _integration_manager is None:
        config = load_integration_config(config_path)
        _integration_manager = IntegrationManager(config)
    return _integration_manager


async def initialize_integrations(config_path: Optional[str] = None) -> IntegrationManager:
    """Initialize all external integrations."""
    manager = get_integration_manager(config_path)

    # Validate all configurations
    validation_results = await manager.validate_all_configurations()

    enabled_integrations = [name for name, valid in validation_results.items() if valid]
    disabled_integrations = [name for name, valid in validation_results.items() if not valid]

    logger.info(f"Initialized external integrations")
    logger.info(f"Enabled integrations: {enabled_integrations}")
    if disabled_integrations:
        logger.warning(f"Disabled integrations (invalid config): {disabled_integrations}")

    return manager
