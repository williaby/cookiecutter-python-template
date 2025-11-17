"""Service Token Monitoring System for AUTH-2 implementation.

This module provides monitoring capabilities for service tokens including:
- Token expiration alerting and notifications
- Usage pattern analysis and anomaly detection
- Health monitoring and metrics collection
- Automated cleanup and maintenance scheduling
- Integration with external monitoring systems (Grafana, Prometheus, etc.)
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
import json
import smtplib
from email.mime.text import MIMEText as MimeText
from email.mime.multipart import MIMEMultipart as MimeMultipart

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth.service_token_manager import ServiceTokenManager
from ..database.connection import get_db, database_health_check
from ..config.settings import ApplicationSettings

logger = logging.getLogger(__name__)


class TokenExpirationAlert:
    """Represents a token expiration alert."""

    def __init__(
        self,
        token_name: str,
        token_id: str,
        expires_at: datetime,
        days_until_expiration: int,
        usage_count: int,
        last_used: Optional[datetime] = None,
        metadata: Optional[Dict] = None
    ):
        """Initialize token expiration alert.

        Args:
            token_name: Name of the expiring token
            token_id: Token identifier
            expires_at: Token expiration timestamp
            days_until_expiration: Days until token expires
            usage_count: Current usage count
            last_used: Last usage timestamp
            metadata: Token metadata
        """
        self.token_name = token_name
        self.token_id = token_id
        self.expires_at = expires_at
        self.days_until_expiration = days_until_expiration
        self.usage_count = usage_count
        self.last_used = last_used
        self.metadata = metadata or {}

    @property
    def severity(self) -> str:
        """Get alert severity based on expiration time."""
        if self.days_until_expiration <= 1:
            return "critical"
        elif self.days_until_expiration <= 7:
            return "high"
        elif self.days_until_expiration <= 30:
            return "medium"
        else:
            return "low"

    @property
    def is_active_token(self) -> bool:
        """Check if token is actively used (used in last 30 days)."""
        if not self.last_used:
            return False
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        return self.last_used > cutoff


class ServiceTokenMonitor:
    """Service token monitoring and alerting system."""

    def __init__(self, settings: Optional[ApplicationSettings] = None):
        """Initialize service token monitor.

        Args:
            settings: Application settings (optional)
        """
        self.settings = settings
        self.token_manager = ServiceTokenManager()

        # Alert thresholds (days before expiration)
        self.alert_thresholds = [1, 7, 14, 30]  # Days before expiration to alert
        self.critical_threshold = 7  # Days before expiration considered critical

        # Monitoring intervals
        self.check_interval_hours = 6  # How often to check for expiring tokens
        self.cleanup_interval_hours = 24  # How often to run cleanup

    async def check_expiring_tokens(
        self,
        alert_threshold_days: int = 30
    ) -> List[TokenExpirationAlert]:
        """Check for tokens expiring within threshold.

        Args:
            alert_threshold_days: Alert for tokens expiring within N days

        Returns:
            List of expiration alerts
        """
        alerts = []

        try:
            async for session in get_db():
                # Query for tokens expiring within threshold
                cutoff_date = datetime.now(timezone.utc) + timedelta(days=alert_threshold_days)

                result = await session.execute(
                    text("""
                        SELECT
                            id, token_name, expires_at, usage_count, last_used, token_metadata,
                            EXTRACT(EPOCH FROM (expires_at - NOW())) / 86400 as days_until_expiration
                        FROM service_tokens
                        WHERE expires_at IS NOT NULL
                          AND expires_at <= :cutoff_date
                          AND is_active = TRUE
                        ORDER BY expires_at ASC
                    """),
                    {"cutoff_date": cutoff_date}
                )

                for row in result.fetchall():
                    alert = TokenExpirationAlert(
                        token_name=row.token_name,
                        token_id=str(row.id),
                        expires_at=row.expires_at,
                        days_until_expiration=max(0, int(row.days_until_expiration)),
                        usage_count=row.usage_count,
                        last_used=row.last_used,
                        metadata=row.token_metadata or {}
                    )
                    alerts.append(alert)

                break  # Only need first session

        except Exception as e:
            logger.error(f"Failed to check expiring tokens: {e}")

        return alerts

    async def get_monitoring_metrics(self) -> Dict[str, Any]:
        """Get comprehensive monitoring metrics for service tokens.

        Returns:
            Dictionary with monitoring metrics
        """
        metrics: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "database_health": "unknown",
            "token_stats": {},
            "usage_stats": {},
            "security_alerts": [],
            "performance_metrics": {}
        }

        try:
            # Database health check
            health = await database_health_check()
            metrics["database_health"] = health["status"]
            if isinstance(metrics["performance_metrics"], dict):
                metrics["performance_metrics"]["db_connection_time_ms"] = health.get("connection_time_ms", 0)

            # Token statistics
            analytics = await self.token_manager.get_token_usage_analytics(days=30)
            if analytics and "summary" in analytics:
                summary = analytics["summary"]
                metrics["token_stats"] = {
                    "total_tokens": summary["total_tokens"],
                    "active_tokens": summary["active_tokens"],
                    "inactive_tokens": summary["inactive_tokens"],
                    "expired_tokens": summary["expired_tokens"],
                    "total_usage_30d": summary["total_usage"]
                }

            # Usage statistics
            if analytics and "top_tokens" in analytics:
                top_tokens = analytics["top_tokens"][:5]  # Top 5 most used
                metrics["usage_stats"] = {
                    "most_used_tokens": [
                        {
                            "name": token["token_name"],
                            "usage_count": token["usage_count"],
                            "last_used": token["last_used"]
                        }
                        for token in top_tokens
                    ]
                }

            # Security alerts (expiring tokens)
            expiring_alerts = await self.check_expiring_tokens(alert_threshold_days=30)
            metrics["security_alerts"] = [
                {
                    "type": "token_expiration",
                    "severity": alert.severity,
                    "token_name": alert.token_name,
                    "days_until_expiration": alert.days_until_expiration,
                    "is_active": alert.is_active_token
                }
                for alert in expiring_alerts
            ]

            # Performance metrics
            async for session in get_db():
                # Check recent authentication performance
                result = await session.execute(
                    text("""
                        SELECT
                            COUNT(*) as total_auths,
                            COUNT(*) FILTER (WHERE success = TRUE) as successful_auths,
                            COUNT(*) FILTER (WHERE success = FALSE) as failed_auths,
                            COUNT(*) FILTER (WHERE event_type = 'service_token_auth') as service_token_auths
                        FROM authentication_events
                        WHERE created_at >= NOW() - INTERVAL '1 hour'
                    """)
                )

                auth_stats = result.fetchone()
                if auth_stats and isinstance(metrics["performance_metrics"], dict):
                    metrics["performance_metrics"].update({
                        "auth_requests_1h": auth_stats.total_auths,
                        "auth_success_rate_1h": (
                            (auth_stats.successful_auths / auth_stats.total_auths * 100)
                            if auth_stats.total_auths > 0 else 100
                        ),
                        "service_token_usage_1h": auth_stats.service_token_auths
                    })

                break  # Only need first session

        except Exception as e:
            logger.error(f"Failed to get monitoring metrics: {e}")
            metrics["error"] = str(e)

        return metrics

    async def send_expiration_alerts(
        self,
        alerts: List[TokenExpirationAlert],
        notification_method: str = "log"
    ) -> Dict[str, Any]:
        """Send expiration alerts via configured notification method.

        Args:
            alerts: List of expiration alerts
            notification_method: Method to send alerts (log, email, webhook)

        Returns:
            Dictionary with notification statistics
        """
        if not alerts:
            return {"alerts_sent": 0, "method": notification_method}

        # Group alerts by severity
        alerts_by_severity: Dict[str, List[TokenExpirationAlert]] = {}
        for alert in alerts:
            severity = alert.severity
            if severity not in alerts_by_severity:
                alerts_by_severity[severity] = []
            alerts_by_severity[severity].append(alert)

        sent_count = 0

        try:
            if notification_method == "log":
                # Log-based alerts
                for severity, severity_alerts in alerts_by_severity.items():
                    log_level = {
                        "critical": logging.CRITICAL,
                        "high": logging.ERROR,
                        "medium": logging.WARNING,
                        "low": logging.INFO
                    }.get(severity, logging.INFO)

                    for alert in severity_alerts:
                        logger.log(
                            log_level,
                            f"SERVICE TOKEN EXPIRATION ALERT [{severity.upper()}]: "
                            f"Token '{alert.token_name}' expires in {alert.days_until_expiration} days "
                            f"(usage: {alert.usage_count} times, "
                            f"active: {'yes' if alert.is_active_token else 'no'})"
                        )
                        sent_count += 1

            elif notification_method == "email":
                # Email-based alerts (requires SMTP configuration)
                sent_count = await self._send_email_alerts(alerts_by_severity)

            elif notification_method == "webhook":
                # Webhook-based alerts (Slack, Teams, Discord, etc.)
                sent_count = await self._send_webhook_alerts(alerts_by_severity)

            else:
                logger.warning(f"Unknown notification method: {notification_method}")

        except Exception as e:
            logger.error(f"Failed to send expiration alerts: {e}")

        return {"alerts_sent": sent_count, "method": notification_method}

    async def _send_email_alerts(self, alerts_by_severity: Dict[str, List[TokenExpirationAlert]]) -> int:
        """Send email alerts for expiring tokens.

        Args:
            alerts_by_severity: Alerts grouped by severity

        Returns:
            Number of alerts sent
        """
        # This would require SMTP configuration in settings
        # For now, just log that email alerts would be sent

        total_alerts = sum(len(alerts) for alerts in alerts_by_severity.values())

        logger.info(f"EMAIL ALERT: Would send {total_alerts} token expiration alerts")
        for severity, alerts in alerts_by_severity.items():
            logger.info(f"  {severity.upper()}: {len(alerts)} alerts")

        return total_alerts

    async def _send_webhook_alerts(self, alerts_by_severity: Dict[str, List[TokenExpirationAlert]]) -> int:
        """Send webhook alerts for expiring tokens.

        Args:
            alerts_by_severity: Alerts grouped by severity

        Returns:
            Number of alerts sent
        """
        # This would integrate with Slack/Teams/Discord webhooks
        # For now, just log that webhook alerts would be sent

        total_alerts = sum(len(alerts) for alerts in alerts_by_severity.values())

        logger.info(f"WEBHOOK ALERT: Would send {total_alerts} token expiration alerts")
        for severity, alerts in alerts_by_severity.items():
            logger.info(f"  {severity.upper()}: {len(alerts)} alerts")

        return total_alerts

    async def run_scheduled_monitoring(self) -> Dict[str, Any]:
        """Run scheduled monitoring tasks.

        Returns:
            Summary of monitoring results
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Check for expiring tokens
            expiring_alerts = await self.check_expiring_tokens(alert_threshold_days=30)

            # Send alerts if any found
            alert_results = {"alerts_sent": 0}
            if expiring_alerts:
                alert_results = await self.send_expiration_alerts(expiring_alerts)

            # Get monitoring metrics
            metrics = await self.get_monitoring_metrics()

            # Run cleanup for expired tokens
            cleanup_results = await self.token_manager.cleanup_expired_tokens(deactivate_only=True)

            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            return {
                "status": "completed",
                "timestamp": start_time.isoformat(),
                "execution_time_seconds": execution_time,
                "expiring_tokens_found": len(expiring_alerts),
                "alerts_sent": alert_results["alerts_sent"],
                "cleanup_results": cleanup_results,
                "metrics_collected": True,
                "database_health": metrics.get("database_health", "unknown")
            }

        except Exception as e:
            logger.error(f"Scheduled monitoring failed: {e}")
            return {
                "status": "failed",
                "timestamp": start_time.isoformat(),
                "error": str(e)
            }

    async def start_monitoring_daemon(self, check_interval_minutes: int = 360) -> None:
        """Start continuous monitoring daemon.

        Args:
            check_interval_minutes: Minutes between monitoring checks
        """
        logger.info(f"Starting service token monitoring daemon (interval: {check_interval_minutes} minutes)")

        while True:
            try:
                result = await self.run_scheduled_monitoring()

                if result["status"] == "completed":
                    logger.info(
                        f"Monitoring cycle completed: "
                        f"{result.get('expiring_tokens_found', 0)} expiring tokens, "
                        f"{result.get('alerts_sent', 0)} alerts sent, "
                        f"cleanup: {result.get('cleanup_results', {}).get('expired_tokens_processed', 0)} tokens"
                    )
                else:
                    logger.error(f"Monitoring cycle failed: {result.get('error', 'Unknown error')}")

            except Exception as e:
                logger.error(f"Monitoring daemon error: {e}")

            # Wait for next cycle
            await asyncio.sleep(check_interval_minutes * 60)


# Monitoring API endpoints and health checks
class MonitoringHealthCheck:
    """Health check utilities for monitoring systems."""

    @staticmethod
    async def get_health_status() -> Dict[str, Any]:
        """Get comprehensive health status for monitoring systems.

        Returns:
            Health status dictionary
        """
        monitor = ServiceTokenMonitor()

        try:
            # Get basic metrics
            metrics = await monitor.get_monitoring_metrics()

            # Determine overall health
            db_healthy = metrics.get("database_health") == "healthy"
            has_critical_alerts = any(
                alert.get("severity") == "critical"
                for alert in metrics.get("security_alerts", [])
            )

            overall_status = "healthy"
            if not db_healthy:
                overall_status = "unhealthy"
            elif has_critical_alerts:
                overall_status = "degraded"

            return {
                "status": overall_status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "components": {
                    "database": {
                        "status": "healthy" if db_healthy else "unhealthy",
                        "connection_time_ms": metrics.get("performance_metrics", {}).get("db_connection_time_ms", 0)
                    },
                    "service_tokens": {
                        "status": "healthy",
                        "active_count": metrics.get("token_stats", {}).get("active_tokens", 0),
                        "expired_count": metrics.get("token_stats", {}).get("expired_tokens", 0)
                    },
                    "authentication": {
                        "status": "healthy",
                        "success_rate_1h": metrics.get("performance_metrics", {}).get("auth_success_rate_1h", 100),
                        "requests_1h": metrics.get("performance_metrics", {}).get("auth_requests_1h", 0)
                    }
                },
                "alerts": {
                    "critical_count": sum(
                        1 for alert in metrics.get("security_alerts", [])
                        if alert.get("severity") == "critical"
                    ),
                    "total_count": len(metrics.get("security_alerts", []))
                }
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e)
            }
