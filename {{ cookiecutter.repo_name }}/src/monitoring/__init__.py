"""
Token Optimization Performance Monitoring System

This module provides comprehensive monitoring, metrics collection, and validation
for the dynamic function loading system's 70% token reduction goal.

Key Components:
- TokenOptimizationMonitor: Core monitoring system
- MetricsCollector: Comprehensive metrics collection and analysis
- PerformanceDashboard: Real-time web dashboard
- IntegrationManager: External system integrations
- AlertManager: Alert management and notifications

Usage:
    from src.monitoring import initialize_monitoring_system
    
    monitor, collector, dashboard = await initialize_monitoring_system()
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.core.token_optimization_monitor import (
    TokenOptimizationMonitor,
    get_token_optimization_monitor,
    initialize_monitoring
)
from src.monitoring.metrics_collector import (
    MetricsCollector,
    get_metrics_collector,
    initialize_metrics_collection
)
from src.monitoring.performance_dashboard import (
    RealTimeDashboard,
    AlertManager,
    get_dashboard_app
)
from src.monitoring.integration_utils import (
    IntegrationManager,
    get_integration_manager,
    initialize_integrations
)
from src.utils.observability import create_structured_logger

logger = logging.getLogger(__name__)


class MonitoringSystemConfig:
    """Configuration for the monitoring system."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        config = config_dict or {}
        
        # Core monitoring settings
        self.enabled = config.get("enabled", True)
        self.validation_target = config.get("validation_target", 0.70)  # 70% token reduction
        self.min_acceptable_reduction = config.get("min_acceptable_reduction", 0.50)  # 50% minimum
        self.max_loading_latency_ms = config.get("max_loading_latency_ms", 200.0)
        
        # Database settings
        self.database_path = config.get("database_path", "monitoring.db")
        
        # Collection intervals
        self.metrics_collection_interval = config.get("metrics_collection_interval", 30.0)  # seconds
        self.aggregation_interval = config.get("aggregation_interval", 300.0)  # 5 minutes
        self.dashboard_update_interval = config.get("dashboard_update_interval", 5.0)  # seconds
        
        # Alert thresholds
        self.alert_thresholds = config.get("alert_thresholds", {
            "token_reduction_min": 50.0,
            "token_reduction_target": 70.0,
            "latency_max_ms": 200.0,
            "success_rate_min": 0.95,
            "task_accuracy_min": 0.80,
            "fallback_rate_max": 0.10
        })
        
        # Dashboard settings
        self.dashboard_enabled = config.get("dashboard_enabled", True)
        self.dashboard_host = config.get("dashboard_host", "0.0.0.0")
        self.dashboard_port = config.get("dashboard_port", 8080)
        
        # External integrations
        self.integrations_enabled = config.get("integrations_enabled", True)
        self.integrations_config_path = config.get("integrations_config_path")
        
        # Privacy and data retention
        self.data_retention_days = config.get("data_retention_days", 30)
        self.anonymize_user_data = config.get("anonymize_user_data", True)
        self.collect_raw_sessions = config.get("collect_raw_sessions", False)
        
        # Performance settings
        self.max_concurrent_sessions = config.get("max_concurrent_sessions", 1000)
        self.metrics_buffer_size = config.get("metrics_buffer_size", 1000)


class MonitoringSystem:
    """Comprehensive monitoring system orchestrator."""
    
    def __init__(self, config: MonitoringSystemConfig):
        self.config = config
        self.logger = create_structured_logger("monitoring_system")
        
        # Core components
        self.monitor: Optional[TokenOptimizationMonitor] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        self.dashboard: Optional[RealTimeDashboard] = None
        self.alert_manager: Optional[AlertManager] = None
        self.integration_manager: Optional[IntegrationManager] = None
        
        # Status tracking
        self.initialized = False
        self.running = False
        
        # Background tasks
        self._monitoring_tasks = []
    
    async def initialize(self) -> bool:
        """Initialize all monitoring components."""
        
        if self.initialized:
            return True
        
        try:
            self.logger.info("Initializing token optimization monitoring system")
            
            # Initialize core monitor
            self.monitor = await initialize_monitoring()
            
            # Initialize metrics collector
            self.metrics_collector = await initialize_metrics_collection()
            
            # Initialize dashboard if enabled
            if self.config.dashboard_enabled:
                self.dashboard = RealTimeDashboard(self.monitor)
                self.alert_manager = AlertManager(self.monitor)
            
            # Initialize external integrations if enabled
            if self.config.integrations_enabled:
                self.integration_manager = await initialize_integrations(
                    self.config.integrations_config_path
                )
                
                # Add integration manager as notification channel for alerts
                if self.alert_manager:
                    self.alert_manager.add_notification_channel(self.integration_manager)
            
            self.initialized = True
            self.logger.info("Monitoring system initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize monitoring system: {e}")
            return False
    
    async def start(self) -> bool:
        """Start the monitoring system."""
        
        if not self.initialized:
            success = await self.initialize()
            if not success:
                return False
        
        if self.running:
            return True
        
        try:
            self.logger.info("Starting token optimization monitoring system")
            
            # Start metrics collection
            if self.metrics_collector:
                await self.metrics_collector.start_collection()
            
            # Start dashboard updates
            if self.dashboard:
                await self.dashboard.start_real_time_updates()
            
            # Start periodic tasks
            self._monitoring_tasks = [
                asyncio.create_task(self._validation_loop()),
                asyncio.create_task(self._alert_check_loop()),
                asyncio.create_task(self._health_check_loop())
            ]
            
            self.running = True
            self.logger.info("Monitoring system started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring system: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop the monitoring system."""
        
        if not self.running:
            return True
        
        try:
            self.logger.info("Stopping token optimization monitoring system")
            
            # Cancel background tasks
            for task in self._monitoring_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
            self._monitoring_tasks.clear()
            
            # Stop metrics collection
            if self.metrics_collector:
                await self.metrics_collector.stop_collection()
            
            # Stop dashboard updates
            if self.dashboard:
                await self.dashboard.stop_real_time_updates()
            
            self.running = False
            self.logger.info("Monitoring system stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping monitoring system: {e}")
            return False
    
    async def _validation_loop(self):
        """Periodic validation of optimization claims."""
        
        while self.running:
            try:
                if self.metrics_collector:
                    # Generate comprehensive validation report
                    report = await self.metrics_collector.generate_comprehensive_report()
                    
                    validation_results = report.get("validation_results", {})
                    overall_validated = validation_results.get("overall_validation_status", False)
                    
                    if overall_validated:
                        self.logger.info("Token optimization validation: PASSED")
                    else:
                        self.logger.warning("Token optimization validation: FAILED")
                        
                        # Send validation failure alert
                        if self.alert_manager:
                            alert = {
                                "level": "warning",
                                "title": "Optimization Validation Failed",
                                "message": "Token optimization claims could not be validated with current data",
                                "timestamp": report["report_timestamp"]
                            }
                            await self.alert_manager._send_notification(alert)
                
                # Run validation every hour
                await asyncio.sleep(3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in validation loop: {e}")
                await asyncio.sleep(3600)
    
    async def _alert_check_loop(self):
        """Periodic alert checking."""
        
        while self.running:
            try:
                if self.alert_manager:
                    alerts = await self.alert_manager.check_alerts()
                    
                    # Send metrics to external integrations
                    if self.integration_manager and self.monitor:
                        metrics = await self.monitor.export_metrics()
                        await self.integration_manager.send_metrics_to_all(metrics)
                
                # Check alerts every 2 minutes
                await asyncio.sleep(120)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in alert check loop: {e}")
                await asyncio.sleep(120)
    
    async def _health_check_loop(self):
        """Periodic health checks of monitoring components."""
        
        while self.running:
            try:
                health_status = {
                    "timestamp": asyncio.get_event_loop().time(),
                    "components": {
                        "monitor": self.monitor is not None,
                        "metrics_collector": self.metrics_collector is not None,
                        "dashboard": self.dashboard is not None,
                        "alert_manager": self.alert_manager is not None,
                        "integration_manager": self.integration_manager is not None
                    }
                }
                
                # Check external integrations health
                if self.integration_manager:
                    integration_health = await self.integration_manager.health_check()
                    health_status["integrations"] = integration_health
                
                # Log health status
                unhealthy_components = [
                    name for name, healthy in health_status["components"].items() 
                    if not healthy
                ]
                
                if unhealthy_components:
                    self.logger.warning(f"Unhealthy monitoring components: {unhealthy_components}")
                else:
                    self.logger.debug("All monitoring components healthy")
                
                # Run health check every 15 minutes
                await asyncio.sleep(900)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(900)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        status = {
            "initialized": self.initialized,
            "running": self.running,
            "config": {
                "validation_target": self.config.validation_target,
                "min_acceptable_reduction": self.config.min_acceptable_reduction,
                "max_loading_latency_ms": self.config.max_loading_latency_ms,
                "dashboard_enabled": self.config.dashboard_enabled,
                "integrations_enabled": self.config.integrations_enabled
            }
        }
        
        if self.monitor:
            health_report = await self.monitor.generate_system_health_report()
            status["health_report"] = {
                "total_sessions": health_report.total_sessions,
                "average_token_reduction": health_report.average_token_reduction_percentage,
                "optimization_validated": self.monitor.optimization_validated,
                "validation_confidence": self.monitor.validation_confidence
            }
        
        if self.integration_manager:
            integration_health = await self.integration_manager.health_check()
            status["integrations"] = integration_health
        
        return status
    
    async def generate_report(self, include_recommendations: bool = True) -> Dict[str, Any]:
        """Generate comprehensive monitoring report."""
        
        if not self.metrics_collector:
            return {"error": "Metrics collector not initialized"}
        
        report = await self.metrics_collector.generate_comprehensive_report()
        
        if include_recommendations:
            # Add actionable recommendations
            recommendations = []
            
            validation_results = report.get("validation_results", {})
            token_validation = validation_results.get("token_reduction_claim", {})
            
            if not token_validation.get("validated", False):
                recommendations.append({
                    "priority": "high",
                    "category": "optimization",
                    "title": "Improve Token Reduction Performance",
                    "description": "Current token reduction is below the 70% target",
                    "actions": [
                        "Review task detection accuracy",
                        "Optimize function loading tiers",
                        "Analyze user behavior patterns",
                        "Consider more aggressive optimization settings"
                    ]
                })
            
            # Performance recommendations
            trend_analysis = report.get("trend_analysis", {})
            latency_trend = trend_analysis.get("loading_latency_ms", {})
            
            if latency_trend.get("trend_direction") == "increasing":
                recommendations.append({
                    "priority": "medium",
                    "category": "performance",
                    "title": "Loading Latency Increasing",
                    "description": "Function loading latency shows an increasing trend",
                    "actions": [
                        "Optimize caching strategies",
                        "Review function tier assignments",
                        "Check system resource usage",
                        "Consider cache size adjustments"
                    ]
                })
            
            report["recommendations"] = recommendations
        
        return report


# Global monitoring system instance
_monitoring_system: Optional[MonitoringSystem] = None


async def initialize_monitoring_system(config: Optional[Dict[str, Any]] = None) -> Tuple[TokenOptimizationMonitor, MetricsCollector, RealTimeDashboard]:
    """Initialize the complete monitoring system."""
    
    global _monitoring_system
    
    if _monitoring_system is None:
        monitoring_config = MonitoringSystemConfig(config)
        _monitoring_system = MonitoringSystem(monitoring_config)
    
    success = await _monitoring_system.start()
    
    if not success:
        raise RuntimeError("Failed to initialize monitoring system")
    
    return (
        _monitoring_system.monitor,
        _monitoring_system.metrics_collector,
        _monitoring_system.dashboard
    )


def get_monitoring_system() -> Optional[MonitoringSystem]:
    """Get the global monitoring system instance."""
    return _monitoring_system


async def shutdown_monitoring_system():
    """Shutdown the monitoring system."""
    
    global _monitoring_system
    
    if _monitoring_system:
        await _monitoring_system.stop()
        _monitoring_system = None


# Export public API
__all__ = [
    "MonitoringSystemConfig",
    "MonitoringSystem", 
    "initialize_monitoring_system",
    "get_monitoring_system",
    "shutdown_monitoring_system",
    "TokenOptimizationMonitor",
    "MetricsCollector",
    "RealTimeDashboard",
    "AlertManager",
    "IntegrationManager"
]
