"""
Integration layer for Conservative Fallback Chain with existing systems

This module provides integration points between the conservative fallback chain
and the existing PromptCraft systems including the task detection, MCP integration,
and UI components. It ensures seamless operation while maintaining backwards
compatibility.

Key Features:
- Backwards compatible integration with existing task detection
- Automatic fallback chain activation based on configuration
- Performance monitoring and health checks integration
- Gradual rollout capabilities for production deployment
- Comprehensive logging and metrics collection

Integration Points:
- TaskDetectionSystem: Enhanced with fallback protection
- MCP Integration: Resilient function loading
- UI Components: Graceful degradation notifications
- Health Check Endpoints: System status monitoring

Architecture:
    The integration follows the decorator pattern to enhance existing
    functionality without breaking changes. It provides multiple integration
    modes for different deployment scenarios.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.core.conservative_fallback_chain import (
    FallbackDecision,
    FallbackLevel,
    create_conservative_fallback_chain,
)
from src.core.task_detection import DetectionResult, TaskDetectionSystem
from src.core.task_detection_config import DetectionMode, TaskDetectionConfig
from src.utils.logging_mixin import LoggerMixin

logger = logging.getLogger(__name__)


class IntegrationMode(Enum):
    """Integration modes for gradual rollout"""
    DISABLED = "disabled"              # No fallback protection
    MONITORING = "monitoring"          # Log only, no intervention
    SHADOW = "shadow"                  # Run in parallel, compare results
    ACTIVE = "active"                  # Full fallback protection
    AGGRESSIVE = "aggressive"          # Prefer fallback over detection


@dataclass
class IntegrationConfig:
    """Configuration for fallback integration"""
    mode: IntegrationMode = IntegrationMode.ACTIVE
    enable_performance_monitoring: bool = True
    enable_health_checks: bool = True
    enable_metrics_collection: bool = True
    enable_user_notifications: bool = True

    # Rollout configuration
    rollout_percentage: float = 100.0  # Percentage of requests to protect
    rollout_user_whitelist: list | None = None
    rollout_query_patterns: list | None = None

    # Performance thresholds
    max_detection_time_ms: float = 5000.0
    max_memory_usage_mb: int = 100
    enable_emergency_mode: bool = True

    # Notification configuration
    notify_on_fallback: bool = True
    notify_on_recovery: bool = True
    notify_on_emergency: bool = True


class FallbackIntegrationMetrics:
    """Metrics collector for integration performance"""

    def __init__(self) -> None:
        self.total_requests = 0
        self.fallback_activations = 0
        self.detection_successes = 0
        self.detection_failures = 0
        self.performance_improvements = 0
        self.user_notifications_sent = 0

        # Timing metrics
        self.total_detection_time = 0.0
        self.total_fallback_time = 0.0

        # Error tracking
        self.errors_prevented = 0
        self.cascade_failures_prevented = 0

    def record_request(self, used_fallback: bool, detection_time: float,
                      fallback_time: float = 0.0, error_prevented: bool = False) -> None:
        """Record a request outcome"""
        self.total_requests += 1
        self.total_detection_time += detection_time

        if used_fallback:
            self.fallback_activations += 1
            self.total_fallback_time += fallback_time

            if error_prevented:
                self.errors_prevented += 1
        else:
            self.detection_successes += 1

    def record_notification(self, notification_type: str) -> None:
        """Record user notification"""
        self.user_notifications_sent += 1

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get comprehensive metrics summary"""
        fallback_rate = self.fallback_activations / max(1, self.total_requests)
        avg_detection_time = self.total_detection_time / max(1, self.detection_successes)
        avg_fallback_time = self.total_fallback_time / max(1, self.fallback_activations)

        return {
            "total_requests": self.total_requests,
            "fallback_rate": fallback_rate,
            "detection_success_rate": self.detection_successes / max(1, self.total_requests),
            "avg_detection_time_ms": avg_detection_time,
            "avg_fallback_time_ms": avg_fallback_time,
            "errors_prevented": self.errors_prevented,
            "cascade_failures_prevented": self.cascade_failures_prevented,
            "user_notifications_sent": self.user_notifications_sent,
            "performance_gain": max(0, avg_detection_time - avg_fallback_time) if avg_detection_time > 0 else 0,
        }


class UserNotificationManager:
    """Manages user notifications for fallback events"""

    def __init__(self, config: IntegrationConfig) -> None:
        self.config = config
        self.notification_history = []
        self.last_notification_time = {}
        self.notification_cooldown = 300.0  # 5 minutes

    async def notify_fallback_activation(self, level: FallbackLevel,
                                       query: str, reason: str) -> bool:
        """Notify user of fallback activation"""
        if not self.config.notify_on_fallback:
            return False

        notification_key = f"fallback_{level.value}"

        # Check cooldown
        if self._is_in_cooldown(notification_key):
            return False

        message = self._create_fallback_message(level, reason)
        success = await self._send_notification(message, "info")

        if success:
            self.last_notification_time[notification_key] = time.time()
            self.notification_history.append({
                "type": "fallback_activation",
                "level": level.value,
                "query": query[:50],
                "reason": reason,
                "timestamp": time.time(),
            })

        return success

    async def notify_emergency_mode(self, reason: str) -> bool:
        """Notify user of emergency mode activation"""
        if not self.config.notify_on_emergency:
            return False


        # Emergency notifications ignore cooldown
        message = (
            "⚠️ EMERGENCY MODE ACTIVATED\n"
            f"Reason: {reason}\n"
            "All functions are now available for maximum reliability.\n"
            "Performance may be reduced but functionality is preserved."
        )

        success = await self._send_notification(message, "warning")

        if success:
            self.notification_history.append({
                "type": "emergency_mode",
                "reason": reason,
                "timestamp": time.time(),
            })

        return success

    async def notify_recovery(self, previous_level: FallbackLevel) -> bool:
        """Notify user of system recovery"""
        if not self.config.notify_on_recovery:
            return False

        notification_key = "recovery"

        if self._is_in_cooldown(notification_key):
            return False

        message = (
            "✅ SYSTEM RECOVERED\n"
            f"Previous fallback level: {previous_level.value}\n"
            "Normal operation has been restored.\n"
            "Optimal performance and functionality available."
        )

        success = await self._send_notification(message, "success")

        if success:
            self.last_notification_time[notification_key] = time.time()
            self.notification_history.append({
                "type": "recovery",
                "previous_level": previous_level.value,
                "timestamp": time.time(),
            })

        return success

    def _is_in_cooldown(self, notification_key: str) -> bool:
        """Check if notification is in cooldown period"""
        last_time = self.last_notification_time.get(notification_key, 0)
        return time.time() - last_time < self.notification_cooldown

    def _create_fallback_message(self, level: FallbackLevel, reason: str) -> str:
        """Create user-friendly fallback message"""
        level_messages = {
            FallbackLevel.HIGH_CONFIDENCE: "🔧 Using optimized function set based on your request",
            FallbackLevel.MEDIUM_CONFIDENCE: "⚙️ Loading additional tools for comprehensive coverage",
            FallbackLevel.LOW_CONFIDENCE: "🛡️ Using safe default tools to ensure full functionality",
            FallbackLevel.DETECTION_FAILURE: "🔄 Temporary issue detected, loading comprehensive toolset",
            FallbackLevel.SYSTEM_EMERGENCY: "🚨 Emergency mode: All tools loaded for maximum reliability",
        }

        base_message = level_messages.get(level, "🔧 Adjusting available tools")

        if level in [FallbackLevel.DETECTION_FAILURE, FallbackLevel.SYSTEM_EMERGENCY]:
            base_message += f"\nReason: {reason}"

        return base_message

    async def _send_notification(self, message: str, notification_type: str) -> bool:
        """Send notification to user (implementation depends on UI framework)"""
        try:
            # In a real implementation, this would integrate with the UI system
            # For now, just log the notification
            logger.info(f"USER_NOTIFICATION [{notification_type.upper()}]: {message}")
            return True
        except Exception as e:
            logger.error(f"Failed to send user notification: {e}")
            return False


class EnhancedTaskDetectionSystem(LoggerMixin):
    """Enhanced task detection system with integrated fallback protection"""

    def __init__(self,
                 original_system: TaskDetectionSystem,
                 integration_config: IntegrationConfig | None = None,
                 detection_config: TaskDetectionConfig | None = None) -> None:
        super().__init__(logger_name="enhanced_task_detection")

        self.original_system = original_system
        self.integration_config = integration_config or IntegrationConfig()
        self.detection_config = detection_config or TaskDetectionConfig()

        # Initialize fallback chain if active
        self.fallback_chain = None
        if self.integration_config.mode in [IntegrationMode.ACTIVE, IntegrationMode.AGGRESSIVE, IntegrationMode.SHADOW]:
            self.fallback_chain = create_conservative_fallback_chain(
                original_system,
                detection_config,
            )

        # Initialize components
        self.metrics = FallbackIntegrationMetrics()
        self.notification_manager = UserNotificationManager(integration_config)

        # State tracking
        self.current_fallback_level = None
        self.last_health_check = time.time()
        self.system_healthy = True

    async def detect_categories(self, query: str,
                              context: dict[str, Any] | None = None) -> DetectionResult:
        """Enhanced detect_categories with fallback protection"""

        # Check if request should use fallback protection
        if not self._should_use_fallback(query, context):
            return await self.original_system.detect_categories(query, context)

        detection_start = time.time()

        try:
            # Attempt original detection based on integration mode
            if self.integration_config.mode == IntegrationMode.AGGRESSIVE:
                # Prefer fallback over detection
                return await self._use_fallback_system(query, context)

            if self.integration_config.mode == IntegrationMode.SHADOW:
                # Run both systems in parallel for comparison
                return await self._run_shadow_mode(query, context)

            # Try original first, fallback on failure
            return await self._try_original_with_fallback(query, context)

        finally:
            # Record metrics
            detection_time = (time.time() - detection_start) * 1000
            self.metrics.record_request(
                used_fallback=self.current_fallback_level is not None,
                detection_time=detection_time,
            )

    async def _try_original_with_fallback(self, query: str,
                                        context: dict[str, Any] | None) -> DetectionResult:
        """Try original detection with fallback on failure"""

        try:
            # Set timeout based on configuration
            result = await asyncio.wait_for(
                self.original_system.detect_categories(query, context),
                timeout=self.integration_config.max_detection_time_ms / 1000,
            )

            # Check if result quality is acceptable
            if self._is_result_acceptable(result):
                # Successful detection - check if we need to notify recovery
                if self.current_fallback_level is not None:
                    await self.notification_manager.notify_recovery(self.current_fallback_level)
                    self.current_fallback_level = None

                return result
            # Poor quality result - use fallback
            return await self._use_fallback_system(query, context)

        except Exception as e:
            self.log_error_with_context(e, {"query": query[:50]}, "_try_original_with_fallback")
            return await self._use_fallback_system(query, context)

    async def _use_fallback_system(self, query: str,
                                 context: dict[str, Any] | None) -> DetectionResult:
        """Use fallback system for detection"""

        if not self.fallback_chain:
            # Fallback not available - return minimal safe result
            return self._create_safe_fallback_result()

        fallback_start = time.time()

        try:
            # Get fallback decision
            categories, decision = await self.fallback_chain.get_function_categories(query, context)

            # Update current fallback level
            previous_level = self.current_fallback_level
            self.current_fallback_level = decision.level

            # Notify user if fallback level changed significantly
            if self._should_notify_fallback_change(previous_level, decision.level):
                await self.notification_manager.notify_fallback_activation(
                    decision.level, query, decision.rationale,
                )

            # Create detection result from fallback decision
            return DetectionResult(
                categories=categories,
                confidence_scores=self._estimate_confidence_from_fallback(decision),
                detection_time_ms=(time.time() - fallback_start) * 1000,
                signals_used={"fallback": decision.rationale},
                fallback_applied=decision.level.value,
            )


        except Exception as e:
            self.log_error_with_context(e, {"query": query[:50]}, "_use_fallback_system")
            return self._create_safe_fallback_result()

    async def _run_shadow_mode(self, query: str,
                             context: dict[str, Any] | None) -> DetectionResult:
        """Run both systems in parallel for comparison"""

        try:
            # Run both systems concurrently
            original_task = asyncio.create_task(
                self.original_system.detect_categories(query, context),
            )
            fallback_task = asyncio.create_task(
                self._use_fallback_system(query, context),
            )

            # Wait for both to complete or timeout
            done, pending = await asyncio.wait(
                [original_task, fallback_task],
                timeout=self.integration_config.max_detection_time_ms / 1000,
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()

            # Get results
            original_result = None
            fallback_result = None

            for task in done:
                try:
                    result = await task
                    if task == original_task:
                        original_result = result
                    else:
                        fallback_result = result
                except Exception as e:
                    self.logger.warning(f"Shadow mode task failed: {e}")

            # Compare results and log differences
            if original_result and fallback_result:
                self._compare_shadow_results(original_result, fallback_result, query)

            # Return original result if available and acceptable
            if original_result and self._is_result_acceptable(original_result):
                return original_result
            if fallback_result:
                return fallback_result
            return self._create_safe_fallback_result()

        except Exception as e:
            self.log_error_with_context(e, {"query": query[:50]}, "_run_shadow_mode")
            return self._create_safe_fallback_result()

    def _should_use_fallback(self, query: str, context: dict[str, Any] | None) -> bool:
        """Determine if request should use fallback protection"""

        if self.integration_config.mode == IntegrationMode.DISABLED:
            return False

        # Check rollout percentage
        if self.integration_config.rollout_percentage < 100.0:
            query_hash = hash(query) % 100
            if query_hash >= self.integration_config.rollout_percentage:
                return False

        # Check user whitelist
        if self.integration_config.rollout_user_whitelist:
            user_id = context.get("user_id") if context else None
            if user_id and user_id not in self.integration_config.rollout_user_whitelist:
                return False

        # Check query patterns
        if self.integration_config.rollout_query_patterns:
            if not any(pattern in query.lower() for pattern in self.integration_config.rollout_query_patterns):
                return False

        return True

    def _is_result_acceptable(self, result: DetectionResult) -> bool:
        """Check if detection result quality is acceptable"""

        # Check detection time
        if result.detection_time_ms > self.integration_config.max_detection_time_ms:
            return False

        # Check confidence scores
        if result.confidence_scores:
            max_confidence = max(result.confidence_scores.values())
            if max_confidence < 0.3:  # Very low confidence
                return False

        # Check if any categories were detected
        return any(result.categories.values())

    def _should_notify_fallback_change(self, previous_level: FallbackLevel | None,
                                     current_level: FallbackLevel) -> bool:
        """Determine if fallback level change should trigger notification"""

        # First time entering fallback
        if previous_level is None:
            return current_level in [FallbackLevel.DETECTION_FAILURE, FallbackLevel.SYSTEM_EMERGENCY]

        # Escalating to more severe fallback
        severity_order = [
            FallbackLevel.HIGH_CONFIDENCE,
            FallbackLevel.MEDIUM_CONFIDENCE,
            FallbackLevel.LOW_CONFIDENCE,
            FallbackLevel.DETECTION_FAILURE,
            FallbackLevel.SYSTEM_EMERGENCY,
        ]

        try:
            prev_idx = severity_order.index(previous_level)
            curr_idx = severity_order.index(current_level)
            return curr_idx > prev_idx + 1  # Notify on significant escalation
        except ValueError:
            return True  # Notify on unknown level changes

    def _estimate_confidence_from_fallback(self, decision: FallbackDecision) -> dict[str, float]:
        """Estimate confidence scores from fallback decision"""

        base_confidence = {
            FallbackLevel.HIGH_CONFIDENCE: 0.9,
            FallbackLevel.MEDIUM_CONFIDENCE: 0.6,
            FallbackLevel.LOW_CONFIDENCE: 0.4,
            FallbackLevel.DETECTION_FAILURE: 0.2,
            FallbackLevel.SYSTEM_EMERGENCY: 0.1,
        }.get(decision.level, 0.5)

        # Create confidence scores for loaded categories
        confidence_scores = {}
        for category, loaded in decision.categories_to_load.items():
            if loaded:
                confidence_scores[category] = base_confidence

        return confidence_scores

    def _compare_shadow_results(self, original: DetectionResult,
                               fallback: DetectionResult, query: str) -> None:
        """Compare shadow mode results and log differences"""

        # Compare categories
        orig_categories = {k for k, v in original.categories.items() if v}
        fallback_categories = {k for k, v in fallback.categories.items() if v}

        category_diff = orig_categories.symmetric_difference(fallback_categories)

        # Compare performance
        time_diff = abs(original.detection_time_ms - fallback.detection_time_ms)

        # Log comparison
        self.logger.info(
            f"Shadow mode comparison - Query: {query[:50]} | "
            f"Category diff: {category_diff} | "
            f"Time diff: {time_diff:.2f}ms | "
            f"Original: {len(orig_categories)} cats, {original.detection_time_ms:.2f}ms | "
            f"Fallback: {len(fallback_categories)} cats, {fallback.detection_time_ms:.2f}ms",
        )

    def _create_safe_fallback_result(self) -> DetectionResult:
        """Create safe fallback result when all else fails"""

        safe_categories = {
            "core": True,
            "git": True,
            "analysis": True,
            "debug": False,
            "test": False,
            "quality": False,
            "security": False,
            "external": False,
            "infrastructure": False,
        }

        return DetectionResult(
            categories=safe_categories,
            confidence_scores={"core": 0.5, "git": 0.5, "analysis": 0.5},
            detection_time_ms=1.0,
            signals_used={"emergency": "safe_fallback"},
            fallback_applied="emergency_safe_fallback",
        )

    async def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive health status including fallback system"""

        # Get original system health if available
        original_health = {}
        try:
            if hasattr(self.original_system, "get_performance_metrics"):
                original_health = self.original_system.get_performance_metrics()
        except Exception as e:
            self.logger.warning(f"Failed to get original system health: {e}")

        # Get fallback system health
        fallback_health = {}
        if self.fallback_chain:
            try:
                fallback_health = self.fallback_chain.get_health_status()
            except Exception as e:
                self.logger.warning(f"Failed to get fallback system health: {e}")

        # Get integration metrics
        integration_metrics = self.metrics.get_metrics_summary()

        return {
            "integration_config": {
                "mode": self.integration_config.mode.value,
                "rollout_percentage": self.integration_config.rollout_percentage,
                "performance_monitoring": self.integration_config.enable_performance_monitoring,
            },
            "original_system": original_health,
            "fallback_system": fallback_health,
            "integration_metrics": integration_metrics,
            "current_state": {
                "fallback_level": self.current_fallback_level.value if self.current_fallback_level else None,
                "system_healthy": self.system_healthy,
                "last_health_check": self.last_health_check,
            },
            "notification_history": self.notification_manager.notification_history[-10:],  # Last 10 notifications
        }


# Factory functions for easy integration
def create_enhanced_task_detection(original_system: TaskDetectionSystem,
                                 integration_mode: IntegrationMode = IntegrationMode.ACTIVE,
                                 detection_config: TaskDetectionConfig | None = None) -> EnhancedTaskDetectionSystem:
    """Create enhanced task detection system with fallback protection"""

    integration_config = IntegrationConfig(mode=integration_mode)

    if detection_config is None:
        detection_config = TaskDetectionConfig()
        # Apply conservative mode for safety
        detection_config.apply_mode_preset(DetectionMode.CONSERVATIVE)

    return EnhancedTaskDetectionSystem(
        original_system=original_system,
        integration_config=integration_config,
        detection_config=detection_config,
    )


def create_monitoring_integration(original_system: TaskDetectionSystem) -> EnhancedTaskDetectionSystem:
    """Create monitoring-only integration for gradual rollout"""

    integration_config = IntegrationConfig(
        mode=IntegrationMode.MONITORING,
        enable_performance_monitoring=True,
        enable_health_checks=True,
        enable_user_notifications=False,
    )

    return EnhancedTaskDetectionSystem(
        original_system=original_system,
        integration_config=integration_config,
    )


def create_shadow_integration(original_system: TaskDetectionSystem,
                            rollout_percentage: float = 10.0) -> EnhancedTaskDetectionSystem:
    """Create shadow mode integration for testing"""

    integration_config = IntegrationConfig(
        mode=IntegrationMode.SHADOW,
        rollout_percentage=rollout_percentage,
        enable_performance_monitoring=True,
        enable_user_notifications=False,
    )

    return EnhancedTaskDetectionSystem(
        original_system=original_system,
        integration_config=integration_config,
    )


# Health check endpoint integration
async def get_fallback_system_health(enhanced_system: EnhancedTaskDetectionSystem) -> dict[str, Any]:
    """Get health status for monitoring systems"""
    return await enhanced_system.get_health_status()


# Backwards compatibility wrapper
class BackwardsCompatibleTaskDetection:
    """Backwards compatible wrapper for existing code"""

    def __init__(self, original_system: TaskDetectionSystem) -> None:
        self.enhanced_system = create_enhanced_task_detection(
            original_system,
            IntegrationMode.ACTIVE,
        )

    async def detect_categories(self, query: str,
                              context: dict[str, Any] | None = None) -> DetectionResult:
        """Backwards compatible detect_categories method"""
        return await self.enhanced_system.detect_categories(query, context)

    def __getattr__(self, name):
        """Delegate other attributes to original system"""
        return getattr(self.enhanced_system.original_system, name)
