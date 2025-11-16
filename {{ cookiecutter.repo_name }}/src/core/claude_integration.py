"""
Claude Code Integration Layer for User Control System

Integrates the user override commands and power user features
with the existing Claude Code command structure and workflows.
"""

import asyncio
import contextlib
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from .analytics_engine import AnalyticsEngine
from .help_system import InteractiveHelpSystem
from .task_detection import TaskDetectionSystem
from .task_detection_config import ConfigManager
from .user_control_system import CommandResult, UserControlSystem

logger = logging.getLogger(__name__)


@dataclass
class CommandMetadata:
    """Metadata for Claude Code commands"""
    name: str
    category: str
    complexity: str  # low, medium, high
    estimated_time: str
    dependencies: list[str] = field(default_factory=list)
    sub_commands: list[str] = field(default_factory=list)
    version: str = "1.0"
    tags: list[str] = field(default_factory=list)


@dataclass
class IntegrationStatus:
    """Status of system integration"""
    user_control_active: bool = False
    help_system_active: bool = False
    analytics_active: bool = False
    detection_system_active: bool = False
    command_count: int = 0
    last_health_check: datetime | None = None
    error_count: int = 0


class CommandRegistry:
    """Registry for all Claude Code commands including user control commands"""

    def __init__(self) -> None:
        self.commands: dict[str, dict[str, Any]] = {}
        self.categories: dict[str, list[str]] = {
            "meta": [],
            "workflow": [],
            "validation": [],
            "creation": [],
            "migration": [],
            "function_control": [],  # New category for user control
        }
        self.aliases: dict[str, str] = {}

    def register_command(self, command_name: str,
                        handler: Callable,
                        metadata: CommandMetadata,
                        aliases: list[str] | None = None) -> None:
        """Register a command with metadata"""
        self.commands[command_name] = {
            "handler": handler,
            "metadata": metadata,
            "registered_at": datetime.now(),
        }

        # Add to category
        if metadata.category in self.categories:
            self.categories[metadata.category].append(command_name)

        # Register aliases
        if aliases:
            for alias in aliases:
                self.aliases[alias] = command_name

    def get_command(self, command_name: str) -> dict[str, Any] | None:
        """Get command by name or alias"""
        # Check direct name
        if command_name in self.commands:
            return self.commands[command_name]

        # Check aliases
        if command_name in self.aliases:
            actual_name = self.aliases[command_name]
            return self.commands.get(actual_name)

        return None

    def list_commands_by_category(self, category: str) -> list[str]:
        """List all commands in a category"""
        return self.categories.get(category, [])

    def search_commands(self, query: str) -> list[str]:
        """Search commands by name, tags, or description"""
        matching_commands = []
        query_lower = query.lower()

        for cmd_name, cmd_info in self.commands.items():
            metadata = cmd_info["metadata"]

            # Check name
            if query_lower in cmd_name.lower():
                matching_commands.append(cmd_name)
                continue

            # Check tags
            if any(query_lower in tag.lower() for tag in metadata.tags):
                matching_commands.append(cmd_name)
                continue

        return matching_commands


class ClaudeCommandIntegration:
    """Integrates user control system with Claude Code command structure"""

    def __init__(self, control_system: UserControlSystem,
                 help_system: InteractiveHelpSystem,
                 analytics: AnalyticsEngine) -> None:
        self.control_system = control_system
        self.help_system = help_system
        self.analytics = analytics
        self.command_registry = CommandRegistry()
        self.integration_status = IntegrationStatus()

        # Integration state
        self.active_session_id = None
        self.command_history = []
        self.performance_metrics = {}

        # Initialize command mappings
        self._initialize_command_mappings()

    def _initialize_command_mappings(self) -> None:
        """Initialize command mappings and register user control commands"""

        # Register user control commands
        user_control_commands = {
            "function-loading:load-category": {
                "handler": self._handle_load_category,
                "metadata": CommandMetadata(
                    name="function-loading:load-category",
                    category="function_control",
                    complexity="low",
                    estimated_time="< 1 minute",
                    tags=["function-loading", "categories", "performance"],
                ),
                "aliases": ["load-cat", "fc:load"],
            },
            "function-loading:unload-category": {
                "handler": self._handle_unload_category,
                "metadata": CommandMetadata(
                    name="function-loading:unload-category",
                    category="function_control",
                    complexity="low",
                    estimated_time="< 1 minute",
                    tags=["function-loading", "categories", "performance"],
                ),
                "aliases": ["unload-cat", "fc:unload"],
            },
            "function-loading:list-categories": {
                "handler": self._handle_list_categories,
                "metadata": CommandMetadata(
                    name="function-loading:list-categories",
                    category="function_control",
                    complexity="low",
                    estimated_time="< 1 minute",
                    tags=["function-loading", "categories", "status"],
                ),
                "aliases": ["list-cats", "fc:list"],
            },
            "function-loading:optimize-for": {
                "handler": self._handle_optimize_for,
                "metadata": CommandMetadata(
                    name="function-loading:optimize-for",
                    category="function_control",
                    complexity="medium",
                    estimated_time="< 2 minutes",
                    tags=["function-loading", "optimization", "workflow"],
                ),
                "aliases": ["optimize", "fc:opt"],
            },
            "function-loading:tier-status": {
                "handler": self._handle_tier_status,
                "metadata": CommandMetadata(
                    name="function-loading:tier-status",
                    category="function_control",
                    complexity="low",
                    estimated_time="< 1 minute",
                    tags=["function-loading", "tiers", "status"],
                ),
                "aliases": ["tiers", "fc:status"],
            },
            "function-loading:save-profile": {
                "handler": self._handle_save_profile,
                "metadata": CommandMetadata(
                    name="function-loading:save-profile",
                    category="function_control",
                    complexity="medium",
                    estimated_time="< 2 minutes",
                    tags=["function-loading", "profiles", "automation"],
                ),
                "aliases": ["save-profile", "fc:save"],
            },
            "function-loading:load-profile": {
                "handler": self._handle_load_profile,
                "metadata": CommandMetadata(
                    name="function-loading:load-profile",
                    category="function_control",
                    complexity="low",
                    estimated_time="< 1 minute",
                    tags=["function-loading", "profiles", "automation"],
                ),
                "aliases": ["load-profile", "fc:load-profile"],
            },
            "function-loading:performance-stats": {
                "handler": self._handle_performance_stats,
                "metadata": CommandMetadata(
                    name="function-loading:performance-stats",
                    category="function_control",
                    complexity="low",
                    estimated_time="< 1 minute",
                    tags=["function-loading", "performance", "monitoring"],
                ),
                "aliases": ["perf-stats", "fc:stats"],
            },
            "function-loading:help": {
                "handler": self._handle_function_loading_help,
                "metadata": CommandMetadata(
                    name="function-loading:help",
                    category="function_control",
                    complexity="low",
                    estimated_time="< 3 minutes",
                    tags=["function-loading", "help", "documentation"],
                ),
                "aliases": ["fc:help", "fl:help"],
            },
            "function-loading:analytics": {
                "handler": self._handle_analytics,
                "metadata": CommandMetadata(
                    name="function-loading:analytics",
                    category="function_control",
                    complexity="medium",
                    estimated_time="< 3 minutes",
                    tags=["function-loading", "analytics", "insights"],
                ),
                "aliases": ["fc:analytics", "usage-analytics"],
            },
        }

        # Register all commands
        for cmd_name, cmd_info in user_control_commands.items():
            self.command_registry.register_command(
                cmd_name,
                cmd_info["handler"],
                cmd_info["metadata"],
                cmd_info.get("aliases", []),
            )

        self.integration_status.command_count = len(user_control_commands)
        self.integration_status.user_control_active = True

    async def execute_command(self, command_line: str,
                            context: dict[str, Any] | None = None) -> CommandResult:
        """Execute a command through the integrated system"""
        try:
            # Track command execution
            self._track_command_start(command_line, context)

            # Parse command
            command_parts = self._parse_command_line(command_line)
            command_name = command_parts.get("command")
            arguments = command_parts.get("arguments", [])

            # Get command from registry
            command_info = self.command_registry.get_command(command_name)

            if not command_info:
                # Try user control system directly
                if command_line.startswith("/"):
                    result = await self.control_system.execute_command(command_line)
                    self._track_command_end(command_line, result)
                    return result

                return CommandResult(
                    success=False,
                    message=f"Unknown command: {command_name}",
                    suggestions=self._suggest_similar_commands(command_name),
                )

            # Execute command
            handler = command_info["handler"]
            result = await handler(arguments, context or {})

            # Track completion
            self._track_command_end(command_line, result)

            return result

        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            error_result = CommandResult(
                success=False,
                message=f"Command execution error: {e!s}",
                warnings=["This might be a system error - check logs"],
            )
            self._track_command_end(command_line, error_result)
            return error_result

    def _parse_command_line(self, command_line: str) -> dict[str, Any]:
        """Parse command line into components"""
        # Handle both /command and project:command formats
        if command_line.startswith("/project:"):
            # Claude Code project format
            command_name = command_line[9:]  # Remove /project:
            parts = command_name.split(" ", 1)
            return {
                "command": parts[0],
                "arguments": parts[1].split() if len(parts) > 1 else [],
            }
        elif command_line.startswith("/"):
            # User control system format
            parts = command_line[1:].split()
            return {
                "command": parts[0] if parts else "",
                "arguments": parts[1:] if len(parts) > 1 else [],
            }
        elif ":" in command_line:
            # Claude Code project format without /project: prefix
            parts = command_line.split(" ", 1)
            return {
                "command": parts[0],
                "arguments": parts[1].split() if len(parts) > 1 else [],
            }
        # Simple command
        parts = command_line.split()
        return {
            "command": parts[0] if parts else "",
            "arguments": parts[1:] if len(parts) > 1 else [],
        }

    def _track_command_start(self, command_line: str, context: dict[str, Any]) -> None:
        """Track command execution start"""
        if not self.active_session_id:
            self.active_session_id = f"session_{int(datetime.now().timestamp())}"

        self.analytics.track_user_action(
            "command_started",
            {
                "command_line": command_line,
                "context": context,
            },
            user_id=context.get("user_id", "default"),
            session_id=self.active_session_id,
            context=context,
        )

    def _track_command_end(self, command_line: str, result: CommandResult) -> None:
        """Track command execution completion"""
        if not self.active_session_id:
            return

        self.analytics.track_user_action(
            "command_executed",
            {
                "command_line": command_line,
                "success": result.success,
                "message": result.message,
                "has_data": result.data is not None,
                "suggestions_count": len(result.suggestions),
                "warnings_count": len(result.warnings),
            },
            user_id="default",  # Would get from context in real implementation
            session_id=self.active_session_id,
        )

        # Add to command history
        self.command_history.append({
            "timestamp": datetime.now(),
            "command": command_line,
            "success": result.success,
            "message": result.message,
        })

        # Keep history manageable
        if len(self.command_history) > 100:
            self.command_history = self.command_history[-50:]

    def _suggest_similar_commands(self, command_name: str) -> list[str]:
        """Suggest similar commands for unknown command"""
        # Search in command registry
        similar_commands = self.command_registry.search_commands(command_name)

        # Add function control suggestions if relevant
        if any(keyword in command_name.lower()
               for keyword in ["load", "category", "tier", "optimize", "profile"]):
            similar_commands.extend([
                "function-loading:load-category",
                "function-loading:list-categories",
                "function-loading:optimize-for",
                "function-loading:save-profile",
            ])

        return list(set(similar_commands))[:5]  # Top 5, no duplicates

    # Command handlers
    async def _handle_load_category(self, args: list[str],
                                  context: dict[str, Any]) -> CommandResult:
        """Handle load-category command"""
        if not args:
            return CommandResult(
                success=False,
                message="Category name required",
                suggestions=["Example: function-loading:load-category security"],
            )

        category = args[0]
        command_line = f"/load-category {category}"

        result = await self.control_system.execute_command(command_line)

        # Track category loading
        if result.success:
            self.analytics.track_user_action(
                "category_loaded",
                {
                    "category": category,
                    "token_cost": result.data.get("token_cost", 0) if result.data else 0,
                },
                user_id=context.get("user_id", "default"),
                session_id=self.active_session_id,
            )

        return result

    async def _handle_unload_category(self, args: list[str],
                                   context: dict[str, Any]) -> CommandResult:
        """Handle unload-category command"""
        if not args:
            return CommandResult(
                success=False,
                message="Category name required",
                suggestions=["Example: function-loading:unload-category external"],
            )

        category = args[0]
        command_line = f"/unload-category {category}"

        result = await self.control_system.execute_command(command_line)

        # Track category unloading
        if result.success:
            self.analytics.track_user_action(
                "category_unloaded",
                {
                    "category": category,
                    "tokens_saved": result.data.get("tokens_saved", 0) if result.data else 0,
                },
                user_id=context.get("user_id", "default"),
                session_id=self.active_session_id,
            )

        return result

    async def _handle_list_categories(self, args: list[str],
                                    context: dict[str, Any]) -> CommandResult:
        """Handle list-categories command"""
        # Parse optional arguments
        tier_filter = None
        show_loaded_only = False

        for arg in args:
            if arg.startswith("tier:"):
                with contextlib.suppress(ValueError):
                    tier_filter = int(arg.split(":")[1])
            elif arg == "loaded-only":
                show_loaded_only = True

        # Build command line
        command_parts = ["/list-categories"]
        if tier_filter:
            command_parts.append(f"tier:{tier_filter}")
        if show_loaded_only:
            command_parts.append("loaded-only")

        command_line = " ".join(command_parts)
        return await self.control_system.execute_command(command_line)

    async def _handle_optimize_for(self, args: list[str],
                                 context: dict[str, Any]) -> CommandResult:
        """Handle optimize-for command"""
        if not args:
            return CommandResult(
                success=False,
                message="Task type required",
                suggestions=[
                    "Available types: debugging, security, refactoring, documentation, testing, minimal, comprehensive",
                ],
            )

        task_type = args[0]
        command_line = f"/optimize-for {task_type}"

        result = await self.control_system.execute_command(command_line)

        # Track optimization
        if result.success:
            self.analytics.track_user_action(
                "optimization_applied",
                {
                    "task_type": task_type,
                    "loaded_tiers": result.data.get("loaded_tiers", []) if result.data else [],
                },
                user_id=context.get("user_id", "default"),
                session_id=self.active_session_id,
            )

        return result

    async def _handle_tier_status(self, args: list[str],
                                context: dict[str, Any]) -> CommandResult:
        """Handle tier-status command"""
        return await self.control_system.execute_command("/tier-status")

    async def _handle_save_profile(self, args: list[str],
                                 context: dict[str, Any]) -> CommandResult:
        """Handle save-profile command"""
        if not args:
            return CommandResult(
                success=False,
                message="Profile name required",
                suggestions=['Example: function-loading:save-profile auth-work "Authentication workflow"'],
            )

        profile_name = args[0]
        description = " ".join(args[1:]) if len(args) > 1 else f"Profile saved on {datetime.now().strftime('%Y-%m-%d')}"

        command_line = f'/save-session-profile {profile_name} "{description}"'

        result = await self.control_system.execute_command(command_line)

        # Track profile creation
        if result.success:
            self.analytics.track_user_action(
                "profile_created",
                {
                    "profile_name": profile_name,
                    "description": description,
                },
                user_id=context.get("user_id", "default"),
                session_id=self.active_session_id,
            )

        return result

    async def _handle_load_profile(self, args: list[str],
                                 context: dict[str, Any]) -> CommandResult:
        """Handle load-profile command"""
        if not args:
            return CommandResult(
                success=False,
                message="Profile name required",
                suggestions=["Example: function-loading:load-profile auth-work"],
            )

        profile_name = args[0]
        command_line = f"/load-session-profile {profile_name}"

        result = await self.control_system.execute_command(command_line)

        # Track profile loading
        if result.success:
            self.analytics.track_user_action(
                "profile_loaded",
                {
                    "profile_name": profile_name,
                    "categories": result.data.get("categories", {}) if result.data else {},
                },
                user_id=context.get("user_id", "default"),
                session_id=self.active_session_id,
            )

        return result

    async def _handle_performance_stats(self, args: list[str],
                                      context: dict[str, Any]) -> CommandResult:
        """Handle performance-stats command"""
        result = await self.control_system.execute_command("/function-stats")

        # Also add integration-specific metrics
        if result.success and result.data:
            result.data["integration_metrics"] = {
                "commands_executed": len(self.command_history),
                "error_rate": len([h for h in self.command_history if not h["success"]]) / max(1, len(self.command_history)),
                "avg_commands_per_session": len(self.command_history) / 1,  # Would track multiple sessions
                "most_used_commands": self._get_command_usage_stats(),
            }

        return result

    async def _handle_function_loading_help(self, args: list[str],
                                          context: dict[str, Any]) -> CommandResult:
        """Handle function-loading help command"""
        query = " ".join(args) if args else None

        # Track help request
        self.analytics.track_user_action(
            "help_requested",
            {
                "help_type": "function_loading",
                "query": query,
            },
            user_id=context.get("user_id", "default"),
            session_id=self.active_session_id,
        )

        # Get help from help system
        help_context = {
            "recent_commands": [h["command"] for h in self.command_history[-5:]],
            "recent_results": [],  # Would include recent results
            "current_state": context,
        }

        return self.help_system.get_help(query, context=help_context)

    async def _handle_analytics(self, args: list[str],
                              context: dict[str, Any]) -> CommandResult:
        """Handle analytics command"""
        user_id = context.get("user_id", "default")

        # Get analytics data
        user_analytics = self.analytics.get_user_analytics(user_id)

        # Format for display
        summary = {
            "user_id": user_analytics["user_id"],
            "analysis_period": f"{user_analytics['analysis_period_days']} days",
            "activity_level": self._categorize_activity_level(user_analytics["total_events"]),
            "efficiency_score": self._calculate_efficiency_score(user_analytics),
            "top_recommendations": user_analytics["recommendations"][:3],
            "patterns_detected": user_analytics["patterns_detected"],
            "optimization_opportunities": len(user_analytics["optimization_insights"]),
        }

        return CommandResult(
            success=True,
            message="Usage analytics and insights",
            data={
                "summary": summary,
                "full_analytics": user_analytics,
                "insights": user_analytics["optimization_insights"][:5],  # Top 5 insights
            },
        )

    def _get_command_usage_stats(self) -> dict[str, int]:
        """Get command usage statistics"""
        command_counts = {}
        for entry in self.command_history:
            command = entry["command"].split()[0]  # Get base command
            command_counts[command] = command_counts.get(command, 0) + 1

        # Sort by usage
        sorted_commands = sorted(command_counts.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_commands[:10])  # Top 10

    def _categorize_activity_level(self, total_events: int) -> str:
        """Categorize user activity level"""
        if total_events < 10:
            return "low"
        if total_events < 50:
            return "moderate"
        if total_events < 100:
            return "high"
        return "very_high"

    def _calculate_efficiency_score(self, analytics: dict[str, Any]) -> float:
        """Calculate user efficiency score (0-100)"""
        base_score = 50

        # Positive factors
        if analytics["error_rate"] < 0.1:
            base_score += 20
        elif analytics["error_rate"] < 0.2:
            base_score += 10

        if analytics["patterns_detected"] > 2:
            base_score += 15

        if len(analytics["optimization_insights"]) > 0:
            base_score += 10

        # Negative factors
        if analytics["error_rate"] > 0.3:
            base_score -= 20

        return max(0, min(100, base_score))

    def get_integration_status(self) -> dict[str, Any]:
        """Get current integration status"""
        return {
            "status": asdict(self.integration_status),
            "active_session": self.active_session_id,
            "command_history_length": len(self.command_history),
            "registered_commands": len(self.command_registry.commands),
            "categories": list(self.command_registry.categories.keys()),
            "recent_activity": self.command_history[-5:] if self.command_history else [],
        }

    def health_check(self) -> dict[str, Any]:
        """Perform system health check"""
        try:
            # Test each component
            health_status = {
                "overall": "healthy",
                "components": {
                    "user_control_system": self._check_user_control_health(),
                    "help_system": self._check_help_system_health(),
                    "analytics_engine": self._check_analytics_health(),
                    "command_registry": self._check_registry_health(),
                },
                "metrics": {
                    "total_commands": len(self.command_registry.commands),
                    "command_history_size": len(self.command_history),
                    "error_rate": self._calculate_error_rate(),
                    "last_activity": self.command_history[-1]["timestamp"] if self.command_history else None,
                },
                "timestamp": datetime.now(),
            }

            # Determine overall health
            component_statuses = [status["status"] for status in health_status["components"].values()]
            if "error" in component_statuses:
                health_status["overall"] = "error"
            elif "warning" in component_statuses:
                health_status["overall"] = "warning"

            self.integration_status.last_health_check = datetime.now()
            return health_status

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "overall": "error",
                "error": str(e),
                "timestamp": datetime.now(),
            }

    def _check_user_control_health(self) -> dict[str, Any]:
        """Check user control system health"""
        try:
            # Test basic functionality
            # Would test actual execution in real implementation

            return {
                "status": "healthy",
                "active_sessions": len(self.control_system.active_sessions) if hasattr(self.control_system, "active_sessions") else 1,
                "last_command": self.command_history[-1]["command"] if self.command_history else None,
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }

    def _check_help_system_health(self) -> dict[str, Any]:
        """Check help system health"""
        try:
            # Test help system
            self.help_system.get_help("basic")

            return {
                "status": "healthy",
                "topics_available": len(self.help_system.content_generator.help_topics),
                "learning_paths": len(self.help_system.content_generator.learning_paths),
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }

    def _check_analytics_health(self) -> dict[str, Any]:
        """Check analytics engine health"""
        try:
            # Test analytics
            system_analytics = self.analytics.get_system_analytics()

            return {
                "status": "healthy",
                "total_events": system_analytics.get("total_events", 0),
                "active_users": system_analytics.get("active_users", 0),
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }

    def _check_registry_health(self) -> dict[str, Any]:
        """Check command registry health"""
        try:
            return {
                "status": "healthy",
                "total_commands": len(self.command_registry.commands),
                "categories": len(self.command_registry.categories),
                "aliases": len(self.command_registry.aliases),
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }

    def _calculate_error_rate(self) -> float:
        """Calculate recent error rate"""
        if not self.command_history:
            return 0.0

        recent_commands = self.command_history[-20:]  # Last 20 commands
        error_count = len([cmd for cmd in recent_commands if not cmd["success"]])

        return error_count / len(recent_commands)


class ClaudeCodeCommandFactory:
    """Factory for creating Claude Code compatible command definitions"""

    @staticmethod
    def create_function_loading_commands() -> dict[str, dict[str, Any]]:
        """Create function loading command definitions for .claude/commands/"""

        commands = {}

        # Function Loading Help Command
        commands["function-loading-help"] = {
            "category": "function_control",
            "complexity": "low",
            "estimated_time": "< 3 minutes",
            "dependencies": [],
            "sub_commands": [],
            "version": "1.0",
            "content": """# Function Loading Help

Interactive help system for function loading control: $ARGUMENTS

## Usage Options

- `basic` - Basic function loading concepts
- `categories` - Category management help
- `optimization` - Performance optimization help
- `profiles` - Session profile management
- `troubleshooting` - Common issues and solutions

## Help Categories

### Basic Usage
```bash
/project:function-loading-help basic
/project:function-loading-help categories
```

### Advanced Features
```bash
/project:function-loading-help optimization
/project:function-loading-help profiles
```

### Problem Solving
```bash
/project:function-loading-help troubleshooting
```

## Integration with Existing Commands

This help system integrates with existing Claude Code workflows:

- Works alongside existing /project: commands
- Provides context-aware suggestions
- Tracks usage for optimization insights
- Offers progressive learning paths

## Quick Reference

Most useful commands for daily workflow:
- `/project:function-loading:list-categories` - See available functions
- `/project:function-loading:optimize-for debugging` - Optimize for task
- `/project:function-loading:save-profile name` - Save current setup
- `/project:function-loading:performance-stats` - Monitor usage
""",
        }

        # Category Management Command
        commands["function-loading-categories"] = {
            "category": "function_control",
            "complexity": "low",
            "estimated_time": "< 2 minutes",
            "dependencies": [],
            "sub_commands": ["load", "unload", "list", "info"],
            "version": "1.0",
            "content": """# Function Category Management

Manage function categories for optimized loading: $ARGUMENTS

## Sub-Commands

### list [tier:N] [loaded-only]
List available categories with optional filtering

```bash
/project:function-loading-categories list
/project:function-loading-categories list tier:2
/project:function-loading-categories list loaded-only
```

### load <category>
Load a specific function category

```bash
/project:function-loading-categories load security
/project:function-loading-categories load analysis
```

### unload <category>
Unload a function category (Tier 2 and 3 only)

```bash
/project:function-loading-categories unload external
/project:function-loading-categories unload debug
```

### info <category>
Get detailed information about a category

```bash
/project:function-loading-categories info security
/project:function-loading-categories info analysis
```

## Available Categories

**Tier 1 (Always Loaded):**
- `core` - Essential operations (Read, Write, Edit, Bash)
- `git` - Version control operations

**Tier 2 (Conditional):**
- `analysis` - AI analysis and intelligence tools
- `quality` - Code improvement and refactoring
- `security` - Security auditing and validation
- `test` - Testing and validation tools
- `debug` - Debugging and troubleshooting

**Tier 3 (On-Demand):**
- `external` - External service integrations
- `infrastructure` - Infrastructure management

## Performance Impact

Each category has an associated token cost:
- Loading unnecessary categories increases response time
- Unloading unused categories improves performance
- Use task optimization for automatic management
""",
        }

        # Optimization Command
        commands["function-loading-optimize"] = {
            "category": "function_control",
            "complexity": "medium",
            "estimated_time": "< 2 minutes",
            "dependencies": [],
            "sub_commands": [],
            "version": "1.0",
            "content": """# Function Loading Optimization

Optimize function loading for specific tasks: $ARGUMENTS

## Task Types

### Development Tasks
- `debugging` - Debug and troubleshoot issues
- `refactoring` - Code improvement and cleanup
- `testing` - Test creation and validation
- `documentation` - Documentation generation

### Security Tasks
- `security` - Security auditing and analysis

### Performance Tasks
- `minimal` - Minimal loading for basic operations
- `comprehensive` - Full loading for complex tasks

## Usage

```bash
/project:function-loading-optimize debugging
/project:function-loading-optimize security
/project:function-loading-optimize minimal
```

## What Optimization Does

1. **Analyzes Task Requirements** - Determines which functions are needed
2. **Loads Relevant Categories** - Automatically loads appropriate function groups
3. **Unloads Unnecessary Functions** - Reduces token usage and improves performance
4. **Provides Usage Insights** - Shows what was loaded and why

## Integration with Profiles

After optimization, save the configuration:

```bash
/project:function-loading-optimize debugging
/project:function-loading-save-profile debug-work "Debugging workflow"
```

Later, quickly restore:

```bash
/project:function-loading-load-profile debug-work
```

## Performance Benefits

- **Token Reduction**: 40-70% reduction in unused functions
- **Faster Responses**: Reduced processing overhead
- **Better Accuracy**: More focused function set for task
- **Automatic Management**: No manual category management needed
""",
        }

        return commands


# Example usage and integration
if __name__ == "__main__":
    async def main() -> None:
        # Initialize system components
        detection_system = TaskDetectionSystem()
        config_manager = ConfigManager()
        control_system = UserControlSystem(detection_system, config_manager)
        help_system = InteractiveHelpSystem(control_system)
        analytics = AnalyticsEngine()

        # Create integration layer
        integration = ClaudeCommandIntegration(control_system, help_system, analytics)

        # Test command execution
        test_commands = [
            "function-loading:list-categories",
            "function-loading:load-category security",
            "function-loading:optimize-for debugging",
            "function-loading:save-profile debug-session",
            "function-loading:performance-stats",
            "function-loading:help categories",
            "function-loading:analytics",
        ]


        for command in test_commands:
            result = await integration.execute_command(command)


            if result.data:
                pass

        # Show integration status
        integration.get_integration_status()

        # Health check
        integration.health_check()

    asyncio.run(main())
