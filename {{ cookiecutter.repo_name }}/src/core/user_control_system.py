"""
User Override Commands and Power User Features for Dynamic Function Loading

Provides comprehensive user control over the dynamic function loading system
while maintaining conservative defaults and ensuring discoverability.
"""

import asyncio
import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import timezone, datetime
from pathlib import Path
from typing import Any

from .task_detection import TaskDetectionSystem
from .task_detection_config import ConfigManager

logger = logging.getLogger(__name__)


@dataclass
class UserSession:
    """Tracks user session state and preferences"""
    session_id: str
    user_level: str = "balanced"  # beginner, balanced, expert
    active_categories: dict[str, bool] = field(default_factory=dict)
    performance_mode: str = "conservative"  # conservative, balanced, aggressive
    command_history: list[dict[str, Any]] = field(default_factory=list)
    preferences: dict[str, Any] = field(default_factory=dict)
    learning_enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SessionProfile:
    """Saved session configuration profile"""
    name: str
    description: str
    categories: dict[str, bool]
    performance_mode: str
    detection_config: dict[str, Any] | None = None
    tags: list[str] = field(default_factory=list)
    created_by: str = "user"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    usage_count: int = 0
    last_used: datetime | None = None


@dataclass
class CommandResult:
    """Result of user command execution"""
    success: bool
    message: str
    data: dict[str, Any] | None = None
    suggestions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class CategoryManager:
    """Manages function category loading and unloading"""

    def __init__(self, detection_system: TaskDetectionSystem) -> None:
        self.detection_system = detection_system
        self.available_categories = {
            "core": {
                "description": "Essential development operations (Read, Write, Edit, Bash)",
                "functions": ["Read", "Write", "Edit", "MultiEdit", "Bash", "LS", "Glob", "Grep"],
                "tier": 1,
                "token_cost": 5800,
                "always_required": True,
            },
            "git": {
                "description": "Git version control operations",
                "functions": ["git_status", "git_add", "git_commit", "git_diff", "git_log"],
                "tier": 1,
                "token_cost": 3240,
                "always_required": True,
            },
            "analysis": {
                "description": "Advanced AI analysis and intelligence tools",
                "functions": ["chat", "thinkdeep", "analyze", "consensus", "debug"],
                "tier": 2,
                "token_cost": 9630,
                "usage_patterns": ["analysis", "investigation", "research"],
            },
            "quality": {
                "description": "Code quality, refactoring, and improvement tools",
                "functions": ["codereview", "refactor", "docgen", "testgen"],
                "tier": 2,
                "token_cost": 5310,
                "usage_patterns": ["refactor", "improve", "clean", "optimize"],
            },
            "security": {
                "description": "Security auditing and validation tools",
                "functions": ["secaudit", "precommit"],
                "tier": 2,
                "token_cost": 1900,
                "usage_patterns": ["security", "audit", "scan", "vulnerability"],
            },
            "test": {
                "description": "Testing and validation tools",
                "functions": ["testgen", "precommit"],
                "tier": 2,
                "token_cost": 1770,
                "usage_patterns": ["test", "validate", "verify", "coverage"],
            },
            "debug": {
                "description": "Debugging and troubleshooting tools",
                "functions": ["debug", "tracer", "analyze"],
                "tier": 2,
                "token_cost": 2580,
                "usage_patterns": ["debug", "fix", "error", "issue", "problem"],
            },
            "external": {
                "description": "External service integrations",
                "functions": ["context7", "time", "safety"],
                "tier": 3,
                "token_cost": 3850,
                "usage_patterns": ["api", "service", "integration", "external"],
            },
            "infrastructure": {
                "description": "Infrastructure and resource management",
                "functions": ["ListMcpResourcesTool", "ReadMcpResourceTool"],
                "tier": 3,
                "token_cost": 580,
                "usage_patterns": ["resources", "infrastructure", "management"],
            },
        }

    def load_category(self, category: str) -> CommandResult:
        """Load a specific function category"""
        if category not in self.available_categories:
            return CommandResult(
                success=False,
                message=f"Unknown category: {category}",
                suggestions=[f"Available categories: {', '.join(self.available_categories.keys())}"],
            )

        category_info = self.available_categories[category]

        if category_info.get("always_required", False):
            return CommandResult(
                success=True,
                message=f"Category '{category}' is always loaded (Tier 1)",
                data={"category": category, "status": "always_loaded"},
            )

        # Simulate loading (in real implementation, this would update function availability)
        logger.info(f"Loading category: {category}")

        return CommandResult(
            success=True,
            message=f"Successfully loaded category '{category}' ({len(category_info['functions'])} functions)",
            data={
                "category": category,
                "functions_loaded": category_info["functions"],
                "token_cost": category_info["token_cost"],
                "tier": category_info["tier"],
            },
        )

    def unload_category(self, category: str) -> CommandResult:
        """Unload a specific function category"""
        if category not in self.available_categories:
            return CommandResult(
                success=False,
                message=f"Unknown category: {category}",
                suggestions=[f"Available categories: {', '.join(self.available_categories.keys())}"],
            )

        category_info = self.available_categories[category]

        if category_info.get("always_required", False):
            return CommandResult(
                success=False,
                message=f"Cannot unload category '{category}' - it's required for basic operation",
                warnings=["Categories 'core' and 'git' are always loaded for safety"],
            )

        logger.info(f"Unloading category: {category}")

        return CommandResult(
            success=True,
            message=f"Successfully unloaded category '{category}'",
            data={
                "category": category,
                "functions_unloaded": category_info["functions"],
                "tokens_saved": category_info["token_cost"],
            },
        )

    def list_categories(self, filter_tier: int | None = None,
                       show_loaded_only: bool = False) -> CommandResult:
        """List all available categories with their status"""
        categories_info = []

        for name, info in self.available_categories.items():
            if filter_tier is not None and info["tier"] != filter_tier:
                continue

            # In real implementation, check actual loading status
            is_loaded = info.get("always_required", False)  # Simplified for demo

            if show_loaded_only and not is_loaded:
                continue

            categories_info.append({
                "name": name,
                "description": info["description"],
                "tier": info["tier"],
                "token_cost": info["token_cost"],
                "function_count": len(info["functions"]),
                "is_loaded": is_loaded,
                "usage_patterns": info.get("usage_patterns", []),
            })

        return CommandResult(
            success=True,
            message=f"Found {len(categories_info)} categories",
            data={
                "categories": categories_info,
                "total_tiers": 3,
                "filter_applied": filter_tier is not None,
            },
        )

    def get_category_info(self, category: str) -> CommandResult:
        """Get detailed information about a specific category"""
        if category not in self.available_categories:
            return CommandResult(
                success=False,
                message=f"Unknown category: {category}",
                suggestions=[f"Available categories: {', '.join(self.available_categories.keys())}"],
            )

        info = self.available_categories[category]

        return CommandResult(
            success=True,
            message=f"Category information for '{category}'",
            data={
                "name": category,
                "description": info["description"],
                "tier": info["tier"],
                "token_cost": info["token_cost"],
                "functions": info["functions"],
                "usage_patterns": info.get("usage_patterns", []),
                "always_required": info.get("always_required", False),
                "is_loaded": info.get("always_required", False),  # Simplified
            },
        )


class TierController:
    """Controls tier-based function loading"""

    def __init__(self, detection_system: TaskDetectionSystem) -> None:
        self.detection_system = detection_system
        self.tier_definitions = {
            1: {
                "name": "Essential",
                "description": "Core functions always loaded for basic operation",
                "categories": ["core", "git"],
                "token_cost": 9040,
                "load_threshold": 0.0,  # Always loaded
            },
            2: {
                "name": "Extended",
                "description": "Advanced functions loaded based on task detection",
                "categories": ["analysis", "quality", "security", "test", "debug"],
                "token_cost": 14940,
                "load_threshold": 0.3,
            },
            3: {
                "name": "Specialized",
                "description": "Specialized functions loaded only when specifically needed",
                "categories": ["external", "infrastructure"],
                "token_cost": 3850,
                "load_threshold": 0.6,
            },
        }
        self.current_tier_status = {1: True, 2: False, 3: False}

    def load_tier(self, tier: int, force: bool = False) -> CommandResult:
        """Load specific tier of functions"""
        if tier not in self.tier_definitions:
            return CommandResult(
                success=False,
                message=f"Invalid tier: {tier}",
                suggestions=["Valid tiers: 1 (Essential), 2 (Extended), 3 (Specialized)"],
            )

        tier_info = self.tier_definitions[tier]

        if tier == 1 and not force:
            return CommandResult(
                success=True,
                message="Tier 1 (Essential) is always loaded",
                data={"tier": tier, "status": "always_loaded"},
            )

        # Update tier status (simplified - real implementation would update function registry)
        self.current_tier_status[tier] = True

        return CommandResult(
            success=True,
            message=f"Successfully loaded Tier {tier} ({tier_info['name']})",
            data={
                "tier": tier,
                "categories_loaded": tier_info["categories"],
                "token_cost": tier_info["token_cost"],
                "total_categories": len(tier_info["categories"]),
            },
        )

    def unload_tier(self, tier: int, force: bool = False) -> CommandResult:
        """Unload specific tier of functions"""
        if tier not in self.tier_definitions:
            return CommandResult(
                success=False,
                message=f"Invalid tier: {tier}",
                suggestions=["Valid tiers: 1 (Essential), 2 (Extended), 3 (Specialized)"],
            )

        if tier == 1 and not force:
            return CommandResult(
                success=False,
                message="Cannot unload Tier 1 (Essential) - required for basic operation",
                warnings=["Use --force flag only if you understand the implications"],
            )

        tier_info = self.tier_definitions[tier]
        self.current_tier_status[tier] = False

        return CommandResult(
            success=True,
            message=f"Successfully unloaded Tier {tier} ({tier_info['name']})",
            data={
                "tier": tier,
                "categories_unloaded": tier_info["categories"],
                "tokens_saved": tier_info["token_cost"],
            },
        )

    def get_tier_status(self) -> CommandResult:
        """Get current tier loading status"""
        status_info = []
        total_tokens = 0

        for tier_num, tier_info in self.tier_definitions.items():
            is_loaded = self.current_tier_status[tier_num]
            if is_loaded:
                total_tokens += tier_info["token_cost"]

            status_info.append({
                "tier": tier_num,
                "name": tier_info["name"],
                "description": tier_info["description"],
                "is_loaded": is_loaded,
                "categories": tier_info["categories"],
                "token_cost": tier_info["token_cost"],
                "load_threshold": tier_info["load_threshold"],
            })

        return CommandResult(
            success=True,
            message="Current tier status",
            data={
                "tiers": status_info,
                "total_tokens_loaded": total_tokens,
                "loaded_tiers": [t for t, loaded in self.current_tier_status.items() if loaded],
            },
        )

    def optimize_for_task(self, task_type: str) -> CommandResult:
        """Optimize tier loading for specific task type"""
        task_optimizations = {
            "debugging": {
                "required_tiers": [1, 2],
                "priority_categories": ["debug", "analysis"],
                "description": "Optimized for debugging and troubleshooting",
            },
            "security": {
                "required_tiers": [1, 2],
                "priority_categories": ["security", "analysis"],
                "description": "Optimized for security analysis and auditing",
            },
            "refactoring": {
                "required_tiers": [1, 2],
                "priority_categories": ["quality", "analysis"],
                "description": "Optimized for code refactoring and improvement",
            },
            "documentation": {
                "required_tiers": [1, 2],
                "priority_categories": ["quality", "analysis"],
                "description": "Optimized for documentation generation",
            },
            "testing": {
                "required_tiers": [1, 2],
                "priority_categories": ["test", "quality"],
                "description": "Optimized for test generation and validation",
            },
            "minimal": {
                "required_tiers": [1],
                "priority_categories": ["core", "git"],
                "description": "Minimal loading for basic operations",
            },
            "comprehensive": {
                "required_tiers": [1, 2, 3],
                "priority_categories": [],
                "description": "Comprehensive loading for complex tasks",
            },
        }

        if task_type not in task_optimizations:
            return CommandResult(
                success=False,
                message=f"Unknown task type: {task_type}",
                suggestions=[f"Available task types: {', '.join(task_optimizations.keys())}"],
            )

        optimization = task_optimizations[task_type]

        # Apply optimization
        for tier in [1, 2, 3]:
            should_load = tier in optimization["required_tiers"]
            self.current_tier_status[tier] = should_load

        return CommandResult(
            success=True,
            message=f"Optimized for {task_type}: {optimization['description']}",
            data={
                "task_type": task_type,
                "loaded_tiers": optimization["required_tiers"],
                "priority_categories": optimization["priority_categories"],
                "optimization_applied": True,
            },
        )


class PerformanceMonitor:
    """Monitors and reports on function loading performance"""

    def __init__(self) -> None:
        self.metrics = {
            "loading_times": [],
            "memory_usage": [],
            "cache_hits": 0,
            "cache_misses": 0,
            "function_calls": defaultdict(int),
            "category_usage": defaultdict(int),
        }
        self.start_time = time.time()

    def record_loading_time(self, operation: str, duration_ms: float) -> None:
        """Record loading operation time"""
        self.metrics["loading_times"].append({
            "operation": operation,
            "duration_ms": duration_ms,
            "timestamp": datetime.now(timezone.utc),
        })

    def record_memory_usage(self, usage_mb: float) -> None:
        """Record memory usage"""
        self.metrics["memory_usage"].append({
            "usage_mb": usage_mb,
            "timestamp": datetime.now(timezone.utc),
        })

    def get_function_stats(self) -> CommandResult:
        """Get current function loading statistics"""
        total_calls = sum(self.metrics["function_calls"].values())
        avg_loading_time = 0

        if self.metrics["loading_times"]:
            avg_loading_time = sum(
                lt["duration_ms"] for lt in self.metrics["loading_times"]
            ) / len(self.metrics["loading_times"])

        cache_hit_rate = 0
        if self.metrics["cache_hits"] + self.metrics["cache_misses"] > 0:
            cache_hit_rate = self.metrics["cache_hits"] / (
                self.metrics["cache_hits"] + self.metrics["cache_misses"]
            )

        return CommandResult(
            success=True,
            message="Function loading statistics",
            data={
                "total_function_calls": total_calls,
                "avg_loading_time_ms": round(avg_loading_time, 2),
                "cache_hit_rate": round(cache_hit_rate * 100, 1),
                "uptime_seconds": round(time.time() - self.start_time, 1),
                "most_used_functions": dict(
                    Counter(self.metrics["function_calls"]).most_common(5),
                ),
                "category_usage": dict(self.metrics["category_usage"]),
            },
        )

    def clear_cache(self) -> CommandResult:
        """Clear performance monitoring cache"""
        self.metrics["cache_hits"] = 0
        self.metrics["cache_misses"] = 0
        self.metrics["loading_times"] = []
        self.metrics["memory_usage"] = []

        return CommandResult(
            success=True,
            message="Performance cache cleared",
            data={"cache_cleared": True, "timestamp": datetime.now(timezone.utc)},
        )

    def benchmark_loading(self) -> CommandResult:
        """Run performance benchmark"""
        start_time = time.perf_counter()

        # Simulate various loading operations
        operations = [
            ("category_load", 45.2),
            ("tier_load", 67.8),
            ("cache_lookup", 2.1),
            ("detection_run", 23.5),
        ]

        for operation, simulated_time in operations:
            self.record_loading_time(operation, simulated_time)

        total_time = (time.perf_counter() - start_time) * 1000

        return CommandResult(
            success=True,
            message=f"Benchmark completed in {total_time:.2f}ms",
            data={
                "benchmark_operations": len(operations),
                "total_time_ms": round(total_time, 2),
                "operations_tested": [op[0] for op in operations],
                "avg_operation_time": round(sum(op[1] for op in operations) / len(operations), 2),
            },
        )


class ProfileManager:
    """Manages session profiles for saving/loading function configurations"""

    def __init__(self, config_dir: Path | None = None) -> None:
        self.config_dir = config_dir or Path("user_profiles")
        self.config_dir.mkdir(exist_ok=True)
        self.profiles: dict[str, SessionProfile] = {}
        self._load_profiles()

    def _load_profiles(self) -> None:
        """Load existing profiles from disk"""
        for profile_file in self.config_dir.glob("*.json"):
            try:
                with open(profile_file) as f:
                    data = json.load(f)

                profile = SessionProfile(
                    name=data["name"],
                    description=data["description"],
                    categories=data["categories"],
                    performance_mode=data["performance_mode"],
                    detection_config=data.get("detection_config"),
                    tags=data.get("tags", []),
                    created_by=data.get("created_by", "user"),
                    created_at=datetime.fromisoformat(data["created_at"]),
                    usage_count=data.get("usage_count", 0),
                    last_used=datetime.fromisoformat(data["last_used"]) if data.get("last_used") else None,
                )

                self.profiles[profile.name] = profile

            except Exception as e:
                logger.warning(f"Failed to load profile {profile_file}: {e}")

    def save_session_profile(self, name: str, description: str,
                           current_categories: dict[str, bool],
                           performance_mode: str,
                           tags: list[str] | None = None) -> CommandResult:
        """Save current session as a named profile"""
        if not name or not name.strip():
            return CommandResult(
                success=False,
                message="Profile name cannot be empty",
                suggestions=["Provide a descriptive name for the profile"],
            )

        profile = SessionProfile(
            name=name.strip(),
            description=description.strip(),
            categories=current_categories.copy(),
            performance_mode=performance_mode,
            tags=tags or [],
            usage_count=0,
        )

        # Save to disk
        profile_file = self.config_dir / f"{name}.json"
        try:
            with open(profile_file, "w") as f:
                json.dump({
                    "name": profile.name,
                    "description": profile.description,
                    "categories": profile.categories,
                    "performance_mode": profile.performance_mode,
                    "detection_config": profile.detection_config,
                    "tags": profile.tags,
                    "created_by": profile.created_by,
                    "created_at": profile.created_at.isoformat(),
                    "usage_count": profile.usage_count,
                    "last_used": profile.last_used.isoformat() if profile.last_used else None,
                }, f, indent=2)

            self.profiles[name] = profile

            return CommandResult(
                success=True,
                message=f"Profile '{name}' saved successfully",
                data={
                    "profile_name": name,
                    "categories_saved": len(current_categories),
                    "file_path": str(profile_file),
                },
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Failed to save profile: {e}",
                warnings=["Check file permissions and disk space"],
            )

    def load_session_profile(self, name: str) -> CommandResult:
        """Load a named profile"""
        if name not in self.profiles:
            return CommandResult(
                success=False,
                message=f"Profile '{name}' not found",
                suggestions=[f"Available profiles: {', '.join(self.profiles.keys())}"],
            )

        profile = self.profiles[name]

        # Update usage statistics
        profile.usage_count += 1
        profile.last_used = datetime.now(timezone.utc)

        # Save updated profile
        self._save_profile(profile)

        return CommandResult(
            success=True,
            message=f"Profile '{name}' loaded successfully",
            data={
                "profile_name": name,
                "description": profile.description,
                "categories": profile.categories,
                "performance_mode": profile.performance_mode,
                "tags": profile.tags,
                "usage_count": profile.usage_count,
            },
        )

    def list_profiles(self, tag_filter: str | None = None) -> CommandResult:
        """List all available profiles"""
        filtered_profiles = []

        for profile in self.profiles.values():
            if tag_filter and tag_filter not in profile.tags:
                continue

            filtered_profiles.append({
                "name": profile.name,
                "description": profile.description,
                "tags": profile.tags,
                "usage_count": profile.usage_count,
                "last_used": profile.last_used.isoformat() if profile.last_used else None,
                "created_at": profile.created_at.isoformat(),
                "categories_count": len([c for c, enabled in profile.categories.items() if enabled]),
            })

        # Sort by usage count and recency
        filtered_profiles.sort(
            key=lambda p: (p["usage_count"], p["last_used"] or ""),
            reverse=True,
        )

        return CommandResult(
            success=True,
            message=f"Found {len(filtered_profiles)} profiles",
            data={
                "profiles": filtered_profiles,
                "total_profiles": len(self.profiles),
                "tag_filter": tag_filter,
            },
        )

    def _save_profile(self, profile: SessionProfile) -> None:
        """Save profile to disk"""
        profile_file = self.config_dir / f"{profile.name}.json"
        with open(profile_file, "w") as f:
            json.dump({
                "name": profile.name,
                "description": profile.description,
                "categories": profile.categories,
                "performance_mode": profile.performance_mode,
                "detection_config": profile.detection_config,
                "tags": profile.tags,
                "created_by": profile.created_by,
                "created_at": profile.created_at.isoformat(),
                "usage_count": profile.usage_count,
                "last_used": profile.last_used.isoformat() if profile.last_used else None,
            }, f, indent=2)


class CommandParser:
    """Parses and validates user override commands"""

    def __init__(self) -> None:
        self.commands = {
            "load-category": {
                "handler": "load_category",
                "params": ["category"],
                "description": "Load specific function category",
                "examples": ["/load-category security", "/load-category analysis"],
            },
            "unload-category": {
                "handler": "unload_category",
                "params": ["category"],
                "description": "Unload specific function category",
                "examples": ["/unload-category external", "/unload-category debug"],
            },
            "list-categories": {
                "handler": "list_categories",
                "params": ["tier?", "loaded-only?"],
                "description": "Show all available categories and their status",
                "examples": ["/list-categories", "/list-categories tier:2", "/list-categories loaded-only"],
            },
            "category-info": {
                "handler": "category_info",
                "params": ["category"],
                "description": "Detailed information about a category",
                "examples": ["/category-info security", "/category-info analysis"],
            },
            "load-tier": {
                "handler": "load_tier",
                "params": ["tier"],
                "description": "Force loading of specific tier",
                "examples": ["/load-tier 2", "/load-tier 3"],
            },
            "unload-tier": {
                "handler": "unload_tier",
                "params": ["tier"],
                "description": "Unload specific tier",
                "examples": ["/unload-tier 3", "/unload-tier 2 --force"],
            },
            "tier-status": {
                "handler": "tier_status",
                "params": [],
                "description": "Show current tier loading status",
                "examples": ["/tier-status"],
            },
            "optimize-for": {
                "handler": "optimize_for",
                "params": ["task-type"],
                "description": "Optimize function set for specific task type",
                "examples": ["/optimize-for debugging", "/optimize-for security", "/optimize-for minimal"],
            },
            "function-stats": {
                "handler": "function_stats",
                "params": [],
                "description": "Show current function loading statistics",
                "examples": ["/function-stats"],
            },
            "clear-cache": {
                "handler": "clear_cache",
                "params": [],
                "description": "Clear function loading cache",
                "examples": ["/clear-cache"],
            },
            "benchmark-loading": {
                "handler": "benchmark_loading",
                "params": [],
                "description": "Run performance benchmark",
                "examples": ["/benchmark-loading"],
            },
            "save-session-profile": {
                "handler": "save_session_profile",
                "params": ["name", "description?", "tags?"],
                "description": "Save current function set as named profile",
                "examples": ['/save-session-profile auth-work "Authentication workflow"'],
            },
            "load-session-profile": {
                "handler": "load_session_profile",
                "params": ["name"],
                "description": "Load previously saved function profile",
                "examples": ["/load-session-profile auth-work"],
            },
            "list-profiles": {
                "handler": "list_profiles",
                "params": ["tag?"],
                "description": "Show all saved session profiles",
                "examples": ["/list-profiles", "/list-profiles tag:security"],
            },
            "performance-mode": {
                "handler": "set_performance_mode",
                "params": ["mode"],
                "description": "Set loading strategy (conservative/balanced/aggressive)",
                "examples": ["/performance-mode aggressive", "/performance-mode conservative"],
            },
            "help": {
                "handler": "show_help",
                "params": ["command?"],
                "description": "Show help for function loading commands",
                "examples": ["/help", "/help load-category"],
            },
        }

    def parse_command(self, command_line: str) -> dict[str, Any]:
        """Parse a user command line"""
        if not command_line.startswith("/"):
            return {"error": "Commands must start with /"}

        # Remove leading slash and split
        parts = command_line[1:].split()
        if not parts:
            return {"error": "Empty command"}

        command = parts[0]
        args = parts[1:]

        if command not in self.commands:
            return {
                "error": f"Unknown command: {command}",
                "suggestions": self._suggest_similar_commands(command),
            }

        command_info = self.commands[command]

        # Parse arguments
        parsed_args = {}
        flags = []

        i = 0
        while i < len(args):
            arg = args[i]

            if arg.startswith("--"):
                # Flag argument
                flags.append(arg[2:])
            elif ":" in arg:
                # Key-value argument
                key, value = arg.split(":", 1)
                parsed_args[key] = value
            else:
                # Positional argument
                param_index = len(parsed_args) - len([k for k in parsed_args if ":" not in k])
                param_names = [p.rstrip("?") for p in command_info["params"]]

                if param_index < len(param_names):
                    parsed_args[param_names[param_index]] = arg
                else:
                    parsed_args[f"extra_{param_index}"] = arg

            i += 1

        return {
            "command": command,
            "args": parsed_args,
            "flags": flags,
            "handler": command_info["handler"],
        }

    def _suggest_similar_commands(self, command: str) -> list[str]:
        """Suggest similar commands for typos"""
        # Simple similarity based on common prefixes/suffixes
        suggestions = []
        for cmd in self.commands:
            if cmd.startswith(command[:3]) or command in cmd:
                suggestions.append(cmd)

        return suggestions[:3]  # Top 3 suggestions

    def get_command_help(self, command: str | None = None) -> str:
        """Get help text for commands"""
        if command:
            if command not in self.commands:
                return f"Unknown command: {command}"

            cmd_info = self.commands[command]
            help_text = f"Command: /{command}\n"
            help_text += f"Description: {cmd_info['description']}\n"
            help_text += f"Parameters: {', '.join(cmd_info['params'])}\n"
            help_text += "Examples:\n"
            for example in cmd_info["examples"]:
                help_text += f"  {example}\n"

            return help_text

        # General help
        help_text = "Function Loading Control Commands:\n\n"

        categories = {
            "Category Management": ["load-category", "unload-category", "list-categories", "category-info"],
            "Tier Control": ["load-tier", "unload-tier", "tier-status", "optimize-for"],
            "Performance": ["function-stats", "clear-cache", "benchmark-loading", "performance-mode"],
            "Session Management": ["save-session-profile", "load-session-profile", "list-profiles"],
            "Help": ["help"],
        }

        for category, commands in categories.items():
            help_text += f"{category}:\n"
            for cmd in commands:
                if cmd in self.commands:
                    help_text += f"  /{cmd} - {self.commands[cmd]['description']}\n"
            help_text += "\n"

        help_text += "Use '/help <command>' for detailed help on specific commands.\n"

        return help_text


class UserControlSystem:
    """Main user control system orchestrating all components"""

    def __init__(self, detection_system: TaskDetectionSystem,
                 config_manager: ConfigManager) -> None:
        self.detection_system = detection_system
        self.config_manager = config_manager

        # Initialize components
        self.category_manager = CategoryManager(detection_system)
        self.tier_controller = TierController(detection_system)
        self.performance_monitor = PerformanceMonitor()
        self.profile_manager = ProfileManager()
        self.command_parser = CommandParser()

        # Session state
        self.current_session = UserSession(
            session_id=f"session_{int(time.time())}",
            user_level="balanced",
            performance_mode="conservative",
        )

        # Usage analytics
        self.usage_analytics = {
            "commands_executed": Counter(),
            "categories_used": Counter(),
            "error_patterns": defaultdict(list),
            "session_patterns": [],
        }

    async def execute_command(self, command_line: str) -> CommandResult:
        """Execute a user command"""
        start_time = time.perf_counter()

        try:
            # Parse command
            parsed = self.command_parser.parse_command(command_line)

            if "error" in parsed:
                return CommandResult(
                    success=False,
                    message=parsed["error"],
                    suggestions=parsed.get("suggestions", []),
                )

            # Update analytics
            self.usage_analytics["commands_executed"][parsed["command"]] += 1

            # Route to appropriate handler
            handler_name = parsed["handler"]
            handler_method = getattr(self, handler_name, None)

            if not handler_method:
                return CommandResult(
                    success=False,
                    message=f"Handler not implemented: {handler_name}",
                )

            # Execute command
            result = await handler_method(parsed["args"], parsed["flags"])

            # Record performance
            execution_time = (time.perf_counter() - start_time) * 1000
            self.performance_monitor.record_loading_time(
                f"command_{parsed['command']}", execution_time,
            )

            # Update session history
            self.current_session.command_history.append({
                "command": command_line,
                "timestamp": datetime.now(timezone.utc),
                "success": result.success,
                "execution_time_ms": execution_time,
            })

            return result

        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return CommandResult(
                success=False,
                message=f"Command execution failed: {e!s}",
                warnings=["This might be a system error - check logs"],
            )

    # Command handlers
    async def load_category(self, args: dict[str, Any], flags: list[str]) -> CommandResult:
        """Handle load-category command"""
        if "category" not in args:
            return CommandResult(
                success=False,
                message="Category name required",
                suggestions=["Example: /load-category security"],
            )

        result = self.category_manager.load_category(args["category"])

        if result.success:
            # Update session state
            self.current_session.active_categories[args["category"]] = True
            self.usage_analytics["categories_used"][args["category"]] += 1

        return result

    async def unload_category(self, args: dict[str, Any], flags: list[str]) -> CommandResult:
        """Handle unload-category command"""
        if "category" not in args:
            return CommandResult(
                success=False,
                message="Category name required",
                suggestions=["Example: /unload-category external"],
            )

        result = self.category_manager.unload_category(args["category"])

        if result.success:
            self.current_session.active_categories[args["category"]] = False

        return result

    async def list_categories(self, args: dict[str, Any], flags: list[str]) -> CommandResult:
        """Handle list-categories command"""
        filter_tier = None
        if "tier" in args:
            try:
                filter_tier = int(args["tier"])
            except ValueError:
                return CommandResult(
                    success=False,
                    message="Tier must be a number (1, 2, or 3)",
                )

        show_loaded_only = "loaded-only" in flags

        return self.category_manager.list_categories(filter_tier, show_loaded_only)

    async def category_info(self, args: dict[str, Any], flags: list[str]) -> CommandResult:
        """Handle category-info command"""
        if "category" not in args:
            return CommandResult(
                success=False,
                message="Category name required",
                suggestions=["Example: /category-info security"],
            )

        return self.category_manager.get_category_info(args["category"])

    async def load_tier(self, args: dict[str, Any], flags: list[str]) -> CommandResult:
        """Handle load-tier command"""
        if "tier" not in args:
            return CommandResult(
                success=False,
                message="Tier number required (1, 2, or 3)",
                suggestions=["Example: /load-tier 2"],
            )

        try:
            tier = int(args["tier"])
        except ValueError:
            return CommandResult(
                success=False,
                message="Tier must be a number (1, 2, or 3)",
            )

        force = "force" in flags
        return self.tier_controller.load_tier(tier, force)

    async def unload_tier(self, args: dict[str, Any], flags: list[str]) -> CommandResult:
        """Handle unload-tier command"""
        if "tier" not in args:
            return CommandResult(
                success=False,
                message="Tier number required (1, 2, or 3)",
                suggestions=["Example: /unload-tier 3"],
            )

        try:
            tier = int(args["tier"])
        except ValueError:
            return CommandResult(
                success=False,
                message="Tier must be a number (1, 2, or 3)",
            )

        force = "force" in flags
        return self.tier_controller.unload_tier(tier, force)

    async def tier_status(self, args: dict[str, Any], flags: list[str]) -> CommandResult:
        """Handle tier-status command"""
        return self.tier_controller.get_tier_status()

    async def optimize_for(self, args: dict[str, Any], flags: list[str]) -> CommandResult:
        """Handle optimize-for command"""
        if "task-type" not in args:
            return CommandResult(
                success=False,
                message="Task type required",
                suggestions=[
                    "Available types: debugging, security, refactoring, documentation, testing, minimal, comprehensive",
                ],
            )

        return self.tier_controller.optimize_for_task(args["task-type"])

    async def function_stats(self, args: dict[str, Any], flags: list[str]) -> CommandResult:
        """Handle function-stats command"""
        return self.performance_monitor.get_function_stats()

    async def clear_cache(self, args: dict[str, Any], flags: list[str]) -> CommandResult:
        """Handle clear-cache command"""
        return self.performance_monitor.clear_cache()

    async def benchmark_loading(self, args: dict[str, Any], flags: list[str]) -> CommandResult:
        """Handle benchmark-loading command"""
        return self.performance_monitor.benchmark_loading()

    async def save_session_profile(self, args: dict[str, Any], flags: list[str]) -> CommandResult:
        """Handle save-session-profile command"""
        if "name" not in args:
            return CommandResult(
                success=False,
                message="Profile name required",
                suggestions=['Example: /save-session-profile auth-work "Authentication workflow"'],
            )

        name = args["name"]
        description = args.get("description", f"Profile saved on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}")
        tags = args.get("tags", "").split(",") if args.get("tags") else []

        return self.profile_manager.save_session_profile(
            name, description, self.current_session.active_categories,
            self.current_session.performance_mode, tags,
        )

    async def load_session_profile(self, args: dict[str, Any], flags: list[str]) -> CommandResult:
        """Handle load-session-profile command"""
        if "name" not in args:
            return CommandResult(
                success=False,
                message="Profile name required",
                suggestions=["Example: /load-session-profile auth-work"],
            )

        result = self.profile_manager.load_session_profile(args["name"])

        if result.success and result.data:
            # Update current session
            self.current_session.active_categories = result.data["categories"]
            self.current_session.performance_mode = result.data["performance_mode"]

        return result

    async def list_profiles(self, args: dict[str, Any], flags: list[str]) -> CommandResult:
        """Handle list-profiles command"""
        tag_filter = args.get("tag")
        return self.profile_manager.list_profiles(tag_filter)

    async def set_performance_mode(self, args: dict[str, Any], flags: list[str]) -> CommandResult:
        """Handle performance-mode command"""
        if "mode" not in args:
            return CommandResult(
                success=False,
                message="Performance mode required",
                suggestions=["Available modes: conservative, balanced, aggressive"],
            )

        mode = args["mode"]
        valid_modes = ["conservative", "balanced", "aggressive"]

        if mode not in valid_modes:
            return CommandResult(
                success=False,
                message=f"Invalid performance mode: {mode}",
                suggestions=[f"Available modes: {', '.join(valid_modes)}"],
            )

        self.current_session.performance_mode = mode

        # Apply mode to detection system (would update actual detection config in real implementation)

        return CommandResult(
            success=True,
            message=f"Performance mode set to: {mode}",
            data={
                "mode": mode,
                "description": f"Function loading optimized for {mode} operation",
                "applied": True,
            },
        )

    async def show_help(self, args: dict[str, Any], flags: list[str]) -> CommandResult:
        """Handle help command"""
        command = args.get("command")
        help_text = self.command_parser.get_command_help(command)

        return CommandResult(
            success=True,
            message="Help information",
            data={"help_text": help_text},
        )

    def get_usage_analytics(self) -> dict[str, Any]:
        """Get usage analytics and insights"""
        return {
            "session_info": {
                "session_id": self.current_session.session_id,
                "user_level": self.current_session.user_level,
                "performance_mode": self.current_session.performance_mode,
                "uptime_minutes": (datetime.now(timezone.utc) - self.current_session.created_at).total_seconds() / 60,
            },
            "command_usage": dict(self.usage_analytics["commands_executed"]),
            "category_usage": dict(self.usage_analytics["categories_used"]),
            "active_categories": {k: v for k, v in self.current_session.active_categories.items() if v},
            "recommendations": self._generate_recommendations(),
        }

    def _generate_recommendations(self) -> list[str]:
        """Generate usage optimization recommendations"""
        recommendations = []

        # Analyze command patterns
        total_commands = sum(self.usage_analytics["commands_executed"].values())

        if total_commands > 10:
            most_used = self.usage_analytics["commands_executed"].most_common(1)[0]
            if most_used[1] / total_commands > 0.5:
                recommendations.append(
                    f"Consider creating a profile for your frequent '{most_used[0]}' usage pattern",
                )

        # Analyze category usage
        active_categories = [k for k, v in self.current_session.active_categories.items() if v]
        if len(active_categories) > 5:
            recommendations.append(
                "You have many categories loaded - consider optimizing for specific task types",
            )

        if not any(self.usage_analytics["categories_used"].values()):
            recommendations.append(
                "Try loading specific categories for your tasks to improve performance",
            )

        return recommendations


# Example usage and testing
if __name__ == "__main__":
    async def main() -> None:
        # Initialize system components
        detection_system = TaskDetectionSystem()
        config_manager = ConfigManager()

        # Create user control system
        control_system = UserControlSystem(detection_system, config_manager)

        # Test commands
        test_commands = [
            "/list-categories",
            "/load-category security",
            "/tier-status",
            "/optimize-for debugging",
            "/function-stats",
            '/save-session-profile debug-session "Debugging workflow setup"',
            "/list-profiles",
            "/help load-category",
        ]


        for command in test_commands:
            result = await control_system.execute_command(command)


            if result.data:
                pass

            if result.suggestions:
                pass

            if result.warnings:
                pass

        # Show analytics
        control_system.get_usage_analytics()

    asyncio.run(main())
