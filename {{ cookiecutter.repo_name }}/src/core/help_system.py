"""
Comprehensive Help System for Dynamic Function Loading

Provides context-aware help, learning guidance, and progressive disclosure
for user override commands and power user features.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .user_control_system import CommandResult, UserControlSystem


@dataclass
class HelpTopic:
    """Individual help topic with progressive disclosure"""

    name: str
    title: str
    description: str
    level: str  # beginner, intermediate, advanced
    content: str
    examples: list[str] = field(default_factory=list)
    related_topics: list[str] = field(default_factory=list)
    prerequisites: list[str] = field(default_factory=list)
    see_also: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass
class LearningPath:
    """Structured learning path for mastering features"""

    name: str
    description: str
    level: str
    topics: list[str]  # Ordered list of topics
    estimated_time_minutes: int
    prerequisites: list[str] = field(default_factory=list)


@dataclass
class UserProgress:
    """Tracks user progress through help system"""

    user_id: str
    completed_topics: set[str] = field(default_factory=set)
    current_level: str = "beginner"
    preferred_help_style: str = "guided"  # guided, reference, minimal
    last_accessed: dict[str, datetime] = field(default_factory=dict)
    help_effectiveness_ratings: dict[str, int] = field(default_factory=dict)


class ContextAnalyzer:
    """Analyzes current context to provide relevant help"""

    def __init__(self) -> None:
        self.context_patterns = {
            "error_indicators": {
                "unknown command": ["basic_commands", "command_syntax"],
                "category not found": ["available_categories", "category_management"],
                "invalid tier": ["tier_system", "tier_control"],
                "profile not found": ["profile_management", "session_profiles"],
                "permission denied": ["troubleshooting", "system_requirements"],
            },
            "success_patterns": {
                "loaded successfully": ["optimization_tips", "advanced_usage"],
                "profile saved": ["profile_management", "workflow_automation"],
                "tier loaded": ["performance_optimization", "advanced_tier_control"],
            },
            "command_patterns": {
                "load-category": ["category_management", "performance_considerations"],
                "optimize-for": ["task_optimization", "workflow_patterns"],
                "save-session-profile": ["profile_management", "automation_tips"],
                "benchmark-loading": ["performance_tuning", "system_optimization"],
            },
        }

    def analyze_context(
        self,
        recent_commands: list[str],
        recent_results: list[CommandResult],
        current_state: dict[str, Any],
    ) -> list[str]:
        """Analyze context and suggest relevant help topics"""
        suggested_topics = set()

        # Analyze recent errors
        for result in recent_results[-3:]:  # Last 3 results
            if not result.success:
                error_message = result.message.lower()
                for pattern, topics in self.context_patterns["error_indicators"].items():
                    if pattern in error_message:
                        suggested_topics.update(topics)

        # Analyze successful patterns
        for result in recent_results[-3:]:
            if result.success:
                success_message = result.message.lower()
                for pattern, topics in self.context_patterns["success_patterns"].items():
                    if pattern in success_message:
                        suggested_topics.update(topics)

        # Analyze command patterns
        for command in recent_commands[-5:]:  # Last 5 commands
            command_name = command.split()[0].lstrip("/")
            if command_name in self.context_patterns["command_patterns"]:
                suggested_topics.update(self.context_patterns["command_patterns"][command_name])

        # Analyze current state
        if current_state.get("many_categories_loaded", False):
            suggested_topics.add("performance_optimization")

        if current_state.get("beginner_user", False):
            suggested_topics.update(["getting_started", "basic_commands"])

        if current_state.get("advanced_user", False):
            suggested_topics.update(["automation_tips", "advanced_usage"])

        return list(suggested_topics)


class HelpContentGenerator:
    """Generates dynamic help content based on context and user level"""

    def __init__(self) -> None:
        self.help_topics = self._initialize_help_topics()
        self.learning_paths = self._initialize_learning_paths()

    def _initialize_help_topics(self) -> dict[str, HelpTopic]:
        """Initialize all help topics"""
        topics = {}

        # Basic Commands
        topics["basic_commands"] = HelpTopic(
            name="basic_commands",
            title="Basic Function Loading Commands",
            description="Essential commands for controlling function loading",
            level="beginner",
            content="""
Function loading commands give you control over which Claude Code functions are available.
This helps optimize performance by only loading what you need.

Essential Commands:
• /list-categories - See all available function categories
• /load-category <name> - Load a specific category
• /unload-category <name> - Unload a category
• /tier-status - See current loading status
• /help - Get help on any command

Categories are organized into tiers:
• Tier 1: Core functions (always loaded)
• Tier 2: Extended functions (loaded on demand)
• Tier 3: Specialized functions (loaded rarely)
            """,
            examples=[
                "/list-categories",
                "/load-category security",
                "/tier-status",
                "/help load-category",
            ],
            related_topics=["category_management", "tier_system"],
            tags=["basics", "commands", "getting-started"],
        )

        # Category Management
        topics["category_management"] = HelpTopic(
            name="category_management",
            title="Managing Function Categories",
            description="How to load, unload, and manage function categories",
            level="beginner",
            content="""
Function categories group related tools together for easier management.

Available Categories:
• core - Essential operations (Read, Write, Edit, Bash)
• git - Version control operations
• analysis - AI analysis and intelligence tools
• quality - Code improvement and refactoring
• security - Security auditing and validation
• test - Testing and validation tools
• debug - Debugging and troubleshooting
• external - External service integrations

Managing Categories:
• Load: /load-category <name>
• Unload: /unload-category <name>
• Info: /category-info <name>
• List: /list-categories

Categories in Tier 1 (core, git) cannot be unloaded as they're essential.
            """,
            examples=[
                "/list-categories",
                "/category-info security",
                "/load-category analysis",
                "/unload-category external",
            ],
            related_topics=["tier_system", "performance_considerations"],
            tags=["categories", "management", "functions"],
        )

        # Tier System
        topics["tier_system"] = HelpTopic(
            name="tier_system",
            title="Understanding the Tier System",
            description="How function tiers optimize performance and loading",
            level="intermediate",
            content="""
Functions are organized into 3 tiers based on usage frequency and importance:

Tier 1 - Essential (Always Loaded):
• Categories: core, git
• Token Cost: ~9,040 tokens
• Usage: 90%+ of sessions
• Cannot be unloaded

Tier 2 - Extended (Conditional Loading):
• Categories: analysis, quality, security, test, debug
• Token Cost: ~14,940 tokens
• Usage: 30-60% of sessions
• Loaded based on task detection or user request

Tier 3 - Specialized (On-Demand):
• Categories: external, infrastructure
• Token Cost: ~3,850 tokens
• Usage: <20% of sessions
• Loaded only when specifically needed

Commands:
• /tier-status - See current tier status
• /load-tier <number> - Force load a tier
• /unload-tier <number> - Unload a tier
            """,
            examples=[
                "/tier-status",
                "/load-tier 2",
                "/unload-tier 3",
            ],
            related_topics=["performance_optimization", "task_optimization"],
            tags=["tiers", "performance", "architecture"],
        )

        # Task Optimization
        topics["task_optimization"] = HelpTopic(
            name="task_optimization",
            title="Optimizing for Specific Tasks",
            description="How to optimize function loading for different work types",
            level="intermediate",
            content="""
The system can automatically optimize function loading for specific task types:

Available Task Types:
• debugging - Loads debug, analysis tools
• security - Loads security, analysis tools
• refactoring - Loads quality, analysis tools
• documentation - Loads quality, analysis tools
• testing - Loads test, quality tools
• minimal - Only essential functions
• comprehensive - All functions for complex tasks

Usage:
/optimize-for <task-type>

This automatically loads the most relevant function categories for your task
while keeping token usage reasonable.

Pro Tip: You can save these optimized configurations as profiles for quick reuse.
            """,
            examples=[
                "/optimize-for debugging",
                "/optimize-for security",
                "/optimize-for minimal",
                '/save-session-profile debug-work "Debugging optimization"',
            ],
            related_topics=["profile_management", "workflow_automation"],
            tags=["optimization", "tasks", "efficiency"],
        )

        # Profile Management
        topics["profile_management"] = HelpTopic(
            name="profile_management",
            title="Session Profiles and Automation",
            description="Save and reuse function loading configurations",
            level="intermediate",
            content="""
Session profiles let you save and quickly restore function loading configurations.

Creating Profiles:
1. Configure your functions (load categories, set performance mode)
2. Save: /save-session-profile <name> "<description>"
3. Optional: Add tags for organization

Using Profiles:
• Load: /load-session-profile <name>
• List: /list-profiles
• Filter: /list-profiles tag:<tag-name>

Profile Tips:
• Create profiles for different types of work
• Use descriptive names and tags
• Profiles track usage statistics
• Most-used profiles appear first in listings

Example Workflow:
1. /optimize-for debugging
2. /load-category security
3. /save-session-profile security-debug "Security debugging workflow"
4. Later: /load-session-profile security-debug
            """,
            examples=[
                '/save-session-profile auth-work "Authentication development"',
                "/list-profiles",
                "/load-session-profile auth-work",
                "/list-profiles tag:security",
            ],
            related_topics=["workflow_automation", "advanced_usage"],
            tags=["profiles", "automation", "workflow"],
        )

        # Performance Optimization
        topics["performance_optimization"] = HelpTopic(
            name="performance_optimization",
            title="Performance Monitoring and Optimization",
            description="Monitor and optimize function loading performance",
            level="advanced",
            content="""
Advanced performance monitoring and optimization features:

Performance Modes:
• conservative - Loads more functions for safety (default)
• balanced - Balance between performance and functionality
• aggressive - Minimal loading for maximum performance

Monitoring Commands:
• /function-stats - See loading statistics and usage
• /benchmark-loading - Run performance benchmark
• /clear-cache - Clear loading cache

Performance Tips:
1. Use /function-stats to identify unused categories
2. Create task-specific profiles to avoid over-loading
3. Use aggressive mode for routine tasks
4. Monitor cache hit rates for optimization opportunities

Performance Considerations:
• Each category has a token cost
• Loading affects response time
• Cache improves repeated operations
• Task detection reduces manual management
            """,
            examples=[
                "/function-stats",
                "/performance-mode aggressive",
                "/benchmark-loading",
                "/clear-cache",
            ],
            related_topics=["system_optimization", "advanced_usage"],
            tags=["performance", "monitoring", "optimization"],
        )

        # Troubleshooting
        topics["troubleshooting"] = HelpTopic(
            name="troubleshooting",
            title="Common Issues and Solutions",
            description="Troubleshoot common function loading issues",
            level="beginner",
            content="""
Common Issues and Solutions:

"Unknown category: xyz"
• Check available categories: /list-categories
• Use exact category names (case-sensitive)
• Try: /category-info <name> for details

"Cannot unload category 'core'"
• Core and git categories are always loaded for safety
• These provide essential functionality
• Focus on managing Tier 2 and 3 categories

"Profile 'name' not found"
• List available profiles: /list-profiles
• Check spelling and case sensitivity
• Profiles are saved locally

Performance Issues:
• Too many categories loaded: /list-categories loaded-only
• Clear cache: /clear-cache
• Use task optimization: /optimize-for <task>
• Switch to aggressive mode: /performance-mode aggressive

Getting Help:
• General help: /help
• Command help: /help <command>
• Examples: Every help topic includes examples
            """,
            examples=[
                "/list-categories",
                "/help load-category",
                "/function-stats",
                "/performance-mode conservative",
            ],
            related_topics=["basic_commands", "performance_optimization"],
            tags=["troubleshooting", "errors", "help"],
        )

        return topics

    def _initialize_learning_paths(self) -> dict[str, LearningPath]:
        """Initialize learning paths for different user types"""
        paths = {}

        paths["beginner"] = LearningPath(
            name="beginner",
            description="Get started with function loading control",
            level="beginner",
            topics=["basic_commands", "category_management", "tier_system"],
            estimated_time_minutes=15,
        )

        paths["power_user"] = LearningPath(
            name="power_user",
            description="Master advanced features and automation",
            level="advanced",
            topics=["task_optimization", "profile_management", "performance_optimization"],
            estimated_time_minutes=25,
            prerequisites=["beginner"],
        )

        paths["troubleshooter"] = LearningPath(
            name="troubleshooter",
            description="Learn to diagnose and fix issues",
            level="intermediate",
            topics=["troubleshooting", "performance_optimization", "system_optimization"],
            estimated_time_minutes=20,
            prerequisites=["beginner"],
        )

        return paths

    def get_contextual_help(
        self,
        context_topics: list[str],
        user_level: str = "beginner",
        help_style: str = "guided",
    ) -> dict[str, Any]:
        """Generate contextual help based on suggested topics"""
        relevant_topics = []

        for topic_name in context_topics:
            if topic_name in self.help_topics:
                topic = self.help_topics[topic_name]
                if self._is_appropriate_level(topic.level, user_level):
                    relevant_topics.append(topic)

        if not relevant_topics:
            # Fallback to basic help
            relevant_topics = [self.help_topics["basic_commands"]]

        # Format based on help style
        if help_style == "minimal":
            return self._format_minimal_help(relevant_topics)
        if help_style == "reference":
            return self._format_reference_help(relevant_topics)
        # guided
        return self._format_guided_help(relevant_topics)

    def _is_appropriate_level(self, topic_level: str, user_level: str) -> bool:
        """Check if topic level is appropriate for user level"""
        level_hierarchy = ["beginner", "intermediate", "advanced"]
        topic_index = level_hierarchy.index(topic_level)
        user_index = level_hierarchy.index(user_level)

        # Show topics at or below user level, plus one level above
        return topic_index <= user_index + 1

    def _format_guided_help(self, topics: list[HelpTopic]) -> dict[str, Any]:
        """Format help in guided, tutorial style"""
        primary_topic = topics[0]

        return {
            "style": "guided",
            "primary_topic": {
                "title": primary_topic.title,
                "description": primary_topic.description,
                "content": primary_topic.content,
                "examples": primary_topic.examples[:3],  # Limit examples
                "next_steps": primary_topic.related_topics[:2],
            },
            "related_topics": [
                {
                    "name": topic.name,
                    "title": topic.title,
                    "description": topic.description,
                }
                for topic in topics[1:3]  # Show 2 more related topics
            ],
            "quick_actions": self._generate_quick_actions(primary_topic),
        }

    def _format_reference_help(self, topics: list[HelpTopic]) -> dict[str, Any]:
        """Format help as comprehensive reference"""
        return {
            "style": "reference",
            "topics": [
                {
                    "name": topic.name,
                    "title": topic.title,
                    "content": topic.content,
                    "examples": topic.examples,
                    "related_topics": topic.related_topics,
                    "tags": topic.tags,
                }
                for topic in topics
            ],
        }

    def _format_minimal_help(self, topics: list[HelpTopic]) -> dict[str, Any]:
        """Format help as minimal quick reference"""
        primary_topic = topics[0]

        return {
            "style": "minimal",
            "quick_reference": {
                "title": primary_topic.title,
                "key_commands": primary_topic.examples[:2],
                "description": primary_topic.description,
            },
            "see_also": primary_topic.related_topics[:3],
        }

    def _generate_quick_actions(self, topic: HelpTopic) -> list[dict[str, str]]:
        """Generate quick action suggestions"""
        actions = []

        if topic.name == "basic_commands":
            actions = [
                {"action": "List categories", "command": "/list-categories"},
                {"action": "Check tier status", "command": "/tier-status"},
            ]
        elif topic.name == "category_management":
            actions = [
                {"action": "Load security tools", "command": "/load-category security"},
                {"action": "Get category info", "command": "/category-info analysis"},
            ]
        elif topic.name == "task_optimization":
            actions = [
                {"action": "Optimize for debugging", "command": "/optimize-for debugging"},
                {"action": "Save current setup", "command": '/save-session-profile my-profile "My workflow"'},
            ]

        return actions


class InteractiveHelpSystem:
    """Interactive help system with learning tracking"""

    def __init__(self, control_system: UserControlSystem) -> None:
        self.control_system = control_system
        self.context_analyzer = ContextAnalyzer()
        self.content_generator = HelpContentGenerator()
        self.user_progress = {}  # Would be persisted in real implementation

    def get_help(
        self,
        query: str | None = None,
        user_id: str = "default",
        context: dict[str, Any] | None = None,
    ) -> CommandResult:
        """Main help entry point"""

        # Get or create user progress
        if user_id not in self.user_progress:
            self.user_progress[user_id] = UserProgress(user_id=user_id)

        user = self.user_progress[user_id]

        # Handle specific help queries
        if query:
            return self._handle_specific_query(query, user)

        # Analyze context for suggestions
        context = context or {}
        recent_commands = context.get("recent_commands", [])
        recent_results = context.get("recent_results", [])
        current_state = context.get("current_state", {})

        suggested_topics = self.context_analyzer.analyze_context(
            recent_commands,
            recent_results,
            current_state,
        )

        # Generate contextual help
        help_content = self.content_generator.get_contextual_help(
            suggested_topics,
            user.current_level,
            user.preferred_help_style,
        )

        return CommandResult(
            success=True,
            message="Contextual help generated based on your recent activity",
            data={
                "help_content": help_content,
                "user_level": user.current_level,
                "suggested_topics": suggested_topics,
                "learning_opportunities": self._get_learning_opportunities(user),
            },
        )

    def _handle_specific_query(self, query: str, user: UserProgress) -> CommandResult:
        """Handle specific help queries"""
        query_lower = query.lower()

        # Check for command-specific help
        if query_lower.startswith("/") or "command" in query_lower:
            return self._get_command_help(query, user)

        # Check for topic-specific help
        topic_keywords = {
            "category": "category_management",
            "tier": "tier_system",
            "profile": "profile_management",
            "performance": "performance_optimization",
            "optimization": "task_optimization",
            "troubleshoot": "troubleshooting",
            "error": "troubleshooting",
        }

        for keyword, topic_name in topic_keywords.items():
            if keyword in query_lower:
                return self._get_topic_help(topic_name, user)

        # Fallback to basic help
        return self._get_topic_help("basic_commands", user)

    def _get_command_help(self, query: str, user: UserProgress) -> CommandResult:
        """Get help for specific commands"""
        # Extract command name
        command_match = re.search(r"/([a-z-]+)", query)
        if command_match:
            command = command_match.group(1)

            # Use the control system's help
            help_text = self.control_system.command_parser.get_command_help(command)

            return CommandResult(
                success=True,
                message=f"Help for command: /{command}",
                data={
                    "help_text": help_text,
                    "command": command,
                    "type": "command_help",
                },
            )

        # General command help
        help_text = self.control_system.command_parser.get_command_help()

        return CommandResult(
            success=True,
            message="General command help",
            data={
                "help_text": help_text,
                "type": "general_command_help",
            },
        )

    def _get_topic_help(self, topic_name: str, user: UserProgress) -> CommandResult:
        """Get help for specific topic"""
        if topic_name not in self.content_generator.help_topics:
            return CommandResult(
                success=False,
                message=f"Help topic not found: {topic_name}",
                suggestions=["Try: /help for general help"],
            )

        topic = self.content_generator.help_topics[topic_name]

        # Track user progress
        user.last_accessed[topic_name] = datetime.now()

        # Format based on user preferences
        help_content = self.content_generator._format_guided_help([topic])

        return CommandResult(
            success=True,
            message=f"Help: {topic.title}",
            data={
                "help_content": help_content,
                "topic": topic_name,
                "level": topic.level,
                "related_topics": topic.related_topics,
            },
        )

    def _get_learning_opportunities(self, user: UserProgress) -> list[dict[str, Any]]:
        """Get learning opportunities for user"""
        opportunities = []

        # Check learning paths
        for _path_name, path in self.content_generator.learning_paths.items():
            if self._is_path_appropriate(path, user):
                completed_topics = len([t for t in path.topics if t in user.completed_topics])
                progress_percent = (completed_topics / len(path.topics)) * 100

                opportunities.append(
                    {
                        "type": "learning_path",
                        "name": path.name,
                        "description": path.description,
                        "progress_percent": progress_percent,
                        "estimated_time_minutes": path.estimated_time_minutes,
                        "next_topic": path.topics[completed_topics] if completed_topics < len(path.topics) else None,
                    },
                )

        # Suggest topics based on recent activity
        recent_topics = set(user.last_accessed.keys())
        for topic_name, topic in self.content_generator.help_topics.items():
            if (
                topic_name not in recent_topics
                and topic_name not in user.completed_topics
                and self.content_generator._is_appropriate_level(topic.level, user.current_level)
            ):

                opportunities.append(
                    {
                        "type": "suggested_topic",
                        "name": topic_name,
                        "title": topic.title,
                        "description": topic.description,
                        "level": topic.level,
                    },
                )

        return opportunities[:5]  # Limit to top 5

    def _is_path_appropriate(self, path: LearningPath, user: UserProgress) -> bool:
        """Check if learning path is appropriate for user"""
        # Check prerequisites
        for prereq in path.prerequisites:
            if prereq not in user.completed_topics:
                return False

        # Check if already completed
        completed_topics = len([t for t in path.topics if t in user.completed_topics])
        return completed_topics != len(path.topics)

    def mark_topic_completed(self, user_id: str, topic_name: str) -> None:
        """Mark a topic as completed for user"""
        if user_id in self.user_progress:
            self.user_progress[user_id].completed_topics.add(topic_name)

    def rate_help_effectiveness(self, user_id: str, topic_name: str, rating: int) -> None:
        """Rate help effectiveness (1-5 scale)"""
        if user_id in self.user_progress:
            self.user_progress[user_id].help_effectiveness_ratings[topic_name] = rating


# Integration with existing command system
class HelpCommandIntegration:
    """Integrates help system with existing Claude command structure"""

    def __init__(self, help_system: InteractiveHelpSystem) -> None:
        self.help_system = help_system

    def create_help_commands(self) -> dict[str, Any]:
        """Create help commands for integration with existing command system"""
        return {
            "/help-function-loading": {
                "handler": self.help_function_loading,
                "description": "Comprehensive help for function loading system",
                "examples": [
                    "/help-function-loading",
                    "/help-function-loading categories",
                    "/help-function-loading optimization",
                ],
            },
            "/learning-path": {
                "handler": self.start_learning_path,
                "description": "Start guided learning path for function loading",
                "examples": [
                    "/learning-path beginner",
                    "/learning-path power_user",
                ],
            },
            "/help-context": {
                "handler": self.contextual_help,
                "description": "Get help based on recent activity",
                "examples": ["/help-context"],
            },
        }

    async def help_function_loading(self, args: list[str]) -> CommandResult:
        """Handle function loading help command"""
        query = " ".join(args) if args else None
        return self.help_system.get_help(query)

    async def start_learning_path(self, args: list[str]) -> CommandResult:
        """Start guided learning path"""
        if not args:
            # List available paths
            paths = self.help_system.content_generator.learning_paths
            return CommandResult(
                success=True,
                message="Available learning paths",
                data={
                    "paths": [
                        {
                            "name": path.name,
                            "description": path.description,
                            "level": path.level,
                            "estimated_time_minutes": path.estimated_time_minutes,
                        }
                        for path in paths.values()
                    ],
                },
            )

        path_name = args[0]
        if path_name not in self.help_system.content_generator.learning_paths:
            return CommandResult(
                success=False,
                message=f"Learning path not found: {path_name}",
                suggestions=list(self.help_system.content_generator.learning_paths.keys()),
            )

        path = self.help_system.content_generator.learning_paths[path_name]
        return CommandResult(
            success=True,
            message=f"Starting learning path: {path.description}",
            data={
                "path": path_name,
                "topics": path.topics,
                "estimated_time": path.estimated_time_minutes,
                "first_topic": path.topics[0] if path.topics else None,
            },
        )

    async def contextual_help(self, args: list[str]) -> CommandResult:
        """Provide contextual help based on recent activity"""
        # In real implementation, would get actual context from control system
        mock_context = {
            "recent_commands": ["/load-category security", "/tier-status"],
            "recent_results": [],
            "current_state": {"beginner_user": True},
        }

        return self.help_system.get_help(context=mock_context)


# Example usage and testing
if __name__ == "__main__":
    from .task_detection import TaskDetectionSystem
    from .task_detection_config import ConfigManager

    # Initialize systems
    detection_system = TaskDetectionSystem()
    config_manager = ConfigManager()
    control_system = UserControlSystem(detection_system, config_manager)

    # Create help system
    help_system = InteractiveHelpSystem(control_system)
    help_integration = HelpCommandIntegration(help_system)

    # Test help queries
    test_queries = [
        None,  # Contextual help
        "categories",
        "/load-category",
        "optimization",
        "troubleshooting",
    ]

    for query in test_queries:
        result = help_system.get_help(query)

        if result.data and "help_content" in result.data:
            content = result.data["help_content"]

            if "primary_topic" in content:
                pass
