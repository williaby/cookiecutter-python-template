"""
Analytics Engine for Dynamic Function Loading

Tracks usage patterns, provides optimization insights, and enables
learning-based improvements to the function loading system.
"""

import asyncio
import json
import logging
import sqlite3
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class UsageEvent:
    """Individual usage event tracking"""
    timestamp: datetime
    event_type: str  # command, function_call, category_load, etc.
    event_data: dict[str, Any]
    user_id: str
    session_id: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionMetrics:
    """Metrics for a user session"""
    session_id: str
    user_id: str
    start_time: datetime
    end_time: datetime | None = None
    commands_executed: int = 0
    functions_used: set[str] = field(default_factory=set)
    categories_loaded: set[str] = field(default_factory=set)
    performance_mode: str = "conservative"
    total_tokens_used: int = 0
    errors_encountered: int = 0
    help_requests: int = 0
    optimization_applied: bool = False


@dataclass
class UserBehaviorPattern:
    """Identified user behavior pattern"""
    pattern_id: str
    pattern_type: str  # sequential, temporal, preference, etc.
    description: str
    frequency: float  # How often this pattern occurs
    confidence: float  # Confidence in pattern detection
    associated_categories: list[str]
    typical_sequence: list[str]
    performance_impact: dict[str, float]
    suggested_optimizations: list[str]


@dataclass
class OptimizationInsight:
    """Optimization recommendation based on analysis"""
    insight_id: str
    insight_type: str  # performance, workflow, learning, etc.
    title: str
    description: str
    impact_estimate: str  # high, medium, low
    effort_estimate: str  # easy, moderate, complex
    suggested_actions: list[str]
    evidence: dict[str, Any]
    applicable_users: list[str]


class UsageTracker:
    """Tracks and stores usage events"""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or Path("analytics.db")
        self.session_events = deque(maxlen=10000)  # In-memory buffer
        self.active_sessions: dict[str, SessionMetrics] = {}
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Initialize SQLite database for persistent storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS usage_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_data TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    context TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_metrics (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    commands_executed INTEGER DEFAULT 0,
                    functions_used TEXT,
                    categories_loaded TEXT,
                    performance_mode TEXT DEFAULT 'conservative',
                    total_tokens_used INTEGER DEFAULT 0,
                    errors_encountered INTEGER DEFAULT 0,
                    help_requests INTEGER DEFAULT 0,
                    optimization_applied BOOLEAN DEFAULT FALSE
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_timestamp
                ON usage_events(timestamp)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_user
                ON usage_events(user_id)
            """)

    def track_event(self, event_type: str, event_data: dict[str, Any],
                   user_id: str, session_id: str,
                   context: dict[str, Any] | None = None) -> None:
        """Track a usage event"""
        event = UsageEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            event_data=event_data,
            user_id=user_id,
            session_id=session_id,
            context=context or {},
        )

        # Add to in-memory buffer
        self.session_events.append(event)

        # Update session metrics
        self._update_session_metrics(event)

        # Persist to database (async in real implementation)
        self._persist_event(event)

    def _update_session_metrics(self, event: UsageEvent) -> None:
        """Update session metrics based on event"""
        session_id = event.session_id

        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = SessionMetrics(
                session_id=session_id,
                user_id=event.user_id,
                start_time=event.timestamp,
            )

        session = self.active_sessions[session_id]

        # Update based on event type
        if event.event_type == "command_executed":
            session.commands_executed += 1

            if not event.event_data.get("success", True):
                session.errors_encountered += 1

        elif event.event_type == "function_called":
            session.functions_used.add(event.event_data.get("function_name", "unknown"))

        elif event.event_type == "category_loaded":
            category = event.event_data.get("category")
            if category:
                session.categories_loaded.add(category)
                # Estimate token usage
                token_cost = event.event_data.get("token_cost", 0)
                session.total_tokens_used += token_cost

        elif event.event_type == "help_requested":
            session.help_requests += 1

        elif event.event_type == "optimization_applied":
            session.optimization_applied = True

        elif event.event_type == "performance_mode_changed":
            session.performance_mode = event.event_data.get("mode", "conservative")

    def _persist_event(self, event: UsageEvent) -> None:
        """Persist event to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO usage_events
                    (timestamp, event_type, event_data, user_id, session_id, context)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    event.timestamp.isoformat(),
                    event.event_type,
                    json.dumps(event.event_data),
                    event.user_id,
                    event.session_id,
                    json.dumps(event.context),
                ))
        except Exception as e:
            logger.error(f"Failed to persist event: {e}")

    def end_session(self, session_id: str) -> SessionMetrics | None:
        """End a session and persist metrics"""
        if session_id not in self.active_sessions:
            return None

        session = self.active_sessions[session_id]
        session.end_time = datetime.now()

        # Persist session metrics
        self._persist_session_metrics(session)

        # Remove from active sessions
        return self.active_sessions.pop(session_id)

    def _persist_session_metrics(self, session: SessionMetrics) -> None:
        """Persist session metrics to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO session_metrics
                    (session_id, user_id, start_time, end_time, commands_executed,
                     functions_used, categories_loaded, performance_mode,
                     total_tokens_used, errors_encountered, help_requests,
                     optimization_applied)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session.session_id,
                    session.user_id,
                    session.start_time.isoformat(),
                    session.end_time.isoformat() if session.end_time else None,
                    session.commands_executed,
                    json.dumps(list(session.functions_used)),
                    json.dumps(list(session.categories_loaded)),
                    session.performance_mode,
                    session.total_tokens_used,
                    session.errors_encountered,
                    session.help_requests,
                    session.optimization_applied,
                ))
        except Exception as e:
            logger.error(f"Failed to persist session metrics: {e}")

    def get_recent_events(self, hours: int = 24,
                         event_type: str | None = None,
                         user_id: str | None = None) -> list[UsageEvent]:
        """Get recent events with optional filtering"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        filtered_events = []
        for event in self.session_events:
            if event.timestamp < cutoff_time:
                continue

            if event_type and event.event_type != event_type:
                continue

            if user_id and event.user_id != user_id:
                continue

            filtered_events.append(event)

        return filtered_events

    def get_session_summary(self, session_id: str) -> dict[str, Any] | None:
        """Get summary of session metrics"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
        else:
            # Try to load from database
            session = self._load_session_from_db(session_id)

        if not session:
            return None

        duration = None
        if session.end_time:
            duration = (session.end_time - session.start_time).total_seconds()

        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "duration_seconds": duration,
            "commands_executed": session.commands_executed,
            "unique_functions_used": len(session.functions_used),
            "categories_loaded": list(session.categories_loaded),
            "performance_mode": session.performance_mode,
            "total_tokens_used": session.total_tokens_used,
            "error_rate": session.errors_encountered / max(1, session.commands_executed),
            "help_requests": session.help_requests,
            "optimization_applied": session.optimization_applied,
        }

    def _load_session_from_db(self, session_id: str) -> SessionMetrics | None:
        """Load session metrics from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM session_metrics WHERE session_id = ?
                """, (session_id,))

                row = cursor.fetchone()
                if not row:
                    return None

                return SessionMetrics(
                    session_id=row[0],
                    user_id=row[1],
                    start_time=datetime.fromisoformat(row[2]),
                    end_time=datetime.fromisoformat(row[3]) if row[3] else None,
                    commands_executed=row[4],
                    functions_used=set(json.loads(row[5])),
                    categories_loaded=set(json.loads(row[6])),
                    performance_mode=row[7],
                    total_tokens_used=row[8],
                    errors_encountered=row[9],
                    help_requests=row[10],
                    optimization_applied=bool(row[11]),
                )
        except Exception as e:
            logger.error(f"Failed to load session from database: {e}")
            return None


class PatternDetector:
    """Detects usage patterns from tracked events"""

    def __init__(self, usage_tracker: UsageTracker) -> None:
        self.usage_tracker = usage_tracker
        self.min_pattern_frequency = 0.3  # 30% of sessions
        self.min_confidence = 0.7

    def detect_patterns(self, user_id: str | None = None,
                       days_back: int = 30) -> list[UserBehaviorPattern]:
        """Detect behavior patterns from usage data"""
        patterns = []

        # Get events for analysis
        events = self.usage_tracker.get_recent_events(
            hours=days_back * 24, user_id=user_id,
        )

        if len(events) < 10:  # Need sufficient data
            return patterns

        # Detect sequential patterns
        patterns.extend(self._detect_sequential_patterns(events))

        # Detect temporal patterns
        patterns.extend(self._detect_temporal_patterns(events))

        # Detect preference patterns
        patterns.extend(self._detect_preference_patterns(events))

        # Detect workflow patterns
        patterns.extend(self._detect_workflow_patterns(events))

        return patterns

    def _detect_sequential_patterns(self, events: list[UsageEvent]) -> list[UserBehaviorPattern]:
        """Detect sequential command patterns"""
        patterns = []

        # Group events by session
        session_commands = defaultdict(list)
        for event in events:
            if event.event_type == "command_executed":
                command = event.event_data.get("command", "unknown")
                session_commands[event.session_id].append(command)

        # Find common sequences
        sequence_counter = Counter()
        for commands in session_commands.values():
            if len(commands) >= 2:
                for i in range(len(commands) - 1):
                    sequence = (commands[i], commands[i + 1])
                    sequence_counter[sequence] += 1

        # Identify significant patterns
        total_sessions = len(session_commands)
        for sequence, count in sequence_counter.items():
            frequency = count / total_sessions

            if frequency >= self.min_pattern_frequency:
                patterns.append(UserBehaviorPattern(
                    pattern_id=f"seq_{hash(sequence)}",
                    pattern_type="sequential",
                    description=f"Users often run '{sequence[0]}' followed by '{sequence[1]}'",
                    frequency=frequency,
                    confidence=min(frequency * 2, 1.0),
                    associated_categories=self._infer_categories_from_commands(sequence),
                    typical_sequence=list(sequence),
                    performance_impact={"token_savings": frequency * 500},
                    suggested_optimizations=[
                        f"Create macro for {sequence[0]} + {sequence[1]}",
                        "Preload categories for this sequence",
                    ],
                ))

        return patterns

    def _detect_temporal_patterns(self, events: list[UsageEvent]) -> list[UserBehaviorPattern]:
        """Detect time-based usage patterns"""
        patterns = []

        # Group events by hour of day
        hourly_usage = defaultdict(list)
        for event in events:
            hour = event.timestamp.hour
            if event.event_type == "category_loaded":
                category = event.event_data.get("category")
                if category:
                    hourly_usage[hour].append(category)

        # Find peak usage times
        peak_hours = []
        total_events = sum(len(categories) for categories in hourly_usage.values())

        for hour, categories in hourly_usage.items():
            if len(categories) / total_events > 0.15:  # 15% of daily usage
                peak_hours.append(hour)

                # Most common categories during this hour
                category_counts = Counter(categories)
                top_categories = [cat for cat, _ in category_counts.most_common(3)]

                patterns.append(UserBehaviorPattern(
                    pattern_id=f"temporal_{hour}",
                    pattern_type="temporal",
                    description=f"High usage at {hour}:00 with focus on {', '.join(top_categories)}",
                    frequency=len(categories) / total_events,
                    confidence=0.8,
                    associated_categories=top_categories,
                    typical_sequence=[],
                    performance_impact={"preload_benefit": len(categories) * 100},
                    suggested_optimizations=[
                        f"Preload {', '.join(top_categories)} categories at {hour}:00",
                        f"Create time-based profile for {hour}:00-{(hour+2)%24}:00",
                    ],
                ))

        return patterns

    def _detect_preference_patterns(self, events: list[UsageEvent]) -> list[UserBehaviorPattern]:
        """Detect user preference patterns"""
        patterns = []

        # Analyze performance mode preferences
        mode_usage = Counter()
        for event in events:
            if event.event_type == "performance_mode_changed":
                mode = event.event_data.get("mode")
                if mode:
                    mode_usage[mode] += 1

        if mode_usage:
            preferred_mode = mode_usage.most_common(1)[0][0]
            frequency = mode_usage[preferred_mode] / sum(mode_usage.values())

            if frequency > 0.6:  # Strong preference
                patterns.append(UserBehaviorPattern(
                    pattern_id=f"preference_mode_{preferred_mode}",
                    pattern_type="preference",
                    description=f"Strong preference for {preferred_mode} performance mode",
                    frequency=frequency,
                    confidence=frequency,
                    associated_categories=[],
                    typical_sequence=[],
                    performance_impact={"consistency_benefit": frequency * 200},
                    suggested_optimizations=[
                        f"Set default performance mode to {preferred_mode}",
                        f"Create profile with {preferred_mode} mode as default",
                    ],
                ))

        # Analyze category preferences
        category_usage = Counter()
        for event in events:
            if event.event_type == "category_loaded":
                category = event.event_data.get("category")
                if category:
                    category_usage[category] += 1

        if category_usage:
            total_loads = sum(category_usage.values())
            for category, count in category_usage.items():
                frequency = count / total_loads

                if frequency > 0.3:  # Frequently used category
                    patterns.append(UserBehaviorPattern(
                        pattern_id=f"preference_category_{category}",
                        pattern_type="preference",
                        description=f"Frequently uses {category} category",
                        frequency=frequency,
                        confidence=frequency,
                        associated_categories=[category],
                        typical_sequence=[],
                        performance_impact={"preload_benefit": count * 50},
                        suggested_optimizations=[
                            f"Add {category} to default loading profile",
                            f"Consider auto-loading {category} for this user",
                        ],
                    ))

        return patterns

    def _detect_workflow_patterns(self, events: list[UsageEvent]) -> list[UserBehaviorPattern]:
        """Detect workflow-specific patterns"""
        patterns = []

        # Group events by session and look for optimization patterns
        session_optimizations = defaultdict(list)
        for event in events:
            if event.event_type == "optimization_applied":
                task_type = event.event_data.get("task_type")
                if task_type:
                    session_optimizations[event.session_id].append(task_type)

        # Find common optimization workflows
        optimization_counter = Counter()
        for optimizations in session_optimizations.values():
            for opt in optimizations:
                optimization_counter[opt] += 1

        total_sessions_with_opt = len(session_optimizations)
        if total_sessions_with_opt > 0:
            for task_type, count in optimization_counter.items():
                frequency = count / total_sessions_with_opt

                if frequency > 0.2:  # 20% of optimization sessions
                    patterns.append(UserBehaviorPattern(
                        pattern_id=f"workflow_{task_type}",
                        pattern_type="workflow",
                        description=f"Frequently optimizes for {task_type} tasks",
                        frequency=frequency,
                        confidence=frequency,
                        associated_categories=self._get_optimization_categories(task_type),
                        typical_sequence=[],
                        performance_impact={"workflow_efficiency": frequency * 300},
                        suggested_optimizations=[
                            f"Create dedicated profile for {task_type} workflow",
                            f"Auto-suggest {task_type} optimization",
                            f"Set up shortcuts for {task_type} tasks",
                        ],
                    ))

        return patterns

    def _infer_categories_from_commands(self, commands: tuple[str, ...]) -> list[str]:
        """Infer likely categories from command sequence"""
        command_to_category = {
            "load-category": ["management"],
            "optimize-for": ["optimization"],
            "save-session-profile": ["automation"],
            "function-stats": ["monitoring"],
            "tier-status": ["monitoring"],
        }

        categories = set()
        for command in commands:
            if command in command_to_category:
                categories.update(command_to_category[command])

        return list(categories)

    def _get_optimization_categories(self, task_type: str) -> list[str]:
        """Get categories associated with task optimization"""
        task_categories = {
            "debugging": ["debug", "analysis"],
            "security": ["security", "analysis"],
            "refactoring": ["quality", "analysis"],
            "testing": ["test", "quality"],
            "documentation": ["quality", "analysis"],
        }

        return task_categories.get(task_type, [])


class InsightGenerator:
    """Generates optimization insights from usage patterns"""

    def __init__(self, usage_tracker: UsageTracker, pattern_detector: PatternDetector) -> None:
        self.usage_tracker = usage_tracker
        self.pattern_detector = pattern_detector

    def generate_insights(self, user_id: str | None = None) -> list[OptimizationInsight]:
        """Generate optimization insights for user or system"""
        insights = []

        # Detect patterns first
        patterns = self.pattern_detector.detect_patterns(user_id)

        # Generate performance insights
        insights.extend(self._generate_performance_insights(patterns, user_id))

        # Generate workflow insights
        insights.extend(self._generate_workflow_insights(patterns, user_id))

        # Generate learning insights
        insights.extend(self._generate_learning_insights(user_id))

        # Generate system insights
        if not user_id:  # System-wide insights
            insights.extend(self._generate_system_insights(patterns))

        return sorted(insights, key=lambda i: self._calculate_insight_priority(i), reverse=True)

    def _generate_performance_insights(self, patterns: list[UserBehaviorPattern],
                                     user_id: str | None) -> list[OptimizationInsight]:
        """Generate performance optimization insights"""
        insights = []

        # Look for high-token usage patterns
        high_usage_patterns = [p for p in patterns
                             if p.performance_impact.get("token_savings", 0) > 200]

        for pattern in high_usage_patterns:
            if pattern.pattern_type == "sequential":
                insights.append(OptimizationInsight(
                    insight_id=f"perf_seq_{pattern.pattern_id}",
                    insight_type="performance",
                    title="Command Sequence Optimization Opportunity",
                    description=f"Create macro for frequent sequence: {' → '.join(pattern.typical_sequence)}",
                    impact_estimate="medium",
                    effort_estimate="easy",
                    suggested_actions=[
                        f"Create profile with optimized loading for {' + '.join(pattern.typical_sequence)}",
                        "Consider adding sequence shortcuts",
                        "Enable predictive loading for this pattern",
                    ],
                    evidence={
                        "frequency": pattern.frequency,
                        "estimated_token_savings": pattern.performance_impact.get("token_savings", 0),
                        "sequence": pattern.typical_sequence,
                    },
                    applicable_users=[user_id] if user_id else ["system-wide"],
                ))

        # Check for over-loading patterns
        category_patterns = [p for p in patterns if p.pattern_type == "preference"]
        loaded_categories = [p.associated_categories for p in category_patterns]
        total_categories = len({cat for cats in loaded_categories for cat in cats})

        if total_categories > 6:  # Many categories loaded
            insights.append(OptimizationInsight(
                insight_id="perf_overload",
                insight_type="performance",
                title="Potential Over-Loading Detected",
                description=f"You're loading {total_categories} categories. Consider task-specific optimization.",
                impact_estimate="high",
                effort_estimate="easy",
                suggested_actions=[
                    "Use /optimize-for <task-type> for focused loading",
                    "Create task-specific profiles instead of loading everything",
                    "Review actual function usage with /function-stats",
                ],
                evidence={
                    "categories_loaded": total_categories,
                    "optimization_potential": min(50, (total_categories - 4) * 10),
                },
                applicable_users=[user_id] if user_id else ["high-usage-users"],
            ))

        return insights

    def _generate_workflow_insights(self, patterns: list[UserBehaviorPattern],
                                  user_id: str | None) -> list[OptimizationInsight]:
        """Generate workflow optimization insights"""
        insights = []

        # Look for workflow patterns
        workflow_patterns = [p for p in patterns if p.pattern_type == "workflow"]

        for pattern in workflow_patterns:
            if pattern.frequency > 0.3:  # Frequent workflow
                insights.append(OptimizationInsight(
                    insight_id=f"workflow_{pattern.pattern_id}",
                    insight_type="workflow",
                    title=f"Automate {pattern.description.split()[-2]} Workflow",
                    description=f"Create dedicated workflow automation for {pattern.description}",
                    impact_estimate="high",
                    effort_estimate="moderate",
                    suggested_actions=[
                        f"Create profile optimized for {pattern.description}",
                        "Set up keyboard shortcuts for this workflow",
                        "Enable auto-optimization detection for this task type",
                    ],
                    evidence={
                        "frequency": pattern.frequency,
                        "workflow_efficiency_gain": pattern.performance_impact.get("workflow_efficiency", 0),
                        "associated_categories": pattern.associated_categories,
                    },
                    applicable_users=[user_id] if user_id else ["workflow-users"],
                ))

        # Check for temporal optimization opportunities
        temporal_patterns = [p for p in patterns if p.pattern_type == "temporal"]

        for pattern in temporal_patterns:
            if pattern.frequency > 0.15:  # Significant time-based usage
                insights.append(OptimizationInsight(
                    insight_id=f"temporal_{pattern.pattern_id}",
                    insight_type="workflow",
                    title="Time-Based Loading Optimization",
                    description=pattern.description,
                    impact_estimate="medium",
                    effort_estimate="easy",
                    suggested_actions=[
                        "Create time-based loading profiles",
                        "Enable automatic category preloading",
                        "Set up scheduled optimization",
                    ],
                    evidence={
                        "time_pattern": pattern.description,
                        "preload_benefit": pattern.performance_impact.get("preload_benefit", 0),
                        "categories": pattern.associated_categories,
                    },
                    applicable_users=[user_id] if user_id else ["time-pattern-users"],
                ))

        return insights

    def _generate_learning_insights(self, user_id: str | None) -> list[OptimizationInsight]:
        """Generate learning and education insights"""
        insights = []

        if not user_id:
            return insights  # Only for specific users

        # Get recent session data
        recent_events = self.usage_tracker.get_recent_events(hours=168, user_id=user_id)  # 1 week

        # Check help request patterns
        help_requests = [e for e in recent_events if e.event_type == "help_requested"]
        error_events = [e for e in recent_events if e.event_type == "command_executed"
                       and not e.event_data.get("success", True)]

        if len(help_requests) > 5:  # Many help requests
            insights.append(OptimizationInsight(
                insight_id="learning_help_frequency",
                insight_type="learning",
                title="Learning Opportunity Detected",
                description="You've requested help frequently. Consider guided learning paths.",
                impact_estimate="medium",
                effort_estimate="easy",
                suggested_actions=[
                    "Start with /learning-path beginner",
                    "Review basic commands with /help-function-loading",
                    "Practice with guided examples",
                ],
                evidence={
                    "help_requests": len(help_requests),
                    "error_rate": len(error_events) / max(1, len(recent_events)),
                },
                applicable_users=[user_id],
            ))

        if len(error_events) > len(recent_events) * 0.2:  # High error rate
            insights.append(OptimizationInsight(
                insight_id="learning_error_rate",
                insight_type="learning",
                title="High Error Rate Detected",
                description="Consider reviewing command syntax and examples.",
                impact_estimate="high",
                effort_estimate="easy",
                suggested_actions=[
                    "Review command examples with /help <command>",
                    "Use tab completion for command parameters",
                    "Start with simpler commands and build up",
                ],
                evidence={
                    "error_rate": len(error_events) / len(recent_events),
                    "common_errors": [e.event_data.get("error_type") for e in error_events],
                },
                applicable_users=[user_id],
            ))

        return insights

    def _generate_system_insights(self, patterns: list[UserBehaviorPattern]) -> list[OptimizationInsight]:
        """Generate system-wide optimization insights"""
        insights = []

        # Analyze most common patterns across all users
        pattern_frequency = Counter()
        for pattern in patterns:
            pattern_frequency[pattern.pattern_type] += pattern.frequency

        # System-wide sequential patterns
        sequential_patterns = [p for p in patterns if p.pattern_type == "sequential"]
        if len(sequential_patterns) > 3:  # Multiple sequential patterns
            insights.append(OptimizationInsight(
                insight_id="system_sequential_optimization",
                insight_type="performance",
                title="System-Wide Sequential Pattern Optimization",
                description="Multiple users show similar command sequences. Consider system optimization.",
                impact_estimate="high",
                effort_estimate="complex",
                suggested_actions=[
                    "Implement predictive loading for common sequences",
                    "Create system-wide command macros",
                    "Add intelligent suggestions based on usage patterns",
                ],
                evidence={
                    "pattern_count": len(sequential_patterns),
                    "common_sequences": [p.typical_sequence for p in sequential_patterns[:3]],
                },
                applicable_users=["system-wide"],
            ))

        # Category usage analysis
        category_usage = Counter()
        for pattern in patterns:
            for category in pattern.associated_categories:
                category_usage[category] += pattern.frequency

        underused_categories = [cat for cat, freq in category_usage.items() if freq < 0.1]
        if underused_categories:
            insights.append(OptimizationInsight(
                insight_id="system_underused_categories",
                insight_type="performance",
                title="Underused Categories Detected",
                description=f"Categories {', '.join(underused_categories)} are rarely used.",
                impact_estimate="medium",
                effort_estimate="easy",
                suggested_actions=[
                    "Consider moving underused categories to higher tiers",
                    "Review category definitions and groupings",
                    "Provide better documentation for underused features",
                ],
                evidence={
                    "underused_categories": underused_categories,
                    "usage_frequencies": dict(category_usage),
                },
                applicable_users=["system-wide"],
            ))

        return insights

    def _calculate_insight_priority(self, insight: OptimizationInsight) -> float:
        """Calculate priority score for insight"""
        impact_scores = {"high": 3, "medium": 2, "low": 1}
        effort_scores = {"easy": 3, "moderate": 2, "complex": 1}  # Inverse - easier is better

        impact_score = impact_scores.get(insight.impact_estimate, 1)
        effort_score = effort_scores.get(insight.effort_estimate, 1)

        # Priority = Impact / Effort ratio
        return impact_score * effort_score


class AnalyticsEngine:
    """Main analytics engine orchestrating all components"""

    def __init__(self, db_path: Path | None = None) -> None:
        self.usage_tracker = UsageTracker(db_path)
        self.pattern_detector = PatternDetector(self.usage_tracker)
        self.insight_generator = InsightGenerator(self.usage_tracker, self.pattern_detector)

        # Performance metrics
        self.last_analysis_time = datetime.now()
        self.analysis_cache = {}
        self.cache_ttl = timedelta(hours=1)

    def track_user_action(self, action_type: str, action_data: dict[str, Any],
                         user_id: str, session_id: str,
                         context: dict[str, Any] | None = None) -> None:
        """Track a user action"""
        self.usage_tracker.track_event(
            event_type=action_type,
            event_data=action_data,
            user_id=user_id,
            session_id=session_id,
            context=context,
        )

    def get_user_analytics(self, user_id: str, days_back: int = 7) -> dict[str, Any]:
        """Get comprehensive analytics for a user"""
        cache_key = f"user_analytics_{user_id}_{days_back}"

        # Check cache
        if (cache_key in self.analysis_cache and
            datetime.now() - self.analysis_cache[cache_key]["timestamp"] < self.cache_ttl):
            return self.analysis_cache[cache_key]["data"]

        # Generate fresh analytics
        recent_events = self.usage_tracker.get_recent_events(
            hours=days_back * 24, user_id=user_id,
        )

        patterns = self.pattern_detector.detect_patterns(user_id, days_back)
        insights = self.insight_generator.generate_insights(user_id)

        # Calculate usage statistics
        command_stats = Counter()
        category_stats = Counter()
        error_count = 0

        for event in recent_events:
            if event.event_type == "command_executed":
                command = event.event_data.get("command", "unknown")
                command_stats[command] += 1

                if not event.event_data.get("success", True):
                    error_count += 1

            elif event.event_type == "category_loaded":
                category = event.event_data.get("category", "unknown")
                category_stats[category] += 1

        analytics = {
            "user_id": user_id,
            "analysis_period_days": days_back,
            "total_events": len(recent_events),
            "unique_commands": len(command_stats),
            "most_used_commands": dict(command_stats.most_common(5)),
            "category_usage": dict(category_stats),
            "error_rate": error_count / max(1, len(recent_events)),
            "patterns_detected": len(patterns),
            "behavior_patterns": [
                {
                    "type": pattern.pattern_type,
                    "description": pattern.description,
                    "frequency": pattern.frequency,
                    "confidence": pattern.confidence,
                    "optimizations": pattern.suggested_optimizations,
                } for pattern in patterns
            ],
            "optimization_insights": [
                {
                    "type": insight.insight_type,
                    "title": insight.title,
                    "description": insight.description,
                    "impact": insight.impact_estimate,
                    "effort": insight.effort_estimate,
                    "actions": insight.suggested_actions,
                } for insight in insights
            ],
            "recommendations": self._generate_recommendations(patterns, insights),
        }

        # Cache results
        self.analysis_cache[cache_key] = {
            "data": analytics,
            "timestamp": datetime.now(),
        }

        return analytics

    def get_system_analytics(self) -> dict[str, Any]:
        """Get system-wide analytics"""
        # Get all recent events (system-wide)
        recent_events = self.usage_tracker.get_recent_events(hours=168)  # 1 week

        patterns = self.pattern_detector.detect_patterns(days_back=7)
        insights = self.insight_generator.generate_insights()

        # User activity analysis
        user_activity = defaultdict(int)
        command_popularity = Counter()
        category_popularity = Counter()

        for event in recent_events:
            user_activity[event.user_id] += 1

            if event.event_type == "command_executed":
                command = event.event_data.get("command", "unknown")
                command_popularity[command] += 1

            elif event.event_type == "category_loaded":
                category = event.event_data.get("category", "unknown")
                category_popularity[category] += 1

        return {
            "analysis_period": "7 days",
            "total_events": len(recent_events),
            "active_users": len(user_activity),
            "avg_events_per_user": sum(user_activity.values()) / max(1, len(user_activity)),
            "most_popular_commands": dict(command_popularity.most_common(10)),
            "category_popularity": dict(category_popularity.most_common()),
            "system_patterns": [
                {
                    "type": pattern.pattern_type,
                    "description": pattern.description,
                    "frequency": pattern.frequency,
                    "users_affected": "multiple" if pattern.frequency > 0.2 else "few",
                } for pattern in patterns
            ],
            "system_insights": [
                {
                    "type": insight.insight_type,
                    "title": insight.title,
                    "description": insight.description,
                    "impact": insight.impact_estimate,
                    "applicable_users": insight.applicable_users,
                } for insight in insights if "system-wide" in insight.applicable_users
            ],
        }

    def _generate_recommendations(self, patterns: list[UserBehaviorPattern],
                                insights: list[OptimizationInsight]) -> list[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # From patterns
        high_frequency_patterns = [p for p in patterns if p.frequency > 0.4]
        for pattern in high_frequency_patterns:
            recommendations.extend(pattern.suggested_optimizations[:2])  # Top 2

        # From insights
        high_impact_insights = [i for i in insights
                              if i.impact_estimate == "high" and i.effort_estimate in ["easy", "moderate"]]
        for insight in high_impact_insights:
            recommendations.extend(insight.suggested_actions[:2])  # Top 2

        # Remove duplicates and limit
        unique_recommendations = list(dict.fromkeys(recommendations))
        return unique_recommendations[:8]  # Top 8 recommendations

    def export_analytics(self, user_id: str | None = None,
                        format: str = "json") -> dict[str, Any]:
        """Export analytics data for external analysis"""
        data = self.get_user_analytics(user_id) if user_id else self.get_system_analytics()

        return {
            "export_timestamp": datetime.now().isoformat(),
            "export_type": "user" if user_id else "system",
            "format_version": "1.0",
            "data": data,
        }



# Example usage and testing
if __name__ == "__main__":
    async def main() -> None:
        # Initialize analytics engine
        analytics = AnalyticsEngine()

        # Simulate user activity
        user_id = "test_user"
        session_id = "test_session"

        # Track some events
        analytics.track_user_action(
            "command_executed",
            {"command": "load-category", "category": "security", "success": True},
            user_id, session_id,
        )

        analytics.track_user_action(
            "category_loaded",
            {"category": "security", "token_cost": 1900},
            user_id, session_id,
        )

        analytics.track_user_action(
            "optimization_applied",
            {"task_type": "debugging"},
            user_id, session_id,
        )

        # Get analytics
        analytics.get_user_analytics(user_id)

        # Get system analytics
        analytics.get_system_analytics()

    asyncio.run(main())
