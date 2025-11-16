"""
Task Detection Algorithm for Dynamic Function Loading

Sophisticated multi-modal task detection system that analyzes user intent
to dynamically load appropriate function categories, achieving 70%+ token
reduction while maintaining functionality.
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result of task detection analysis"""

    categories: dict[str, bool]
    confidence_scores: dict[str, float]
    detection_time_ms: float
    signals_used: dict[str, dict[str, float]]
    fallback_applied: str | None = None


@dataclass
class SignalData:
    """Container for detection signals"""

    signal_type: str
    category_scores: dict[str, float]
    confidence: float
    source: str


class KeywordAnalyzer:
    """Analyzes queries for category-specific keywords"""

    def __init__(self) -> None:
        self.keyword_patterns = {
            "git": {
                "direct": ["git", "commit", "branch", "merge", "pull", "push", "checkout", "clone"],
                "contextual": ["repository", "version control", "staging", "diff", "remote"],
                "action": ["add", "status", "log", "reset", "revert", "cherry-pick"],
                "confidence": 0.9,
            },
            "debug": {
                "direct": ["debug", "error", "bug", "issue", "problem", "broken", "exception"],
                "contextual": ["trace", "investigate", "root cause", "troubleshoot", "stack trace"],
                "action": ["fix", "solve", "diagnose", "analyze", "investigate"],
                "confidence": 0.85,
            },
            "test": {
                "direct": ["test", "testing", "spec", "coverage", "pytest", "unittest"],
                "contextual": ["unit test", "integration", "mock", "assertion", "test case"],
                "action": ["validate", "verify", "check", "ensure", "assert"],
                "confidence": 0.8,
            },
            "security": {
                "direct": ["security", "vulnerability", "audit", "scan", "auth", "permission"],
                "contextual": ["role", "token", "encrypt", "decrypt", "certificate", "ssl"],
                "action": ["secure", "protect", "validate", "authorize", "authenticate"],
                "confidence": 0.85,
            },
            "analysis": {
                "direct": ["analyze", "review", "understand", "investigate", "examine"],
                "contextual": ["pattern", "architecture", "design", "structure", "flow"],
                "action": ["study", "evaluate", "assess", "explore", "research", "improve"],
                "confidence": 0.75,
            },
            "quality": {
                "direct": ["refactor", "clean", "improve", "optimize", "enhance"],
                "contextual": ["code quality", "best practice", "maintainability", "readability"],
                "action": ["restructure", "modernize", "organize", "simplify"],
                "confidence": 0.8,
            },
        }

    def analyze(self, query: str) -> dict[str, float]:
        """Analyze query for keyword signals"""
        query_lower = query.lower()
        category_scores = defaultdict(float)

        for category, patterns in self.keyword_patterns.items():
            # Direct keyword matches (highest weight)
            for keyword in patterns["direct"]:
                if keyword in query_lower:
                    category_scores[category] += patterns["confidence"] * 1.0

            # Contextual keyword matches (medium weight)
            for keyword in patterns["contextual"]:
                if keyword in query_lower:
                    category_scores[category] += patterns["confidence"] * 0.7

            # Action keyword matches (lower weight)
            for keyword in patterns["action"]:
                if keyword in query_lower:
                    category_scores[category] += patterns["confidence"] * 0.5

        # Normalize scores to prevent inflation
        return {k: min(1.0, v) for k, v in category_scores.items()}


class ContextAnalyzer:
    """Analyzes context clues from environment and query patterns"""

    def __init__(self) -> None:
        self.context_patterns = {
            "file_extensions": {
                ".py": ["test", "quality", "security"],
                ".js": ["test", "quality", "analysis"],
                ".ts": ["test", "quality", "analysis"],
                ".md": ["quality", "analysis"],
                ".yml": ["security", "quality"],
                ".yaml": ["security", "quality"],
                ".json": ["security", "analysis"],
                ".sql": ["security", "quality"],
                ".dockerfile": ["security", "quality"],
                ".sh": ["security", "debug"],
            },
            "error_indicators": {
                "traceback": ["debug", "analysis"],
                "exception": ["debug", "test"],
                "failed": ["debug", "test"],
                "error:": ["debug", "analysis"],
                "warning:": ["quality", "security"],
                "500": ["debug", "analysis"],
                "404": ["debug", "analysis"],
                "timeout": ["debug", "analysis"],
            },
            "performance_indicators": {
                "slow": ["analysis", "quality"],
                "memory": ["debug", "analysis"],
                "timeout": ["debug", "analysis"],
                "performance": ["analysis", "quality"],
                "optimization": ["quality", "analysis"],
                "bottleneck": ["debug", "analysis"],
            },
        }

    def analyze(self, query: str, context: dict[str, Any]) -> dict[str, float]:
        """Analyze context clues for category signals"""
        category_scores = defaultdict(float)

        # File extension analysis
        file_extensions = context.get("file_extensions", [])
        for ext in file_extensions:
            if ext in self.context_patterns["file_extensions"]:
                for category in self.context_patterns["file_extensions"][ext]:
                    category_scores[category] += 0.3

        # Error indicator analysis
        query_lower = query.lower()
        for indicator, categories in self.context_patterns["error_indicators"].items():
            if indicator in query_lower:
                for category in categories:
                    category_scores[category] += 0.6

        # Performance indicator analysis
        for indicator, categories in self.context_patterns["performance_indicators"].items():
            if indicator in query_lower:
                for category in categories:
                    category_scores[category] += 0.5

        return dict(category_scores)


class EnvironmentAnalyzer:
    """Analyzes environment state for implicit task detection"""

    def analyze_git_state(self, context: dict[str, Any]) -> dict[str, float]:
        """Analyze git repository state for context clues"""
        signals = {}

        # Check for uncommitted changes
        if context.get("has_uncommitted_changes", False):
            signals["git"] = 0.7
            signals["quality"] = 0.4  # Potential pre-commit validation

        # Check for merge conflicts
        if context.get("has_merge_conflicts", False):
            signals["git"] = max(signals.get("git", 0), 0.9)
            signals["debug"] = 0.6

        # Check for recent commits
        if context.get("recent_commits", 0) > 0:
            signals["git"] = max(signals.get("git", 0), 0.5)

        return signals

    def analyze_project_structure(self, context: dict[str, Any]) -> dict[str, float]:
        """Analyze project structure for implicit needs"""
        signals = {}

        # Check for test directories
        if context.get("has_test_directories", False):
            signals["test"] = 0.6

        # Check for security-related files
        if context.get("has_security_files", False):
            signals["security"] = 0.5

        # Check for CI/CD files
        if context.get("has_ci_files", False):
            signals["quality"] = 0.4
            signals["test"] = 0.4

        # Check for documentation files
        if context.get("has_docs", False):
            signals["quality"] = 0.3

        return signals

    def analyze(self, context: dict[str, Any]) -> dict[str, float]:
        """Comprehensive environment analysis"""
        git_signals = self.analyze_git_state(context)
        structure_signals = self.analyze_project_structure(context)

        # Combine signals
        combined = defaultdict(float)
        for signals in [git_signals, structure_signals]:
            for category, score in signals.items():
                combined[category] += score

        return dict(combined)


class SessionAnalyzer:
    """Analyzes conversation history and session patterns"""

    def __init__(self) -> None:
        self.function_categories = {
            "git": ["git_status", "git_add", "git_commit", "git_push", "git_pull"],
            "debug": ["debug", "analyze", "trace"],
            "test": ["testgen", "test"],
            "security": ["secaudit", "security"],
            "quality": ["codereview", "refactor", "docgen"],
            "analysis": ["analyze", "thinkdeep", "consensus"],
        }

    def analyze_recent_functions(self, history: list[dict]) -> dict[str, float]:
        """Analyze recently used functions for pattern recognition"""
        if not history:
            return {}

        recent_functions = []
        for entry in history[-10:]:  # Last 10 interactions
            functions_used = entry.get("functions_used", [])
            recent_functions.extend(functions_used)

        # Categorize functions
        category_usage = defaultdict(int)
        for func in recent_functions:
            for category, funcs in self.function_categories.items():
                if any(f in func for f in funcs):
                    category_usage[category] += 1

        # Convert to scores (higher usage = higher probability of continued use)
        max_usage = max(category_usage.values()) if category_usage else 1
        return {cat: min(0.6, count / max_usage * 0.6) for cat, count in category_usage.items()}

    def analyze_query_evolution(self, history: list[dict]) -> float:
        """Analyze how similar current query is to recent queries"""
        if len(history) < 2:
            return 0.0

        current_query = history[-1].get("query", "")
        recent_queries = [entry.get("query", "") for entry in history[-5:-1]]

        # Simple similarity based on shared keywords
        current_words = set(current_query.lower().split())
        similarity_scores = []

        for query in recent_queries:
            query_words = set(query.lower().split())
            if current_words and query_words:
                intersection = len(current_words & query_words)
                union = len(current_words | query_words)
                similarity = intersection / union if union > 0 else 0
                similarity_scores.append(similarity)

        return max(similarity_scores) if similarity_scores else 0.0

    def analyze(self, context: dict[str, Any]) -> dict[str, float]:
        """Analyze session context for patterns"""
        history = context.get("session_history", [])

        recent_function_signals = self.analyze_recent_functions(history)

        # If high query similarity, boost recent function signals
        query_similarity = self.analyze_query_evolution(history)
        if query_similarity > 0.7:
            # User is continuing similar work
            for category in recent_function_signals:
                recent_function_signals[category] = min(0.8, recent_function_signals[category] + 0.3)

        return recent_function_signals


class ConfidenceCalibrator:
    """Calibrates confidence scores based on historical accuracy"""

    def __init__(self) -> None:
        # Default calibration curves (would be learned from data in production)
        self.calibration_curves = {
            "git": {0.3: 0.8, 0.5: 0.9, 0.7: 0.95, 1.0: 1.0},
            "debug": {0.3: 0.7, 0.5: 0.8, 0.7: 0.9, 1.0: 0.95},
            "test": {0.3: 0.6, 0.5: 0.75, 0.7: 0.85, 1.0: 0.9},
            "security": {0.3: 0.7, 0.5: 0.8, 0.7: 0.9, 1.0: 0.95},
            "analysis": {0.3: 0.5, 0.5: 0.65, 0.7: 0.8, 1.0: 0.9},
            "quality": {0.3: 0.6, 0.5: 0.7, 0.7: 0.85, 1.0: 0.9},
        }

    def calibrate_scores(self, raw_scores: dict[str, float], query_complexity: float = 0.5) -> dict[str, float]:
        """Calibrate scores based on historical accuracy data"""
        calibrated_scores = {}

        for category, score in raw_scores.items():
            # Apply calibration curve
            calibrated_score = self.apply_calibration_curve(category, score)

            # Adjust for query complexity
            complexity_modifier = self.get_complexity_modifier(query_complexity)
            calibrated_score *= complexity_modifier

            calibrated_scores[category] = min(1.0, calibrated_score)

        return calibrated_scores

    def apply_calibration_curve(self, category: str, score: float) -> float:
        """Apply learned calibration curve to improve accuracy"""
        curve = self.calibration_curves.get(category, {})

        if not curve:
            return score

        # Piecewise linear interpolation
        sorted_points = sorted(curve.items())

        for i, (threshold, adjustment) in enumerate(sorted_points):
            if score <= threshold:
                if i == 0:
                    return score * adjustment / threshold
                prev_threshold, prev_adjustment = sorted_points[i - 1]
                # Linear interpolation between points
                ratio = (score - prev_threshold) / (threshold - prev_threshold)
                return prev_adjustment + ratio * (adjustment - prev_adjustment)

        return score * sorted_points[-1][1]  # Use last adjustment

    def get_complexity_modifier(self, complexity: float) -> float:
        """Adjust confidence based on query complexity"""
        # For complex queries, be more conservative (load more categories)
        if complexity > 0.8:
            return 0.8  # Reduce confidence, encouraging more loading
        if complexity < 0.3:
            return 1.1  # Increase confidence for simple queries
        return 1.0


class TaskDetectionScorer:
    """Main scoring engine for task detection"""

    def __init__(self) -> None:
        self.signal_weights = {
            "keyword_direct": 1.0,
            "keyword_contextual": 0.7,
            "keyword_action": 0.5,
            "context_files": 0.6,
            "context_errors": 0.8,
            "environment_git": 0.7,
            "environment_structure": 0.5,
            "session_recent": 0.6,
            "session_pattern": 0.8,
        }

        self.category_modifiers = {
            ("git", "keyword_direct"): 1.2,
            ("debug", "context_errors"): 1.3,
            ("security", "context_files"): 1.1,
            ("test", "environment_structure"): 1.2,
        }

    def calculate_category_scores(self, signals: list[SignalData]) -> dict[str, float]:
        """Calculate weighted scores for each function category"""
        category_scores = defaultdict(float)

        for signal in signals:
            base_weight = self.signal_weights.get(signal.signal_type, 0.5)

            for category, confidence in signal.category_scores.items():
                # Apply signal-specific weighting
                weighted_score = confidence * base_weight * signal.confidence

                # Apply category-specific modifiers
                modifier = self.category_modifiers.get((category, signal.signal_type), 1.0)
                final_score = weighted_score * modifier

                category_scores[category] += final_score

        # Normalize scores to prevent inflation
        return self.normalize_scores(dict(category_scores))

    def normalize_scores(self, scores: dict[str, float]) -> dict[str, float]:
        """Normalize scores to reasonable ranges"""
        if not scores:
            return {}

        max_score = max(scores.values())
        if max_score > 1.0:
            # Scale down all scores proportionally
            scale_factor = 1.0 / max_score
            return {k: v * scale_factor for k, v in scores.items()}

        return scores


class FunctionLoader:
    """Manages tier-based function loading decisions"""

    def __init__(self) -> None:
        # Import here to avoid circular imports
        from .task_detection_config import default_config

        config = default_config
        self.tier_definitions = {
            "tier1": {"categories": ["core", "git"], "threshold": 0.0, "token_cost": 9040},  # Always loaded
            "tier2": {
                "categories": ["analysis", "quality", "debug", "test", "security"],
                "threshold": config.thresholds.tier2_base_threshold,  # Use config value
                "token_cost": 14940,
            },
            "tier3": {
                "categories": ["external", "infrastructure"],
                "threshold": config.thresholds.tier3_base_threshold,  # Use config value
                "token_cost": 3850,
            },
        }

    def make_loading_decision(
        self,
        scores: dict[str, float],
        context: dict[str, Any],
    ) -> tuple[dict[str, bool], str | None]:
        """Make tier-based loading decisions with fallback logic"""
        decisions = {}
        fallback_applied = None

        # Tier 1: Always load
        for category in self.tier_definitions["tier1"]["categories"]:
            decisions[category] = True

        # Tier 2: Conditional loading
        for category in self.tier_definitions["tier2"]["categories"]:
            score = scores.get(category, 0.0)
            threshold = self.tier_definitions["tier2"]["threshold"]

            # Apply conservative bias
            adjusted_threshold = self.apply_conservative_bias(threshold, context)

            if score >= adjusted_threshold:
                decisions[category] = True
            else:
                decisions[category] = False

        # Tier 3: High-confidence only
        for category in self.tier_definitions["tier3"]["categories"]:
            score = scores.get(category, 0.0)
            threshold = self.tier_definitions["tier3"]["threshold"]
            decisions[category] = score >= threshold

        # Apply fallback logic
        decisions, fallback_applied = self.apply_fallback_logic(decisions, scores, context)

        return decisions, fallback_applied

    def apply_conservative_bias(self, threshold: float, context: dict[str, Any]) -> float:
        """Apply conservative bias to prevent functionality loss"""
        # Reduce threshold for new users or complex queries
        if context.get("user_experience", "expert") == "new":
            return threshold * 0.7

        if context.get("query_complexity", 0.5) > 0.8:
            return threshold * 0.8

        return threshold

    def apply_fallback_logic(
        self,
        initial_decisions: dict[str, bool],
        scores: dict[str, float],
        context: dict[str, Any],
    ) -> tuple[dict[str, bool], str | None]:
        """Apply fallback chain for edge cases and ambiguous situations"""

        # 1. High-confidence detection → Use as-is
        max_score = max(scores.values()) if scores else 0
        if max_score >= 0.8:
            return initial_decisions, None

        # 2. Medium-confidence → Load multiple likely categories
        if max_score >= 0.4:
            expanded = self.expand_medium_confidence(initial_decisions, scores)
            return expanded, "medium_confidence_expansion"

        # 3. Low-confidence/ambiguous → Check if conservative bias enabled any functions
        if max_score < 0.4 or self.is_ambiguous(scores):
            # If conservative bias enabled any tier2+ functions, respect those decisions
            tier2_enabled = any(
                initial_decisions.get(cat, False) for cat in self.tier_definitions["tier2"]["categories"]
            )
            tier3_enabled = any(
                initial_decisions.get(cat, False) for cat in self.tier_definitions["tier3"]["categories"]
            )

            if tier2_enabled or tier3_enabled:
                return initial_decisions, "conservative_bias"
            safe_default = self.load_safe_default(context)
            return safe_default, "safe_default"

        # 4. Detection failure → Full load with learning capture
        full_load = self.full_load_with_learning(context, scores)
        return full_load, "full_load_fallback"

    def expand_medium_confidence(self, decisions: dict[str, bool], scores: dict[str, float]) -> dict[str, bool]:
        """Expand loading for medium-confidence scenarios"""
        expanded = decisions.copy()

        # Load top 2 scoring tier2 categories (reduced from 3 to improve precision)
        tier2_categories = self.tier_definitions["tier2"]["categories"]
        tier2_scores = {k: v for k, v in scores.items() if k in tier2_categories}

        top_categories = sorted(tier2_scores.items(), key=lambda x: x[1], reverse=True)[:2]

        for category, score in top_categories:
            if score >= 0.3:  # Increased threshold from 0.2 to 0.3 for better precision
                expanded[category] = True

        return expanded

    def is_ambiguous(self, scores: dict[str, float]) -> bool:
        """Detect ambiguous scoring patterns"""
        if not scores:
            return True

        # Check if multiple categories have similar scores
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) >= 2:
            return sorted_scores[0] - sorted_scores[1] < 0.2

        return False

    def load_safe_default(self, context: dict[str, Any]) -> dict[str, bool]:
        """Load conservative safe default for ambiguous cases"""
        safe_default = {
            "core": True,
            "git": True,
            "analysis": True,  # Conservative inclusion
            "debug": False,
            "test": False,
            "quality": False,
            "security": False,
            "external": False,
            "infrastructure": False,
        }

        # Context-based adjustments
        if context.get("project_type") == "security":
            safe_default["security"] = True

        if context.get("has_tests", False):
            safe_default["test"] = True

        if context.get("file_extensions", []):
            # If working with code files, enable quality tools
            code_extensions = [".py", ".js", ".ts", ".java", ".cpp", ".c"]
            if any(ext in context["file_extensions"] for ext in code_extensions):
                safe_default["quality"] = True

        return safe_default

    def full_load_with_learning(self, context: dict[str, Any], scores: dict[str, float]) -> dict[str, bool]:
        """Full load for detection failures with learning capture"""
        # Load everything in tier 1 and 2, selective tier 3
        full_load = {
            "core": True,
            "git": True,
            "analysis": True,
            "debug": True,
            "test": True,
            "quality": True,
            "security": True,
            "external": False,  # Still selective on tier 3
            "infrastructure": False,
        }

        # Log for learning system
        logger.warning(f"Detection failed, using full load fallback. Scores: {scores}, Context: {context}")

        return full_load


class TaskDetectionSystem:
    """Main task detection system orchestrating all components"""

    def __init__(self) -> None:
        self.keyword_analyzer = KeywordAnalyzer()
        self.context_analyzer = ContextAnalyzer()
        self.environment_analyzer = EnvironmentAnalyzer()
        self.session_analyzer = SessionAnalyzer()
        self.scorer = TaskDetectionScorer()
        self.calibrator = ConfidenceCalibrator()
        self.loader = FunctionLoader()

        # Performance tracking
        self.cache = {}
        self.cache_timestamps = {}
        self.max_cache_age = timedelta(hours=1)

    @lru_cache(maxsize=1000)
    def _cached_keyword_analysis(self, query: str) -> dict[str, float]:
        """Cached keyword analysis for performance"""
        return self.keyword_analyzer.analyze(query)

    async def detect_categories(self, query: str, context: dict[str, Any] | None = None) -> DetectionResult:
        """Main entry point for task detection"""
        start_time = time.perf_counter()

        if context is None:
            context = {}

        # Check cache first
        cache_key = self._generate_cache_key(query, context)
        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            if datetime.now() - self.cache_timestamps[cache_key] < self.max_cache_age:
                return cached_result

        try:
            # Extract signals in parallel for performance
            signal_tasks = [
                self._extract_keyword_signals(query),
                self._extract_context_signals(query, context),
                self._extract_environment_signals(context),
                self._extract_session_signals(context),
            ]

            signals = await asyncio.gather(*signal_tasks, return_exceptions=True)

            # Filter out exceptions and flatten signals
            valid_signals = []
            signals_dict = {}

            for i, signal_result in enumerate(signals):
                if not isinstance(signal_result, Exception):
                    valid_signals.extend(signal_result)
                    signals_dict[f"signal_{i}"] = signal_result
                else:
                    logger.warning(f"Signal extraction failed: {signal_result}")

            # Calculate scores
            raw_scores = self.scorer.calculate_category_scores(valid_signals)

            # Apply calibration
            query_complexity = self._estimate_query_complexity(query)
            calibrated_scores = self.calibrator.calibrate_scores(raw_scores, query_complexity)

            # Make loading decisions
            decisions, fallback_applied = self.loader.make_loading_decision(calibrated_scores, context)

            # Calculate detection time
            detection_time = (time.perf_counter() - start_time) * 1000

            # Create result
            result = DetectionResult(
                categories=decisions,
                confidence_scores=calibrated_scores,
                detection_time_ms=detection_time,
                signals_used=signals_dict,
                fallback_applied=fallback_applied,
            )

            # Cache result
            self.cache[cache_key] = result
            self.cache_timestamps[cache_key] = datetime.now()

            return result

        except Exception as e:
            logger.error(f"Task detection failed: {e}")
            # Return safe fallback
            return DetectionResult(
                categories=self.loader.load_safe_default(context),
                confidence_scores={},
                detection_time_ms=(time.perf_counter() - start_time) * 1000,
                signals_used={},
                fallback_applied="error_fallback",
            )

    async def _extract_keyword_signals(self, query: str) -> list[SignalData]:
        """Extract keyword-based signals"""
        scores = self._cached_keyword_analysis(query)
        return [SignalData(signal_type="keyword", category_scores=scores, confidence=0.9, source="keyword_analyzer")]

    async def _extract_context_signals(self, query: str, context: dict[str, Any]) -> list[SignalData]:
        """Extract context-based signals"""
        scores = self.context_analyzer.analyze(query, context)
        return [SignalData(signal_type="context", category_scores=scores, confidence=0.7, source="context_analyzer")]

    async def _extract_environment_signals(self, context: dict[str, Any]) -> list[SignalData]:
        """Extract environment-based signals"""
        scores = self.environment_analyzer.analyze(context)
        return [
            SignalData(
                signal_type="environment",
                category_scores=scores,
                confidence=0.6,
                source="environment_analyzer",
            ),
        ]

    async def _extract_session_signals(self, context: dict[str, Any]) -> list[SignalData]:
        """Extract session-based signals"""
        scores = self.session_analyzer.analyze(context)
        return [SignalData(signal_type="session", category_scores=scores, confidence=0.8, source="session_analyzer")]

    def _generate_cache_key(self, query: str, context: dict[str, Any]) -> str:
        """Generate cache key for results"""
        # Use query and key context elements for caching
        context_key = str(sorted(context.items())) if context else ""
        return f"{hash(query)}_{hash(context_key)}"

    def _estimate_query_complexity(self, query: str) -> float:
        """Estimate query complexity for calibration"""
        # Simple heuristics for complexity estimation
        word_count = len(query.split())

        # Complexity indicators
        complexity_indicators = [
            "and",
            "or",
            "but",
            "also",
            "multiple",
            "various",
            "complex",
            "analyze",
            "understand",
            "investigate",
        ]

        indicator_count = sum(1 for indicator in complexity_indicators if indicator in query.lower())

        # Normalize to 0-1 scale
        base_complexity = min(1.0, word_count / 20.0)
        indicator_bonus = min(0.3, indicator_count * 0.1)

        return min(1.0, base_complexity + indicator_bonus)

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics for monitoring"""
        return {
            "cache_size": len(self.cache),
            "cache_hit_rate": self._calculate_cache_hit_rate(),
            "avg_detection_time": self._calculate_avg_detection_time(),
            "memory_usage": self._estimate_memory_usage(),
        }

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        # This would be tracked with proper counters in production
        return 0.0  # Placeholder

    def _calculate_avg_detection_time(self) -> float:
        """Calculate average detection time"""
        # This would be tracked with proper metrics in production
        return 0.0  # Placeholder

    def _estimate_memory_usage(self) -> int:
        """Estimate current memory usage in bytes"""
        # This would use actual memory profiling in production
        return len(self.cache) * 1024  # Rough estimate


# Example usage and testing
if __name__ == "__main__":

    async def main() -> None:
        detector = TaskDetectionSystem()

        # Test cases
        test_cases = [
            {
                "query": "debug the failing tests in the authentication module",
                "context": {
                    "project_type": "web_app",
                    "has_tests": True,
                    "file_extensions": [".py"],
                    "has_security_files": True,
                },
            },
            {"query": "help me improve this code", "context": {"file_extensions": [".py"], "project_size": "large"}},
            {
                "query": 'git commit -m "fix security vulnerability"',
                "context": {"has_uncommitted_changes": True, "has_security_files": True},
            },
        ]

        for _i, test_case in enumerate(test_cases):

            result = await detector.detect_categories(test_case["query"], test_case["context"])

            if result.fallback_applied:
                pass

    asyncio.run(main())
