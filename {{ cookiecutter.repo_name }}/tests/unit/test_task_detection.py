"""
Comprehensive test suite for task detection algorithm

Tests all components of the task detection system including edge cases,
performance requirements, and fallback mechanisms.
"""

import asyncio
import time
from unittest.mock import patch

import pytest

from src.core.task_detection import (
    ConfidenceCalibrator,
    ContextAnalyzer,
    DetectionResult,
    EnvironmentAnalyzer,
    FunctionLoader,
    KeywordAnalyzer,
    SessionAnalyzer,
    SignalData,
    TaskDetectionScorer,
    TaskDetectionSystem,
)


class TestKeywordAnalyzer:
    """Test keyword analysis component"""

    def setup_method(self):
        self.analyzer = KeywordAnalyzer()

    def test_direct_keyword_detection(self):
        """Test direct keyword matching"""
        query = "debug this git issue"
        result = self.analyzer.analyze(query)

        assert "debug" in result
        assert "git" in result
        assert result["debug"] > 0.8  # High confidence for direct match
        assert result["git"] > 0.8

    def test_contextual_keyword_detection(self):
        """Test contextual keyword matching"""
        query = "investigate the repository merge conflicts"
        result = self.analyzer.analyze(query)

        assert "debug" in result or "git" in result
        assert any(score > 0.5 for score in result.values())

    def test_action_keyword_detection(self):
        """Test action keyword matching"""
        query = "analyze and fix the authentication flow"
        result = self.analyzer.analyze(query)

        assert "analysis" in result or "debug" in result or "security" in result

    def test_empty_query(self):
        """Test handling of empty queries"""
        result = self.analyzer.analyze("")
        assert result == {}

    def test_no_keywords_query(self):
        """Test query with no relevant keywords"""
        query = "hello world foo bar"
        result = self.analyzer.analyze(query)
        assert len(result) == 0 or all(score < 0.1 for score in result.values())


class TestContextAnalyzer:
    """Test context analysis component"""

    def setup_method(self):
        self.analyzer = ContextAnalyzer()

    def test_file_extension_analysis(self):
        """Test file extension context analysis"""
        query = "look at this file"
        context = {"file_extensions": [".py", ".js"]}

        result = self.analyzer.analyze(query, context)

        assert any(cat in result for cat in ["test", "quality", "analysis"])

    def test_error_indicator_analysis(self):
        """Test error indicator detection"""
        query = "exception occurred in traceback"
        context = {}

        result = self.analyzer.analyze(query, context)

        assert "debug" in result
        assert result["debug"] > 0.5

    def test_performance_indicator_analysis(self):
        """Test performance indicator detection"""
        query = "application is slow and has memory issues"
        context = {}

        result = self.analyzer.analyze(query, context)

        assert any(cat in result for cat in ["analysis", "debug", "quality"])

    def test_multiple_indicators(self):
        """Test multiple context indicators"""
        query = "debug slow performance in .py files"
        context = {"file_extensions": [".py"]}

        result = self.analyzer.analyze(query, context)

        # Should detect multiple signals
        assert len(result) >= 2


class TestEnvironmentAnalyzer:
    """Test environment analysis component"""

    def setup_method(self):
        self.analyzer = EnvironmentAnalyzer()

    def test_git_state_analysis(self):
        """Test git state analysis"""
        context = {"has_uncommitted_changes": True, "has_merge_conflicts": False, "recent_commits": 3}

        result = self.analyzer.analyze_git_state(context)

        assert "git" in result
        assert result["git"] > 0.5

    def test_project_structure_analysis(self):
        """Test project structure analysis"""
        context = {"has_test_directories": True, "has_security_files": True, "has_ci_files": True}

        result = self.analyzer.analyze_project_structure(context)

        assert "test" in result
        assert "security" in result
        assert "quality" in result

    def test_combined_analysis(self):
        """Test combined environment analysis"""
        context = {"has_uncommitted_changes": True, "has_test_directories": True, "has_security_files": True}

        result = self.analyzer.analyze(context)

        assert len(result) >= 2  # Should detect multiple signals


class TestSessionAnalyzer:
    """Test session analysis component"""

    def setup_method(self):
        self.analyzer = SessionAnalyzer()

    def test_recent_function_analysis(self):
        """Test analysis of recently used functions"""
        history = [
            {"functions_used": ["git_status", "git_add"]},
            {"functions_used": ["debug", "analyze"]},
            {"functions_used": ["git_commit"]},
        ]

        result = self.analyzer.analyze_recent_functions(history)

        assert "git" in result
        assert "debug" in result or "analysis" in result

    def test_query_evolution_analysis(self):
        """Test query similarity analysis"""
        history = [{"query": "debug git issues"}, {"query": "fix git issues"}, {"query": "resolve git issues"}]

        similarity = self.analyzer.analyze_query_evolution(history)

        assert similarity >= 0.5  # Should detect high similarity

    def test_empty_history(self):
        """Test handling of empty history"""
        result = self.analyzer.analyze({"session_history": []})
        assert result == {}

    def test_session_continuation_pattern(self):
        """Test detection of session continuation patterns"""
        context = {
            "session_history": [
                {"functions_used": ["git_status"], "query": "check git status"},
                {"functions_used": ["git_add"], "query": "add files to git"},
                {"query": "now commit the changes"},  # Current query
            ],
        }

        result = self.analyzer.analyze(context)

        # Should boost git category due to continuation pattern
        assert "git" in result
        assert result["git"] > 0.4


class TestTaskDetectionScorer:
    """Test the main scoring engine"""

    def setup_method(self):
        self.scorer = TaskDetectionScorer()

    def test_single_signal_scoring(self):
        """Test scoring with single signal"""
        signals = [
            SignalData(
                signal_type="keyword",
                category_scores={"git": 0.9, "debug": 0.7},
                confidence=0.9,
                source="test",
            ),
        ]

        result = self.scorer.calculate_category_scores(signals)

        assert "git" in result
        assert "debug" in result
        assert result["git"] > result["debug"]  # Should maintain relative order

    def test_multiple_signal_fusion(self):
        """Test fusion of multiple signals"""
        signals = [
            SignalData(signal_type="keyword", category_scores={"git": 0.8}, confidence=0.9, source="keyword"),
            SignalData(signal_type="context", category_scores={"git": 0.6}, confidence=0.7, source="context"),
        ]

        result = self.scorer.calculate_category_scores(signals)

        # Should combine signals for higher confidence
        assert result["git"] > 0.5  # Realistic expectation with current weights

    def test_score_normalization(self):
        """Test score normalization prevents inflation"""
        signals = [
            SignalData(
                signal_type="keyword",
                category_scores={"git": 2.0},
                confidence=1.0,
                source="test",  # Artificially high
            ),
        ]

        result = self.scorer.calculate_category_scores(signals)

        assert all(score <= 1.0 for score in result.values())

    def test_category_modifiers(self):
        """Test category-specific modifiers"""
        signals = [
            SignalData(signal_type="keyword_direct", category_scores={"git": 0.5}, confidence=1.0, source="test"),
        ]

        result = self.scorer.calculate_category_scores(signals)

        # Git with direct keywords should get modifier boost
        assert result["git"] > 0.5


class TestConfidenceCalibrator:
    """Test confidence calibration"""

    def setup_method(self):
        self.calibrator = ConfidenceCalibrator()

    def test_calibration_curves(self):
        """Test calibration curve application"""
        raw_scores = {"git": 0.5, "debug": 0.3}

        result = self.calibrator.calibrate_scores(raw_scores)

        # Scores should be adjusted but maintain relative order
        assert "git" in result
        assert "debug" in result
        assert result["git"] > result["debug"]

    def test_complexity_modifier(self):
        """Test query complexity modification"""
        raw_scores = {"git": 0.5}

        # High complexity should reduce confidence
        result_complex = self.calibrator.calibrate_scores(raw_scores, 0.9)
        result_simple = self.calibrator.calibrate_scores(raw_scores, 0.2)

        assert result_complex["git"] < result_simple["git"]

    def test_score_bounds(self):
        """Test that calibrated scores stay within bounds"""
        raw_scores = {"git": 0.8, "debug": 1.5}  # One over 1.0

        result = self.calibrator.calibrate_scores(raw_scores)

        assert all(0.0 <= score <= 1.0 for score in result.values())


class TestFunctionLoader:
    """Test function loading decision logic"""

    def setup_method(self):
        self.loader = FunctionLoader()

    def test_tier1_always_loaded(self):
        """Test that tier 1 categories are always loaded"""
        scores = {}  # Empty scores
        context = {}

        decisions, fallback = self.loader.make_loading_decision(scores, context)

        # Tier 1 should always be loaded
        assert decisions["core"] is True
        assert decisions["git"] is True

    def test_tier2_conditional_loading(self):
        """Test tier 2 conditional loading"""
        scores = {"debug": 0.8, "test": 0.2}  # One above, one below threshold
        context = {}

        decisions, fallback = self.loader.make_loading_decision(scores, context)

        assert decisions["debug"] is True  # Above threshold
        assert decisions["test"] is False  # Below threshold

    def test_conservative_bias_new_user(self):
        """Test conservative bias for new users"""
        scores = {"debug": 0.25}  # Just below normal threshold
        context = {"user_experience": "new"}

        decisions, fallback = self.loader.make_loading_decision(scores, context)

        # Should load due to conservative bias
        assert decisions["debug"] is True

    def test_fallback_logic_high_confidence(self):
        """Test fallback logic for high confidence"""
        scores = {"git": 0.9}
        context = {}

        decisions, fallback = self.loader.make_loading_decision(scores, context)

        assert fallback is None  # No fallback needed

    def test_fallback_logic_medium_confidence(self):
        """Test fallback logic for medium confidence"""
        scores = {"debug": 0.5, "test": 0.4}
        context = {}

        decisions, fallback = self.loader.make_loading_decision(scores, context)

        assert fallback == "medium_confidence_expansion"
        # Should load more categories
        loaded_categories = [k for k, v in decisions.items() if v]
        assert len(loaded_categories) >= 3

    def test_fallback_logic_safe_default(self):
        """Test safe default fallback"""
        scores = {"unknown": 0.1}  # Very low confidence
        context = {}

        decisions, fallback = self.loader.make_loading_decision(scores, context)

        assert fallback == "safe_default"
        # Should include core, git, analysis at minimum
        assert decisions["core"] is True
        assert decisions["git"] is True
        assert decisions["analysis"] is True

    def test_ambiguous_detection(self):
        """Test detection of ambiguous patterns"""
        scores = {"debug": 0.5, "test": 0.49, "quality": 0.48}  # Very similar scores

        is_ambiguous = self.loader.is_ambiguous(scores)
        assert is_ambiguous is True

    def test_context_based_adjustments(self):
        """Test context-based safe default adjustments"""
        context = {"project_type": "security", "has_tests": True, "file_extensions": [".py"]}

        safe_default = self.loader.load_safe_default(context)

        assert safe_default["security"] is True  # Security project
        assert safe_default["test"] is True  # Has tests
        assert safe_default["quality"] is True  # Code files


class TestTaskDetectionSystem:
    """Test the main detection system"""

    def setup_method(self):
        self.system = TaskDetectionSystem()

    @pytest.mark.asyncio
    async def test_basic_detection(self):
        """Test basic detection functionality"""
        query = "debug git merge conflicts"
        context = {"file_extensions": [".py"]}

        result = await self.system.detect_categories(query, context)

        assert isinstance(result, DetectionResult)
        assert "git" in result.categories
        assert "debug" in result.categories
        assert result.detection_time_ms > 0

    @pytest.mark.asyncio
    async def test_performance_requirements(self):
        """Test that detection meets performance requirements"""
        query = "analyze this code for security issues"
        context = {}

        start_time = time.perf_counter()
        result = await self.system.detect_categories(query, context)
        end_time = time.perf_counter()

        detection_time_ms = (end_time - start_time) * 1000

        # Should be under 50ms requirement
        assert detection_time_ms < 50.0
        assert result.detection_time_ms < 50.0

    @pytest.mark.asyncio
    async def test_cache_functionality(self):
        """Test caching functionality"""
        query = "test query"
        context = {}

        # First call
        result1 = await self.system.detect_categories(query, context)

        # Second call should be faster (cached)
        start_time = time.perf_counter()
        result2 = await self.system.detect_categories(query, context)
        end_time = time.perf_counter()

        cache_time_ms = (end_time - start_time) * 1000

        assert result1.categories == result2.categories
        assert cache_time_ms < 5.0  # Cache retrieval should be very fast

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling and fallback"""
        # Mock the scorer to raise an exception to trigger actual error fallback
        with patch.object(self.system.scorer, "calculate_category_scores", side_effect=Exception("Test error")):

            query = "test error handling"
            result = await self.system.detect_categories(query)

            # Should still return a result with error fallback
            assert isinstance(result, DetectionResult)
            assert result.fallback_applied == "error_fallback"
            assert result.categories["core"] is True  # Safe default

    @pytest.mark.asyncio
    async def test_complex_query_complexity_estimation(self):
        """Test query complexity estimation"""
        simple_query = "git status"
        complex_query = "analyze and debug the complex security issues in multiple authentication modules"

        simple_complexity = self.system._estimate_query_complexity(simple_query)
        complex_complexity = self.system._estimate_query_complexity(complex_query)

        assert complex_complexity > simple_complexity
        assert 0.0 <= simple_complexity <= 1.0
        assert 0.0 <= complex_complexity <= 1.0


class TestEdgeCases:
    """Test edge cases and special scenarios"""

    def setup_method(self):
        self.system = TaskDetectionSystem()

    @pytest.mark.asyncio
    async def test_multi_domain_detection(self):
        """Test detection of multi-domain tasks"""
        query = "debug the security tests that are failing in authentication"
        context = {"has_tests": True, "has_security_files": True, "file_extensions": [".py"]}

        result = await self.system.detect_categories(query, context)

        # Should detect multiple relevant categories
        loaded_categories = [k for k, v in result.categories.items() if v]
        assert "debug" in loaded_categories
        assert "test" in loaded_categories
        assert "security" in loaded_categories

    @pytest.mark.asyncio
    async def test_vague_request_handling(self):
        """Test handling of vague requests"""
        query = "help me improve this"
        context = {"file_extensions": [".py"]}

        result = await self.system.detect_categories(query, context)

        # Should apply conservative loading for vague requests
        assert result.fallback_applied in ["safe_default", "medium_confidence_expansion"]
        assert result.categories["core"] is True
        assert result.categories["git"] is True

    @pytest.mark.asyncio
    async def test_context_dependent_tasks(self):
        """Test context-dependent task detection"""
        query = "analyze this security vulnerability"

        # Context 1: Python file with tests (should favor test + quality)
        context1 = {"file_extensions": [".py"], "has_test_directories": True}

        # Context 2: Strong security context (should favor security + analysis)
        context2 = {"file_extensions": [".yml"], "has_security_files": True, "project_type": "security"}

        result1 = await self.system.detect_categories(query, context1)
        result2 = await self.system.detect_categories(query, context2)

        # Results should differ based on context - specifically security should be different
        assert result1.categories != result2.categories or result1.confidence_scores != result2.confidence_scores

    @pytest.mark.asyncio
    async def test_new_pattern_handling(self):
        """Test handling of completely novel patterns"""
        query = "flibbertigibbet the quantum flux capacitor"
        context = {}

        result = await self.system.detect_categories(query, context)

        # Should fall back to safe default
        assert result.fallback_applied in ["safe_default", "full_load_fallback"]
        assert result.categories["core"] is True

    @pytest.mark.asyncio
    async def test_empty_input_handling(self):
        """Test handling of empty inputs"""
        result = await self.system.detect_categories("", {})

        # Should handle gracefully
        assert isinstance(result, DetectionResult)
        assert result.categories["core"] is True

    @pytest.mark.asyncio
    async def test_minimal_context(self):
        """Test with minimal context information"""
        query = "fix this"
        context = {}  # Minimal context

        result = await self.system.detect_categories(query, context)

        # Should still make reasonable decisions
        assert isinstance(result, DetectionResult)
        assert any(result.categories.values())  # At least something should be loaded


class TestPerformanceBenchmarks:
    """Performance and scalability tests"""

    def setup_method(self):
        self.system = TaskDetectionSystem()

    @pytest.mark.asyncio
    async def test_latency_benchmark(self):
        """Benchmark detection latency"""
        queries = [
            "debug git issues",
            "analyze security vulnerabilities",
            "test authentication flow",
            "refactor code quality",
            "help me improve this code",
        ]

        latencies = []

        for query in queries:
            start_time = time.perf_counter()
            await self.system.detect_categories(query, {})
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)

        # Performance requirements
        assert avg_latency < 30.0  # Average under 30ms
        assert max_latency < 50.0  # Max under 50ms

    @pytest.mark.asyncio
    async def test_throughput_benchmark(self):
        """Test detection throughput"""
        num_requests = 100
        query = "debug authentication issues"
        context = {"file_extensions": [".py"]}

        start_time = time.perf_counter()

        tasks = []
        for _ in range(num_requests):
            task = self.system.detect_categories(query, context)
            tasks.append(task)

        await asyncio.gather(*tasks)

        end_time = time.perf_counter()
        total_time = end_time - start_time
        throughput = num_requests / total_time

        # Should handle at least 500 requests per second
        assert throughput > 500

    def test_memory_usage(self):
        """Test memory usage remains reasonable"""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Create system and run many detections
        system = TaskDetectionSystem()

        # Simulate heavy usage
        for i in range(1000):
            # This would normally be async, but for memory testing we use sync
            query = f"test query {i}"
            context = {"iteration": i}
            # Just create the cache entries
            cache_key = system._generate_cache_key(query, context)
            system.cache[cache_key] = DetectionResult(
                categories={"core": True},
                confidence_scores={},
                detection_time_ms=1.0,
                signals_used={},
            )

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Should stay under 10MB increase
        assert memory_increase < 10 * 1024 * 1024

    @pytest.mark.asyncio
    async def test_concurrent_detection(self):
        """Test concurrent detection handling"""
        query = "analyze code quality"
        context = {}

        # Run multiple detections concurrently
        num_concurrent = 50
        tasks = [self.system.detect_categories(query, context) for _ in range(num_concurrent)]

        start_time = time.perf_counter()
        results = await asyncio.gather(*tasks)
        end_time = time.perf_counter()

        total_time = end_time - start_time

        # All should complete successfully
        assert len(results) == num_concurrent
        assert all(isinstance(r, DetectionResult) for r in results)

        # Should complete concurrently faster than sequentially
        assert total_time < 1.0  # Under 1 second for 50 concurrent


class TestAccuracyValidation:
    """Test accuracy against known scenarios"""

    def setup_method(self):
        self.system = TaskDetectionSystem()

        # Known test scenarios with expected outcomes
        # NOTE: core and git are tier1 categories (always loaded)
        self.test_scenarios = [
            {
                "query": "debug the failing tests in the authentication module",
                "context": {"project_type": "web_app", "has_tests": True},
                "expected_categories": ["core", "git", "debug", "test", "security"],
                "scenario_type": "multi_domain",
            },
            {
                "query": "help me improve this code",
                "context": {"file_extensions": [".py"], "project_size": "large"},
                "expected_categories": ["core", "git", "quality", "analysis"],
                "scenario_type": "vague",
            },
            {
                "query": 'git commit -m "fix security vulnerability"',
                "context": {"has_uncommitted_changes": True, "has_security_files": True},
                "expected_categories": ["core", "git", "security"],
                "scenario_type": "standard",
            },
            {
                "query": "analyze performance bottlenecks in the database queries",
                "context": {"file_extensions": [".sql", ".py"], "project_type": "backend"},
                "expected_categories": ["core", "git", "analysis", "debug", "quality"],
                "scenario_type": "performance",
            },
            {
                "query": "refactor the authentication code to improve security",
                "context": {"has_security_files": True, "file_extensions": [".py"]},
                "expected_categories": ["core", "git", "quality", "security", "analysis"],
                "scenario_type": "refactoring",
            },
        ]

    @pytest.mark.asyncio
    async def test_scenario_accuracy(self):
        """Test accuracy against predefined scenarios"""
        total_precision = 0
        total_recall = 0
        scenario_count = 0

        for scenario in self.test_scenarios:
            result = await self.system.detect_categories(scenario["query"], scenario["context"])

            predicted_categories = {k for k, v in result.categories.items() if v}
            expected_categories = set(scenario["expected_categories"])

            # Calculate metrics
            true_positives = len(predicted_categories & expected_categories)
            precision = true_positives / len(predicted_categories) if predicted_categories else 0
            recall = true_positives / len(expected_categories) if expected_categories else 1

            total_precision += precision
            total_recall += recall
            scenario_count += 1

            # Individual scenario assertions
            assert recall >= 0.8, f"Low recall for scenario: {scenario['scenario_type']}"

        # Overall accuracy requirements
        avg_precision = total_precision / scenario_count
        avg_recall = total_recall / scenario_count

        assert avg_precision >= 0.7, f"Average precision {avg_precision} below threshold"
        assert avg_recall >= 0.85, f"Average recall {avg_recall} below threshold"

    @pytest.mark.asyncio
    async def test_conservative_bias_validation(self):
        """Validate that the system maintains conservative bias"""
        over_inclusion_rates = []
        under_inclusion_rates = []

        for scenario in self.test_scenarios:
            result = await self.system.detect_categories(scenario["query"], scenario["context"])

            predicted_categories = {k for k, v in result.categories.items() if v}
            expected_categories = set(scenario["expected_categories"])

            over_included = predicted_categories - expected_categories
            under_included = expected_categories - predicted_categories

            over_inclusion_rate = len(over_included) / len(predicted_categories) if predicted_categories else 0
            under_inclusion_rate = len(under_included) / len(expected_categories) if expected_categories else 0

            over_inclusion_rates.append(over_inclusion_rate)
            under_inclusion_rates.append(under_inclusion_rate)

        avg_over_inclusion = sum(over_inclusion_rates) / len(over_inclusion_rates)
        avg_under_inclusion = sum(under_inclusion_rates) / len(under_inclusion_rates)

        # Conservative bias: prefer over-inclusion to under-inclusion
        assert avg_under_inclusion < 0.15, "Too much under-inclusion (missing functionality)"
        # Allow reasonable over-inclusion for safety
        assert avg_over_inclusion < 0.40, "Excessive over-inclusion hurts performance"


if __name__ == "__main__":
    # Run basic test
    import asyncio

    async def run_basic_test():
        system = TaskDetectionSystem()

        test_query = "debug the failing authentication tests"
        test_context = {"file_extensions": [".py"], "has_tests": True, "has_security_files": True}

        result = await system.detect_categories(test_query, test_context)

        print(f"Query: {test_query}")
        print(f"Loaded categories: {[k for k, v in result.categories.items() if v]}")
        print(f"Confidence scores: {result.confidence_scores}")
        print(f"Detection time: {result.detection_time_ms:.2f}ms")
        print(f"Fallback applied: {result.fallback_applied}")

    asyncio.run(run_basic_test())
