"""Benchmark tests for core algorithms and performance-critical functions."""

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from examples.task_detection_integration import IntelligentFunctionLoader
from src.core.task_detection import DetectionResult


class TestQueryProcessingBenchmarks:
    """Benchmark tests for query processing components."""

    @pytest.mark.benchmark
    @pytest.mark.performance
    def test_query_parsing_benchmark(self, benchmark):
        """Benchmark query parsing performance."""

        def parse_query():
            # Simulate query parsing logic
            query = "What are the best practices for authentication in web applications?"
            tokens = query.lower().split()
            return {
                "tokens": tokens,
                "length": len(tokens),
                "has_question": query.endswith("?"),
                "keywords": [t for t in tokens if len(t) > 3],
            }

        result = benchmark(parse_query)
        assert result["length"] > 0
        assert isinstance(result["tokens"], list)

    @pytest.mark.benchmark
    @pytest.mark.performance
    def test_embedding_generation_simulation_benchmark(self, benchmark):
        """Benchmark embedding generation simulation."""

        def generate_mock_embedding():
            # Simulate embedding generation (mock CPU-intensive operation)
            "This is a sample text for embedding generation"
            # Simulate processing time
            time.sleep(0.001)  # 1ms simulation
            return [0.1 * i for i in range(384)]  # Mock 384-dim embedding

        result = benchmark(generate_mock_embedding)
        assert len(result) == 384
        assert all(isinstance(x, float) for x in result)

    @pytest.mark.benchmark
    @pytest.mark.performance
    def test_vector_similarity_calculation_benchmark(self, benchmark):
        """Benchmark vector similarity calculation."""

        def calculate_similarity():
            # Mock vectors for similarity calculation
            vector_a = [0.1 * i for i in range(384)]
            vector_b = [0.2 * i for i in range(384)]

            # Dot product calculation
            dot_product = sum(a * b for a, b in zip(vector_a, vector_b, strict=False))

            # Magnitude calculation
            mag_a = sum(a * a for a in vector_a) ** 0.5
            mag_b = sum(b * b for b in vector_b) ** 0.5

            # Cosine similarity
            if mag_a > 0 and mag_b > 0:
                return dot_product / (mag_a * mag_b)
            return 0.0

        result = benchmark(calculate_similarity)
        assert isinstance(result, float)
        assert -1.0 <= result <= 1.0


class TestDataProcessingBenchmarks:
    """Benchmark tests for data processing operations."""

    @pytest.mark.benchmark
    @pytest.mark.performance
    def test_json_serialization_benchmark(self, benchmark):
        """Benchmark JSON serialization performance."""

        def serialize_large_data():
            # Create mock large data structure
            data = {
                "metadata": {"version": "1.0", "timestamp": "2024-01-01"},
                "results": [
                    {
                        "id": i,
                        "text": f"Sample text content {i}",
                        "score": 0.1 * i,
                        "embeddings": [0.01 * j for j in range(100)],
                    }
                    for i in range(100)
                ],
            }
            return json.dumps(data)

        result = benchmark(serialize_large_data)
        assert isinstance(result, str)
        assert len(result) > 1000

    @pytest.mark.benchmark
    @pytest.mark.performance
    def test_text_preprocessing_benchmark(self, benchmark):
        """Benchmark text preprocessing operations."""

        def preprocess_text():
            text = """
            This is a sample text document that needs preprocessing.
            It contains multiple sentences, punctuation marks, and various formatting.
            We need to clean, tokenize, and normalize this text efficiently.
            Performance matters when processing large volumes of text data.
            """

            # Simulate text preprocessing steps
            # 1. Clean whitespace
            cleaned = " ".join(text.split())

            # 2. Remove punctuation (simplified)
            chars_to_remove = ".,!?;:"
            for char in chars_to_remove:
                cleaned = cleaned.replace(char, "")

            # 3. Lowercase
            cleaned = cleaned.lower()

            # 4. Tokenize
            tokens = cleaned.split()

            # 5. Filter short tokens
            filtered_tokens = [token for token in tokens if len(token) > 2]

            return {
                "original_length": len(text),
                "cleaned_length": len(cleaned),
                "token_count": len(tokens),
                "filtered_count": len(filtered_tokens),
            }

        result = benchmark(preprocess_text)
        assert result["filtered_count"] > 0
        assert result["cleaned_length"] > 0

    @pytest.mark.benchmark
    @pytest.mark.performance
    def test_list_processing_benchmark(self, benchmark):
        """Benchmark list processing operations."""

        def process_large_list():
            # Create large list
            data = list(range(10000))

            # Perform various operations
            filtered = [x for x in data if x % 2 == 0]
            mapped = [x * 2 for x in filtered[:1000]]
            sorted_data = sorted(mapped, reverse=True)

            return {
                "original_size": len(data),
                "filtered_size": len(filtered),
                "mapped_size": len(mapped),
                "final_size": len(sorted_data),
            }

        result = benchmark(process_large_list)
        assert result["original_size"] == 10000
        assert result["filtered_size"] == 5000


class TestAlgorithmBenchmarks:
    """Benchmark tests for core algorithms."""

    @pytest.mark.benchmark
    @pytest.mark.performance
    def test_search_algorithm_benchmark(self, benchmark):
        """Benchmark search algorithm performance."""

        def binary_search():
            # Create sorted list for binary search
            data = list(range(0, 10000, 2))  # Even numbers
            target = 5000

            left, right = 0, len(data) - 1

            while left <= right:
                mid = (left + right) // 2
                if data[mid] == target:
                    return mid
                if data[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1

            return -1

        result = benchmark(binary_search)
        assert isinstance(result, int)

    @pytest.mark.benchmark
    @pytest.mark.performance
    def test_sorting_algorithm_benchmark(self, benchmark):
        """Benchmark sorting algorithm performance."""

        def sort_large_dataset():
            # Create unsorted data
            import random

            data = [random.randint(1, 1000) for _ in range(1000)]
            return sorted(data)

        result = benchmark(sort_large_dataset)
        assert len(result) == 1000
        assert result == sorted(result)  # Verify it's actually sorted

    @pytest.mark.benchmark
    @pytest.mark.performance
    def test_hash_computation_benchmark(self, benchmark):
        """Benchmark hash computation performance."""

        def compute_hashes():
            import hashlib

            data = ["sample_text_" + str(i) for i in range(1000)]
            hashes = []

            for item in data:
                hash_obj = hashlib.sha256(item.encode())
                hashes.append(hash_obj.hexdigest())

            return hashes

        result = benchmark(compute_hashes)
        assert len(result) == 1000
        assert all(len(h) == 64 for h in result)  # SHA256 hex length


class TestConcurrencyBenchmarks:
    """Benchmark tests for concurrency-related operations."""

    @pytest.mark.benchmark
    @pytest.mark.performance
    @pytest.mark.no_parallel  # Don't run this in parallel
    def test_sequential_processing_benchmark(self, benchmark):
        """Benchmark sequential processing for comparison."""

        def sequential_task():
            results = []
            for i in range(100):
                # Simulate work
                result = sum(j * j for j in range(i + 1))
                results.append(result)
            return results

        result = benchmark(sequential_task)
        assert len(result) == 100
        assert all(isinstance(x, int) for x in result)

    @pytest.mark.benchmark
    @pytest.mark.performance
    def test_memory_allocation_benchmark(self, benchmark):
        """Benchmark memory allocation patterns."""

        def allocate_memory():
            # Test different allocation patterns
            data = {
                "lists": [[i] * 100 for i in range(100)],
                "dicts": [{"key_" + str(i): i} for i in range(100)],
                "strings": ["string_" + str(i) for i in range(100)],
            }

            # Calculate total elements
            return sum(len(lst) for lst in data["lists"]) + len(data["dicts"]) + len(data["strings"])

        result = benchmark(allocate_memory)
        assert result > 0


class TestRegressionBenchmarks:
    """Benchmark tests for performance regression detection."""

    @pytest.mark.benchmark
    @pytest.mark.performance
    def test_baseline_performance_benchmark(self, benchmark):
        """Establish baseline performance metrics."""

        def baseline_operation():
            # Standard operation that should maintain consistent performance
            data = list(range(1000))
            processed = [x * 2 + 1 for x in data if x % 3 == 0]
            return sum(processed)

        result = benchmark(baseline_operation)
        assert result > 0

        # This benchmark serves as a baseline for performance regression testing

    @pytest.mark.benchmark
    @pytest.mark.performance
    def test_cpu_intensive_benchmark(self, benchmark):
        """Benchmark CPU-intensive operations."""

        def cpu_intensive_task():
            # CPU-bound calculation
            result = 0
            for i in range(10000):
                result += i**2 % 1000
            return result

        result = benchmark(cpu_intensive_task)
        assert isinstance(result, int)

    @pytest.mark.benchmark
    @pytest.mark.performance
    def test_io_simulation_benchmark(self, benchmark):
        """Benchmark I/O simulation operations."""

        def io_simulation():
            # Simulate I/O operations
            data = []
            for i in range(100):
                # Simulate reading/writing data
                content = f"Content line {i}" * 10
                data.append(len(content))
                # Small delay to simulate I/O
                time.sleep(0.0001)

            return sum(data)

        result = benchmark(io_simulation)
        assert result > 0


class TestTaskDetectionPerformance:
    """Performance tests for task detection and function loading."""

    @pytest.mark.benchmark
    @pytest.mark.performance
    @pytest.mark.asyncio
    @patch("examples.task_detection_integration.TaskDetectionSystem")
    @patch("examples.task_detection_integration.ConfigManager")
    @patch("examples.task_detection_integration.Path")
    async def test_memory_usage_with_large_history(self, mock_path, mock_config_manager, mock_detection_system):
        """Test memory usage with large loading history - moved from examples test suite to prevent timeout."""
        # Mock file system operations to prevent expensive directory scanning
        mock_cwd = Mock()
        mock_cwd.return_value = Path("/test/project")
        mock_path.cwd = mock_cwd
        
        # Mock working directory operations
        mock_working_dir = Mock()
        mock_working_dir.__truediv__ = Mock(return_value=Mock(exists=Mock(return_value=True)))
        mock_working_dir.rglob = Mock(return_value=[])  # Empty file list to prevent timeout
        
        # Mock config
        mock_config = Mock()
        mock_config.performance.max_detection_time_ms = 100.0
        mock_config_manager.return_value.get_config.return_value = mock_config

        # Mock detection result
        mock_detection_result = DetectionResult(
            categories={"core": True},
            confidence_scores={"core": 0.8},
            detection_time_ms=25.0,
            signals_used={},
            fallback_applied=None,
        )

        mock_detection_system.return_value.detect_categories = AsyncMock(return_value=mock_detection_result)

        loader = IntelligentFunctionLoader("production")

        # Simulate many loading decisions (reduced from 2000 to 1500 for performance test suite)
        for i in range(1500):  # More than the 1000 limit
            with patch.object(Path, '__new__', return_value=mock_working_dir):
                await loader.load_functions_for_query(f"query {i}")

        # Check that history is properly limited
        assert len(loader.loading_history) == 1000

        # Check that performance summary still works
        summary = loader.get_performance_summary()
        assert "average_detection_time_ms" in summary

    @pytest.mark.benchmark
    @pytest.mark.performance
    @pytest.mark.asyncio
    @patch("examples.task_detection_integration.TaskDetectionSystem")
    @patch("examples.task_detection_integration.ConfigManager")
    async def test_function_loading_performance(self, mock_config_manager, mock_detection_system):
        """Benchmark function loading performance."""
        # Mock config
        mock_config = Mock()
        mock_config.performance.max_detection_time_ms = 100.0
        mock_config_manager.return_value.get_config.return_value = mock_config

        # Mock detection result
        mock_detection_result = DetectionResult(
            categories={"core": True, "git": True, "analysis": True},
            confidence_scores={"core": 0.9, "git": 0.8, "analysis": 0.7},
            detection_time_ms=15.0,
            signals_used={},
            fallback_applied=None,
        )

        mock_detection_system.return_value.detect_categories = AsyncMock(return_value=mock_detection_result)

        loader = IntelligentFunctionLoader("production")

        # Mock file system to prevent directory scanning
        with patch.object(Path, 'cwd', return_value=Path("/test")):
            with patch.object(Path, 'rglob', return_value=[]):
                result = await loader.load_functions_for_query("test query for performance")

        # Verify result structure
        assert "functions" in result
        assert "detection_result" in result
        assert "stats" in result
        assert "context_used" in result

        # Verify performance characteristics
        stats = result["stats"]
        assert stats.detection_time_ms < 100.0  # Should be fast
        assert stats.token_savings_percent > 0  # Should provide savings
