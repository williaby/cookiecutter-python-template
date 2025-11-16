"""
Mock utility helpers for testing async/sync dual usage patterns.

This module provides utilities to handle complex mocking scenarios where
methods are sometimes awaited and sometimes called synchronously.
"""

import contextlib
from typing import Any
from unittest.mock import MagicMock


class AwaitableCallableMock:
    """
    A mock that can be both called synchronously and awaited asynchronously.

    This handles the dual usage pattern where the same method is sometimes
    awaited and sometimes called synchronously, as seen in QdrantVectorStore
    where upsert() is awaited in insert_documents() but called synchronously
    in update_document().

    Example:
        mock_result = MagicMock()
        mock_result.status = "completed"
        mock_client.upsert = AwaitableCallableMock(mock_result)

        # Both patterns work:
        result = await mock_client.upsert(args)  # Async usage
        result = mock_client.upsert(args)        # Sync usage
    """

    def __init__(self, return_value: Any):
        """
        Initialize the mock with a return value.

        Args:
            return_value: The value to return when called or awaited
        """
        self.return_value = return_value
        self.call_count = 0
        self.call_args_list: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.side_effect = None  # Support for side_effect like MagicMock

    def __call__(self, *args, **kwargs):
        """
        Handle synchronous calls and prepare for potential awaiting.

        Returns an AwaitableResult that can be used directly or awaited.
        """
        # Track calls for assertion purposes
        self.call_count += 1
        self.call_args_list.append((args, kwargs))

        # Handle side_effect if set
        if self.side_effect is not None:
            if callable(self.side_effect):
                # Call the side effect function
                side_result = self.side_effect(*args, **kwargs)
                # Check if it's a coroutine
                if hasattr(side_result, "__await__"):
                    # For async side effects, we need to create a special awaitable that wraps the coroutine
                    class AsyncSideEffectResult:
                        def __init__(self, coro):
                            self.coro = coro

                        def __await__(self):
                            # When awaited, return the coroutine result
                            return self.coro.__await__()

                        def __getattr__(self, name):
                            # For sync access, we can't access coroutine attributes
                            # but we need to provide some fallback behavior
                            if name == "status":
                                # Return a mock-like status for sync access
                                return "completed"
                            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

                    return AsyncSideEffectResult(side_result)
                return AwaitableResult(side_result)
            # side_effect is an exception or iterable
            if isinstance(self.side_effect, Exception):
                raise self.side_effect
            # Assume it's an iterable
            try:
                result = next(iter(self.side_effect))
                return AwaitableResult(result)
            except (StopIteration, TypeError):
                pass

        # Return an object that can be both used directly AND awaited
        return AwaitableResult(self.return_value)

    # Mock assertion methods for test compatibility
    def assert_called(self):
        """Assert that the mock has been called at least once."""
        assert self.call_count > 0, "Expected to be called but was not called"

    def assert_called_once(self):
        """Assert that the mock has been called exactly once."""
        assert self.call_count == 1, f"Expected to be called once but was called {self.call_count} times"

    def assert_not_called(self):
        """Assert that the mock has not been called."""
        assert self.call_count == 0, f"Expected not to be called but was called {self.call_count} times"

    @property
    def call_args(self) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
        """Get the arguments of the last call."""
        return self.call_args_list[-1] if self.call_args_list else None

    @property
    def called(self) -> bool:
        """Check if the mock has been called."""
        return self.call_count > 0


class AwaitableResult:
    """
    A result object that can be both used directly and awaited.

    This wraps a return value and makes it accessible both synchronously
    and asynchronously.
    """

    def __init__(self, value: Any):
        """
        Initialize with a value to wrap.

        Args:
            value: The value to wrap and make awaitable
        """
        self.value = value

        # Copy all attributes from the original value
        for attr_name in dir(value):
            if not attr_name.startswith("_"):
                with contextlib.suppress(AttributeError, TypeError):
                    setattr(self, attr_name, getattr(value, attr_name))

    def __getattr__(self, name):
        """Delegate any missing attributes to the wrapped value."""
        return getattr(self.value, name)

    def __await__(self):
        """Make this object awaitable, returning the wrapped value."""

        async def _async():
            return self.value

        return _async().__await__()


def create_dual_usage_mock(return_value: Any) -> AwaitableCallableMock:
    """
    Create a mock that can handle both sync and async usage patterns.

    Args:
        return_value: The value to return when called or awaited

    Returns:
        AwaitableCallableMock instance
    """
    return AwaitableCallableMock(return_value)


def create_qdrant_client_mock() -> MagicMock:
    """
    Create a comprehensive mock for QdrantClient with dual usage patterns.

    Returns:
        MagicMock configured for QdrantClient with AwaitableCallableMock
        for methods that have dual usage patterns.
    """
    mock_client = MagicMock()

    # Mock successful collection operations
    mock_collections_response = MagicMock()

    # Create collection objects with proper .name attributes
    test_collection = MagicMock()
    test_collection.name = "test_collection"

    default_collection = MagicMock()
    default_collection.name = "default"

    mock_collections_response.collections = [test_collection, default_collection]
    mock_client.get_collections.return_value = mock_collections_response

    # Mock successful collection info
    mock_collection_info = MagicMock()
    mock_collection_info.config = MagicMock()
    mock_collection_info.config.params = MagicMock()
    mock_collection_info.config.params.vectors = MagicMock()
    mock_collection_info.config.params.vectors.size = 384  # DEFAULT_VECTOR_DIMENSIONS
    mock_collection_info.config.params.vectors.distance = MagicMock()
    mock_collection_info.config.params.vectors.distance.value = "Cosine"
    mock_collection_info.points_count = 42
    mock_collection_info.segments_count = 1
    mock_collection_info.status = MagicMock()
    mock_collection_info.status.value = "green"
    mock_client.get_collection.return_value = mock_collection_info

    # Mock successful search results
    mock_search_hit = MagicMock()
    mock_search_hit.id = "test_doc_1"
    mock_search_hit.score = 0.95
    mock_search_hit.payload = {
        "content": "Test document content about Python caching strategies",
        "metadata": {"category": "performance", "language": "python"},
    }
    mock_search_hit.vector = [0.1] * 384
    mock_client.search.return_value = [mock_search_hit]

    # Mock successful upsert operation with dual usage pattern
    mock_upsert_result = MagicMock()
    mock_upsert_result.status = "completed"
    mock_client.upsert = create_dual_usage_mock(mock_upsert_result)

    # Mock successful delete operation with dual usage pattern
    mock_delete_result = MagicMock()
    mock_delete_result.status = "completed"
    mock_client.delete = create_dual_usage_mock(mock_delete_result)

    return mock_client
