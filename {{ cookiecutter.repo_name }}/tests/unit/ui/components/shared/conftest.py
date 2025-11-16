"""
Pytest fixtures for UI components shared module tests.

This module provides common fixtures for testing UI components,
including export utilities and other shared components.
"""

from unittest.mock import patch

import pytest


@pytest.fixture
def mock_datetime():
    """Mock datetime.now() to return consistent timestamp for testing."""
    from unittest.mock import Mock

    with patch("src.ui.components.shared.export_utils.datetime") as mock_dt:
        # Create a mock datetime object with proper method mocking
        mock_now = Mock()
        mock_now.isoformat.return_value = "2024-01-01T12:00:00"
        mock_now.strftime.return_value = "20240101_120000"

        mock_dt.now.return_value = mock_now

        yield mock_dt


@pytest.fixture
def sample_export_data():
    """Sample data for testing export utilities."""
    return {
        "enhanced_prompt": "Create a Python function that calculates the factorial of a number using recursion.",
        "create_breakdown": {
            "context": "You are a Python programming assistant helping with algorithmic solutions.",
            "request": "Create a factorial function using recursion with proper error handling.",
            "examples": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
            "augmentations": "Include input validation and documentation.",
            "tone_format": "Professional and educational with clear comments.",
            "evaluation": "Test with various inputs including edge cases.",
        },
        "model_info": {"model": "claude-3-5-sonnet-20241022", "response_time": 1.234, "cost": 0.0123},
        "file_sources": [
            {"name": "test_export.py", "type": "python", "size": 1024},
            {"name": "utils.py", "type": "python", "size": 2048},
        ],
        "session_data": {"total_cost": 0.0456, "request_count": 3, "avg_response_time": 1.567},
    }


@pytest.fixture
def complex_content_sample():
    """Complex content with multiple code blocks for testing extraction."""
    return '''Here's some Python code:

```python
def calculate_score(data):
    """Calculate the average score from data."""
    total = sum(data)  # Sum all values
    return total / len(data)
```

And here's some JavaScript:

```javascript
function processData(items) {
    // Process each item
    return items.map(item => item.value * 2);
}
```

Some SQL query:

```sql
SELECT * FROM users
WHERE active = true
ORDER BY created_at DESC;
```

Plain text block:

```
No special formatting here.
Just plain text content.
```

Regular text continues here...
'''


@pytest.fixture
def sample_code_blocks():
    """Sample code blocks data for testing formatting functions."""
    return [
        {
            "id": 1,
            "language": "python",
            "content": """def test_example():
    \"\"\"Test function docstring.\"\"\"
    return 42""",
            "line_count": 3,
            "char_count": 58,
            "comments": ["Test function docstring."],
            "has_functions": True,
            "complexity": "simple",
            "preview": "def test_example()...",
        },
        {
            "id": 2,
            "language": "javascript",
            "content": """function getData() {
    // Fetch data from API
    return fetch('/api/data');
}""",
            "line_count": 4,
            "char_count": 76,
            "comments": ["Fetch data from API"],
            "has_functions": True,
            "complexity": "simple",
            "preview": "function getData() {",
        },
    ]
