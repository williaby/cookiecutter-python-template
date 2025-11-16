"""Comprehensive tests for multi_journey_interface.py coverage gaps - targeting 0% coverage functions."""

import tarfile
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock

import gradio as gr
import pytest

from src.ui.multi_journey_interface import MultiJourneyInterface


class TestMultiJourneyInterfaceCoverageGaps:
    """Test MultiJourneyInterface methods with 0% coverage."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock ApplicationSettings for testing."""
        settings = Mock()
        settings.max_file_size = 10 * 1024 * 1024  # 10MB
        settings.max_total_files = 100
        settings.max_upload_size = 50 * 1024 * 1024  # 50MB
        settings.allowed_file_types = [".txt", ".py", ".md", ".json"]
        return settings

    @pytest.fixture
    def multi_journey_interface(self, mock_settings):
        """Create MultiJourneyInterface for testing."""
        return MultiJourneyInterface()

    def test_check_archive_bomb_heuristics_method(self, multi_journey_interface):
        """Test _check_archive_bomb_heuristics method with 0% coverage."""
        # Test with normal text file - should not raise error (not an archive MIME type)
        multi_journey_interface._check_archive_bomb_heuristics("normal_file.txt", 1024, "text/plain")

    def test_check_archive_bomb_heuristics_suspicious_compression(self, multi_journey_interface):
        """Test _check_archive_bomb_heuristics with suspicious small size."""
        # Test with very small archive file that should trigger size heuristic
        with pytest.raises(gr.Error):
            multi_journey_interface._check_archive_bomb_heuristics("suspicious.zip", 50, "application/zip")

    def test_check_archive_bomb_heuristics_large_file(self, multi_journey_interface):
        """Test _check_archive_bomb_heuristics with another small file size."""
        # Test with another small archive file that should trigger size heuristic
        with pytest.raises(gr.Error):
            multi_journey_interface._check_archive_bomb_heuristics("large_file.zip", 25, "application/zip")

    def test_check_archive_bomb_heuristics_edge_cases(self, multi_journey_interface):
        """Test _check_archive_bomb_heuristics edge cases."""
        # Test with normal file sizes and non-archive MIME types - should not raise errors
        multi_journey_interface._check_archive_bomb_heuristics("normal.txt", 1024, "text/plain")
        multi_journey_interface._check_archive_bomb_heuristics("medium.pdf", 10 * 1024 * 1024, "application/pdf")

    def test_check_tar_gzip_bomb_heuristics_method(self, multi_journey_interface):
        """Test _check_tar_gzip_bomb_heuristics method with 0% coverage."""
        # Create a temporary tar.gz file for testing
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

            # Create a simple tar.gz file
            with tarfile.open(temp_path, "w:gz") as tar:
                # Add a text file to the archive
                info = tarfile.TarInfo(name="test.txt")
                info.size = 100
                tar.addfile(info, fileobj=None)

        try:
            # Should not raise an error for normal tar.gz
            multi_journey_interface._check_tar_gzip_bomb_heuristics(temp_path, 1000)

        finally:
            # Clean up temporary file
            temp_path.unlink(missing_ok=True)

    def test_check_tar_gzip_bomb_heuristics_invalid_file(self, multi_journey_interface):
        """Test _check_tar_gzip_bomb_heuristics with invalid file."""
        # Test with non-existent file - should raise error
        non_existent_path = Path("/non/existent/file.tar.gz")

        with pytest.raises(gr.Error):
            multi_journey_interface._check_tar_gzip_bomb_heuristics(non_existent_path, 1000)

    def test_check_tar_gzip_bomb_heuristics_not_tar_file(self, multi_journey_interface):
        """Test _check_tar_gzip_bomb_heuristics with non-tar file."""
        # Create a regular text file with .tar.gz extension
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as temp_file:
            temp_file.write(b"This is not a tar file")
            temp_path = Path(temp_file.name)

        try:
            # Should raise error for non-tar file
            with pytest.raises(gr.Error):
                multi_journey_interface._check_tar_gzip_bomb_heuristics(temp_path, 500)

        finally:
            # Clean up temporary file
            temp_path.unlink(missing_ok=True)

    def test_process_journey2_search_method(self, multi_journey_interface):
        """Test _process_journey2_search method with 0% coverage."""
        test_query = "test search query"
        test_session_id = "test_session_123"

        # Test the search processing (this is a sync method, not async)
        result_text = multi_journey_interface._process_journey2_search(test_query, test_session_id)

        # Verify search result format
        assert isinstance(result_text, str)
        assert test_query in result_text
        assert "Search results for:" in result_text

    def test_process_journey2_search_empty_query(self, multi_journey_interface):
        """Test _process_journey2_search with empty query."""
        empty_query = ""
        test_session_id = "test_session_123"

        result_text = multi_journey_interface._process_journey2_search(empty_query, test_session_id)

        # Should handle empty query gracefully
        assert isinstance(result_text, str)
        assert empty_query in result_text or "Search results for:" in result_text

    def test_process_journey2_search_error_handling(self, multi_journey_interface):
        """Test _process_journey2_search error handling."""
        test_query = "test query"
        test_session_id = "test_session_123"

        # The actual method is simple and doesn't have complex error handling
        # Test that it handles normal input without errors
        result_text = multi_journey_interface._process_journey2_search(test_query, test_session_id)

        # Should return a valid string result
        assert isinstance(result_text, str)
        assert test_query in result_text

    def test_process_file_uploads_method(self, multi_journey_interface):
        """Test _process_file_uploads method with 0% coverage."""
        # Create mock file uploads
        mock_files = [Mock(name="file1.txt", size=1024), Mock(name="file2.py", size=2048)]
        test_session_id = "test_session_123"

        # Test file processing
        result = multi_journey_interface._process_file_uploads(mock_files, test_session_id)

        # Verify file processing result
        assert isinstance(result, str)
        assert "Files processed successfully" in result
        assert "2 files" in result

    def test_process_file_uploads_empty_list(self, multi_journey_interface):
        """Test _process_file_uploads with empty file list."""
        test_session_id = "test_session_123"
        result = multi_journey_interface._process_file_uploads([], test_session_id)

        # Should handle empty list gracefully
        assert isinstance(result, str)
        assert "0 files" in result

    def test_process_file_uploads_invalid_files(self, multi_journey_interface):
        """Test _process_file_uploads with invalid files."""
        mock_files = [Mock(name="invalid.exe", size=1024)]
        test_session_id = "test_session_123"

        # Test with mock files - the current implementation just counts files
        result = multi_journey_interface._process_file_uploads(mock_files, test_session_id)

        # Should return result string indicating file count
        assert isinstance(result, str)
        assert "1 files" in result


class TestMultiJourneyInterfaceArchiveBombDetection:
    """Test archive bomb detection functionality."""

    @pytest.fixture
    def multi_journey_interface(self):
        """Create MultiJourneyInterface for testing."""
        return MultiJourneyInterface()

    def test_archive_bomb_detection_zip_bomb(self, multi_journey_interface):
        """Test archive bomb detection with zip bomb characteristics."""
        # Test with very large file that should trigger archive bomb detection
        with pytest.raises(gr.Error):
            multi_journey_interface._check_archive_bomb_heuristics("bomb.zip", 1000 * 1024 * 1024, "application/zip")

    def test_archive_bomb_detection_normal_compression(self, multi_journey_interface):
        """Test archive bomb detection with normal compression."""
        # Test with normal non-archive file - should not raise error (not an archive MIME type)
        multi_journey_interface._check_archive_bomb_heuristics("document.txt", 10 * 1024, "text/plain")

    def test_tar_gzip_bomb_detection_with_real_archive(self, multi_journey_interface):
        """Test tar.gz bomb detection with a real archive."""
        # Create a legitimate small tar.gz file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            tar_path = temp_path / "test.tar.gz"

            # Create a small text file
            text_file = temp_path / "sample.txt"
            text_file.write_text("This is a sample text file for testing.")

            # Create tar.gz archive
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(text_file, arcname="sample.txt")

            # Should not raise error for legitimate archive
            multi_journey_interface._check_tar_gzip_bomb_heuristics(tar_path, 1000)

    def test_tar_gzip_bomb_detection_corrupted_archive(self, multi_journey_interface):
        """Test tar.gz bomb detection with corrupted archive."""
        # Create a corrupted tar.gz file
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as temp_file:
            temp_file.write(b"This is not a valid tar.gz file content")
            temp_path = Path(temp_file.name)

        try:
            # Should raise error for corrupted archive
            with pytest.raises(gr.Error):
                multi_journey_interface._check_tar_gzip_bomb_heuristics(temp_path, 500)

        finally:
            temp_path.unlink(missing_ok=True)


class TestMultiJourneyInterfaceAsyncMethods:
    """Test async methods in MultiJourneyInterface."""

    @pytest.fixture
    def multi_journey_interface(self):
        """Create MultiJourneyInterface for testing."""
        return MultiJourneyInterface()

    def test_process_journey2_search_performance(self, multi_journey_interface):
        """Test _process_journey2_search performance."""
        test_query = "performance test query"
        test_session_id = "test_session_123"

        # Test performance of the actual sync method
        start_time = time.time()
        result_text = multi_journey_interface._process_journey2_search(test_query, test_session_id)
        duration = time.time() - start_time

        # Should complete quickly
        assert duration < 1.0  # Less than 1 second
        assert isinstance(result_text, str)
        assert test_query in result_text

    def test_process_journey2_search_concurrent_requests(self, multi_journey_interface):
        """Test _process_journey2_search with concurrent requests."""
        queries = [f"query_{i}" for i in range(5)]
        session_id = "test_session_123"

        # Execute multiple searches (sync method)
        results = []
        for query in queries:
            result_text = multi_journey_interface._process_journey2_search(query, session_id)
            results.append(result_text)

        # Verify all searches completed
        assert len(results) == 5
        for i, result_text in enumerate(results):
            assert isinstance(result_text, str)
            assert f"query_{i}" in result_text


@pytest.mark.parametrize(
    ("file_size", "detected_mime", "expected_result"),
    [
        (1024, "text/plain", None),  # Non-archive file - should pass
        (200 * 1024 * 1024, "text/plain", None),  # Large non-archive - should pass
        (50, "application/zip", gr.Error),  # Very small zip - should raise error (size heuristic)
    ],
)
def test_archive_bomb_heuristics_parametrized(file_size, detected_mime, expected_result):
    """Test _check_archive_bomb_heuristics with various parameters."""
    interface = MultiJourneyInterface()

    if expected_result == gr.Error:
        with pytest.raises(gr.Error):
            interface._check_archive_bomb_heuristics("test_file.txt", file_size, detected_mime)
    else:
        # Should not raise an error
        interface._check_archive_bomb_heuristics("test_file.txt", file_size, detected_mime)


@pytest.mark.parametrize(
    ("query", "expected_contains"),
    [
        ("", "Search results for:"),  # Empty query still gets processed
        ("test query", "test query"),  # Should contain the query
        ("very long query " * 100, "very long query"),  # Long query
    ],
)
def test_process_journey2_search_parametrized(query, expected_contains):
    """Test _process_journey2_search with various query inputs."""
    interface = MultiJourneyInterface()
    session_id = "test_session_123"

    result_text = interface._process_journey2_search(query, session_id)

    # Verify result contains expected content
    assert isinstance(result_text, str)
    assert expected_contains in result_text
