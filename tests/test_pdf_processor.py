"""
Unit tests for PDF Processor Service

Tests the core functionality of the PDF processing service including
PDF to Markdown conversion and event publishing.
"""

import os
import pytest
import tempfile
from unittest.mock import Mock, patch, MagicMock
from core_services.pdf_processor import (
    convert_pdf_to_markdown,
    save_processing_state,
    publish_conversion_complete,
    process_pdf,
    handle_pdf_upload_event
)


class TestPDFProcessor:
    """Test suite for PDF processor functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def sample_pdf_path(self, temp_dir):
        """Create a sample PDF file path."""
        pdf_path = os.path.join(temp_dir, "test_document.pdf")
        # Create an empty file to simulate a PDF
        with open(pdf_path, 'w') as f:
            f.write("Mock PDF content")
        return pdf_path

    @patch('core_services.pdf_processor.requests.post')
    def test_save_processing_state_success(self, mock_post):
        """Test successful state saving."""
        mock_post.return_value.status_code = 204

        # Should not raise an exception
        save_processing_state("test-pdf-123", "processing")

        # Verify the request was made
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert "statestore" in args[0]
        assert kwargs['json'][0]['key'] == "test-pdf-123"
        assert kwargs['json'][0]['value'] == "processing"

    @patch('core_services.pdf_processor.requests.post')
    def test_save_processing_state_failure(self, mock_post):
        """Test state saving failure handling."""
        mock_post.return_value.status_code = 500
        mock_post.return_value.text = "Internal server error"

        with pytest.raises(Exception) as exc_info:
            save_processing_state("test-pdf-123", "processing")

        assert "Failed to save state" in str(exc_info.value)

    @patch('core_services.pdf_processor.requests.post')
    def test_publish_conversion_complete_success(self, mock_post):
        """Test successful event publishing."""
        mock_post.return_value.status_code = 204

        publish_conversion_complete("test-pdf-123", "/path/to/output.md")

        # Verify the event was published
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert "conversion-complete" in args[0]
        assert kwargs['json']['pdf_id'] == "test-pdf-123"
        assert kwargs['json']['markdown_file'] == "/path/to/output.md"

    @patch('core_services.pdf_processor.requests.post')
    def test_publish_conversion_complete_failure(self, mock_post):
        """Test event publishing failure handling."""
        mock_post.return_value.status_code = 500
        mock_post.return_value.text = "Internal server error"

        with pytest.raises(Exception) as exc_info:
            publish_conversion_complete("test-pdf-123", "/path/to/output.md")

        assert "Failed to publish event" in str(exc_info.value)

    def test_process_pdf_creates_markdown(self, temp_dir):
        """Test that process_pdf creates a markdown file."""
        pdf_path = os.path.join(temp_dir, "test.pdf")
        with open(pdf_path, 'w') as f:
            f.write("Mock PDF")

        markdown_path = process_pdf(pdf_path)

        # Check that markdown file was created
        assert os.path.exists(markdown_path)
        assert markdown_path.endswith(".md")

        # Check content
        with open(markdown_path, 'r') as f:
            content = f.read()
            assert "Converted Markdown" in content

    def test_handle_pdf_upload_event_missing_file(self):
        """Test handling of event with missing PDF file."""
        event_data = {"pdf_file": "/nonexistent/file.pdf"}

        with pytest.raises(Exception) as exc_info:
            handle_pdf_upload_event(event_data)

        assert "not found" in str(exc_info.value)

    @patch('core_services.pdf_processor.DaprClient')
    def test_handle_pdf_upload_event_success(self, mock_dapr_client, temp_dir):
        """Test successful PDF upload event handling."""
        # Create a test PDF file
        pdf_path = os.path.join(temp_dir, "test.pdf")
        with open(pdf_path, 'w') as f:
            f.write("Mock PDF")

        event_data = {"pdf_file": pdf_path}

        # Mock the Dapr client
        mock_client_instance = MagicMock()
        mock_dapr_client.return_value.__enter__.return_value = mock_client_instance

        # Handle the event
        handle_pdf_upload_event(event_data)

        # Verify that publish_event was called
        mock_client_instance.publish_event.assert_called_once()
        call_args = mock_client_instance.publish_event.call_args
        assert call_args[1]['topic_name'] == "conversion-complete"


class TestPDFConversion:
    """Test suite for PDF to Markdown conversion."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @patch('core_services.pdf_processor.subprocess.run')
    @patch('core_services.pdf_processor.requests.post')
    @patch('core_services.pdf_processor.shutil.move')
    def test_convert_pdf_to_markdown_success(self, mock_move, mock_post, mock_subprocess, temp_dir):
        """Test successful PDF to Markdown conversion."""
        # Setup mocks
        mock_post.return_value.status_code = 204
        mock_subprocess.return_value = None

        pdf_path = os.path.join(temp_dir, "test.pdf")
        output_folder = os.path.join(temp_dir, "output")

        # Create the PDF file
        with open(pdf_path, 'w') as f:
            f.write("Mock PDF")

        # Run conversion
        result = convert_pdf_to_markdown(pdf_path, output_folder)

        # Verify subprocess was called with marker command
        mock_subprocess.assert_called_once()
        assert 'marker' in str(mock_subprocess.call_args)

        # Verify state was saved
        assert mock_post.call_count >= 2  # Initial and completed states

    @patch('core_services.pdf_processor.subprocess.run')
    @patch('core_services.pdf_processor.requests.post')
    def test_convert_pdf_to_markdown_subprocess_failure(self, mock_post, mock_subprocess, temp_dir):
        """Test handling of subprocess failure during conversion."""
        # Setup mocks
        mock_post.return_value.status_code = 204
        mock_subprocess.side_effect = Exception("Marker failed")

        pdf_path = os.path.join(temp_dir, "test.pdf")
        output_folder = os.path.join(temp_dir, "output")

        # Create the PDF file
        with open(pdf_path, 'w') as f:
            f.write("Mock PDF")

        # Conversion should raise an exception
        with pytest.raises(Exception) as exc_info:
            convert_pdf_to_markdown(pdf_path, output_folder)

        assert "PDF conversion failed" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
