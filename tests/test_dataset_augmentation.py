"""
Unit tests for Dataset Augmentation Service

Tests the dataset augmentation pipeline and event handling.
"""

import os
import pytest
import tempfile
from unittest.mock import Mock, patch, MagicMock
from core_services.dataset_augmentation_service import (
    run_augmentation_pipeline,
    handle_conversion_complete_event
)


class TestDatasetAugmentation:
    """Test suite for dataset augmentation functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def sample_markdown_file(self, temp_dir):
        """Create a sample markdown file."""
        md_path = os.path.join(temp_dir, "test_document.md")
        with open(md_path, 'w') as f:
            f.write("# Test Document\n\nThis is a test document.")
        return md_path

    @patch('core_services.dataset_augmentation_service.subprocess.run')
    def test_run_augmentation_pipeline_success(self, mock_subprocess, sample_markdown_file):
        """Test successful augmentation pipeline execution."""
        mock_subprocess.return_value = None

        result = run_augmentation_pipeline(sample_markdown_file)

        # Verify subprocess was called
        mock_subprocess.assert_called_once()
        assert 'processing.py' in str(mock_subprocess.call_args)

        # Verify result path
        assert result is not None
        assert "augmented_dataset" in result

    @patch('core_services.dataset_augmentation_service.subprocess.run')
    def test_run_augmentation_pipeline_failure(self, mock_subprocess, sample_markdown_file):
        """Test handling of augmentation pipeline failure."""
        from subprocess import CalledProcessError

        mock_subprocess.side_effect = CalledProcessError(1, 'python')

        with pytest.raises(Exception) as exc_info:
            run_augmentation_pipeline(sample_markdown_file)

        assert "Dataset augmentation failed" in str(exc_info.value)

    def test_handle_conversion_complete_event_missing_file(self):
        """Test handling of event with missing markdown file."""
        event_data = {"markdown_file": "/nonexistent/file.md"}

        with pytest.raises(Exception) as exc_info:
            handle_conversion_complete_event(event_data)

        assert "not found" in str(exc_info.value)

    @patch('core_services.dataset_augmentation_service.subprocess.run')
    @patch('core_services.dataset_augmentation_service.DaprClient')
    def test_handle_conversion_complete_event_success(
        self, mock_dapr_client, mock_subprocess, sample_markdown_file
    ):
        """Test successful handling of conversion complete event."""
        # Setup mocks
        mock_subprocess.return_value = None
        mock_client_instance = MagicMock()
        mock_dapr_client.return_value.__enter__.return_value = mock_client_instance

        event_data = {"markdown_file": sample_markdown_file}

        # Handle the event
        handle_conversion_complete_event(event_data)

        # Verify subprocess was called
        mock_subprocess.assert_called_once()

        # Verify event was published
        mock_client_instance.publish_event.assert_called_once()
        call_args = mock_client_instance.publish_event.call_args
        assert call_args[1]['topic_name'] == "dataset-augmented"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
