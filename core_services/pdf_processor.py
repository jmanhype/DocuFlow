# PDF processing service
# core_services/pdf_processor.py
import os
import subprocess
import requests
import shutil
import json
import logging
from typing import Dict, Any, Optional
from dapr.clients import DaprClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dapr endpoints
DAPR_STATE_URL = "http://localhost:3500/v1.0/state/statestore"
DAPR_PUBSUB_URL = "http://localhost:3500/v1.0/publish/pubsub/conversion-complete"

# Raw text input folder for dataset augmentation
RAW_TEXT_INPUT_FOLDER = "/augmenttool/raw_text_input"

# Set up OpenTelemetry tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4317")
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

def save_processing_state(pdf_id: str, state: str) -> None:
    """
    Save the processing state of the PDF in Dapr's state store.

    Args:
        pdf_id: The ID of the PDF document being processed.
        state: The current state of the processing (e.g., 'processing', 'completed').

    Raises:
        Exception: If saving state fails.
    """
    state_data = [
        {
            "key": pdf_id,
            "value": state
        }
    ]
    try:
        logger.info(f"Saving processing state for PDF {pdf_id}: {state}")
        response = requests.post(DAPR_STATE_URL, json=state_data, timeout=10)
        if response.status_code != 204:
            logger.error(f"Failed to save state for PDF {pdf_id}. Status: {response.status_code}")
            raise Exception(f"Failed to save state for PDF {pdf_id}: {response.text}")
        logger.debug(f"State saved successfully for PDF {pdf_id}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error while saving state for PDF {pdf_id}: {e}")
        raise Exception(f"Network error saving state for PDF {pdf_id}: {e}")

def publish_conversion_complete(pdf_id: str, markdown_file: str) -> None:
    """
    Publish an event to notify other services that PDF processing is complete.

    Args:
        pdf_id: The ID of the PDF document being processed.
        markdown_file: The path to the converted Markdown file.

    Raises:
        Exception: If publishing the event fails.
    """
    event_data = {
        "pdf_id": pdf_id,
        "markdown_file": markdown_file
    }
    try:
        logger.info(f"Publishing conversion complete event for PDF {pdf_id}")
        response = requests.post(DAPR_PUBSUB_URL, json=event_data, timeout=10)
        if response.status_code != 204:
            logger.error(f"Failed to publish event for PDF {pdf_id}. Status: {response.status_code}")
            raise Exception(f"Failed to publish event for PDF {pdf_id}: {response.text}")
        logger.debug(f"Event published successfully for PDF {pdf_id}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error while publishing event for PDF {pdf_id}: {e}")
        raise Exception(f"Network error publishing event for PDF {pdf_id}: {e}")

def convert_pdf_to_markdown(pdf_path: str, output_folder: str) -> str:
    """
    Convert a PDF file to Markdown using the Marker library and move it to the raw text input folder.

    Args:
        pdf_path: The path to the PDF file.
        output_folder: The folder where the Markdown file will be temporarily saved.

    Returns:
        The path to the generated Markdown file in the raw text input folder.

    Raises:
        FileNotFoundError: If the PDF file doesn't exist.
        Exception: If conversion or file operations fail.
    """
    # Validate PDF path
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not pdf_path.lower().endswith('.pdf'):
        logger.error(f"Invalid PDF file extension: {pdf_path}")
        raise ValueError(f"File must have .pdf extension: {pdf_path}")

    # Extract the PDF ID (could be the file name without extension)
    pdf_id = os.path.basename(pdf_path).replace('.pdf', '')
    logger.info(f"Starting conversion for PDF: {pdf_id}")

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Define output markdown file path
    temp_markdown_file = os.path.join(output_folder, pdf_id + '.md')

    # Save initial processing state
    save_processing_state(pdf_id, 'processing')

    # Call Marker to convert PDF to Markdown
    try:
        logger.info(f"Running Marker conversion for {pdf_id}")
        result = subprocess.run(
            ['marker', pdf_path, '-o', temp_markdown_file],
            check=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        logger.debug(f"Marker conversion completed for {pdf_id}")
    except subprocess.CalledProcessError as e:
        save_processing_state(pdf_id, 'failed')
        logger.error(f"PDF conversion failed for {pdf_id}: {e.stderr if e.stderr else str(e)}")
        raise Exception(f"PDF conversion failed for {pdf_id}: {e.stderr if e.stderr else str(e)}")
    except subprocess.TimeoutExpired:
        save_processing_state(pdf_id, 'failed')
        logger.error(f"PDF conversion timed out for {pdf_id}")
        raise Exception(f"PDF conversion timed out for {pdf_id}")
    except FileNotFoundError:
        save_processing_state(pdf_id, 'failed')
        logger.error("Marker tool not found. Please ensure it is installed.")
        raise Exception("Marker tool not found. Please install with: pip install marker-pdf")

    # Verify the markdown file was created
    if not os.path.exists(temp_markdown_file):
        save_processing_state(pdf_id, 'failed')
        logger.error(f"Markdown file was not created: {temp_markdown_file}")
        raise Exception(f"Markdown file was not created: {temp_markdown_file}")

    # Ensure the raw text input folder exists
    os.makedirs(RAW_TEXT_INPUT_FOLDER, exist_ok=True)

    # Move the generated Markdown file to the raw text input folder
    final_markdown_file = os.path.join(RAW_TEXT_INPUT_FOLDER, pdf_id + '.md')
    try:
        shutil.move(temp_markdown_file, final_markdown_file)
        logger.info(f"Moved markdown file to: {final_markdown_file}")
    except Exception as e:
        save_processing_state(pdf_id, 'failed')
        logger.error(f"Failed to move markdown file: {e}")
        raise

    # Save the completed state
    save_processing_state(pdf_id, 'completed')

    # Publish an event that the conversion is complete
    publish_conversion_complete(pdf_id, final_markdown_file)

    logger.info(f"Conversion completed successfully for {pdf_id}")
    return final_markdown_file

def process_pdf(pdf_file: str) -> str:
    with tracer.start_as_current_span("process_pdf"):
        # Simulate PDF processing
        markdown_file = pdf_file.replace('.pdf', '.md')
        
        with tracer.start_as_current_span("convert_pdf_to_markdown"):
            # TODO: Implement actual PDF to Markdown conversion
            with open(markdown_file, 'w') as f:
                f.write("# Converted Markdown\n\nThis is a placeholder for the converted content.")
        
        return markdown_file

def handle_pdf_upload_event(event_data: Dict[str, str]):
    with tracer.start_as_current_span("handle_pdf_upload_event"):
        pdf_file = event_data.get("pdf_file")
        if not pdf_file or not os.path.exists(pdf_file):
            raise Exception(f"PDF file {pdf_file} not found.")
        
        markdown_file = process_pdf(pdf_file)
        
        with tracer.start_as_current_span("publish_conversion_complete_event"):
            with DaprClient() as dapr_client:
                dapr_client.publish_event(
                    pubsub_name="pubsub",
                    topic_name="conversion-complete",
                    data=json.dumps({"markdown_file": markdown_file})
                )

# TODO: Add the subscription mechanism to listen for the "pdf-uploaded" event
# and pass the event data to `handle_pdf_upload_event`.

if __name__ == "__main__":
    from dapr.ext.fastapi import DaprApp
    from fastapi import FastAPI
    import uvicorn

    app = FastAPI()
    dapr_app = DaprApp(app)

    @dapr_app.subscribe(pubsub_name="pubsub", topic="pdf-uploaded")
    def pdf_uploaded(event: Dict[str, Any]):
        handle_pdf_upload_event(event.get('data', {}))

    uvicorn.run(app, host="0.0.0.0", port=8001)

# Example usage
# TODO: Create command-line argument parsing to make this a more flexible CLI tool.
# pdf_file_path = 'path/to/your/file.pdf'
# output_dir = 'path/to/temp/output/folder'
# convert_pdf_to_markdown(pdf_file_path, output_dir)
