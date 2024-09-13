# Dataset Augmentation Service
# core_services/data_augmentation_service.py
import os
import subprocess
import json
from dapr.clients import DaprClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Set up OpenTelemetry tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4317")
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

def run_augmentation_pipeline(markdown_file):
    """
    Run the processing.py script from the augmentoolkit/original folder.
    This script augments the dataset using the converted Markdown file.
    
    Args:
        markdown_file (str): The path to the Markdown file.
    
    Returns:
        str: The path to the augmented dataset.
    """
    with tracer.start_as_current_span("run_augmentation_pipeline"):
        augmenttool_root = "/augmenttool"  # Adjust to your actual path if necessary
        processing_script = os.path.join(augmenttool_root, 'augmentoolkit', 'original', 'processing.py')

        try:
            subprocess.run(['python', processing_script], check=True)
        except subprocess.CalledProcessError as e:
            # TODO: Add better error handling for failed augmentation
            raise Exception(f"Dataset augmentation failed: {e}")

        # Assuming the augmented dataset is saved in the same directory as the Markdown file
        return os.path.join(os.path.dirname(markdown_file), "augmented_dataset")

def handle_conversion_complete_event(event_data: dict):
    """
    Handles the conversion complete event and triggers the dataset augmentation pipeline.
    
    Args:
        event_data (dict): The event data containing the markdown_file path and other info.
    """
    with tracer.start_as_current_span("handle_conversion_complete_event"):
        markdown_file = event_data.get("markdown_file")
        if not markdown_file or not os.path.exists(markdown_file):
            raise Exception(f"Markdown file {markdown_file} not found.")
        
        # Trigger the dataset augmentation pipeline
        augmented_dataset = run_augmentation_pipeline(markdown_file)
        
        # Publish an event to trigger model training
        with DaprClient() as dapr_client:
            dapr_client.publish_event(
                pubsub_name="pubsub",
                topic_name="dataset-augmented",
                data=json.dumps({"augmented_dataset_path": augmented_dataset})
            )

# TODO: Add the subscription mechanism to listen for the "conversion-complete" event
# and pass the event data to `handle_conversion_complete_event`.
