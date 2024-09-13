# Monitoring service
# monitoring_service.py

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, BatchSpanProcessor
from opentelemetry.exporter.zipkin.json import ZipkinExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._logs import LoggingHandler

import logging

# Set up OpenTelemetry Tracer
trace.set_tracer_provider(
    TracerProvider(resource=Resource.create({SERVICE_NAME: "monitoring_service"}))
)

# Set up Zipkin Exporter (or Jaeger/OTLP)
zipkin_exporter = ZipkinExporter(endpoint="http://<zipkin-endpoint>:9411/api/v2/spans")

# Export spans to console (for testing)
console_exporter = ConsoleSpanExporter()

# Configure span processor (either Simple or Batch)
span_processor = BatchSpanProcessor(zipkin_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Optionally, log traces to console
trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(console_exporter))

# Instrument requests to collect traces from HTTP calls
RequestsInstrumentor().instrument()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a tracer
tracer = trace.get_tracer(__name__)

def track_pdf_conversion(pdf_id):
    """
    Example function to track PDF processing and emit traces
    """
    with tracer.start_as_current_span(f"PDF Conversion - {pdf_id}"):
        logger.info(f"Tracking PDF conversion process for {pdf_id}")
        # Simulate some processing steps with spans
        with tracer.start_as_current_span("Extracting PDF content"):
            # Simulate PDF extraction
            pass
        
        with tracer.start_as_current_span("Converting to Markdown"):
            # Simulate conversion
            pass

        logger.info(f"PDF conversion completed for {pdf_id}")

def track_dataset_augmentation(markdown_file):
    """
    Example function to track dataset augmentation process and emit traces
    """
    with tracer.start_as_current_span(f"Dataset Augmentation - {markdown_file}"):
        logger.info(f"Tracking dataset augmentation for {markdown_file}")
        # Simulate dataset augmentation steps
        pass
        logger.info(f"Dataset augmentation completed for {markdown_file}")

def track_model_training(dataset_path):
    """
    Example function to track model training and emit traces
    """
    with tracer.start_as_current_span(f"Model Training - {dataset_path}"):
        logger.info(f"Tracking model training for {dataset_path}")
        # Simulate model training
        pass
        logger.info(f"Model training completed for {dataset_path}")

if __name__ == "__main__":
    # Example usage
    track_pdf_conversion("pdf-1234")
    track_dataset_augmentation("/path/to/markdown.md")
    track_model_training("/path/to/augmented_dataset")
