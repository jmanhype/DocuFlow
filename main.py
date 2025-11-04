"""
DocuFlow Service Colony - Main Entry Point

This is the main entry point for the DocuFlow service colony framework.
It orchestrates the startup of core services and handles service discovery.
"""

import logging
import os
from typing import Optional
from fastapi import FastAPI
from dapr.ext.fastapi import DaprApp
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DocuFlow Service Colony",
    description="Distributed service colony for document processing and ML workflows",
    version="1.0.0"
)

# Initialize Dapr app
dapr_app = DaprApp(app)


@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {
        "service": "DocuFlow Service Colony",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/info")
async def service_info():
    """Return service information."""
    return {
        "services": [
            "pdf_processor",
            "dataset_augmentation",
            "model_training",
            "data_retrieval",
            "monitoring",
            "strategic_reasoning"
        ],
        "description": "DocuFlow Service Colony - Distributed document processing and ML framework"
    }


def main():
    """Main entry point for the application."""
    logger.info("Starting DocuFlow Service Colony...")

    # Get configuration from environment
    host = os.getenv("SERVICE_HOST", "0.0.0.0")
    port = int(os.getenv("SERVICE_PORT", "8000"))

    logger.info(f"Service will be available at http://{host}:{port}")
    logger.info("Dashboard available at http://localhost:8000/dashboard")
    logger.info("API documentation available at http://localhost:8000/docs")

    # Start the service
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
