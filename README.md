# DocuFlow Service Colony

DocuFlowServiceColony is an advanced service colony framework designed for distributed, autonomous services working together to process documents, augment datasets, perform strategic reasoning, and retrieve relevant data. The system is self-adaptive, highly modular, and incorporates state-of-the-art machine learning techniques for efficient information retrieval and routing.

## Key Features

- PDF to Markdown conversion
- Dataset augmentation
- Strategic reasoning
- Advanced retrieval using RouterRetriever
- Self-adaptive mechanisms
- Distributed tracing with OpenTelemetry
- Integration with Dapr for microservices architecture

## Project Structure

- **core_services/**: Houses the main services like PDF processing, dataset augmentation, model training, etc.
- **communication/**: Handles communication logic, message queuing, and service registry.
- **self_adaptation/**: Implements self-adaptive mechanisms to optimize performance.
- **monitoring/**: Manages monitoring dashboards and log aggregation for the colony.
- **config/**: Central configuration files and utility scripts.
- **deployment/**: Docker and Kubernetes deployment configurations.
- **tests/**: Contains unit tests and integration tests for validating the services.

## Key Components

1. **PDF Processor**: Converts PDF documents to Markdown format.
2. **Dataset Augmentation Service**: Enhances datasets using advanced NLP techniques.
3. **Model Training Service**: Trains both SentenceTransformer and RouterRetriever models.
4. **RouterRetriever**: A novel approach for routing queries to the most appropriate expert model.
5. **Strategic Reasoning Service**: Performs high-level decision making based on processed data.

## Usage Guide

### 1. Running the System Locally
To run the entire system locally, first, install the dependencies:

```bash
pip install -r requirements.txt
```

Then, use **Docker Compose** to start all services:

```bash
docker-compose up
```

### 2. Kubernetes Deployment
For production deployment, use Kubernetes:

```bash
kubectl apply -f deployment/k8s-deployment.yaml
```

Ensure that each service is properly configured to use the service registry for service discovery.

### 3. Adding a New Service
To add a new service to the colony, follow these steps:
- Create a new folder in `core_services/` for the service.
- Implement the core functionality in `app.py`.
- Add a `Dockerfile` and update the deployment configuration in `deployment/k8s-deployment.yaml`.

### 4. Monitoring
Access the monitoring dashboard via the following:

```
http://localhost:8000/dashboard
```

This dashboard provides insights into service performance, logs, and real-time statistics.

### 5. Running Tests
To validate the services, run unit tests:

```bash
pytest tests/
```

This will execute all tests in the `tests/` folder, including load simulations and integration tests.

## Model Training

The system uses two main types of models:

1. **SentenceTransformer**: Used for generating embeddings from text data.
2. **RouterRetriever**: A novel approach that routes queries to the most appropriate expert model.

The training process is triggered by a "dataset-augmented" event and includes:
- Loading and preparing the synthetic dataset
- Training the SentenceTransformer model
- Training the RouterRetriever model
- Evaluating both models and saving the results

For more details on the training process, refer to the `train_model` function in the `model_training_service.py` file.

## Dependencies

- Python 3.9+
- Docker
- Kubernetes
- FastAPI
- Dapr
- OpenTelemetry
- PyTorch
- Sentence-Transformers
- Scikit-learn

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Specify your license here]

This updated README provides a comprehensive overview of your project, including the new RouterRetriever component and the model training process. It maintains the structure of the original README while incorporating the new information about your project's advanced features and components.
