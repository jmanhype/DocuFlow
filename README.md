# DocuFlow Service Colony

DocuFlowServiceColony is a service colony framework designed for distributed, autonomous services working together to process documents (PDF to Markdown conversion), augment datasets, perform strategic reasoning, and retrieve relevant data. The system is self-adaptive and highly modular.

## Project Structure

- **core_services/**: Houses the main services like PDF processing, dataset augmentation, etc.
- **communication/**: Handles communication logic, message queuing, and service registry.
- **self_adaptation/**: Implements self-adaptive mechanisms to optimize performance.
- **monitoring/**: Manages monitoring dashboards and log aggregation for the colony.
- **config/**: Central configuration files and utility scripts.
- **deployment/**: Docker and Kubernetes deployment configurations.
- **tests/**: Contains unit tests and integration tests for validating the services.

---

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

---

## Dependencies

- Python 3.9+
- Docker
- Kubernetes
