import json
import random
import logging
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch
from fastapi import FastAPI
from dapr.clients import DaprClient
from dapr.ext.fastapi import DaprApp
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .router_retriever import RouterRetriever, train_router_retriever, evaluate_router_retriever, CustomRouterRetrieverLoss

# Load environment variables and setup logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()
dapr_app = DaprApp(app)

# Set up OpenTelemetry tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4317")
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

def load_synthetic_dataset(filepath: str) -> List[Dict]:
    """Load synthetic dataset from a .jsonl file."""
    with open(filepath, 'r') as f:
        return [json.loads(line) for line in f if json.loads(line).get('conversations')]

def prepare_dataset_for_embedding(data: List[Dict]) -> List[InputExample]:
    """Prepare dataset for embedding model training."""
    prepared_data = []
    for item in data:
        if len(item['conversations']) >= 3:
            query = item['conversations'][1].get('value', '')
            response = item['conversations'][2].get('value', '')
            prepared_data.append(InputExample(texts=[query, response]))
    return prepared_data

def evaluate_embeddings(model, test_data):
    correct = 0
    total = len(test_data)
    
    for example in test_data:
        query_embedding = model.encode(example.texts[0])
        response_embedding = model.encode(example.texts[1])
        
        # Calculate cosine similarity
        similarity = cosine_similarity([query_embedding], [response_embedding])[0][0]
        
        # If similarity is above a threshold (e.g., 0.5), consider it correct
        if similarity > 0.5:
            correct += 1
    
    accuracy = correct / total
    return accuracy

@dapr_app.subscribe(pubsub_name="pubsub", topic="dataset-augmented")
def train_model(event: Dict[str, Any]):
    with tracer.start_as_current_span("train_model"):
        try:
            augmented_dataset_path = event.get('data', {}).get('augmented_dataset_path')
            if not augmented_dataset_path:
                logging.error("No augmented dataset path provided")
                return
            
            with tracer.start_as_current_span("load_and_prepare_dataset"):
                logging.info(f"Loading synthetic dataset from {augmented_dataset_path}")
                synthetic_data = load_synthetic_dataset(augmented_dataset_path)
                prepared_data = prepare_dataset_for_embedding(synthetic_data)
                logging.info(f"Prepared {len(prepared_data)} examples for embedding training")

            with tracer.start_as_current_span("split_dataset"):
                train_data, test_data = train_test_split(prepared_data, test_size=0.2, random_state=42)

            # Train SentenceTransformer model
            with tracer.start_as_current_span("train_sentence_transformer"):
                model = SentenceTransformer('distilbert-base-nli-mean-tokens')
                train_dataloader = DataLoader(train_data, shuffle=True, batch_size=16)
                train_loss = losses.MultipleNegativesRankingLoss(model)

                logging.info("Starting embedding model training")
                model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3, warmup_steps=100)
                logging.info("Embedding model training completed")

                model_save_path = "./embedding_model"
                model.save(model_save_path)
                logging.info(f"SentenceTransformer model saved to {model_save_path}")

                test_accuracy = evaluate_embeddings(model, test_data)
                logging.info(f"\nSentenceTransformer Test Accuracy: {test_accuracy:.2f}")

            # Train RouterRetriever
            with tracer.start_as_current_span("train_router_retriever"):
                base_model_name = "facebook/contriever"
                num_experts = 7  # AR, MS, HO, NF, SF, QU, FI
                router_model = RouterRetriever(base_model_name, num_experts)
                
                router_train_dataloader = DataLoader(train_data, shuffle=True, batch_size=16, collate_fn=collate_batch)
                router_test_dataloader = DataLoader(test_data, batch_size=16, collate_fn=collate_batch)
                
                logging.info("Starting RouterRetriever training")
                train_router_retriever(router_model, [router_train_dataloader], num_epochs=3)
                logging.info("RouterRetriever training completed")

                router_model_save_path = "./router_retriever_model"
                torch.save(router_model.state_dict(), router_model_save_path)
                logging.info(f"RouterRetriever model saved to {router_model_save_path}")

                router_test_loss, router_test_accuracy = evaluate_router_retriever(router_model, router_test_dataloader)
                logging.info(f"\nRouterRetriever Test Loss: {router_test_loss:.4f}, Test Accuracy: {router_test_accuracy:.4f}")

            logging.info("All model training, testing, and evaluation completed")

        except Exception as e:
            logging.error(f"An error occurred during model training: {str(e)}")
            with DaprClient() as dapr_client:
                dapr_client.publish_event(
                    pubsub_name="pubsub",
                    topic_name="model-training-failed",
                    data=json.dumps({"error": str(e)})
                )

def collate_batch(batch):
    """Custom collate function for DataLoader."""
    input_ids = torch.stack([item.texts[0] for item in batch])
    attention_mask = torch.stack([item.texts[1] for item in batch])
    expert_idx = torch.tensor([0 for _ in batch])  # Placeholder, replace with actual expert_idx if available
    target_embedding = torch.stack([torch.randn(768) for _ in batch])  # Placeholder, replace with actual target embeddings
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'expert_idx': expert_idx,
        'target_embedding': target_embedding
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)