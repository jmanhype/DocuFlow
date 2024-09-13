import torch
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
from typing import List
from torch.utils.data import DataLoader

class RouterRetriever(torch.nn.Module):
    def __init__(self, base_model_name, num_experts):
        super().__init__()
        self.base_encoder = AutoModel.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        self.expert_gates = torch.nn.ModuleList([
            get_peft_model(self.base_encoder, LoraConfig(
                r=8, lora_alpha=32, target_modules=["query", "key", "value"]
            ))
            for _ in range(num_experts)
        ])
        
        self.pilot_embeddings = None

    def forward(self, input_ids, attention_mask):
        base_embedding = self.base_encoder(input_ids, attention_mask).last_hidden_state[:, 0]
        
        similarities = torch.matmul(base_embedding, self.pilot_embeddings.T)
        expert_idx = similarities.mean(dim=1).argmax()
        
        expert_embedding = self.expert_gates[expert_idx](input_ids, attention_mask).last_hidden_state[:, 0]
        
        return expert_embedding, expert_idx

def train_router_retriever(model: RouterRetriever, datasets: List[DataLoader], num_epochs: int):
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(num_epochs):
        for dataset_idx, dataset in enumerate(datasets):
            for batch in dataset:
                loss = train_step(model, batch, dataset_idx)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    model.pilot_embeddings = create_pilot_embeddings(model, datasets)

def train_step(model: RouterRetriever, batch, expert_idx: int):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    
    expert_embedding, selected_expert = model(input_ids, attention_mask)
    
    # Calculate loss based on whether the correct expert was selected
    expert_selection_loss = torch.nn.functional.cross_entropy(
        selected_expert.unsqueeze(0),
        torch.tensor([expert_idx], device=selected_expert.device)
    )
    
    # Calculate embedding quality loss (e.g., using cosine similarity)
    target_embedding = batch['target_embedding']  # Assuming this is provided in the batch
    embedding_quality_loss = 1 - torch.nn.functional.cosine_similarity(expert_embedding, target_embedding).mean()
    
    # Combine losses
    loss = expert_selection_loss + embedding_quality_loss
    
    return loss

def create_pilot_embeddings(model: RouterRetriever, datasets: List[DataLoader]):
    pilot_embeddings = []
    for dataset in datasets:
        embeddings = []
        for batch in dataset:
            with torch.no_grad():
                emb = model.base_encoder(batch['input_ids'], batch['attention_mask']).last_hidden_state[:, 0]
                embeddings.append(emb)
        pilot_embeddings.append(torch.mean(torch.cat(embeddings), dim=0))
    return torch.stack(pilot_embeddings)

def evaluate_router_retriever(model: RouterRetriever, test_data: DataLoader):
    model.eval()
    total_loss = 0
    correct_expert_selections = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in test_data:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            true_expert_idx = batch['expert_idx']  # Assuming this is provided in the batch
            
            expert_embedding, selected_expert = model(input_ids, attention_mask)
            
            # Calculate expert selection accuracy
            correct_expert_selections += (selected_expert == true_expert_idx).sum().item()
            total_samples += input_ids.size(0)
            
            # Calculate embedding quality (using cosine similarity)
            target_embedding = batch['target_embedding']  # Assuming this is provided in the batch
            embedding_quality = torch.nn.functional.cosine_similarity(expert_embedding, target_embedding).mean()
            
            # Combine metrics into a single loss value
            loss = 1 - embedding_quality  # Lower is better
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_data)
    expert_selection_accuracy = correct_expert_selections / total_samples
    
    return avg_loss, expert_selection_accuracy

class CustomRouterRetrieverLoss(torch.nn.Module):
    def __init__(self, expert_selection_weight=0.5, embedding_quality_weight=0.5):
        super().__init__()
        self.expert_selection_weight = expert_selection_weight
        self.embedding_quality_weight = embedding_quality_weight

    def forward(self, expert_embedding, selected_expert, true_expert_idx, target_embedding):
        # Expert selection loss
        expert_selection_loss = torch.nn.functional.cross_entropy(
            selected_expert.unsqueeze(0),
            true_expert_idx
        )
        
        # Embedding quality loss (using cosine similarity)
        embedding_quality_loss = 1 - torch.nn.functional.cosine_similarity(expert_embedding, target_embedding).mean()
        
        # Combine losses
        total_loss = (
            self.expert_selection_weight * expert_selection_loss +
            self.embedding_quality_weight * embedding_quality_loss
        )
        
        return total_loss