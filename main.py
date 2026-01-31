from config import Config
from dataset import AmazonCrossDomainDataset, ClientDataset
from federated import Server, Client
import torch
import numpy as np

def split_data_for_clients(full_dataset, num_clients):
    """ Partitions data into clients (Simulating federated distribution) """
    data_len = len(full_dataset.train_data_a)
    indices = np.random.permutation(data_len)
    split_indices = np.array_split(indices, num_clients)
    
    clients_data = []
    for idxs in split_indices:
        subset_a = [full_dataset.train_data_a[i] for i in idxs]
        subset_b = [full_dataset.train_data_b[i] for i in idxs]
        clients_data.append((ClientDataset(subset_a), ClientDataset(subset_b)))
    return clients_data

if __name__ == "__main__":
    # 1. Setup
    config = Config()
    print(f"Running FedSCOPE on device: {config.device}")
    
    # 2. Data Preparation
    # Note: Requires 'reviews_Movies_and_TV_5.json.gz' and 'reviews_Books_5.json.gz'
    # Or it will generate mock data.
    full_dataset = AmazonCrossDomainDataset(config.domain_a_path, config.domain_b_path, config)
    
    # 3. Create Server
    server = Server(full_dataset, config)
    
    # 4. Create Clients (Distribute data)
    client_datasets = split_data_for_clients(full_dataset, config.num_clients)
    clients = []
    for i, (ds_a, ds_b) in enumerate(client_datasets):
        client = Client(
            client_id=i, 
            dataset_a=ds_a, 
            dataset_b=ds_b, 
            semantic_emb_a=full_dataset.semantic_emb_a.to(config.device),
            semantic_emb_b=full_dataset.semantic_emb_b.to(config.device),
            config=config
        )
        clients.append(client)
        
    # 5. Federated Training Loop
    print("Starting Federated Training...")
    for round_idx in range(config.rounds):
        print(f"--- Communication Round {round_idx+1}/{config.rounds} ---")
        
        client_updates = []
        
        # Broadcast global model is implicit by passing params to train_round
        for client in clients:
            # Client Local Training + Adaptive DP
            up_a, up_b, size, loss = client.train_round(server.params_a, server.params_b)
            client_updates.append((up_a, up_b, size, loss))
            print(f"Client {client.id} Loss: {loss:.4f}")
            
        # Secure Aggregation
        avg_loss = server.aggregate(client_updates)
        print(f"Round {round_idx+1} Global Avg Loss: {avg_loss:.4f}")
        
    print("Training Complete. Models updated with privacy-preserving collaborative learning.")