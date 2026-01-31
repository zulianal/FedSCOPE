import gzip
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer

class AmazonCrossDomainDataset:
    def __init__(self, path_a, path_b, config):
        self.config = config
        print("Loading and preprocessing data (this may take time)...")
        try:
            self.df_a = self.load_json(path_a)
            self.df_b = self.load_json(path_b)
        except FileNotFoundError:
            print("Dataset files not found. Generating Mock Data for demonstration...")
            self.df_a, self.df_b = self.generate_mock_data()

        # Filtering
        self.df_a = self.filter_k_core(self.df_a)
        self.df_b = self.filter_k_core(self.df_b)
        
        # Find overlapping users [cite: 581]
        users_a = set(self.df_a['reviewerID'].unique())
        users_b = set(self.df_b['reviewerID'].unique())
        common_users = list(users_a.intersection(users_b))
        
        # Map IDs
        self.user_map = {u: i for i, u in enumerate(common_users)}
        # Filter data to only common users
        self.df_a = self.df_a[self.df_a['reviewerID'].isin(common_users)]
        self.df_b = self.df_b[self.df_b['reviewerID'].isin(common_users)]
        
        # Item Mapping
        self.item_map_a = {item: i+1 for i, item in enumerate(self.df_a['asin'].unique())} # 0 is padding
        self.item_map_b = {item: i+1 for i, item in enumerate(self.df_b['asin'].unique())}
        
        # Semantic Augmentation (Offline Phase) 
        print("Simulating Offline LLM Semantic Augmentation...")
        self.semantic_emb_a = self.generate_semantic_embeddings(self.df_a, self.item_map_a)
        self.semantic_emb_b = self.generate_semantic_embeddings(self.df_b, self.item_map_b)
        
        # Create Sequences
        self.train_data_a = self.create_sequences(self.df_a, self.item_map_a)
        self.train_data_b = self.create_sequences(self.df_b, self.item_map_b)

    def load_json(self, path):
        data = []
        with gzip.open(path, 'r') as f:
            for l in f:
                data.append(json.loads(l.strip()))
        return pd.DataFrame(data)[['reviewerID', 'asin', 'unixReviewTime', 'reviewText']]

    def generate_mock_data(self):
        # Create dummy data if files don't exist
        users = [f'User{i}' for i in range(100)]
        items_a = [f'Movie{i}' for i in range(200)]
        items_b = [f'Book{i}' for i in range(200)]
        
        data_a, data_b = [], []
        for u in users:
            # Domain A interactions
            for _ in range(np.random.randint(10, 20)):
                data_a.append({'reviewerID': u, 'asin': np.random.choice(items_a), 
                               'unixReviewTime': np.random.randint(1e9, 2e9), 'reviewText': "Good movie"})
            # Domain B interactions
            for _ in range(np.random.randint(10, 20)):
                data_b.append({'reviewerID': u, 'asin': np.random.choice(items_b), 
                               'unixReviewTime': np.random.randint(1e9, 2e9), 'reviewText': "Nice book"})
        return pd.DataFrame(data_a), pd.DataFrame(data_b)

    def filter_k_core(self, df):
        user_counts = df['reviewerID'].value_counts()
        return df[df['reviewerID'].isin(user_counts[user_counts >= self.config.min_interactions].index)]

    def generate_semantic_embeddings(self, df, item_map):
        # Simulating Eq 2: Instead of calling GPT-4, we use a small BERT to encode item text/metadata
        # In the paper, LLM generates JSON attributes, which are then embedded. 
        # Here we embed the review text/metadata as a proxy for semantic features.
        model = SentenceTransformer('all-MiniLM-L6-v2')
        unique_items = df.drop_duplicates(subset=['asin'])
        embeddings = torch.zeros(len(item_map) + 1, self.config.semantic_dim)
        
        # Batch processing for speed
        batch_texts = []
        indices = []
        for _, row in unique_items.iterrows():
            if row['asin'] in item_map:
                # Use review text or metadata as "Semantic Attributes"
                text = row.get('reviewText', '') or "No description"
                batch_texts.append(text[:128]) # Truncate for speed
                indices.append(item_map[row['asin']])
        
        if batch_texts:
            vecs = model.encode(batch_texts)
            for idx, vec in zip(indices, vecs):
                embeddings[idx] = torch.tensor(vec)
        return embeddings

    def create_sequences(self, df, item_map):
        dataset = []
        grouped = df.sort_values('unixReviewTime').groupby('reviewerID')
        for user, group in grouped:
            if user not in self.user_map: continue
            
            user_id = self.user_map[user]
            items = [item_map[x] for x in group['asin'].tolist()]
            
            # Sequence truncation/padding
            seq = items[:-1]
            target = items[1:]
            
            seq = seq[-self.config.max_seq_len:]
            target = target[-self.config.max_seq_len:]
            
            pad_len = self.config.max_seq_len - len(seq)
            seq = [0] * pad_len + seq
            target = [0] * pad_len + target
            
            dataset.append((user_id, torch.tensor(seq), torch.tensor(target)))
        return dataset

class ClientDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]