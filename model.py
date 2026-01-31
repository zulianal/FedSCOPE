import torch
import torch.nn as nn
import torch.nn.functional as F

class SASRecEncoder(nn.Module):
    def __init__(self, num_items, config):
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 1, config.embedding_size, padding_idx=0)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.embedding_size)
        self.emb_dropout = nn.Dropout(config.dropout_rate)
        
        # Transformer Blocks [cite: 190, 191]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_size, 
            nhead=config.num_heads, 
            dim_feedforward=config.hidden_size, 
            dropout=config.dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_blocks)
        self.device = config.device

    def forward(self, seqs):
        seqs = seqs.to(self.device)
        batch_size, seq_len = seqs.shape
        
        # Embeddings + Positional
        seq_emb = self.item_emb(seqs)
        positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_emb(positions)
        
        x = self.emb_dropout(seq_emb + pos_emb)
        
        # Causal Mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=self.device) * float('-inf'), diagonal=1)
        
        output = self.transformer(x, mask=mask)
        return output # (Batch, Seq_Len, Dim)

class FeatureFusion(nn.Module):
    """ Fuses behavioral embedding with LLM semantic embedding  """
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.embedding_size + config.semantic_dim, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.embedding_size)
        )
    
    def forward(self, behavior_emb, semantic_emb):
        # behavior_emb: (Batch, Dim), semantic_emb: (Batch, Semantic_Dim)
        concat = torch.cat([behavior_emb, semantic_emb], dim=-1)
        return self.mlp(concat)

class FedSCOPEModel(nn.Module):
    def __init__(self, num_items, semantic_embeddings, config):
        super().__init__()
        self.encoder = SASRecEncoder(num_items, config)
        self.fusion = FeatureFusion(config)
        self.semantic_embeddings = nn.Parameter(semantic_embeddings, requires_grad=False) # Fixed offline features
        self.config = config
        
    def forward(self, seqs):
        # Get sequence representation (using the last item's hidden state for user rep)
        seq_out = self.encoder(seqs) # (Batch, Len, Dim)
        behavior_user_rep = seq_out[:, -1, :] # User rep from sequence
        
        # Get Semantic Features for the last item in sequence (Proxy for user interest semantics)
        # Note: Paper aligns user semantics. Here we use the items in history to query semantic table.
        last_items = seqs[:, -1]
        semantic_feats = self.semantic_embeddings[last_items]
        
        # Fusion 
        final_user_rep = self.fusion(behavior_user_rep, semantic_feats)
        
        return final_user_rep, behavior_user_rep, seq_out

    def predict(self, user_rep, target_item_indices):
        # Simple Dot Product for prediction [cite: 411]
        # Get target item embeddings (Behavioral + Semantic)
        target_behav = self.encoder.item_emb(target_item_indices)
        target_sem = self.semantic_embeddings[target_item_indices]
        target_fused = self.fusion(target_behav, target_sem)
        
        logits = (user_rep * target_fused).sum(dim=1)
        return logits