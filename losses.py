import torch
import torch.nn.functional as F

def calc_contrastive_loss(anchor, positive, negatives, temperature):
    """ Generic InfoNCE loss """
    # Cosine similarity
    pos_sim = F.cosine_similarity(anchor, positive) / temperature
    
    # Negative similarity
    # negatives: (Batch, Num_Neg, Dim)
    anchor_expanded = anchor.unsqueeze(1)
    neg_sim = F.cosine_similarity(anchor_expanded, negatives, dim=2) / temperature
    
    # LogSoftmax
    numerator = torch.exp(pos_sim)
    denominator = numerator + torch.sum(torch.exp(neg_sim), dim=1)
    return -torch.log(numerator / denominator).mean()

def iidcl_loss(batch_reps, other_domain_reps, user_ids, config):
    """
    Implementation of IIDCL [cite: 333, 381]
    batch_reps: (Batch, Dim) - Current domain fused representations
    other_domain_reps: (Batch, Dim) - Representations of SAME users in other domain
    """
    batch_size = batch_reps.size(0)
    
    # 1. Intra-Domain Contrastive Learning
    # Find similar users within the batch as "Top-K" approximation for demo efficiency
    # In full implementation, this uses a memory bank or full dataset scan.
    sim_matrix = F.cosine_similarity(batch_reps.unsqueeze(1), batch_reps.unsqueeze(0), dim=2)
    # Mask self
    sim_matrix.fill_diagonal_(-1e9)
    
    # Select positive (most similar) and negatives
    # Simplified: taking the single most similar user as positive for InfoNCE
    _, pos_idx = sim_matrix.max(dim=1)
    positives_intra = batch_reps[pos_idx]
    
    # Random negatives from batch
    neg_indices = torch.randint(0, batch_size, (batch_size, 5)) # 5 negatives
    negatives_intra = batch_reps[neg_indices]
    
    loss_intra = calc_contrastive_loss(batch_reps, positives_intra, negatives_intra, config.tau_intra)
    
    # 2. Inter-Domain Contrastive Learning
    # Positive: The same user's representation in the other domain
    # Negative: Other users in the other domain
    if other_domain_reps is not None:
        neg_indices_inter = torch.randint(0, batch_size, (batch_size, 5))
        negatives_inter = other_domain_reps[neg_indices_inter]
        
        loss_inter = calc_contrastive_loss(batch_reps, other_domain_reps, negatives_inter, config.tau_inter)
    else:
        loss_inter = torch.tensor(0.0, device=config.device)
        
    return loss_intra, loss_inter