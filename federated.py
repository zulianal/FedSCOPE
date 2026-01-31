import copy
import numpy as np

class Client:
    def __init__(self, client_id, dataset_a, dataset_b, semantic_emb_a, semantic_emb_b, config):
        self.id = client_id
        self.config = config
        self.data_a = DataLoader(dataset_a, batch_size=config.batch_size, shuffle=True)
        self.data_b = DataLoader(dataset_b, batch_size=config.batch_size, shuffle=True)
        self.dataset_size = len(dataset_a) + len(dataset_b)
        
        # Local Models
        self.model_a = FedSCOPEModel(len(dataset_a.dataset.item_map_a), semantic_emb_a, config).to(config.device)
        self.model_b = FedSCOPEModel(len(dataset_b.dataset.item_map_b), semantic_emb_b, config).to(config.device)
        
        self.optimizer_a = torch.optim.Adam(self.model_a.parameters(), lr=config.lr, weight_decay=config.l2_reg)
        self.optimizer_b = torch.optim.Adam(self.model_b.parameters(), lr=config.lr, weight_decay=config.l2_reg)

    def train_round(self, global_params_a, global_params_b):
        # Load global parameters
        self.model_a.load_state_dict(global_params_a)
        self.model_b.load_state_dict(global_params_b)
        self.model_a.train()
        self.model_b.train()
        
        total_loss = 0
        
        # Simulating simultaneous training (assuming aligned batches for user matching)
        # In real world, use PSI to align. Here datasets are pre-aligned by user index.
        for (u_a, seq_a, tar_a), (u_b, seq_b, tar_b) in zip(self.data_a, self.data_b):
            seq_a, tar_a = seq_a.to(self.config.device), tar_a.to(self.config.device)
            seq_b, tar_b = seq_b.to(self.config.device), tar_b.to(self.config.device)
            
            # Forward
            rep_a, _, _ = self.model_a(seq_a)
            rep_b, _, _ = self.model_b(seq_b)
            
            # Prediction Loss [cite: 413]
            pred_a = self.model_a.predict(rep_a, tar_a)
            loss_pred_a = F.binary_cross_entropy_with_logits(pred_a, torch.ones_like(pred_a)) # Simplified positive only for demo
            
            pred_b = self.model_b.predict(rep_b, tar_b)
            loss_pred_b = F.binary_cross_entropy_with_logits(pred_b, torch.ones_like(pred_b))
            
            # IIDCL Loss [cite: 398]
            l_intra_a, l_inter_a = iidcl_loss(rep_a, rep_b.detach(), u_a, self.config)
            l_intra_b, l_inter_b = iidcl_loss(rep_b, rep_a.detach(), u_b, self.config)
            
            # Total Loss 
            loss_a = loss_pred_a + self.config.alpha * (self.config.lambda_intra * l_intra_a + self.config.lambda_inter * l_inter_a)
            loss_b = loss_pred_b + self.config.alpha * (self.config.lambda_intra * l_intra_b + self.config.lambda_inter * l_inter_b)
            
            # Optimization
            self.optimizer_a.zero_grad()
            loss_a.backward()
            self.optimizer_a.step()
            
            self.optimizer_b.zero_grad()
            loss_b.backward()
            self.optimizer_b.step()
            
            total_loss += (loss_a.item() + loss_b.item())
            
        # Get Updates
        update_a = {k: v - global_params_a[k] for k, v in self.model_a.state_dict().items()}
        update_b = {k: v - global_params_b[k] for k, v in self.model_b.state_dict().items()}
        
        # Apply Adaptive Personalized DP [cite: 466, 490]
        update_a = self.apply_adaptive_dp(update_a)
        update_b = self.apply_adaptive_dp(update_b)
        
        return update_a, update_b, self.dataset_size, total_loss

    def apply_adaptive_dp(self, update_dict):
        """ Implements Eq 10-13 from the paper """
        # 1. Adaptive Budget Allocation based on data size (Eq 13 simplified logic)
        # Larger dataset -> larger budget -> smaller noise
        epsilon_k = self.config.dp_epsilon_total * (self.dataset_size ** self.config.dp_beta) 
        
        # 2. Adaptive Clipping (Eq 13)
        # Calculate median norm (simplified: using global norm here for demo)
        total_norm = torch.norm(torch.stack([torch.norm(p) for p in update_dict.values()]))
        clip_threshold = 1.0 # In real impl, this updates dynamically via median
        
        scaling_factor = min(1, clip_threshold / (total_norm + 1e-6))
        
        # 3. Noise Injection
        sigma = (np.sqrt(2 * np.log(1.25 / self.config.dp_delta)) * clip_threshold) / epsilon_k
        
        privatized_update = {}
        for k, v in update_dict.items():
            clipped = v * scaling_factor
            noise = torch.normal(0, sigma, size=v.shape).to(v.device)
            privatized_update[k] = clipped + noise
            
        return privatized_update

class Server:
    def __init__(self, dataset_obj, config):
        self.config = config
        # Initialize Global Models
        self.global_model_a = FedSCOPEModel(len(dataset_obj.item_map_a), dataset_obj.semantic_emb_a, config).to(config.device)
        self.global_model_b = FedSCOPEModel(len(dataset_obj.item_map_b), dataset_obj.semantic_emb_b, config).to(config.device)
        
        self.params_a = self.global_model_a.state_dict()
        self.params_b = self.global_model_b.state_dict()

    def aggregate(self, updates_list):
        """ Weighted Aggregation  """
        total_samples = sum([u[2] for u in updates_list])
        
        # Aggregate Domain A
        new_params_a = copy.deepcopy(self.params_a)
        for key in new_params_a.keys():
            weighted_sum = sum([update[0][key] * update[2] for update in updates_list])
            new_params_a[key] += weighted_sum / total_samples
        self.params_a = new_params_a
        
        # Aggregate Domain B
        new_params_b = copy.deepcopy(self.params_b)
        for key in new_params_b.keys():
            weighted_sum = sum([update[1][key] * update[2] for update in updates_list])
            new_params_b[key] += weighted_sum / total_samples
        self.params_b = new_params_b
        
        return sum([u[3] for u in updates_list]) / len(updates_list) # Avg loss