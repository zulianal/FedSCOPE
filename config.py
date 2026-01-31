import torch

class Config:
    def __init__(self):
        # Data settings
        self.domain_a_path = 'reviews_Movies_and_TV_5.json.gz' 
        self.domain_b_path = 'reviews_Books_5.json.gz'        
        self.min_interactions = 5
        self.max_seq_len = 50
        
        # Model hyperparameters [cite: 627, 628]
        self.embedding_size = 64
        self.hidden_size = 64
        self.num_blocks = 2
        self.num_heads = 2
        self.dropout_rate = 0.3
        self.batch_size = 256
        self.lr = 0.001
        self.l2_reg = 1e-4
        
        # LLM Augmentation
        self.semantic_dim = 384 # Dimension for 'all-MiniLM-L6-v2' (simulating LLM)
        
        # IIDCL Hyperparameters [cite: 331]
        self.rho = 0.02 
        self.k_max = 50
        self.tau_intra = 1.0
        self.tau_inter = 1.0
        self.lambda_intra = 0.1
        self.lambda_inter = 0.1
        self.alpha = 0.1 # Balance between prediction and contrastive 
        
        # Federated & Privacy 
        self.num_clients = 5  # Reduced for demo purpose, paper uses 50
        self.rounds = 10
        self.local_epochs = 1
        self.dp_epsilon_total = 10.0
        self.dp_delta = 1e-5
        self.dp_beta = 0.5 # Personalization factor
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Config()