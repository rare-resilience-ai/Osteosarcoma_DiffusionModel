"""
Biology-Aware Diffusion Model for Disease Progression
Incorporates pathway constraints and biological knowledge
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
import numpy as np
from typing import Optional, Tuple


class PathwayGraphEncoder(nn.Module):
    """
    Graph Neural Network that encodes gene/pathway relationships
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Initial projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList()
        
        for i in range(num_layers):
            in_channels = hidden_dim if i == 0 else hidden_dim * heads
            out_channels = hidden_dim
            
            self.gat_layers.append(
                GATConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=heads if i < num_layers - 1 else 1,
                    dropout=dropout,
                    concat=True if i < num_layers - 1 else False
                )
            )
        
        # Output projection to latent space
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, batch=None):
        """
        Args:
            x: Node features (n_nodes, input_dim)
            edge_index: Graph connectivity (2, n_edges)
            batch: Batch assignment for graph pooling
            
        Returns:
            latent: Latent representation (batch_size, latent_dim)
        """
        
        # Initial projection
        h = F.elu(self.input_proj(x))
        h = self.dropout(h)
        
        # Graph convolutions
        for gat_layer in self.gat_layers:
            h = gat_layer(h, edge_index)
            h = F.elu(h)
            h = self.dropout(h)
        
        # Global pooling
        if batch is not None:
            h = global_mean_pool(h, batch)
        else:
            h = h.mean(dim=0, keepdim=True)
        
        # Project to latent space
        z = self.output_proj(h)
        
        return z


class ConditionalEmbedding(nn.Module):
    """Embed clinical conditions (survival, age, etc.)"""
    
    def __init__(self, num_continuous: int, embedding_dim: int):
        super().__init__()
        
        self.num_continuous = num_continuous
        self.embedding_dim = embedding_dim
        
        # MLP for continuous variables
        self.mlp = nn.Sequential(
            nn.Linear(num_continuous, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def forward(self, conditions):
        """
        Args:
            conditions: (batch_size, num_continuous)
        Returns:
            embeddings: (batch_size, embedding_dim)
        """
        return self.mlp(conditions)


class TimeEmbedding(nn.Module):
    """Sinusoidal time embeddings for diffusion timestep"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        """
        Args:
            t: (batch_size,) timestep values [0, 1]
        Returns:
            embeddings: (batch_size, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        
        return embeddings


class DiffusionUNet(nn.Module):
    """
    U-Net style architecture for diffusion denoising
    Conditioned on time, clinical features, and biological constraints
    """
    
    def __init__(
        self,
        data_dim: int,
        time_dim: int = 128,
        condition_dim: int = 64,
        hidden_dims: list = [256, 512, 256],
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.data_dim = data_dim
        
        # Time embedding
        self.time_embed = TimeEmbedding(time_dim)
        
        # Input projection
        self.input_proj = nn.Linear(data_dim, hidden_dims[0])
        
        # Conditioning projection
        self.cond_proj = nn.Linear(condition_dim, hidden_dims[0])
        self.time_proj = nn.Linear(time_dim, hidden_dims[0])
        
        # Encoder (downsampling path)
        self.encoder = nn.ModuleList()
        in_dim = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            self.encoder.append(self._make_block(in_dim, h_dim, dropout))
            in_dim = h_dim
        
        # Bottleneck
        self.bottleneck = self._make_block(in_dim, in_dim, dropout)
        
        # Decoder (upsampling path with skip connections)
        self.decoder = nn.ModuleList()
        # We need to reverse hidden_dims but handle skip connections correctly
        # If hidden_dims is [256, 512, 256], encoder outputs [512, 256]
        for i in range(len(hidden_dims) - 2, -1, -1):
            skip_dim = hidden_dims[i+1] 
            out_dim = hidden_dims[i]
            # Create the block: current_dim + skip_dim -> out_dim
            self.decoder.append(self._make_block(in_dim + skip_dim, out_dim, dropout))
            in_dim = out_dim
        
        # Output projection
        self.output_proj = nn.Linear(in_dim, data_dim)

    def _make_block(self, in_dim, out_dim, dropout):
        """Helper to create consistent residual-like blocks"""
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GroupNorm(8, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.GroupNorm(8, out_dim),
            nn.SiLU()
        )
    
    def forward(self, x, t, condition):
        """
        Args:
            x: Noisy data (batch_size, data_dim)
            t: Timestep (batch_size,) values in [0, 1]
            condition: Clinical conditions (batch_size, condition_dim)
            
        Returns:
            noise_pred: Predicted noise (batch_size, data_dim)
        """
        
        # Embed time
        t_emb = self.time_embed(t)
        t_emb = self.time_proj(t_emb)
        
        # Project condition
        c_emb = self.cond_proj(condition)
        
        # Input projection
        h = self.input_proj(x)
        
        # Add time and condition embeddings
        h = h + t_emb + c_emb
        
        # Encoder with skip connections
        skips = []
        for enc_block in self.encoder:
            h = enc_block(h)
            skips.append(h)
        
        # Bottleneck
        h = self.bottleneck(h)
        
        # Decoder with skip connections
        skips.pop() 
        
        for dec_block in self.decoder:
            if not skips:
                break
            # Get the matching skip from the encoder stack (Last-In, First-Out)
            skip = skips.pop() 
            h = torch.cat([h, skip], dim=-1)
            h = dec_block(h) # Note: Activation is already inside _make_block
        
        # Output
        noise_pred = self.output_proj(h)
        
        return noise_pred


class BiologyAwareDiffusionModel(nn.Module):
    """
    Complete diffusion model with biological constraints
    """
    
    def __init__(
        self,
        mutation_dim: int,
        expression_dim: int,
        pathway_dim: int,
        condition_dim: int,
        config: dict
    ):
        super().__init__()
        
        self.mutation_dim = mutation_dim
        self.expression_dim = expression_dim
        self.pathway_dim = pathway_dim
        self.condition_dim = condition_dim
        
        # Total data dimension (mutations + expression + pathways)
        self.data_dim = mutation_dim + expression_dim + pathway_dim
        
        # Conditional embedding
        self.condition_embed = ConditionalEmbedding(
            num_continuous=condition_dim,
            embedding_dim=config['model']['latent_dim'] // 2
        )
        
        # Denoising U-Net
        self.unet = DiffusionUNet(
            data_dim=self.data_dim,
            time_dim=config['model']['latent_dim'],
            condition_dim=config['model']['latent_dim'] // 2,
            hidden_dims=config['model']['hidden_dims'],
            dropout=config['model']['gnn']['dropout']
        )
        
        # Diffusion parameters
        self.num_steps = config['model']['diffusion']['num_steps']
        self.register_buffer('betas', self._get_beta_schedule(
            config['model']['diffusion']['beta_schedule'],
            self.num_steps
        ))
        
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', 
                           torch.sqrt(1.0 - alphas_cumprod))
        
    def _get_beta_schedule(self, schedule_type: str, num_steps: int):
        """Generate beta schedule for diffusion"""
        
        if schedule_type == 'linear':
            return torch.linspace(1e-4, 0.02, num_steps)
        
        elif schedule_type == 'cosine':
            steps = torch.arange(num_steps + 1, dtype=torch.float32) / num_steps
            alphas_cumprod = torch.cos((steps + 0.008) / 1.008 * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        
        else:
            raise ValueError(f"Unknown schedule: {schedule_type}")
    
    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion: add noise to data
        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        """
        
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise
        
        return x_t, noise
    
    def forward(self, x_0, conditions, return_loss=True):
        """
        Training forward pass
        
        Args:
            x_0: Clean data (batch_size, data_dim)
            conditions: Clinical conditions (batch_size, condition_dim)
            return_loss: Whether to compute loss
            
        Returns:
            loss or predicted noise
        """
        
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # Sample random timesteps
        t = torch.randint(0, self.num_steps, (batch_size,), device=device)
        
        # Add noise
        x_t, noise = self.q_sample(x_0, t)
        
        # Normalize timesteps to [0, 1]
        t_normalized = t.float() / self.num_steps
        
        # Embed conditions
        c_emb = self.condition_embed(conditions)
        
        # Predict noise
        noise_pred = self.unet(x_t, t_normalized, c_emb)
        
        if return_loss:
            # Simple MSE loss
            loss = F.mse_loss(noise_pred, noise)
            return loss
        else:
            return noise_pred
    
    @torch.no_grad()
    def p_sample(self, x_t, t, conditions):
        """
        Single reverse diffusion step
        """
        
        batch_size = x_t.shape[0]
        device = x_t.device
        
        # Normalize timestep
        t_normalized = torch.full((batch_size,), t / self.num_steps, device=device)
        
        # Embed conditions
        c_emb = self.condition_embed(conditions)
        
        # Predict noise
        noise_pred = self.unet(x_t, t_normalized, c_emb)
        
        # Compute coefficients
        alpha_t = 1.0 - self.betas[t]
        alpha_bar_t = self.alphas_cumprod[t]
        
        # Predict x_0
        x_0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
        
        # Compute x_{t-1}
        if t > 0:
            noise = torch.randn_like(x_t)
            alpha_bar_t_prev = self.alphas_cumprod[t - 1]
            
            # Mean
            mean = (
                torch.sqrt(alpha_bar_t_prev) * self.betas[t] * x_0_pred / (1 - alpha_bar_t) +
                torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev) * x_t / (1 - alpha_bar_t)
            )
            
            # Variance
            variance = (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * self.betas[t]
            
            x_t_minus_1 = mean + torch.sqrt(variance) * noise
        else:
            x_t_minus_1 = x_0_pred
        
        return x_t_minus_1
    
    @torch.no_grad()
    def sample(self, conditions, num_samples: int = 1):
        """
        Generate samples via reverse diffusion
        
        Args:
            conditions: Clinical conditions (num_samples, condition_dim)
            num_samples: Number of samples to generate
            
        Returns:
            samples: Generated data (num_samples, data_dim)
        """
        
        device = next(self.parameters()).device
        
        # Start from pure noise
        x_t = torch.randn(num_samples, self.data_dim, device=device)
        
        # Reverse diffusion
        for t in reversed(range(self.num_steps)):
            x_t = self.p_sample(x_t, t, conditions)
        
        return x_t


if __name__ == "__main__":
    # Test model
    config = {
        'model': {
            'latent_dim': 128,
            'hidden_dims': [256, 512, 256],
            'gnn': {'dropout': 0.2},
            'diffusion': {
                'num_steps': 1000,
                'beta_schedule': 'cosine'
            }
        }
    }
    
    model = BiologyAwareDiffusionModel(
        mutation_dim=100,
        expression_dim=200,
        pathway_dim=50,
        condition_dim=5,
        config=config
    )
    
    # Test forward pass
    x = torch.randn(4, 350)
    conditions = torch.randn(4, 5)
    
    loss = model(x, conditions)
    print(f"Loss: {loss.item()}")
    
    # Test sampling
    samples = model.sample(conditions, num_samples=4)
    print(f"Generated samples shape: {samples.shape}")
