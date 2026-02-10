"""
Training Pipeline for Disease Progression Models
Includes data augmentation, early stopping, and checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Optional
from tqdm import tqdm
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OsteosarcomaDataset(Dataset):
    """PyTorch dataset for osteosarcoma data"""
    
    def __init__(
        self,
        mutation_matrix: pd.DataFrame,
        expression_matrix: pd.DataFrame,
        pathway_scores: pd.DataFrame,
        clinical_data: pd.DataFrame,
        condition_features: list
    ):
        """
        Args:
            mutation_matrix: (n_samples, n_genes)
            expression_matrix: (n_samples, n_genes)
            pathway_scores: (n_samples, n_pathways)
            clinical_data: Contains survival_days, event_occurred, etc.
            condition_features: List of column names to use as conditions
        """
        
        # Align all dataframes by index
        common_idx = (
            mutation_matrix.index
            .intersection(expression_matrix.index)
            .intersection(pathway_scores.index)
        )
        
        self.mutations = torch.FloatTensor(
            mutation_matrix.loc[common_idx].values
        )
        
        self.expression = torch.FloatTensor(
            expression_matrix.loc[common_idx].values
        )
        
        self.pathways = torch.FloatTensor(
            pathway_scores.loc[common_idx].values
        )
        
        # Combine all features
        self.data = torch.cat([self.mutations, self.expression, self.pathways], dim=1)
        
        # Extract conditions from clinical data
        clinical_aligned = clinical_data.set_index('submitter_id').loc[common_idx]
        
        conditions = clinical_aligned[condition_features].values
        
        # Handle missing values
        conditions = np.nan_to_num(conditions, nan=0.0)
        
        self.conditions = torch.FloatTensor(conditions)
        
        # Store survival for auxiliary loss
        self.survival_days = torch.FloatTensor(
            clinical_aligned['survival_days'].fillna(0).values
        )
        
        logger.info(f"Dataset: {len(self)} samples")
        logger.info(f"Data dim: {self.data.shape[1]}")
        logger.info(f"Condition dim: {self.conditions.shape[1]}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'data': self.data[idx],
            'conditions': self.conditions[idx],
            'survival': self.survival_days[idx]
        }


class MixupAugmentation:
    """Mixup data augmentation for small datasets"""
    
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        
    def __call__(self, batch):
        """
        Apply mixup to a batch
        
        Args:
            batch: Dict with 'data', 'conditions', 'survival'
            
        Returns:
            Mixed batch
        """
        
        data = batch['data']
        conditions = batch['conditions']
        survival = batch['survival']
        
        batch_size = data.size(0)
        
        # Sample lambda from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        # Random permutation
        index = torch.randperm(batch_size)
        
        # Mix data
        mixed_data = lam * data + (1 - lam) * data[index]
        mixed_conditions = lam * conditions + (1 - lam) * conditions[index]
        mixed_survival = lam * survival + (1 - lam) * survival[index]
        
        return {
            'data': mixed_data,
            'conditions': mixed_conditions,
            'survival': mixed_survival
        }


class EarlyStopping:
    """Early stopping handler"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class Trainer:
    """Training pipeline for disease progression models"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config['training']['patience'],
            min_delta=config['training']['min_delta']
        )
        
        # Data augmentation
        self.mixup = MixupAugmentation(
            alpha=config['training']['augmentation']['mixup_alpha']
        ) if config['training']['augmentation']['mixup_alpha'] > 0 else None
        
        # Checkpointing
        self.save_dir = Path(config['training']['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
        
    def train_epoch(self):
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc='Training')
        
        for batch in pbar:
            # Move to device
            data = batch['data'].to(self.device)
            conditions = batch['conditions'].to(self.device)
            survival = batch['survival'].to(self.device)
            
            # Apply mixup if enabled
            if self.mixup is not None:
                batch_mixed = self.mixup({
                    'data': data,
                    'conditions': conditions,
                    'survival': survival
                })
                data = batch_mixed['data']
                conditions = batch_mixed['conditions']
                survival = batch_mixed['survival']
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Different models have different signatures
            if hasattr(self.model, 'vae'):  # BiologyConstrainedVAE
                loss = self.model(data, conditions, survival)
            else:  # BiologyAwareDiffusionModel
                loss = self.model(data, conditions, return_loss=True)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
        """Validate model"""
        
        self.model.eval()
        total_loss = 0.0
        
        for batch in self.val_loader:
            data = batch['data'].to(self.device)
            conditions = batch['conditions'].to(self.device)
            survival = batch['survival'].to(self.device)
            
            # Forward pass
            if hasattr(self.model, 'vae'):
                loss = self.model(data, conditions, survival)
            else:
                loss = self.model(data, conditions, return_loss=True)
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        path = self.save_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        
        # Save best model
        if is_best:
            best_path = self.save_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def train(self):
        """Complete training loop"""
        
        logger.info("Starting training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Val batches: {len(self.val_loader)}")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config['training']['num_epochs']):
            logger.info(f"\nEpoch {epoch + 1}/{self.config['training']['num_epochs']}")
            
            # Train
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.history['val_loss'].append(val_loss)
            
            logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            if (epoch + 1) % self.config['training']['save_frequency'] == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)
            
            # Early stopping
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        logger.info("Training complete!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        
        return self.history


def prepare_data(config: dict):
    """Load and prepare data for training"""
    
    logger.info("Loading processed data...")
    
    processed_dir = Path(config['data']['processed_dir'])
    
    # Load aligned data
    mutation_matrix = pd.read_csv(
        processed_dir / 'mutation_matrix_aligned.csv',
        index_col=0
    )
    
    expression_matrix = pd.read_csv(
        processed_dir / 'expression_matrix_aligned.csv',
        index_col=0
    )
    
    clinical_data = pd.read_csv(
        processed_dir / 'clinical_aligned.csv'
    )
    
    # Load pathway scores (need to compute if not exists)
    pathway_scores_path = processed_dir / 'pathway_scores.csv'
    
    if not pathway_scores_path.exists():
        logger.info("Computing pathway scores...")
        from utils.pathway_features import PathwayFeatureEngineering
        
        pathway_eng = PathwayFeatureEngineering(
            pathway_database=config['data']['pathway_database']
        )
        pathway_eng.load_gene_sets()
        
        pathway_scores = pathway_eng.compute_pathway_scores_from_expression(
            expression_matrix
        )
        pathway_scores.to_csv(pathway_scores_path)
    else:
        pathway_scores = pd.read_csv(pathway_scores_path, index_col=0)
    
    # Normalize expression (if not already)
    # expression_matrix = (expression_matrix - expression_matrix.mean()) / (expression_matrix.std() + 1e-8)
    
    # Normalize pathway scores
    pathway_scores = (pathway_scores - pathway_scores.mean()) / (pathway_scores.std() + 1e-8)
    
    # Normalize survival time
    clinical_data['survival_days_norm'] = (
        clinical_data['survival_days'] - clinical_data['survival_days'].mean()
    ) / (clinical_data['survival_days'].std() + 1e-8)
    
    # Condition features
    condition_features = ['survival_days_norm', 'event_occurred', 'age_years', 'metastasis_at_diagnosis']
    
    # Available features in clinical data
    condition_features = [f for f in condition_features if f in clinical_data.columns]
    
    logger.info(f"Condition features: {condition_features}")
    
    # Create dataset
    dataset = OsteosarcomaDataset(
        mutation_matrix=mutation_matrix,
        expression_matrix=expression_matrix,
        pathway_scores=pathway_scores,
        clinical_data=clinical_data,
        condition_features=condition_features
    )
    
    # Train/val split
    total_size = len(dataset)
    val_size = int(total_size * config['training']['val_split'])
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config['training']['random_seed'])
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Store dimensions in config for model initialization
    config['model']['n_genes_mutation'] = mutation_matrix.shape[1]
    config['model']['n_genes_expression'] = expression_matrix.shape[1]
    config['model']['n_pathways'] = pathway_scores.shape[1]
    config['model']['n_conditions'] = len(condition_features)
    
    return train_loader, val_loader, config


if __name__ == "__main__":
    # Load config
    with open("../config/config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Prepare data
    train_loader, val_loader, config = prepare_data(config)
    
    # Initialize model
    if config['model']['architecture'] == 'diffusion':
        from models.diffusion import BiologyAwareDiffusionModel
        
        model = BiologyAwareDiffusionModel(
            mutation_dim=config['model']['n_genes_mutation'],
            expression_dim=config['model']['n_genes_expression'],
            pathway_dim=config['model']['n_pathways'],
            condition_dim=config['model']['n_conditions'],
            config=config
        )
    elif config['model']['architecture'] == 'cvae':
        from models.cvae import BiologyConstrainedVAE
        
        model = BiologyConstrainedVAE(
            mutation_dim=config['model']['n_genes_mutation'],
            expression_dim=config['model']['n_genes_expression'],
            pathway_dim=config['model']['n_pathways'],
            condition_dim=config['model']['n_conditions'],
            config=config
        )
    else:
        raise ValueError(f"Unknown architecture: {config['model']['architecture']}")
    
    # Train
    trainer = Trainer(model, train_loader, val_loader, config)
    history = trainer.train()
    
    print("Training complete!")
