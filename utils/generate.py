"""
Generation Pipeline
Create synthetic patients with conditional generation
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Optional
from tqdm import tqdm
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyntheticPatientGenerator:
    """Generate synthetic patients using trained model"""
    
    def __init__(
        self,
        model,
        config: dict,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.model.eval()
        self.config = config
        self.device = device
        
        # Data dimensions
        self.mutation_dim = model.mutation_dim
        self.expression_dim = model.expression_dim
        self.pathway_dim = model.pathway_dim
        self.condition_dim = model.condition_dim
        
    def create_conditions(
        self,
        num_samples: int,
        scenario: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Create condition vectors for generation
        
        Args:
            num_samples: Number of samples to generate
            scenario: Dictionary with condition values
                     e.g., {'survival_time': 1000, 'event_occurred': 0, ...}
        
        Returns:
            conditions: (num_samples, condition_dim)
        """
        
        if scenario is not None:
            # Use specified scenario
            condition_values = []
            
            # Expected order: survival_days_norm, event_occurred, age_years, metastasis_at_diagnosis
            # Need to normalize survival_days
            
            if 'survival_time' in scenario:
                # Normalize (assuming mean=800, std=500 from typical data)
                survival_norm = (scenario['survival_time'] - 800) / 500
                condition_values.append(survival_norm)
            else:
                condition_values.append(0.0)
            
            condition_values.append(scenario.get('event_occurred', 0))
            condition_values.append(scenario.get('age', 15.0))  # Default pediatric age
            condition_values.append(scenario.get('metastasis_at_diagnosis', 0))
            
            # Repeat for all samples
            conditions = torch.tensor(
                [condition_values] * num_samples,
                dtype=torch.float32,
                device=self.device
            )
        else:
            # Random conditions
            conditions = torch.randn(num_samples, self.condition_dim, device=self.device)
        
        return conditions
    
    @torch.no_grad()
    def generate(
        self,
        num_samples: int,
        scenario: Optional[Dict] = None,
        guidance_scale: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """
        Generate synthetic patients
        
        Args:
            num_samples: Number of synthetic patients to generate
            scenario: Conditional scenario (survival, metastasis, etc.)
            guidance_scale: Guidance strength for conditional generation
        
        Returns:
            synthetic_data: Dictionary with mutations, expression, pathways
        """
        
        logger.info(f"Generating {num_samples} synthetic patients...")
        
        if scenario:
            logger.info(f"Scenario: {scenario}")
        
        # Create conditions
        conditions = self.create_conditions(num_samples, scenario)
        
        # Generate
        synthetic_samples = self.model.sample(conditions, num_samples=num_samples)
        
        # Convert to numpy
        synthetic_samples = synthetic_samples.cpu().numpy()
        
        # Split into components
        mutations = synthetic_samples[:, :self.mutation_dim]
        expression = synthetic_samples[:, self.mutation_dim:self.mutation_dim + self.expression_dim]
        pathways = synthetic_samples[:, self.mutation_dim + self.expression_dim:]
        
        # Post-process mutations (binarize)
        mutations = (mutations > 0.5).astype(float)
        
        logger.info("Generation complete!")
        
        return {
            'mutations': mutations,
            'expression': expression,
            'pathways': pathways,
            'conditions': conditions.cpu().numpy()
        }
    
    def generate_scenarios(
        self,
        scenarios: List[Dict],
        samples_per_scenario: int
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Generate synthetic patients for multiple scenarios
        
        Args:
            scenarios: List of scenario dictionaries
            samples_per_scenario: Number of samples per scenario
        
        Returns:
            all_synthetic_data: Dictionary keyed by scenario name
        """
        
        all_synthetic_data = {}
        
        for scenario in scenarios:
            scenario_name = scenario['name']
            logger.info(f"\nGenerating scenario: {scenario_name}")
            
            synthetic_data = self.generate(
                num_samples=samples_per_scenario,
                scenario=scenario['conditions']
            )
            
            all_synthetic_data[scenario_name] = synthetic_data
        
        return all_synthetic_data
    
    def save_synthetic_data(
        self,
        synthetic_data: Dict[str, np.ndarray],
        output_dir: Path,
        gene_names: Dict[str, List[str]],
        prefix: str = "synthetic"
    ):
        """
        Save synthetic data to files
        
        Args:
            synthetic_data: Dictionary with mutations, expression, pathways
            output_dir: Directory to save files
            gene_names: Dictionary with gene/pathway names for columns
            prefix: Filename prefix
        """
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save mutations
        if 'mutation_genes' in gene_names:
            mut_df = pd.DataFrame(
                synthetic_data['mutations'],
                columns=gene_names['mutation_genes']
            )
            mut_path = output_dir / f"{prefix}_mutations.csv"
            mut_df.to_csv(mut_path, index=False)
            logger.info(f"Saved mutations to {mut_path}")
        
        # Save expression
        if 'expression_genes' in gene_names:
            expr_df = pd.DataFrame(
                synthetic_data['expression'],
                columns=gene_names['expression_genes']
            )
            expr_path = output_dir / f"{prefix}_expression.csv"
            expr_df.to_csv(expr_path, index=False)
            logger.info(f"Saved expression to {expr_path}")
        
        # Save pathways
        if 'pathway_names' in gene_names:
            pathway_df = pd.DataFrame(
                synthetic_data['pathways'],
                columns=gene_names['pathway_names']
            )
            pathway_path = output_dir / f"{prefix}_pathways.csv"
            pathway_df.to_csv(pathway_path, index=False)
            logger.info(f"Saved pathways to {pathway_path}")
        
        # Save conditions
        cond_df = pd.DataFrame(
            synthetic_data['conditions'],
            columns=['survival_days_norm', 'event_occurred', 'age_years', 'metastasis_at_diagnosis']
        )
        cond_path = output_dir / f"{prefix}_conditions.csv"
        cond_df.to_csv(cond_path, index=False)
        logger.info(f"Saved conditions to {cond_path}")


def load_trained_model(checkpoint_path: Path, config: dict, device: str):
    """Load trained model from checkpoint"""
    
    logger.info(f"Loading model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    # Detect actual dimension from the saved weights
    # This looks at the weight matrix of the condition projection layer
    saved_cond_dim = state_dict['condition_embed.mlp.0.weight'].shape[1]
    logger.info(f"Checkpoint condition dimension: {saved_cond_dim}")
    
    # Initialize model
    if config['model']['architecture'] == 'diffusion':
        from models.diffusion import BiologyAwareDiffusionModel
        
        model = BiologyAwareDiffusionModel(
            mutation_dim=config['model']['n_genes_mutation'],
            expression_dim=config['model']['n_genes_expression'],
            pathway_dim=config['model']['n_pathways'],
            condition_dim=saved_cond_dim, # Use the detected dim
            config=config
        )
    elif config['model']['architecture'] == 'cvae':
        from models.cvae import BiologyConstrainedVAE
        model = BiologyConstrainedVAE(
            mutation_dim=config['model']['n_genes_mutation'],
            expression_dim=config['model']['n_genes_expression'],
            pathway_dim=config['model']['n_pathways'],
            condition_dim=saved_cond_dim, # Update this line too!
            config=config
        )
    else:
        raise ValueError(f"Unknown architecture: {config['model']['architecture']}")
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info("Model loaded successfully!")
    
    return model


def main():
    """Main generation pipeline"""
    
    # Load config
    config_path = Path("../config/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Load model
    checkpoint_path = Path(config['training']['save_dir']) / 'best_model.pt'
    
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.error("Please train the model first!")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_trained_model(checkpoint_path, config, device)
    
    # Initialize generator
    generator = SyntheticPatientGenerator(model, config, device)
    
    # Generate scenarios
    scenarios = config['generation']['scenarios']
    samples_per_scenario = config['generation']['num_synthetic_samples'] // len(scenarios)
    
    all_synthetic_data = generator.generate_scenarios(scenarios, samples_per_scenario)
    
    # Load gene names for column headers
    processed_dir = Path(config['data']['processed_dir'])
    
    mutation_matrix = pd.read_csv(
        processed_dir / 'mutation_matrix_aligned.csv',
        index_col=0
    )
    expression_matrix = pd.read_csv(
        processed_dir / 'expression_matrix_aligned.csv',
        index_col=0
    )
    pathway_scores = pd.read_csv(
        processed_dir / 'pathway_scores.csv',
        index_col=0
    )
    
    gene_names = {
        'mutation_genes': mutation_matrix.columns.tolist(),
        'expression_genes': expression_matrix.columns.tolist(),
        'pathway_names': pathway_scores.columns.tolist()
    }
    
    # Save synthetic data for each scenario
    output_dir = Path(config['output']['synthetic_data_dir'])
    
    for scenario_name, synthetic_data in all_synthetic_data.items():
        scenario_dir = output_dir / scenario_name
        generator.save_synthetic_data(
            synthetic_data,
            scenario_dir,
            gene_names,
            prefix=scenario_name
        )
    
    logger.info("\nGeneration pipeline complete!")
    logger.info(f"Synthetic data saved to {output_dir}")


if __name__ == "__main__":
    main()
