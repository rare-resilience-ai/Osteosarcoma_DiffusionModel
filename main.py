#!/usr/bin/env python3
"""
Main Pipeline for Pediatric Osteosarcoma Disease Progression Model
Orchestrates the complete workflow from data download to validation
"""

import argparse
import logging
from pathlib import Path
import yaml
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_data(config):
    """Step 1: Download TARGET-OS data from GDC"""
    logger.info("=" * 60)
    logger.info("STEP 1: Downloading TARGET-OS data from GDC")
    logger.info("=" * 60)
    
    from data.gdc_loader import GDCDataLoader
    
    loader = GDCDataLoader(
        project_id=config['data']['gdc_project'],
        data_dir=config['data']['data_dir']
    )
    
    results = loader.download_all()
    logger.info(f"Downloaded data to: {results}")
    
    return results


def preprocess_data(config):
    """Step 2: Preprocess raw data into ML-ready format"""
    logger.info("=" * 60)
    logger.info("STEP 2: Preprocessing data")
    logger.info("=" * 60)
    
    from data.preprocessor import OsteosarcomaDataProcessor
    
    processor = OsteosarcomaDataProcessor(
        raw_dir=Path(config['data']['raw_dir']),
        processed_dir=Path(config['data']['processed_dir']),
        config=config
    )
    
    processed_data = processor.process_all()
    
    logger.info(f"Processed {len(processed_data['mutation_matrix'])} samples")
    
    return processed_data


def compute_pathway_features(config):
    """Step 3: Compute pathway-level features"""
    logger.info("=" * 60)
    logger.info("STEP 3: Computing pathway features")
    logger.info("=" * 60)
    
    import pandas as pd
    from utils.pathway_features import PathwayFeatureEngineering
    
    processed_dir = Path(config['data']['processed_dir'])
    
    # Load expression data
    expression_matrix = pd.read_csv(
        processed_dir / 'expression_matrix_aligned.csv',
        index_col=0
    )
    
    mutation_matrix = pd.read_csv(
        processed_dir / 'mutation_matrix_aligned.csv',
        index_col=0
    )
    
    # Compute pathway scores
    pathway_eng = PathwayFeatureEngineering(
        pathway_database=config['data']['pathway_database']
    )
    pathway_eng.load_gene_sets()
    
    # Expression-based pathway scores
    pathway_scores = pathway_eng.compute_pathway_scores_from_expression(
        expression_matrix
    )
    pathway_scores.to_csv(processed_dir / 'pathway_scores.csv')
    
    # Mutation-based pathway scores
    pathway_mut_scores = pathway_eng.compute_pathway_scores_from_mutations(
        mutation_matrix
    )
    pathway_mut_scores.to_csv(processed_dir / 'pathway_mutation_scores.csv')
    
    # Gene-pathway matrix for graph construction
    gene_pathway_matrix = pathway_eng.create_gene_pathway_matrix()
    gene_pathway_matrix.to_csv(processed_dir / 'gene_pathway_matrix.csv')
    
    logger.info(f"Computed {len(pathway_scores.columns)} pathway features")
    
    return pathway_scores


def train_model(config, resume=False):
    """Step 4: Train the disease progression model"""
    logger.info("=" * 60)
    logger.info("STEP 4: Training model")
    logger.info("=" * 60)
    
    from utils.train import prepare_data, Trainer
    
    # Prepare data
    train_loader, val_loader, config_updated = prepare_data(config)

    # 2. DYNAMICALLY UPDATE CONFIG 
    # We add an extra '.dataset' to unwrap the Subset object
    base_dataset = train_loader.dataset.dataset

    config_updated['model']['n_genes_mutation'] = base_dataset.mutations.shape[1]
    config_updated['model']['n_genes_expression'] = base_dataset.expression.shape[1]
    config_updated['model']['n_pathways'] = base_dataset.pathways.shape[1]
    config_updated['model']['n_conditions'] = base_dataset.conditions.shape[1]

    logger.info(f"Model configured with: Mut={config_updated['model']['n_genes_mutation']}, "
                f"Expr={config_updated['model']['n_genes_expression']}, "
                f"Path={config_updated['model']['n_pathways']}")
    
    # Initialize model
    if config_updated['model']['architecture'] == 'diffusion':
        from models.diffusion import BiologyAwareDiffusionModel
        
        model = BiologyAwareDiffusionModel(
            mutation_dim=config_updated['model']['n_genes_mutation'],
            expression_dim=config_updated['model']['n_genes_expression'],
            pathway_dim=config_updated['model']['n_pathways'],
            condition_dim=config_updated['model']['n_conditions'],
            config=config_updated
        )
    elif config_updated['model']['architecture'] == 'cvae':
        from models.cvae import BiologyConstrainedVAE
        
        model = BiologyConstrainedVAE(
            mutation_dim=config_updated['model']['n_genes_mutation'],
            expression_dim=config_updated['model']['n_genes_expression'],
            pathway_dim=config_updated['model']['n_pathways'],
            condition_dim=config_updated['model']['n_conditions'],
            config=config_updated
        )
    else:
        raise ValueError(f"Unknown architecture: {config_updated['model']['architecture']}")
    
    # Train
    trainer = Trainer(model, train_loader, val_loader, config_updated)
    history = trainer.train()
    
    logger.info("Training complete!")
    
    # Save updated config
    with open('config/config_updated.yaml', 'w') as f:
        yaml.dump(config_updated, f)
    
    return history


def generate_synthetic_patients(config):
    """Step 5: Generate synthetic patients"""
    logger.info("=" * 60)
    logger.info("STEP 5: Generating synthetic patients")
    logger.info("=" * 60)
    
    from utils.generate import load_trained_model, SyntheticPatientGenerator
    import pandas as pd
    
    # 1. Load updated config if exists
    updated_config_path = Path('config/config_updated.yaml')
    if updated_config_path.exists():
        with open(updated_config_path) as f:
            config = yaml.safe_load(f)
    
    # 2. DYNAMICALLY DETECT CONDITION DIMENSION BEFORE LOADING MODEL
    # Load the processed conditions to see how many columns there actually are
    processed_dir = Path(config['data']['processed_dir'])
    conditions_path = processed_dir / 'clinical_data_cleaned.csv' # Adjust filename if different
    if conditions_path.exists():
        temp_df = pd.read_csv(conditions_path)
        # Drop ID or non-feature columns if your processor leaves them in
        if 'patient_id' in temp_df.columns:
            temp_df = temp_df.drop(columns=['patient_id'])
        
        actual_n_conditions = temp_df.shape[1]
        config['model']['n_conditions'] = actual_n_conditions
        logger.info(f"Detected {actual_n_conditions} clinical conditions. Updating config.")

    # 3. Load model with the corrected config
    checkpoint_path = Path(config['training']['save_dir']) / 'best_model.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_trained_model(checkpoint_path, config, device)
    
    # Initialize generator
    generator = SyntheticPatientGenerator(model, config, device)
    
    # Generate scenarios
    scenarios = config['generation']['scenarios']
    samples_per_scenario = config['generation']['num_synthetic_samples'] // len(scenarios)
    
    all_synthetic_data = generator.generate_scenarios(scenarios, samples_per_scenario)
    
    # Load gene names
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
    
    # Save
    output_dir = Path(config['output']['synthetic_data_dir'])
    
    for scenario_name, synthetic_data in all_synthetic_data.items():
        scenario_dir = output_dir / scenario_name
        generator.save_synthetic_data(
            synthetic_data,
            scenario_dir,
            gene_names,
            prefix=scenario_name
        )
    
    logger.info(f"Synthetic data saved to {output_dir}")
    
    return all_synthetic_data


def validate_synthetic_patients(config):
    """Step 6: Validate synthetic patients"""
    logger.info("=" * 60)
    logger.info("STEP 6: Validating synthetic patients")
    logger.info("=" * 60)
    
    import pandas as pd
    from utils.validation import BiologicalValidator
    
    # Load real data
    processed_dir = Path(config['data']['processed_dir'])
    
    real_mutations = pd.read_csv(
        processed_dir / 'mutation_matrix_aligned.csv',
        index_col=0
    )
    real_expression = pd.read_csv(
        processed_dir / 'expression_matrix_aligned.csv',
        index_col=0
    )
    real_pathways = pd.read_csv(
        processed_dir / 'pathway_scores.csv',
        index_col=0
    )
    
    # Load synthetic data (combine all scenarios)
    output_dir = Path(config['output']['synthetic_data_dir'])
    
    all_synth_mutations = []
    all_synth_expression = []
    all_synth_pathways = []
    
    for scenario in config['generation']['scenarios']:
        scenario_name = scenario['name']
        scenario_dir = output_dir / scenario_name
        
        synth_mut = pd.read_csv(scenario_dir / f"{scenario_name}_mutations.csv")
        synth_expr = pd.read_csv(scenario_dir / f"{scenario_name}_expression.csv")
        synth_path = pd.read_csv(scenario_dir / f"{scenario_name}_pathways.csv")
        
        all_synth_mutations.append(synth_mut)
        all_synth_expression.append(synth_expr)
        all_synth_pathways.append(synth_path)
    
    synth_mutations = pd.concat(all_synth_mutations, ignore_index=True)
    synth_expression = pd.concat(all_synth_expression, ignore_index=True)
    synth_pathways = pd.concat(all_synth_pathways, ignore_index=True)
    
    # Validate
    validator = BiologicalValidator(config)
    
    results = validator.validate_all(
        real_mutations=real_mutations,
        real_expression=real_expression,
        real_pathways=real_pathways,
        synth_mutations=synth_mutations,
        synth_expression=synth_expression,
        synth_pathways=synth_pathways
    )
    
    # Save results
    results_dir = Path(config['output']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame([results])
    results_df.to_csv(results_dir / 'validation_results.csv', index=False)
    
    logger.info(f"Validation results saved to {results_dir / 'validation_results.csv'}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Pediatric Osteosarcoma Disease Progression Model Pipeline"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--steps',
        nargs='+',
        default=['all'],
        choices=['download', 'preprocess', 'pathways', 'train', 'generate', 'validate', 'all'],
        help='Pipeline steps to run'
    )
    
    parser.add_argument(
        '--resume-training',
        action='store_true',
        help='Resume training from checkpoint'
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    logger.info("Starting Osteosarcoma Disease Progression Pipeline")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Steps: {args.steps}")
    
    steps = args.steps
    if 'all' in steps:
        steps = ['download', 'preprocess', 'pathways', 'train', 'generate', 'validate']
    
    try:
        # Execute pipeline steps
        if 'download' in steps:
            download_data(config)
        
        if 'preprocess' in steps:
            preprocess_data(config)
        
        if 'pathways' in steps:
            compute_pathway_features(config)
        
        if 'train' in steps:
            train_model(config, resume=args.resume_training)
        
        if 'generate' in steps:
            generate_synthetic_patients(config)
        
        if 'validate' in steps:
            validate_synthetic_patients(config)
        
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    import torch
    main()
