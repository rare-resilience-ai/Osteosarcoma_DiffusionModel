"""
Debug script to check model state and data dimensions
"""

import torch
import pandas as pd
import yaml
from pathlib import Path

def main():
    print("="*60)
    print("OSTEOSARCOMA MODEL DEBUGGER")
    print("="*60)
    
    # 1. Check config
    print("\n1. CHECKING CONFIG...")
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        print(f"❌ Config not found at {config_path}")
        return
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print(f"✓ Config loaded")
    print(f"  Architecture: {config['model']['architecture']}")
    print(f"  n_conditions in config: {config['model']['n_conditions']}")
    print(f"  condition_on: {config['model']['condition_on']}")
    print(f"  Number of conditions in list: {len(config['model']['condition_on'])}")
    
    # 2. Check processed data
    print("\n2. CHECKING PROCESSED DATA...")
    processed_dir = Path(config['data']['processed_dir'])
    
    files_to_check = [
        'mutation_matrix_aligned.csv',
        'expression_matrix_aligned.csv',
        'pathway_scores.csv',
        'clinical_data.csv'
    ]
    
    data_dims = {}
    for filename in files_to_check:
        filepath = processed_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath, index_col=0)
            print(f"✓ {filename}")
            print(f"  Shape: {df.shape}")
            data_dims[filename] = df.shape
        else:
            print(f"❌ {filename} NOT FOUND")
    
    if 'mutation_matrix_aligned.csv' in data_dims:
        print(f"\n  → Mutation genes: {data_dims['mutation_matrix_aligned.csv'][1]}")
    if 'expression_matrix_aligned.csv' in data_dims:
        print(f"  → Expression genes: {data_dims['expression_matrix_aligned.csv'][1]}")
    if 'pathway_scores.csv' in data_dims:
        print(f"  → Pathways: {data_dims['pathway_scores.csv'][1]}")
    
    # 3. Check checkpoint
    print("\n3. CHECKING CHECKPOINT...")
    checkpoint_path = Path(config['training']['save_dir']) / 'best_model.pt'
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        print("\n⚠️  You need to train the model first!")
        print("   Run: python main.py --steps train")
        return
    
    print(f"✓ Checkpoint found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"\n  Checkpoint keys: {list(checkpoint.keys())}")
    
    if 'model_config' in checkpoint:
        print(f"\n  ✓ Model config found in checkpoint:")
        for key, value in checkpoint['model_config'].items():
            print(f"    {key}: {value}")
    else:
        print(f"\n  ⚠️  No model_config in checkpoint")
    
    # 4. Check state_dict for dimensions
    print("\n4. INSPECTING STATE_DICT...")
    state_dict = checkpoint['model_state_dict']
    
    print(f"  Total parameters: {len(state_dict)}")
    
    # Find condition embedding layer
    cond_keys = [k for k in state_dict.keys() if 'condition' in k.lower()]
    print(f"\n  Condition-related layers:")
    for key in cond_keys[:5]:  # Show first 5
        print(f"    {key}: {state_dict[key].shape}")
    
    # Detect condition dimension
    if 'condition_embed.mlp.0.weight' in state_dict:
        cond_dim = state_dict['condition_embed.mlp.0.weight'].shape[1]
        print(f"\n  → Detected condition dimension: {cond_dim}")
    else:
        print(f"\n  ❌ Could not find condition_embed.mlp.0.weight")
        print(f"     Available keys starting with 'condition':")
        for key in sorted(state_dict.keys()):
            if 'condition' in key:
                print(f"       {key}")
    
    # 5. Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if 'mutation_matrix_aligned.csv' in data_dims and checkpoint_path.exists():
        mut_dim = data_dims['mutation_matrix_aligned.csv'][1]
        expr_dim = data_dims['expression_matrix_aligned.csv'][1]
        path_dim = data_dims['pathway_scores.csv'][1]
        
        print(f"\n✓ Data dimensions from processed files:")
        print(f"  mutation_dim = {mut_dim}")
        print(f"  expression_dim = {expr_dim}")
        print(f"  pathway_dim = {path_dim}")
        
        if 'condition_embed.mlp.0.weight' in state_dict:
            cond_dim = state_dict['condition_embed.mlp.0.weight'].shape[1]
            print(f"  condition_dim = {cond_dim} (from checkpoint)")
            
            print(f"\n✓ Config should have:")
            print(f"  n_genes_mutation: {mut_dim}")
            print(f"  n_genes_expression: {expr_dim}")
            print(f"  n_pathways: {path_dim}")
            print(f"  n_conditions: {cond_dim}")
            
            # Check mismatch
            config_cond = config['model']['n_conditions']
            if config_cond != cond_dim:
                print(f"\n⚠️  MISMATCH DETECTED!")
                print(f"  Config says {config_cond} conditions")
                print(f"  Model was trained with {cond_dim} conditions")
                print(f"\n  → Fix: Change n_conditions to {cond_dim} in config.yaml")
            else:
                print(f"\n✓ Condition dimensions match!")
    
    # 6. Check scenarios
    print("\n" + "="*60)
    print("CHECKING SCENARIOS")
    print("="*60)
    
    scenarios = config['generation']['scenarios']
    for scenario in scenarios:
        name = scenario['name']
        conditions = scenario['conditions']
        num_conds = len(conditions)
        print(f"\n  Scenario: {name}")
        print(f"  Conditions: {list(conditions.keys())}")
        print(f"  Number of conditions: {num_conds}")
        
        if 'condition_embed.mlp.0.weight' in state_dict:
            expected = state_dict['condition_embed.mlp.0.weight'].shape[1]
            if num_conds != expected:
                print(f"  ⚠️  MISMATCH: Has {num_conds}, expected {expected}")
            else:
                print(f"  ✓ Matches model")

if __name__ == "__main__":
    main()
