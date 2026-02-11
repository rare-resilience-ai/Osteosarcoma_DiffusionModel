# Pediatric Osteosarcoma Disease Progression Model

A biology-aware generative model for pediatric osteosarcoma that learns disease progression trajectories, latent biological manifolds, and therapy-response patterns from multi-omic data.

## 🎯 Overview

This project implements a **biology-aware diffusion model** (with cVAE baseline) for generating synthetic pediatric osteosarcoma patients. The model is designed for small-sample scenarios (n=80-120) and incorporates:

- **Pathway-level biological constraints**
- **Mutation-expression correlations**
- **Disease progression trajectories**
- **Conditional generation** based on clinical outcomes

### Key Features

- ✅ **Small-data aware**: Designed for TARGET-OS cohort (n~100)
- ✅ **Biology-constrained**: Enforces pathway coherence and known mutation effects
- ✅ **Multi-modal**: Integrates mutations, RNA-seq, and pathway activity
- ✅ **Conditional generation**: Generate patients with specific outcomes
- ✅ **Rigorous validation**: Biological plausibility + statistical tests
- ✅ **Production-ready**: Modular architecture for restricted data integration

## 📁 Project Structure

```
osteosarcoma_diffusion/
├── config/
│   └── config.yaml                 # Configuration file
├── data/
│   ├── gdc_loader.py              # Download TARGET-OS from GDC
│   └── preprocessor.py            # Data preprocessing
├── models/
│   ├── diffusion.py               # Biology-aware diffusion model
│   └── cvae.py                    # Conditional VAE baseline
├── utils/
│   ├── pathway_features.py        # Pathway feature engineering
│   ├── train.py                   # Training pipeline
│   ├── generate.py                # Synthetic patient generation
│   └── validation.py              # Biological validation
├── notebooks/
│   └── analysis.ipynb             # Analysis and visualization
├── main.py                         # Main pipeline orchestrator
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/rare-resilience-ai/osteosarcoma_diffusion.git
cd osteosarcoma_diffusion

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download and Preprocess Data

```bash
# Download TARGET-OS data from GDC
python main.py --steps download preprocess pathways

# This will:
# 1. Download mutation (MAF), RNA-seq, and clinical data
# 2. Process into mutation matrix, expression matrix, clinical outcomes
# 3. Compute pathway-level features
```

### 3. Train Model

```bash
# Train diffusion model (default)
python main.py --steps train

# Or train cVAE baseline
# Edit config.yaml: model.architecture = "cvae"
python main.py --steps train
```

### 4. Generate Synthetic Patients

```bash
# Generate synthetic patients for all scenarios
python main.py --steps generate

# Scenarios defined in config.yaml:
# - Early stage, good prognosis
# - Metastatic, poor prognosis
# - Typical patient
```

### 5. Validate Synthetic Patients

```bash
# Run biological validation
python main.py --steps validate

# This checks:
# - Mutation co-occurrence patterns
# - Pathway coherence
# - Mutation-expression correlations
# - Statistical distribution matching (MMD, KS tests)
```

### 6. Run Complete Pipeline

```bash
# Run all steps
python main.py --steps all
```

## 📊 Model Architectures

### Biology-Aware Diffusion Model

**Key Innovation**: Conditional diffusion with biological constraints

```
Architecture:
┌─────────────────────────────────────────┐
│  Input: Mutations + Expression + Pathways │
├─────────────────────────────────────────┤
│  Conditional Embedding (survival, age)   │
├─────────────────────────────────────────┤
│  Diffusion U-Net                         │
│  - Time embedding                        │
│  - Skip connections                      │
│  - Biological constraint layers          │
├─────────────────────────────────────────┤
│  Multi-task Loss:                        │
│  - Diffusion loss                        │
│  - Pathway coherence                     │
│  - Mutation-expression correlation       │
│  - Survival prediction (auxiliary)       │
└─────────────────────────────────────────┘
```

**Advantages**:
- Generates high-quality samples
- Captures complex distributions
- Handles multi-modal data well

**Challenges**:
- Slower sampling
- Can oversmooth rare events
- More parameters to tune

### Conditional VAE (Baseline)

**Architecture**:
```
Encoder: (X, conditions) → (μ, σ²)
Latent: z ~ N(μ, σ²)
Decoder: (z, conditions) → X'

Loss:
- Reconstruction: ||X - X'||²
- KL divergence: KL(q(z|X) || p(z))
- Biological constraints (same as diffusion)
```

**Advantages**:
- Faster training and sampling
- Interpretable latent space
- Easier to control generation

## 🧬 Biological Constraints

### 1. Pathway Coherence
- Genes in the same pathway should be co-expressed
- Implemented via within-pathway correlation loss

### 2. Mutation-Expression Correlation
- Known driver mutations → specific pathway changes
- Example: TP53 mutation → p53 pathway downregulation

### 3. Mutually Exclusive Mutations
- Certain mutation pairs rarely co-occur
- Example: TP53 and MDM2 are typically mutually exclusive

### 4. Driver Gene Frequencies
- Osteosarcoma driver genes: TP53, RB1, ATRX, DLG2, PTEN
- Model learns appropriate mutation frequencies

## 📈 Validation Framework

### Statistical Tests

1. **Kolmogorov-Smirnov Test**: Per-feature distribution matching
2. **Maximum Mean Discrepancy (MMD)**: Overall distribution similarity
3. **Wasserstein Distance**: Earth mover's distance on principal components

### Biological Validation

1. **Mutation Co-occurrence**: Chi-square tests for mutation pairs
2. **Pathway Coherence**: Within-pathway gene correlation
3. **Mutation-Expression**: Verify known regulatory relationships
4. **Driver Gene Rates**: Match real mutation frequencies

### Overall Score

```
Biological Score = mean([
    mutation_frequency_correlation,
    cooccurrence_pattern_correlation,
    1 - mutual_exclusivity_violation_rate,
    1 - mutation_expression_violation_rate
])
```

**Target**: > 0.85 for production use

## 🔧 Configuration

Edit `config/config.yaml` to customize:

### Data Parameters
```yaml
data:
  gdc_project: "TARGET-OS"
  min_samples_per_gene: 3
  min_var_expression: 0.1
  pathway_database: "msigdb_hallmark"
```

### Model Parameters
```yaml
model:
  architecture: "diffusion"  # or "cvae"
  latent_dim: 128
  hidden_dims: [256, 512, 256]
  
  diffusion:
    num_steps: 1000
    beta_schedule: "cosine"
  
  constraints:
    pathway_coherence_weight: 1.0
    mutation_expression_weight: 0.5
    survival_prediction_weight: 0.3
```

### Training Parameters
```yaml
training:
  batch_size: 16
  num_epochs: 500
  learning_rate: 0.0001
  
  augmentation:
    mixup_alpha: 0.2
    cross_cancer_pretrain: true
```

## 🎯 Conditional Generation

Generate patients with specific characteristics:

```python
from utils.generate import SyntheticPatientGenerator

# Load trained model
model = load_trained_model("results/checkpoints/best_model.pt")
generator = SyntheticPatientGenerator(model, config)

# Generate good-prognosis patients
good_prognosis = generator.generate(
    num_samples=100,
    scenario={
        'survival_time': 2000,  # days
        'event_occurred': 0,    # alive
        'metastasis_at_diagnosis': 0
    }
)

# Generate poor-prognosis patients
poor_prognosis = generator.generate(
    num_samples=100,
    scenario={
        'survival_time': 300,
        'event_occurred': 1,    # deceased
        'metastasis_at_diagnosis': 1
    }
)
```

## 📊 Results and Outputs

### Generated Files

After running the pipeline:

```
results/
├── checkpoints/
│   ├── best_model.pt              # Best model checkpoint
│   └── checkpoint_epoch_*.pt      # Regular checkpoints
├── synthetic/
│   ├── early_stage_good_prognosis/
│   │   ├── mutations.csv
│   │   ├── expression.csv
│   │   ├── pathways.csv
│   │   └── conditions.csv
│   ├── metastatic_poor_prognosis/
│   └── typical_patient/
├── figures/
│   ├── training_curves.png
│   ├── latent_space_umap.png
│   └── validation_metrics.png
└── validation_results.csv         # Biological validation scores
```

### Example Validation Results

```
Metric                                    Value
─────────────────────────────────────────────────
mutation_frequency_correlation            0.92
cooccurrence_pattern_correlation          0.87
driver_gene_frequency_diff                0.03
mutual_exclusivity_violation_rate         0.02
mutation_expression_violation_rate        0.05
pathway_coherence_correlation             0.89
mmd                                       0.12
overall_biological_score                  0.88
```

## 🔬 Advanced Usage

### Cross-Cancer Pretraining

```python
# In config.yaml
training:
  augmentation:
    cross_cancer_pretrain: true
    pretrain_datasets: ["TCGA-SARC", "TARGET-EWS"]
```

### Custom Pathway Definitions

```python
from utils.pathway_features import PathwayFeatureEngineering

# Add custom pathways
custom_pathways = {
    'CUSTOM_OSTEOSARCOMA_SIGNATURE': [
        'TP53', 'RB1', 'ATRX', 'DLG2', 'PTEN', 'CDKN2A'
    ]
}

pathway_eng = PathwayFeatureEngineering()
pathway_eng.gene_sets.update(custom_pathways)
```

### Integration with Restricted Data

When transitioning to proprietary data:

```python
# 1. Create custom DataLoader for your data format
class ProprietaryDataLoader:
    def load_mutations(self):
        # Your data loading logic
        return mutation_matrix
    
    def load_expression(self):
        # Your data loading logic
        return expression_matrix

# 2. Update config
config['data']['data_loader'] = 'proprietary'

# 3. Pipeline automatically uses your loader
python main.py --config config_proprietary.yaml
```

## 🐛 Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**
```bash
# Reduce batch size
# In config.yaml:
training:
  batch_size: 8  # or 4
```

**2. Data Download Fails**
```bash
# GDC API sometimes has issues
# Retry or download manually from:
# https://portal.gdc.cancer.gov/projects/TARGET-OS
```

**3. Poor Validation Scores**
```bash
# Increase biological constraint weights
model:
  constraints:
    pathway_coherence_weight: 2.0
    mutation_expression_weight: 1.0
```

## 📚 References

1. **TARGET-OS**: [GDC Data Portal](https://portal.gdc.cancer.gov/projects/TARGET-OS)
2. **MSigDB**: [Molecular Signatures Database](http://www.gsea-msigdb.org)
3. **Diffusion Models**: Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
4. **Biology-Aware Generation**: Chen et al., "Generating Realistic Biology Data", Nature Methods 2023

## 📝 Citation

If you use this code, please cite:

```bibtex
@software{osteosarcoma_diffusion,
  title={Biology-Aware Disease Progression Model for Pediatric Osteosarcoma},
  author={Shu Liu},
  year={2026},
  url={https://github.com/rare-resilience-ai/osteosarcoma_diffusion}
}
```

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- TARGET Consortium for open osteosarcoma data
- GDC for data infrastructure
- MSigDB for pathway databases

## 💬 Contact

For questions or issues:
- GitHub Issues: [github.com/rare-resilience-ai/osteosarcoma_diffusion/issues](https://github.com/rare-resilience-ai/osteosarcoma_diffusion/issues)
- Email: shuliu@rare-resilience-ai.com

---

**Note**: This is a research prototype. Synthetic patients should NOT be used for clinical decision-making without extensive validation.
# Osteosarcoma_DiffusionModel

