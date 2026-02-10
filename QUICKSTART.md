# Quick Start Guide

## Get Started in 5 Minutes

This guide will help you run the complete pipeline on TARGET-OS data.

## Prerequisites

- Python 3.8+
- 16GB RAM minimum
- GPU recommended (but not required)
- Internet connection (for data download)

## Installation

```bash
# Navigate to project directory
cd osteosarcoma_diffusion

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Run the Complete Pipeline

### Option 1: Run Everything (Recommended for First Time)

```bash
# This will:
# 1. Download TARGET-OS data from GDC (~30 min)
# 2. Preprocess data (~5 min)
# 3. Compute pathway features (~2 min)
# 4. Train model (~2-4 hours on GPU, ~8-12 hours on CPU)
# 5. Generate synthetic patients (~5 min)
# 6. Validate synthetic patients (~5 min)

python main.py --steps all
```

### Option 2: Run Step-by-Step

```bash
# Step 1: Download data
python main.py --steps download

# Step 2: Preprocess data
python main.py --steps preprocess

# Step 3: Compute pathway features
python main.py --steps pathways

# Step 4: Train model
python main.py --steps train

# Step 5: Generate synthetic patients
python main.py --steps generate

# Step 6: Validate synthetic patients
python main.py --steps validate
```

## Analyze Results

```bash
# Launch Jupyter notebook
jupyter notebook notebooks/analysis.ipynb

# Or view results files directly
cat results/validation_results.csv
ls results/synthetic/
```

## Customize Configuration

Edit `config/config.yaml` to change:

```yaml
# Model architecture
model:
  architecture: "cvae"  # Change to "cvae" for faster training

# Training parameters
training:
  num_epochs: 100  # Reduce for faster training
  batch_size: 8    # Reduce if out of memory
```

## Expected Output

After running the complete pipeline, you'll have:

```
results/
├── checkpoints/
│   └── best_model.pt              # Trained model
├── synthetic/
│   ├── early_stage_good_prognosis/
│   │   ├── mutations.csv          # Synthetic mutations
│   │   ├── expression.csv         # Synthetic expression
│   │   ├── pathways.csv           # Synthetic pathways
│   │   └── conditions.csv         # Clinical conditions
│   ├── metastatic_poor_prognosis/
│   └── typical_patient/
├── figures/
│   ├── mutation_frequency_comparison.png
│   ├── pathway_distributions.png
│   ├── umap_real_vs_synthetic.png
│   └── validation_metrics.png
└── validation_results.csv         # Validation scores
```

## Troubleshooting

### Issue: Out of Memory

```bash
# Reduce batch size in config.yaml
training:
  batch_size: 4  # or even 2
```

### Issue: Data Download Fails

```bash
# Check internet connection
# Or download manually from:
# https://portal.gdc.cancer.gov/projects/TARGET-OS

# Place files in:
# data/raw/mutations/
# data/raw/rna_seq/
# data/raw/clinical.csv
```

### Issue: Validation Score Low

```bash
# Increase biological constraint weights in config.yaml
model:
  constraints:
    pathway_coherence_weight: 2.0      # Increase from 1.0
    mutation_expression_weight: 1.0    # Increase from 0.5
```

## Next Steps

1. **Review Results**: Check `results/validation_results.csv`
2. **Analyze Synthetic Data**: Use `notebooks/analysis.ipynb`
3. **Fine-tune Model**: Adjust `config/config.yaml` and retrain
4. **Production Deployment**: See `docs/DEPLOYMENT.md`

## Getting Help

- **Documentation**: See `README.md` for detailed guide
- **Issues**: Check GitHub issues or create new one
- **Email**: your.email@example.com

## Example: Generate Specific Patient Profile

```python
from utils.generate import SyntheticPatientGenerator, load_trained_model
import yaml

# Load config
with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

# Load model
model = load_trained_model('results/checkpoints/best_model.pt', config, 'cpu')

# Generate patient
generator = SyntheticPatientGenerator(model, config, 'cpu')

# Good prognosis patient
good_prognosis = generator.generate(
    num_samples=10,
    scenario={
        'survival_time': 2000,  # 2000 days
        'event_occurred': 0,    # Alive
        'age': 14.0,           # 14 years old
        'metastasis_at_diagnosis': 0  # No metastasis
    }
)

print(f"Generated {len(good_prognosis['mutations'])} patients")
```

## Benchmarks

On a typical system:

| Step | Time (GPU) | Time (CPU) |
|------|-----------|-----------|
| Download | 30 min | 30 min |
| Preprocess | 5 min | 5 min |
| Pathways | 2 min | 2 min |
| Train (500 epochs) | 2-4 hours | 8-12 hours |
| Generate (1000 samples) | 5 min | 10 min |
| Validate | 5 min | 5 min |
| **Total** | **3-5 hours** | **9-13 hours** |

## Minimal Example

Want to skip data download and train on dummy data?

```python
# Generate dummy data for testing
python -c "
import numpy as np
import pandas as pd
from pathlib import Path

# Create dummy data
n_samples = 100
Path('data/processed').mkdir(parents=True, exist_ok=True)

# Mutations
mut = pd.DataFrame(np.random.randint(0, 2, (n_samples, 50)))
mut.to_csv('data/processed/mutation_matrix_aligned.csv')

# Expression
expr = pd.DataFrame(np.random.randn(n_samples, 100))
expr.to_csv('data/processed/expression_matrix_aligned.csv')

# Pathways
path = pd.DataFrame(np.random.randn(n_samples, 30))
path.to_csv('data/processed/pathway_scores.csv')

# Clinical
clin = pd.DataFrame({
    'submitter_id': [f'P{i}' for i in range(n_samples)],
    'survival_days': np.random.randint(100, 2000, n_samples),
    'event_occurred': np.random.randint(0, 2, n_samples),
    'age_years': np.random.uniform(10, 18, n_samples),
    'metastasis_at_diagnosis': np.random.randint(0, 2, n_samples)
})
clin.to_csv('data/processed/clinical_aligned.csv', index=False)
"

# Train on dummy data
python main.py --steps train generate validate
```

This will run much faster (~30 minutes total) for testing the pipeline.

---

**Ready to start?** Run `python main.py --steps all` and come back in a few hours! ☕
