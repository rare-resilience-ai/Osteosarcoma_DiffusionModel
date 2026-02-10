# Production Deployment Guide

## Overview

This guide covers deploying the Osteosarcoma Disease Progression Model for production use with restricted/proprietary data.

## Architecture for Production

```
┌─────────────────────────────────────────────────────────┐
│                    Production Pipeline                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Data Sources         Processing          Model          │
│  ┌──────────┐        ┌──────────┐      ┌──────────┐    │
│  │ Internal │───────>│ Secure   │─────>│ Trained  │    │
│  │ Hospital │        │ Pipeline │      │ Model    │    │
│  │ Data     │        └──────────┘      └──────────┘    │
│  └──────────┘             │                  │          │
│  ┌──────────┐        ┌──────────┐      ┌──────────┐    │
│  │ Clinical │───────>│ Feature  │─────>│ API      │    │
│  │ Trials   │        │ Engineer │      │ Service  │    │
│  └──────────┘        └──────────┘      └──────────┘    │
│                           │                  │          │
│                      ┌──────────┐      ┌──────────┐    │
│                      │ Quality  │<─────│ Monitor  │    │
│                      │ Control  │      │ Service  │    │
│                      └──────────┘      └──────────┘    │
└─────────────────────────────────────────────────────────┘
```

## 1. Data Integration

### Custom Data Loader

Create a custom loader for your proprietary data:

```python
# data/proprietary_loader.py

class ProprietaryDataLoader:
    """
    Load data from hospital systems / clinical trials
    """
    
    def __init__(self, config):
        self.config = config
        self.db_connection = self.setup_connection()
    
    def setup_connection(self):
        """Setup secure database connection"""
        # Implement your database connection
        # Use environment variables for credentials
        import os
        from sqlalchemy import create_engine
        
        db_url = os.getenv('DATABASE_URL')
        engine = create_engine(db_url, echo=False)
        return engine
    
    def load_mutations(self):
        """Load mutation data from database"""
        query = """
        SELECT patient_id, gene, mutation_type, variant_classification
        FROM mutations
        WHERE cancer_type = 'osteosarcoma'
        AND patient_age < 18
        """
        
        df = pd.read_sql(query, self.db_connection)
        
        # Convert to binary mutation matrix
        mutation_matrix = df.pivot_table(
            index='patient_id',
            columns='gene',
            values='mutation_type',
            aggfunc=lambda x: 1,
            fill_value=0
        )
        
        return mutation_matrix
    
    def load_expression(self):
        """Load RNA-seq expression data"""
        # Implement based on your data format
        pass
    
    def load_clinical(self):
        """Load clinical outcomes"""
        # Implement based on your data format
        pass
```

### Update Configuration

```yaml
# config/config_production.yaml

data:
  data_loader: "proprietary"
  
  # Database connection (set via environment variables)
  database:
    host: "${DB_HOST}"
    port: "${DB_PORT}"
    database: "${DB_NAME}"
    
  # Data security
  encryption:
    enabled: true
    key_path: "/secure/keys/data_key.pem"
  
  # PHI handling
  phi_removal:
    enabled: true
    fields_to_remove: ["patient_name", "mrn", "ssn"]
```

## 2. Model Training in Production

### Secure Training Environment

```bash
# Use Docker for reproducible environment
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy code
COPY . /app
WORKDIR /app

# Environment variables for credentials
ENV DATABASE_URL=postgresql://user:pass@host:5432/db
ENV ENCRYPTION_KEY=/secure/keys/data_key.pem

# Run training
CMD ["python", "main.py", "--config", "config/config_production.yaml", "--steps", "train"]
```

### Training Pipeline

```python
# production_train.py

import logging
from pathlib import Path
from data.proprietary_loader import ProprietaryDataLoader

# Setup logging to secure location
logging.basicConfig(
    filename='/secure/logs/training.log',
    level=logging.INFO
)

def train_production_model():
    """Train model on proprietary data"""
    
    # Load data
    loader = ProprietaryDataLoader(config)
    mutation_matrix = loader.load_mutations()
    expression_matrix = loader.load_expression()
    clinical_data = loader.load_clinical()
    
    # PHI removal
    clinical_data = remove_phi(clinical_data, config)
    
    # Data quality checks
    assert len(mutation_matrix) > 100, "Insufficient samples"
    assert not contains_phi(clinical_data), "PHI detected!"
    
    # Train model
    # ... (same as prototype)
    
    # Encrypt model before saving
    model_path = save_encrypted_model(model, config)
    
    # Log training metrics (without PHI)
    log_training_metrics(history, model_path)
```

## 3. API Service

### FastAPI Implementation

```python
# api/service.py

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import torch

app = FastAPI(title="Osteosarcoma Progression API")
security = HTTPBearer()

# Load model once at startup
model = None

@app.on_event("startup")
async def load_model():
    global model
    model = load_encrypted_model("results/models/best_model_encrypted.pt")

# Request/response models
class PatientProfile(BaseModel):
    survival_time: float
    event_occurred: int
    age: float
    metastasis: int

class SyntheticPatient(BaseModel):
    mutations: dict
    expression: dict
    pathways: dict

# Authentication
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token"""
    token = credentials.credentials
    # Implement token verification
    if not is_valid_token(token):
        raise HTTPException(status_code=401, detail="Invalid token")
    return token

# Endpoints
@app.post("/generate", response_model=SyntheticPatient)
async def generate_patient(
    profile: PatientProfile,
    token: str = Depends(verify_token)
):
    """Generate synthetic patient"""
    
    try:
        # Prepare conditions
        conditions = torch.tensor([[
            profile.survival_time,
            profile.event_occurred,
            profile.age,
            profile.metastasis
        ]])
        
        # Generate
        synthetic = model.sample(conditions, num_samples=1)
        
        # Parse output
        result = parse_synthetic_patient(synthetic)
        
        # Log request (without PHI)
        log_api_request(profile, token)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}
```

### Deploy with Docker

```bash
# Run API service
docker build -t osteosarcoma-api .
docker run -d \
  -p 8000:8000 \
  -v /secure/models:/models \
  -e MODEL_PATH=/models/best_model.pt \
  -e API_TOKEN_SECRET=${API_TOKEN_SECRET} \
  osteosarcoma-api
```

## 4. Monitoring & Logging

### Monitoring Dashboard

```python
# monitoring/dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Osteosarcoma Model Monitoring")

# Load metrics
metrics = load_production_metrics()

# Model performance over time
fig = px.line(metrics, x='timestamp', y='validation_loss',
             title='Model Performance Over Time')
st.plotly_chart(fig)

# Data drift detection
drift_scores = detect_data_drift(metrics)
st.metric("Data Drift Score", f"{drift_scores['overall']:.3f}")

# Alert if drift detected
if drift_scores['overall'] > 0.1:
    st.warning("⚠️ Data drift detected! Consider retraining.")

# Request statistics
st.subheader("API Usage")
request_stats = get_request_stats()
st.dataframe(request_stats)
```

### Logging Strategy

```python
# monitoring/logger.py

import logging
from logging.handlers import RotatingFileHandler
import json

class SecureLogger:
    """Logger that ensures PHI is not logged"""
    
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        
        # Rotating file handler (max 10MB, keep 5 backups)
        handler = RotatingFileHandler(
            '/secure/logs/app.log',
            maxBytes=10*1024*1024,
            backupCount=5
        )
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_generation(self, conditions, result):
        """Log generation request (sanitized)"""
        
        # Remove any potential PHI
        safe_conditions = {
            'survival_time_normalized': conditions['survival_time'],
            'event_occurred': conditions['event_occurred']
            # DO NOT log: patient_id, names, etc.
        }
        
        self.logger.info(f"Generation: {json.dumps(safe_conditions)}")
```

## 5. Security Checklist

### Pre-Deployment

- [ ] Remove all PHI from data
- [ ] Encrypt model files
- [ ] Secure database credentials (use environment variables)
- [ ] Enable API authentication
- [ ] Setup HTTPS/TLS
- [ ] Configure firewall rules
- [ ] Implement rate limiting
- [ ] Setup audit logging
- [ ] Data backup strategy
- [ ] Disaster recovery plan

### Compliance

- [ ] HIPAA compliance review
- [ ] IRB approval (if using patient data)
- [ ] Data use agreements signed
- [ ] Privacy impact assessment
- [ ] Security audit completed

### Code Security

```bash
# Scan for secrets
trufflehog --regex --entropy=True .

# Dependency vulnerability scan
pip-audit

# Code security scan
bandit -r .
```

## 6. Performance Optimization

### Model Optimization

```python
# Quantize model for faster inference
import torch.quantization

model_fp32 = load_model('best_model.pt')
model_quantized = torch.quantization.quantize_dynamic(
    model_fp32,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Save quantized model
torch.save(model_quantized.state_dict(), 'best_model_quantized.pt')
```

### Caching Strategy

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def generate_for_profile(survival_time, event, age, metastasis):
    """Cached generation for common profiles"""
    # Round inputs to reduce cache misses
    conditions = round_conditions(survival_time, event, age, metastasis)
    return model.sample(conditions)
```

## 7. Continuous Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml

name: CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: pip install -r requirements.txt
    
    - name: Run tests
      run: pytest tests/
    
    - name: Security scan
      run: |
        pip install bandit
        bandit -r .
    
    - name: Code quality
      run: |
        pip install flake8
        flake8 .

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Deploy to production
      run: |
        # Deploy to your infrastructure
        echo "Deploying..."
```

## 8. Testing in Production

```python
# tests/test_production.py

def test_model_loading():
    """Ensure model loads correctly"""
    model = load_encrypted_model("results/models/best_model_encrypted.pt")
    assert model is not None

def test_generation_quality():
    """Validate generated samples"""
    conditions = create_test_conditions()
    samples = model.sample(conditions)
    
    # Biological validation
    validator = BiologicalValidator(config)
    results = validator.validate_all(real_data, samples)
    
    assert results['overall_biological_score'] > 0.85

def test_api_response_time():
    """Ensure API responds within SLA"""
    import time
    
    start = time.time()
    response = client.post("/generate", json=test_profile)
    elapsed = time.time() - start
    
    assert elapsed < 1.0  # 1 second SLA
    assert response.status_code == 200
```

## 9. Rollback Strategy

If issues arise in production:

```bash
# Rollback to previous version
docker pull osteosarcoma-api:v1.2.3
docker stop osteosarcoma-api-current
docker run -d --name osteosarcoma-api-current osteosarcoma-api:v1.2.3

# Restore database backup
pg_restore -d osteosarcoma_db backup_20240115.dump

# Reload previous model
cp /backups/best_model_v1.2.3.pt /models/best_model.pt
systemctl restart api-service
```

## 10. Documentation Requirements

Maintain the following documentation:

1. **Data Dictionary**: All features, their meanings, and valid ranges
2. **Model Card**: Model architecture, training data, performance metrics
3. **API Documentation**: OpenAPI spec, examples, rate limits
4. **Runbook**: Deployment steps, troubleshooting guide
5. **Change Log**: Version history with changes
6. **Compliance Documents**: HIPAA, IRB approvals, etc.

---

## Support

For production deployment assistance:
- Email: production-support@example.com
- Slack: #osteosarcoma-model-support
- On-call: Pagerduty rotation
