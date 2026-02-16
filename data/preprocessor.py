"""
Data Preprocessing Module
Converts raw GDC data into ML-ready matrices with biological features
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict
import gzip

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OsteosarcomaDataProcessor:
    """Process TARGET-OS data into mutation matrix, expression matrix, and clinical outcomes"""
    
    def __init__(self, raw_dir: Path, processed_dir: Path, config: dict):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        
    def process_mutations(self) -> pd.DataFrame:
        """
        Process MAF files into binary mutation matrix
        Returns: DataFrame with shape (n_samples, n_genes)
        """
        
        logger.info("Processing mutation data...")
        
        maf_dir = self.raw_dir / "mutations"
        maf_files = list(maf_dir.glob("*.maf*"))
        
        if not maf_files:
            raise FileNotFoundError(f"No MAF files found in {maf_dir}")
        
        # Concatenate all MAF files
        all_mutations = []
        
        for maf_file in maf_files:
            logger.info(f"Reading {maf_file.name}...")
            
            # Handle gzipped files
            if maf_file.suffix == '.gz':
                with gzip.open(maf_file, 'rt') as f:
                    df = pd.read_csv(f, sep='\t', comment='#', low_memory=False)
            else:
                df = pd.read_csv(maf_file, sep='\t', comment='#', low_memory=False)
            
            all_mutations.append(df)
        
        mutations_df = pd.concat(all_mutations, ignore_index=True)
        logger.info(f"Total mutations: {len(mutations_df)}")
        
        # Filter for high-confidence variants
        # Keep only protein-altering mutations
        variant_classes = [
            'Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del',
            'Frame_Shift_Ins', 'In_Frame_Del', 'In_Frame_Ins', 'Splice_Site'
        ]
        
        mutations_df = mutations_df[
            mutations_df['Variant_Classification'].isin(variant_classes)
        ]
        
        logger.info(f"Protein-altering mutations: {len(mutations_df)}")
        
        # Create binary mutation matrix
        # Pivot: rows = samples, columns = genes, values = 1 (mutated) or 0
        mutation_matrix = mutations_df.pivot_table(
            index='Tumor_Sample_Barcode',
            columns='Hugo_Symbol',
            values='Variant_Classification',
            aggfunc=lambda x: 1,  # Binary: mutated or not
            fill_value=0
        )
        
        # Filter genes: keep only those mutated in >= min_samples_per_gene samples
        min_samples = self.config['data']['min_samples_per_gene']
        gene_counts = mutation_matrix.sum(axis=0)
        genes_to_keep = gene_counts[gene_counts >= min_samples].index
        
        mutation_matrix = mutation_matrix[genes_to_keep]
        
        logger.info(f"Mutation matrix shape: {mutation_matrix.shape}")
        logger.info(f"Samples: {mutation_matrix.shape[0]}, Genes: {mutation_matrix.shape[1]}")
        
        # Save
        output_path = self.processed_dir / "mutation_matrix.csv"
        mutation_matrix.to_csv(output_path)
        logger.info(f"Saved to {output_path}")
        
        return mutation_matrix
    
    def process_rna_seq(self) -> pd.DataFrame:
        """
        Process RNA-seq count files into expression matrix
        Returns: DataFrame with shape (n_samples, n_genes)
        """
        
        logger.info("Processing RNA-seq data...")
        
        rna_dir = self.raw_dir / "rna_seq"
        metadata_path = rna_dir / "metadata.csv"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"RNA-seq metadata not found at {metadata_path}")
        
        metadata = pd.read_csv(metadata_path)
        
        # Read all count files
        expression_data = []
        
        for _, row in metadata.iterrows():
            file_path = Path(row['file_path'])
            
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue
            
            # Read STAR counts file
            if file_path.suffix == '.gz':
                counts = pd.read_csv(file_path, sep='\t', compression='gzip', comment='#')
            else:
                counts = pd.read_csv(file_path, sep='\t', comment='#')
            
            # GDC 2.0 column names are usually: 
            # gene_id, gene_name, gene_type, unstranded, stranded_first, ...
            # If 'gene_name' is missing, we use 'gene_id'
            id_col = 'gene_name' if 'gene_name' in counts.columns else 'gene_id'

            if 'unstranded' in counts.columns:
                count_col = 'unstranded'
            elif 'tpm_unstranded' in counts.columns: # Sometimes TPM is preferred
                count_col = 'tpm_unstranded'
            else:
                count_col = counts.columns[3] # Fallback to the 4th column (index 3)

            gene_counts = counts.set_index(id_col)[count_col]
            gene_counts.name = row['submitter_id']
            
            expression_data.append(gene_counts)
        
        # Combine into matrix
        expression_matrix = pd.concat(expression_data, axis=1).T
        
        # Remove version numbers from gene names (e.g., ENSG00000000003.14 -> ENSG00000000003)
        expression_matrix.columns = expression_matrix.columns.str.split('.').str[0]
        
        # Filter low-variance genes
        gene_variances = expression_matrix.var(axis=0)
        # Take the top 5000 most informative genes
        top_genes = gene_variances.sort_values(ascending=False).head(5000).index
        expression_matrix = expression_matrix[top_genes]
        
        logger.info(f"RNA-seq matrix reduced to {expression_matrix.shape[1]} high-variance genes.")
        
        # Log-transform: log2(counts + 1)
        expression_matrix = np.log2(expression_matrix + 1)
        
        logger.info(f"Expression matrix shape: {expression_matrix.shape}")
        
        # Save
        output_path = self.processed_dir / "expression_matrix.csv"
        expression_matrix.to_csv(output_path)
        logger.info(f"Saved to {output_path}")
        
        return expression_matrix
    
    def process_clinical(self) -> pd.DataFrame:
        """
        Process clinical data into structured format
        """
        
        logger.info("Processing clinical data...")
        
        clinical_path = self.raw_dir / "clinical.csv"
        clinical_df = pd.read_csv(clinical_path)

        # Standardize column names (GDC API vs Portal CSVs vary)
        clinical_df.columns = [c.lower() for c in clinical_df.columns]

        # ---  FORCE NUMERIC CONVERSION ---
        # GDC often puts 'not reported' or '--' in these columns. 
      
        clinical_df['days_to_death'] = pd.to_numeric(clinical_df['days_to_death'], errors='coerce')
        clinical_df['days_to_last_follow_up'] = pd.to_numeric(clinical_df['days_to_last_follow_up'], errors='coerce')
        clinical_df['age_at_diagnosis'] = pd.to_numeric(clinical_df['age_at_diagnosis'], errors='coerce')

        # Fix Vital Status
        clinical_df['vital_status_clean'] = clinical_df['vital_status'].fillna('Unknown').str.capitalize()
        clinical_df['event_occurred'] = (clinical_df['vital_status_clean'] == 'Dead').astype(int)

        # Survival calculation
        clinical_df['survival_days'] = clinical_df['days_to_death'].fillna(
            clinical_df['days_to_last_follow_up']
        )
        
        # Extract metastasis at diagnosis (from tumor stage)
        def has_metastasis(stage):
            if pd.isna(stage): return 0
            stage = str(stage).upper()
            return 1 if ('IV' in stage or 'M1' in stage) else 0
        clinical_df['metastasis_at_diagnosis'] = clinical_df['tumor_stage'].apply(has_metastasis)
    
        # Encode age
        clinical_df['age_years'] = clinical_df['age_at_diagnosis'] / 365.25

        # --- NEW: GENDER ENCODING (Fixes the PyTorch TypeError) ---
        # PyTorch cannot handle "male"/"female" strings.
        clinical_df['gender_bin'] = clinical_df['gender'].map({'female': 0, 'male': 1, 'Female': 0, 'Male': 1}).fillna(0)

        # SELECT AND MAP IDs
        # Use 'gender_bin' instead of 'gender'
        features = ['submitter_id', 'survival_days', 'event_occurred', 'age_years', 'gender_bin']
    
        # Drop rows where survival_days is NaN (if fallback wasn't used)
        clinical_processed = clinical_df[features].dropna(subset=['survival_days']).copy()
    
        logger.info(f"Clinical data shape: {clinical_processed.shape}")
        logger.info(f"Events: {clinical_processed['event_occurred'].sum()}/{len(clinical_processed)}")
    
        output_path = self.processed_dir / "clinical.csv"
        clinical_processed.to_csv(output_path, index=False)
    
        return clinical_processed
    
    def align_datasets(
        self,
        mutation_matrix: pd.DataFrame,
        expression_matrix: pd.DataFrame,
        clinical_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Align all datasets to common samples
        """
        
        logger.info("Aligning datasets...")
        
        # Get sample IDs from each dataset
        # Note: mutation_matrix uses Tumor_Sample_Barcode, expression uses submitter_id
        # Need to map between them

        # 1. Standardize Mutation IDs (Keep only the first 3 parts: Project-TSS-Participant)
        # Example: TARGET-40-0A1234-01A -> TARGET-40-0A1234
        # This is more robust for TARGET/TCGA barcodes
        mutation_matrix.index = mutation_matrix.index.map(lambda x: "-".join(x.split("-")[:3]))

        # 2. Handle duplicates (If a patient has two samples, we'll take the first one)
        mutation_matrix = mutation_matrix[~mutation_matrix.index.duplicated(keep='first')]

        mutation_samples = set(mutation_matrix.index)
        expression_samples = set(expression_matrix.index)
        clinical_samples = set(clinical_df['submitter_id'])
        
        # Find common samples (this is simplified; in practice need proper ID mapping)
        # For TARGET data, often submitter_id is in the barcode
        common_samples = mutation_samples.intersection(expression_samples).intersection(clinical_samples)
        
        logger.info(f"Common samples across all datasets: {len(common_samples)}")
        
        if len(common_samples) < 20:
            logger.warning("Very few common samples! Check ID mapping.")
        
        # Filter to common samples
        mutation_aligned = mutation_matrix.loc[list(common_samples)]
        expression_aligned = expression_matrix.loc[list(common_samples)]
        clinical_aligned = clinical_df[clinical_df['submitter_id'].isin(common_samples)]
        
        # Sort all by same order
        sample_order = sorted(common_samples)
        mutation_aligned = mutation_aligned.loc[sample_order]
        expression_aligned = expression_aligned.loc[sample_order]
        clinical_aligned = clinical_aligned.set_index('submitter_id').loc[sample_order].reset_index()
        
        # Save aligned data
        mutation_aligned.to_csv(self.processed_dir / "mutation_matrix_aligned.csv")
        expression_aligned.to_csv(self.processed_dir / "expression_matrix_aligned.csv")
        clinical_aligned.to_csv(self.processed_dir / "clinical_aligned.csv", index=False)
        
        logger.info(f"Final aligned dataset: {len(sample_order)} samples")
        
        return mutation_aligned, expression_aligned, clinical_aligned
    
    def process_all(self) -> Dict:
        """Run complete preprocessing pipeline"""
        
        logger.info("Starting data preprocessing pipeline...")
        
        # Process each data type
        mutation_matrix = self.process_mutations()
        expression_matrix = self.process_rna_seq()
        clinical_df = self.process_clinical()
        
        # Align datasets
        mutation_aligned, expression_aligned, clinical_aligned = self.align_datasets(
            mutation_matrix, expression_matrix, clinical_df
        )
        
        logger.info("Preprocessing complete!")
        
        return {
            'mutation_matrix': mutation_aligned,
            'expression_matrix': expression_aligned,
            'clinical': clinical_aligned
        }


if __name__ == "__main__":
    import yaml
    
    with open("../config/config.yaml") as f:
        config = yaml.safe_load(f)
    
    processor = OsteosarcomaDataProcessor(
        raw_dir=Path("./raw"),
        processed_dir=Path("./processed"),
        config=config
    )
    
    data = processor.process_all()
    print(f"Processed {len(data['mutation_matrix'])} samples")
