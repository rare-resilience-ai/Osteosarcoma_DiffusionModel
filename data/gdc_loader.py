"""
Data Acquisition Module for TARGET-OS from GDC
Downloads mutation (MAF), RNA-seq, and clinical data
"""

import os
import json
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GDCDataLoader:
    """Download and organize TARGET-OS data from GDC"""
    
    GDC_API = "https://api.gdc.cancer.gov"
    
    def __init__(self, project_id="TARGET-OS", data_dir="./data"):
        self.project_id = project_id
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
    def query_files(self, data_category, data_type, workflow_type=None):
        """Query GDC for files matching criteria"""
        
        filters = {
            "op": "and",
            "content": [
                {"op": "in", "content": {"field": "cases.project.project_id", "value": [self.project_id]}},
                {"op": "in", "content": {"field": "files.data_category", "value": [data_category]}},
                {"op": "in", "content": {"field": "files.data_type", "value": [data_type]}}
            ]
        }
        
        if workflow_type:
            filters["content"].append({
                "op": "in",
                "content": {"field": "files.analysis.workflow_type", "value": [workflow_type]}
            })
        
        params = {
            "filters": json.dumps(filters),
            "fields": "file_id,file_name,cases.submitter_id,cases.case_id",
            "format": "JSON",
            "size": 1000
        }
        
        response = requests.post(f"{self.GDC_API}/files", json=params)
        return response.json()["data"]["hits"]
    
    def download_file(self, file_id, output_path):
        """Download a single file from GDC"""
        
        url = f"{self.GDC_API}/data/{file_id}"
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            total=total_size, unit='iB', unit_scale=True, desc=output_path.name
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)
    
    def download_mutations(self):
        """Download somatic mutation data (MAF files)"""
        
        logger.info("Querying mutation data...")
        files = self.query_files(
            data_category="Simple Nucleotide Variation",
            data_type="Masked Somatic Mutation",
            workflow_type="MuTect2 Variant Aggregation and Masking"
        )
        
        logger.info(f"Found {len(files)} mutation files")
        
        maf_dir = self.raw_dir / "mutations"
        maf_dir.mkdir(exist_ok=True)
        
        for file_info in files:
            file_id = file_info["file_id"]
            file_name = file_info["file_name"]
            output_path = maf_dir / file_name
            
            if not output_path.exists():
                logger.info(f"Downloading {file_name}...")
                self.download_file(file_id, output_path)
        
        return maf_dir
    
    def download_rna_seq(self):
        """Download RNA-seq expression data"""
        
        logger.info("Querying RNA-seq data...")
        files = self.query_files(
            data_category="Transcriptome Profiling",
            data_type="Gene Expression Quantification",
            workflow_type="STAR - Counts"
        )
        
        logger.info(f"Found {len(files)} RNA-seq files")
        
        rna_dir = self.raw_dir / "rna_seq"
        rna_dir.mkdir(exist_ok=True)
        
        # Store file metadata for later processing
        metadata = []
        
        for file_info in files:
            file_id = file_info["file_id"]
            file_name = file_info["file_name"]
            case_id = file_info["cases"][0]["case_id"] if file_info.get("cases") else None
            submitter_id = file_info["cases"][0]["submitter_id"] if file_info.get("cases") else None
            
            output_path = rna_dir / file_name
            
            metadata.append({
                "file_id": file_id,
                "file_name": file_name,
                "case_id": case_id,
                "submitter_id": submitter_id,
                "file_path": str(output_path)
            })
            
            if not output_path.exists():
                logger.info(f"Downloading {file_name}...")
                self.download_file(file_id, output_path)
        
        # Save metadata
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(rna_dir / "metadata.csv", index=False)
        
        return rna_dir
    
    def download_clinical(self):
        """Download clinical data"""
        
        logger.info("Querying clinical data...")
        
        # Query for clinical data
        filters = {
            "op": "in",
            "content": {
                "field": "cases.project.project_id",
                "value": [self.project_id]
            }
        }
        
        params = {
            "filters": json.dumps(filters),
            "expand": "diagnoses,demographic,exposures,follow_ups",
            "format": "JSON",
            "size": 1000
        }
        
        response = requests.get(f"{self.GDC_API}/cases", params=params)
        cases = response.json()["data"]["hits"]
        
        logger.info(f"Found {len(cases)} cases with clinical data")
        
        # Parse clinical data
        clinical_data = []
        
        for case in cases:
            case_id = case["case_id"]
            submitter_id = case["submitter_id"]
            
            # Demographic
            demographic = case.get("demographic", {})
            age = demographic.get("age_at_diagnosis")
            gender = demographic.get("gender")
            race = demographic.get("race")
            ethnicity = demographic.get("ethnicity")
            
            # Diagnosis
            diagnoses = case.get("diagnoses", [])
            if diagnoses:
                diag = diagnoses[0]
                tumor_stage = diag.get("tumor_stage")
                primary_diagnosis = diag.get("primary_diagnosis")
                site_of_resection = diag.get("site_of_resection_or_biopsy")
                morphology = diag.get("morphology")
            else:
                tumor_stage = primary_diagnosis = site_of_resection = morphology = None
            
            # Follow-up / Survival
            follow_ups = case.get("follow_ups", [])
            if follow_ups:
                fu = follow_ups[-1]  # Most recent
                days_to_death = fu.get("days_to_death")
                days_to_last_follow_up = fu.get("days_to_last_follow_up")
                vital_status = fu.get("vital_status")
            else:
                days_to_death = days_to_last_follow_up = vital_status = None
            
            clinical_data.append({
                "case_id": case_id,
                "submitter_id": submitter_id,
                "age_at_diagnosis": age,
                "gender": gender,
                "race": race,
                "ethnicity": ethnicity,
                "tumor_stage": tumor_stage,
                "primary_diagnosis": primary_diagnosis,
                "site_of_resection": site_of_resection,
                "morphology": morphology,
                "days_to_death": days_to_death,
                "days_to_last_follow_up": days_to_last_follow_up,
                "vital_status": vital_status
            })
        
        clinical_df = pd.DataFrame(clinical_data)
        clinical_path = self.raw_dir / "clinical.csv"
        clinical_df.to_csv(clinical_path, index=False)
        
        logger.info(f"Clinical data saved to {clinical_path}")
        return clinical_path
    
    def download_all(self):
        """Download all data types"""
        
        logger.info(f"Starting download for project {self.project_id}")
        
        results = {
            "mutations": self.download_mutations(),
            "rna_seq": self.download_rna_seq(),
            "clinical": self.download_clinical()
        }
        
        logger.info("Download complete!")
        return results


if __name__ == "__main__":
    loader = GDCDataLoader(project_id="TARGET-OS", data_dir="./data")
    results = loader.download_all()
    print(f"Data downloaded to: {results}")
