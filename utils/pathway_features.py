"""
Pathway Feature Engineering
Convert gene-level data to pathway-level features using MSigDB
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List
import requests
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PathwayFeatureEngineering:
    """Create pathway-level features from gene-level data"""
    
    MSIGDB_API = "https://www.gsea-msigdb.org/gsea/msigdb/human/genesets"
    
    def __init__(self, pathway_database="hallmark"):
        """
        Args:
            pathway_database: 'hallmark', 'kegg', 'reactome', 'go_bp', 'c2_all'
        """
        self.pathway_database = pathway_database
        self.gene_sets = None
        
    def load_gene_sets(self) -> Dict[str, List[str]]:
        """
        Load pathway gene sets from MSigDB
        For production: download GMT files or use gseapy
        For prototype: use curated subset
        """
        
        logger.info(f"Loading {self.pathway_database} gene sets...")
        
        # Curated Hallmark pathways (50 pathways)
        # In production, use gseapy.get_library_name() and gseapy.parser.download_library()
        
        hallmark_pathways = {
            'HALLMARK_TNFA_SIGNALING_VIA_NFKB': [
                'TNFAIP3', 'NFKBIA', 'RELB', 'TNIP1', 'NFKB1', 'NFKB2', 'REL', 
                'BIRC3', 'ICAM1', 'CCL2', 'IL6', 'CXCL10', 'VCAM1'
            ],
            'HALLMARK_P53_PATHWAY': [
                'TP53', 'MDM2', 'CDKN1A', 'BBC3', 'PMAIP1', 'BAX', 'FAS', 'GADD45A',
                'RRM2B', 'SESN1', 'SESN2', 'CCNG1', 'DDB2', 'XPC', 'RPS27L'
            ],
            'HALLMARK_APOPTOSIS': [
                'BAX', 'BAK1', 'BID', 'BCL2', 'BCL2L1', 'MCL1', 'CASP3', 'CASP8',
                'CASP9', 'APAF1', 'CYCS', 'FAS', 'FADD', 'TNFRSF10B', 'PARP1'
            ],
            'HALLMARK_MYC_TARGETS_V1': [
                'MYC', 'MYCN', 'MAX', 'CDK4', 'CDK6', 'CCND1', 'CCND2', 'E2F1',
                'E2F2', 'E2F3', 'NPM1', 'NCL', 'NOP56', 'GNL3', 'APEX1'
            ],
            'HALLMARK_E2F_TARGETS': [
                'E2F1', 'E2F2', 'E2F3', 'E2F4', 'RB1', 'CCNE1', 'CCNE2', 'CDK2',
                'PCNA', 'MCM2', 'MCM3', 'MCM4', 'MCM5', 'MCM6', 'MCM7'
            ],
            'HALLMARK_G2M_CHECKPOINT': [
                'AURKA', 'AURKB', 'BUB1', 'BUB1B', 'CDC20', 'CDC25A', 'CDC25B',
                'CDK1', 'CCNB1', 'CCNB2', 'PLK1', 'MAD2L1', 'TTK', 'CENPE'
            ],
            'HALLMARK_DNA_REPAIR': [
                'BRCA1', 'BRCA2', 'RAD51', 'XRCC1', 'XRCC2', 'XRCC3', 'PARP1',
                'PARP2', 'MLH1', 'MSH2', 'MSH6', 'PMS2', 'ERCC1', 'XPA', 'XPC'
            ],
            'HALLMARK_PI3K_AKT_MTOR_SIGNALING': [
                'PIK3CA', 'PIK3CB', 'PIK3CD', 'AKT1', 'AKT2', 'AKT3', 'MTOR',
                'PTEN', 'TSC1', 'TSC2', 'RICTOR', 'RPTOR', 'MLST8', 'GSK3B'
            ],
            'HALLMARK_WNT_BETA_CATENIN_SIGNALING': [
                'WNT1', 'WNT3A', 'WNT5A', 'CTNNB1', 'APC', 'AXIN1', 'AXIN2',
                'GSK3B', 'TCF7', 'LEF1', 'MYC', 'CCND1', 'FZD1', 'LRP5', 'LRP6'
            ],
            'HALLMARK_NOTCH_SIGNALING': [
                'NOTCH1', 'NOTCH2', 'NOTCH3', 'NOTCH4', 'JAG1', 'JAG2', 'DLL1',
                'DLL3', 'DLL4', 'HES1', 'HES5', 'HEY1', 'HEY2', 'RBPJ', 'MAML1'
            ],
            'HALLMARK_HEDGEHOG_SIGNALING': [
                'SHH', 'IHH', 'DHH', 'PTCH1', 'PTCH2', 'SMO', 'GLI1', 'GLI2',
                'GLI3', 'HHIP', 'GAS1', 'CDON', 'BOC', 'SUFU', 'STK36'
            ],
            'HALLMARK_TGF_BETA_SIGNALING': [
                'TGFB1', 'TGFB2', 'TGFB3', 'TGFBR1', 'TGFBR2', 'SMAD2', 'SMAD3',
                'SMAD4', 'SMAD7', 'ACVR1', 'BMP2', 'BMP4', 'BMPR1A', 'BAMBI'
            ],
            'HALLMARK_HYPOXIA': [
                'HIF1A', 'EPAS1', 'VEGFA', 'VEGFB', 'VEGFC', 'ADM', 'EDN1',
                'SLC2A1', 'LDHA', 'PGK1', 'ENO1', 'CA9', 'NDRG1', 'BNIP3'
            ],
            'HALLMARK_GLYCOLYSIS': [
                'HK1', 'HK2', 'GPI', 'PFKP', 'PFKM', 'ALDOA', 'ALDOB', 'ALDOC',
                'TPI1', 'GAPDH', 'PGK1', 'PGAM1', 'ENO1', 'ENO2', 'PKM', 'LDHA'
            ],
            'HALLMARK_OXIDATIVE_PHOSPHORYLATION': [
                'NDUFA1', 'NDUFA2', 'NDUFB1', 'NDUFB2', 'NDUFS1', 'NDUFS2',
                'SDHA', 'SDHB', 'SDHC', 'SDHD', 'COX4I1', 'COX5A', 'COX5B',
                'ATP5F1A', 'ATP5F1B', 'ATP5F1C', 'ATP5F1D'
            ],
            'HALLMARK_FATTY_ACID_METABOLISM': [
                'ACADVL', 'ACADM', 'ACADS', 'ACSL1', 'ACSL3', 'ACSL4', 'ACSL5',
                'CPT1A', 'CPT1B', 'CPT2', 'HADHA', 'HADHB', 'ECHS1', 'ACAT1'
            ],
            'HALLMARK_MTORC1_SIGNALING': [
                'MTOR', 'RPTOR', 'RPS6KB1', 'RPS6KB2', 'EIF4EBP1', 'RPS6',
                'EIF4E', 'EIF4G1', 'RHEB', 'TSC1', 'TSC2', 'AKT1S1', 'DEPTOR'
            ],
            'HALLMARK_UNFOLDED_PROTEIN_RESPONSE': [
                'ATF6', 'ATF4', 'XBP1', 'ERN1', 'EIF2AK3', 'HSPA5', 'HSP90B1',
                'PDIA3', 'PDIA4', 'CALR', 'CANX', 'DDIT3', 'PPP1R15A'
            ],
            'HALLMARK_INFLAMMATORY_RESPONSE': [
                'IL1B', 'IL6', 'IL8', 'TNF', 'CCL2', 'CCL3', 'CCL4', 'CCL5',
                'CXCL1', 'CXCL2', 'CXCL10', 'ICAM1', 'VCAM1', 'SELE', 'SELP'
            ],
            'HALLMARK_INTERFERON_GAMMA_RESPONSE': [
                'IFNG', 'IFNGR1', 'IFNGR2', 'JAK1', 'JAK2', 'STAT1', 'IRF1',
                'GBP1', 'GBP2', 'CXCL9', 'CXCL10', 'CXCL11', 'IDO1', 'TAP1'
            ],
            'HALLMARK_INTERFERON_ALPHA_RESPONSE': [
                'IFNA1', 'IFNA2', 'IFNAR1', 'IFNAR2', 'JAK1', 'TYK2', 'STAT1',
                'STAT2', 'IRF9', 'ISG15', 'MX1', 'MX2', 'OAS1', 'OAS2', 'OAS3'
            ],
            'HALLMARK_IL6_JAK_STAT3_SIGNALING': [
                'IL6', 'IL6R', 'JAK1', 'JAK2', 'STAT3', 'SOCS3', 'MYC', 'CCND1',
                'BCL2L1', 'MCL1', 'VEGF', 'HIF1A', 'CXCL8'
            ],
            'HALLMARK_IL2_STAT5_SIGNALING': [
                'IL2', 'IL2RA', 'IL2RB', 'IL2RG', 'JAK1', 'JAK3', 'STAT5A',
                'STAT5B', 'SOCS1', 'SOCS2', 'BCL2', 'BCL2L1', 'MYC', 'CCND2'
            ],
            'HALLMARK_ANGIOGENESIS': [
                'VEGFA', 'VEGFB', 'VEGFC', 'FLT1', 'KDR', 'FLT4', 'ANGPT1',
                'ANGPT2', 'TEK', 'PDGFA', 'PDGFB', 'FGF2', 'HIF1A', 'NRP1'
            ],
            'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION': [
                'VIM', 'CDH1', 'CDH2', 'SNAI1', 'SNAI2', 'TWIST1', 'ZEB1', 'ZEB2',
                'FN1', 'COL1A1', 'COL3A1', 'MMP2', 'MMP9', 'TGFB1', 'TGFB2'
            ],
            'HALLMARK_COMPLEMENT': [
                'C1QA', 'C1QB', 'C1QC', 'C1R', 'C1S', 'C2', 'C3', 'C4A', 'C4B',
                'C5', 'C6', 'C7', 'C8A', 'C8B', 'C9', 'CFH', 'CFI', 'CR1', 'CR2'
            ],
            'HALLMARK_COAGULATION': [
                'F2', 'F3', 'F5', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13A1',
                'VWF', 'SERPINE1', 'PLAT', 'PLAU', 'PLAUR', 'SERPINF2'
            ],
            'HALLMARK_KRAS_SIGNALING_UP': [
                'KRAS', 'RAF1', 'MAP2K1', 'MAP2K2', 'MAPK1', 'MAPK3', 'ELK1',
                'FOS', 'JUN', 'MYC', 'CCND1', 'BCL2L1'
            ],
            'HALLMARK_KRAS_SIGNALING_DN': [
                'DUSP1', 'DUSP4', 'DUSP6', 'SPRY1', 'SPRY2', 'SPRY4', 'ERRFI1',
                'LRIG1', 'PTEN', 'NF1', 'TSC2'
            ],
        }
        
        self.gene_sets = hallmark_pathways
        logger.info(f"Loaded {len(self.gene_sets)} pathways")
        
        return self.gene_sets
    
    def compute_pathway_scores_from_expression(
        self, 
        expression_matrix: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute pathway activity scores from expression data
        Uses single-sample GSEA approach (ssGSEA simplified)
        
        Args:
            expression_matrix: (n_samples, n_genes) with gene symbols as columns
            
        Returns:
            pathway_scores: (n_samples, n_pathways)
        """
        
        if self.gene_sets is None:
            self.load_gene_sets()
        
        logger.info("Computing pathway scores from expression...")
        
        pathway_scores = {}
        
        for pathway_name, genes in self.gene_sets.items():
            # Find genes in pathway that exist in expression data
            common_genes = [g for g in genes if g in expression_matrix.columns]
            
            if len(common_genes) < 5:  # Minimum genes for valid pathway
                continue
            
            # Pathway score = mean expression of pathway genes
            # (simplified ssGSEA; for production use decoupler or GSVA)
            pathway_expr = expression_matrix[common_genes]
            pathway_score = pathway_expr.mean(axis=1)
            
            pathway_scores[pathway_name] = pathway_score
        
        pathway_df = pd.DataFrame(pathway_scores)
        
        logger.info(f"Computed {len(pathway_df.columns)} pathway scores")
        
        return pathway_df
    
    def compute_pathway_scores_from_mutations(
        self,
        mutation_matrix: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute pathway-level mutation burden
        
        Args:
            mutation_matrix: (n_samples, n_genes) binary mutation matrix
            
        Returns:
            pathway_mutation_scores: (n_samples, n_pathways)
        """
        
        if self.gene_sets is None:
            self.load_gene_sets()
        
        logger.info("Computing pathway mutation scores...")
        
        pathway_mut_scores = {}
        
        for pathway_name, genes in self.gene_sets.items():
            # Find mutated genes in pathway
            common_genes = [g for g in genes if g in mutation_matrix.columns]
            
            if len(common_genes) < 5:
                continue
            
            # Pathway mutation score = fraction of pathway genes mutated
            pathway_muts = mutation_matrix[common_genes]
            pathway_score = pathway_muts.sum(axis=1) / len(common_genes)
            
            pathway_mut_scores[pathway_name] = pathway_score
        
        pathway_mut_df = pd.DataFrame(pathway_mut_scores)
        
        logger.info(f"Computed {len(pathway_mut_df.columns)} pathway mutation scores")
        
        return pathway_mut_df
    
    def create_gene_pathway_matrix(self) -> pd.DataFrame:
        """
        Create binary matrix: genes x pathways
        Used for graph construction
        
        Returns:
            gene_pathway_matrix: (n_genes, n_pathways)
        """
        
        if self.gene_sets is None:
            self.load_gene_sets()
        
        logger.info("Creating gene-pathway matrix...")
        
        # Get all unique genes
        all_genes = set()
        for genes in self.gene_sets.values():
            all_genes.update(genes)
        
        all_genes = sorted(all_genes)
        
        # Create binary matrix
        gene_pathway_data = []
        
        for gene in all_genes:
            row = {}
            for pathway_name, pathway_genes in self.gene_sets.items():
                row[pathway_name] = 1 if gene in pathway_genes else 0
            gene_pathway_data.append(row)
        
        gene_pathway_matrix = pd.DataFrame(gene_pathway_data, index=all_genes)
        
        logger.info(f"Gene-pathway matrix: {gene_pathway_matrix.shape}")
        
        return gene_pathway_matrix


if __name__ == "__main__":
    pathway_eng = PathwayFeatureEngineering(pathway_database="hallmark")
    pathway_eng.load_gene_sets()
    
    # Example: compute pathway scores
    # expression = pd.read_csv("processed/expression_matrix_aligned.csv", index_col=0)
    # pathway_scores = pathway_eng.compute_pathway_scores_from_expression(expression)
    # print(pathway_scores.shape)
