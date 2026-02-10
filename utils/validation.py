"""
Biological Validation Framework
Validates that synthetic patients respect biological constraints
"""

import torch
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import distance
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BiologicalValidator:
    """Validate synthetic patients against biological knowledge"""
    
    def __init__(self, config: dict):
        self.config = config
        self.driver_genes = config['evaluation']['driver_genes']
        self.mutually_exclusive_pairs = config['evaluation']['mutually_exclusive_pairs']
        self.required_correlations = config['evaluation']['required_correlations']
        
    def validate_mutation_cooccurrence(
        self,
        real_mutations: pd.DataFrame,
        synthetic_mutations: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Check if mutation co-occurrence patterns match real data
        
        Returns:
            metrics: Dictionary with validation scores
        """
        
        logger.info("Validating mutation co-occurrence patterns...")
        
        results = {}
        
        # 1. Mutation frequencies
        real_freq = real_mutations.mean(axis=0)
        synth_freq = synthetic_mutations.mean(axis=0)
        
        # Correlation of mutation frequencies
        common_genes = real_freq.index.intersection(synth_freq.index)
        freq_corr = np.corrcoef(
            real_freq[common_genes],
            synth_freq[common_genes]
        )[0, 1]
        
        results['mutation_frequency_correlation'] = freq_corr
        logger.info(f"Mutation frequency correlation: {freq_corr:.3f}")
        
        # 2. Driver gene mutation rates
        driver_genes_present = [g for g in self.driver_genes if g in real_mutations.columns]
        
        if driver_genes_present:
            real_driver_freq = real_mutations[driver_genes_present].mean(axis=0)
            synth_driver_freq = synthetic_mutations[driver_genes_present].mean(axis=0)
            
            driver_freq_diff = np.abs(real_driver_freq - synth_driver_freq).mean()
            results['driver_gene_frequency_diff'] = driver_freq_diff
            logger.info(f"Driver gene frequency difference: {driver_freq_diff:.3f}")
        
        # 3. Mutually exclusive pairs
        if self.mutually_exclusive_pairs:
            violations = 0
            total_pairs = 0
            
            for gene1, gene2 in self.mutually_exclusive_pairs:
                if gene1 in synthetic_mutations.columns and gene2 in synthetic_mutations.columns:
                    # Count samples with both mutations
                    both_mutated = (
                        (synthetic_mutations[gene1] == 1) &
                        (synthetic_mutations[gene2] == 1)
                    ).sum()
                    
                    violations += both_mutated
                    total_pairs += 1
            
            if total_pairs > 0:
                violation_rate = violations / (len(synthetic_mutations) * total_pairs)
                results['mutual_exclusivity_violation_rate'] = violation_rate
                logger.info(f"Mutual exclusivity violation rate: {violation_rate:.3f}")
        
        # 4. Pairwise mutation co-occurrence
        # Compute chi-square test for independence
        chi2_scores_real = []
        chi2_scores_synth = []
        
        sample_genes = np.random.choice(
            common_genes, size=min(50, len(common_genes)), replace=False
        )
        
        for i, gene1 in enumerate(sample_genes):
            for gene2 in sample_genes[i+1:]:
                # Real data
                contingency_real = pd.crosstab(
                    real_mutations[gene1],
                    real_mutations[gene2]
                )
                chi2_real, _ = stats.chi2_contingency(contingency_real)[:2]
                
                # Synthetic data
                contingency_synth = pd.crosstab(
                    synthetic_mutations[gene1],
                    synthetic_mutations[gene2]
                )
                chi2_synth, _ = stats.chi2_contingency(contingency_synth)[:2]
                
                chi2_scores_real.append(chi2_real)
                chi2_scores_synth.append(chi2_synth)
        
        # Correlation of chi-square scores
        if chi2_scores_real:
            chi2_corr = np.corrcoef(chi2_scores_real, chi2_scores_synth)[0, 1]
            results['cooccurrence_pattern_correlation'] = chi2_corr
            logger.info(f"Co-occurrence pattern correlation: {chi2_corr:.3f}")
        
        return results
    
    def validate_pathway_coherence(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        pathway_gene_matrix: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Validate that pathway activity is coherent
        (genes in same pathway should be correlated)
        """
        
        logger.info("Validating pathway coherence...")
        
        results = {}
        
        # For each pathway, compute within-pathway correlation
        real_coherence_scores = []
        synth_coherence_scores = []
        
        for pathway in pathway_gene_matrix.columns[:10]:  # Sample pathways
            pathway_genes = pathway_gene_matrix[pathway_gene_matrix[pathway] == 1].index
            pathway_genes = [g for g in pathway_genes if g in real_data.columns]
            
            if len(pathway_genes) < 3:
                continue
            
            # Real data: mean pairwise correlation
            real_corr_matrix = real_data[pathway_genes].corr()
            real_mean_corr = real_corr_matrix.values[np.triu_indices_from(real_corr_matrix.values, k=1)].mean()
            
            # Synthetic data
            synth_corr_matrix = synthetic_data[pathway_genes].corr()
            synth_mean_corr = synth_corr_matrix.values[np.triu_indices_from(synth_corr_matrix.values, k=1)].mean()
            
            real_coherence_scores.append(real_mean_corr)
            synth_coherence_scores.append(synth_mean_corr)
        
        if real_coherence_scores:
            # Average coherence
            results['real_pathway_coherence'] = np.mean(real_coherence_scores)
            results['synthetic_pathway_coherence'] = np.mean(synth_coherence_scores)
            
            # Correlation of coherence scores
            coherence_corr = np.corrcoef(real_coherence_scores, synth_coherence_scores)[0, 1]
            results['pathway_coherence_correlation'] = coherence_corr
            
            logger.info(f"Real pathway coherence: {results['real_pathway_coherence']:.3f}")
            logger.info(f"Synthetic pathway coherence: {results['synthetic_pathway_coherence']:.3f}")
            logger.info(f"Coherence correlation: {coherence_corr:.3f}")
        
        return results
    
    def validate_mutation_expression_correlation(
        self,
        mutations: pd.DataFrame,
        expression: pd.DataFrame,
        pathway_scores: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Validate known mutation-expression relationships
        E.g., TP53 mutation → p53 pathway downregulation
        """
        
        logger.info("Validating mutation-expression correlations...")
        
        results = {}
        violations = 0
        total_checks = 0
        
        for corr_rule in self.required_correlations:
            gene = corr_rule['mutation']
            pathway = corr_rule['pathway']
            expected_direction = corr_rule['direction']
            
            if gene not in mutations.columns or pathway not in pathway_scores.columns:
                continue
            
            # Compute correlation
            mut_status = mutations[gene]
            pathway_activity = pathway_scores[pathway]
            
            corr = mut_status.corr(pathway_activity)
            
            # Check if direction matches expectation
            if expected_direction == 'positive' and corr < 0:
                violations += 1
            elif expected_direction == 'negative' and corr > 0:
                violations += 1
            
            total_checks += 1
            
            logger.info(f"{gene} vs {pathway}: corr={corr:.3f} (expected: {expected_direction})")
        
        if total_checks > 0:
            violation_rate = violations / total_checks
            results['mutation_expression_violation_rate'] = violation_rate
            logger.info(f"Mutation-expression violation rate: {violation_rate:.3f}")
        
        return results
    
    def statistical_tests(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray
    ) -> Dict[str, float]:
        """
        Statistical tests for distribution matching
        """
        
        logger.info("Running statistical tests...")
        
        results = {}
        
        # 1. Kolmogorov-Smirnov test (per feature)
        ks_pvalues = []
        
        for i in range(min(real_data.shape[1], 100)):  # Sample features
            ks_stat, ks_pval = stats.ks_2samp(real_data[:, i], synthetic_data[:, i])
            ks_pvalues.append(ks_pval)
        
        results['ks_test_mean_pvalue'] = np.mean(ks_pvalues)
        results['ks_test_fraction_significant'] = (np.array(ks_pvalues) < 0.05).mean()
        
        logger.info(f"KS test mean p-value: {results['ks_test_mean_pvalue']:.3f}")
        logger.info(f"KS test fraction significant: {results['ks_test_fraction_significant']:.3f}")
        
        # 2. Maximum Mean Discrepancy (MMD)
        mmd = self.compute_mmd(real_data, synthetic_data)
        results['mmd'] = mmd
        logger.info(f"MMD: {mmd:.4f}")
        
        # 3. Wasserstein distance (on first 10 principal components)
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=10)
        real_pca = pca.fit_transform(real_data)
        synth_pca = pca.transform(synthetic_data)
        
        wasserstein_dists = []
        for i in range(10):
            wd = stats.wasserstein_distance(real_pca[:, i], synth_pca[:, i])
            wasserstein_dists.append(wd)
        
        results['wasserstein_distance_mean'] = np.mean(wasserstein_dists)
        logger.info(f"Mean Wasserstein distance: {results['wasserstein_distance_mean']:.3f}")
        
        return results
    
    def compute_mmd(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        kernel: str = 'rbf',
        gamma: float = None
    ) -> float:
        """
        Compute Maximum Mean Discrepancy between two distributions
        """
        
        if gamma is None:
            gamma = 1.0 / X.shape[1]
        
        # RBF kernel
        def rbf_kernel(X, Y, gamma):
            pairwise_sq_dists = distance.cdist(X, Y, 'sqeuclidean')
            return np.exp(-gamma * pairwise_sq_dists)
        
        XX = rbf_kernel(X, X, gamma).mean()
        YY = rbf_kernel(Y, Y, gamma).mean()
        XY = rbf_kernel(X, Y, gamma).mean()
        
        mmd = XX + YY - 2 * XY
        
        return np.sqrt(max(mmd, 0))
    
    def validate_all(
        self,
        real_mutations: pd.DataFrame,
        real_expression: pd.DataFrame,
        real_pathways: pd.DataFrame,
        synth_mutations: pd.DataFrame,
        synth_expression: pd.DataFrame,
        synth_pathways: pd.DataFrame,
        pathway_gene_matrix: pd.DataFrame = None
    ) -> Dict[str, float]:
        """
        Run all validation tests
        """
        
        logger.info("=" * 50)
        logger.info("BIOLOGICAL VALIDATION")
        logger.info("=" * 50)
        
        all_results = {}
        
        # Mutation co-occurrence
        mut_results = self.validate_mutation_cooccurrence(
            real_mutations, synth_mutations
        )
        all_results.update(mut_results)
        
        # Pathway coherence (if expression data available)
        if pathway_gene_matrix is not None:
            pathway_results = self.validate_pathway_coherence(
                real_expression, synth_expression, pathway_gene_matrix
            )
            all_results.update(pathway_results)
        
        # Mutation-expression correlation (if pathway scores available)
        mut_expr_results = self.validate_mutation_expression_correlation(
            synth_mutations, synth_expression, synth_pathways
        )
        all_results.update(mut_expr_results)
        
        # Statistical tests
        real_data_combined = np.concatenate([
            real_mutations.values,
            real_expression.values,
            real_pathways.values
        ], axis=1)
        
        synth_data_combined = np.concatenate([
            synth_mutations.values,
            synth_expression.values,
            synth_pathways.values
        ], axis=1)
        
        stat_results = self.statistical_tests(
            real_data_combined, synth_data_combined
        )
        all_results.update(stat_results)
        
        # Summary
        logger.info("=" * 50)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 50)
        
        for key, value in all_results.items():
            logger.info(f"{key}: {value:.4f}")
        
        # Overall score (simple average of normalized metrics)
        # Higher is better for correlations, lower is better for violations/distances
        
        score_components = []
        
        if 'mutation_frequency_correlation' in all_results:
            score_components.append(all_results['mutation_frequency_correlation'])
        
        if 'cooccurrence_pattern_correlation' in all_results:
            score_components.append(all_results['cooccurrence_pattern_correlation'])
        
        if 'mutual_exclusivity_violation_rate' in all_results:
            score_components.append(1 - all_results['mutual_exclusivity_violation_rate'])
        
        if 'mutation_expression_violation_rate' in all_results:
            score_components.append(1 - all_results['mutation_expression_violation_rate'])
        
        if score_components:
            overall_score = np.mean(score_components)
            all_results['overall_biological_score'] = overall_score
            logger.info(f"\nOverall Biological Score: {overall_score:.3f}")
        
        return all_results


if __name__ == "__main__":
    # Example usage
    import yaml
    
    with open("../config/config.yaml") as f:
        config = yaml.safe_load(f)
    
    validator = BiologicalValidator(config)
    
    # Load real and synthetic data
    # real_mutations = pd.read_csv("data/processed/mutation_matrix_aligned.csv", index_col=0)
    # synth_mutations = pd.read_csv("results/synthetic/mutations.csv", index_col=0)
    # ...
    
    # results = validator.validate_all(...)
    # print(results)
