import os
import sys
import pandas as pd
import numpy as np
import scanpy as sc
import anndata
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

# Ensure Python can see modules in the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_import import convert_anndata_to_pseudobulk, convert_pseudobulk_to_tensor
from .predict import run_coupled_tpls_classification

def evaluate_pipeline(X_rna_in, X_atac_in, num_genes, num_peaks, eval_rank=5):
    """Helper function to run the pipeline steps and return cross-validation accuracy."""
    X_rna_temp = X_rna_in.copy()
    X_atac_temp = X_atac_in.copy()
    
    # Filter out low-expression features (<3 cells)
    sc.pp.filter_genes(X_rna_temp, min_cells=3)
    sc.pp.filter_genes(X_atac_temp, min_cells=3)

    # Identify highly variable features (seurat flavor avoids bin crashes)
    try:
        sc.pp.highly_variable_genes(X_rna_temp, n_top_genes=min(num_genes, X_rna_temp.n_vars), flavor="seurat", inplace=True)
    except Exception:
        sc.pp.highly_variable_genes(X_rna_temp, n_top_genes=min(num_genes, X_rna_temp.n_vars), flavor="cell_ranger", n_bins=5, inplace=True)
    X_rna_small = X_rna_temp[:, X_rna_temp.var["highly_variable"]].copy()

    try:
        sc.pp.highly_variable_genes(X_atac_temp, n_top_genes=min(num_peaks, X_atac_temp.n_vars), flavor="seurat", inplace=True)
    except Exception:
        sc.pp.highly_variable_genes(X_atac_temp, n_top_genes=min(num_peaks, X_atac_temp.n_vars), flavor="cell_ranger", n_bins=5, inplace=True)
    X_atac_small = X_atac_temp[:, X_atac_temp.var["highly_variable"]].copy()

    # Shared Cell Types Intersection
    shared_cell_types = set(X_atac_small.obs['reannotated_predicted_id']).intersection(set(X_rna_small.obs['cell_type']))
    if not shared_cell_types:
        return 0.0
    X_atac_small = X_atac_small[X_atac_small.obs['reannotated_predicted_id'].isin(shared_cell_types)].copy()
    X_rna_small = X_rna_small[X_rna_small.obs['cell_type'].isin(shared_cell_types)].copy()

    # Pseudobulk Conversion
    pseudobulk_atac_df = convert_anndata_to_pseudobulk(adata=X_atac_small, sample_col="joint_sample_id", cell_type_col="reannotated_predicted_id", outcome_col="lesion_response")
    pseudobulk_rna_df = convert_anndata_to_pseudobulk(adata=X_rna_small, sample_col="joint_sample_id", cell_type_col="cell_type", outcome_col="lesion_response")

    # Tensor Conversion
    tensor_atac, _ = convert_pseudobulk_to_tensor(pseudobulk_df=pseudobulk_atac_df, outcome_col="lesion_response")
    tensor_rna, labels_rna = convert_pseudobulk_to_tensor(pseudobulk_df=pseudobulk_rna_df, outcome_col="lesion_response")
    if 0 in tensor_atac.shape or 0 in tensor_rna.shape:
        return 0.0

    y_raw = labels_rna["lesion_response"]
    y = np.array([0 if v == "R" else 1 for v in y_raw], dtype=int)
    tensors = [tensor_atac, tensor_rna]

    # Run standard model evaluation at locked rank 5
    try:
        _, final_acc, _, _ = run_coupled_tpls_classification(
        tensors=tensors, 
        labels=y, 
        rank=eval_rank, 
        return_proba=False
    )
        return final_acc
    except Exception as e:
        print(f"Warning: Classification failed for this parameter set ({e}). Defaulting accuracy to 0.0")
    return 0.0

def main():
    print("--- Starting Automated Multi-Environment Stress-Testing Pipeline ---")
    
    # LOAD BASE DATASETS
    print("Loading raw base datasets...")
    X_rna_base = anndata.read_h5ad("/mnt/yang_lab/ar3023/ctf_melanoma/data/scrna_non_tumor.h5ad")
    X_atac_base = anndata.read_h5ad("/mnt/yang_lab/ar3023/ctf_melanoma/data/scatac_non_tumor_gene_activities.h5ad")

    rna_to_joint = {
        'D19-11960': 'S1',  'D19-11966': 'S2',  'D19-11971': 'S3',
        'D19-11989': 'S4',  'D19-11990': 'S5',  'D19-11994': 'S6',
        'D19-11995': 'S7',  'D19-11997': 'S8',  'D19-11999': 'S9',
        'D19-9123':  'S10', 'D21-194004': 'S11', 'D21-194011': 'S12'
    }
    atac_to_joint = {
        'D19-11983': 'S1', 'D19-11984': 'S1', 'D19-11985': 'S2', 'D19-11988': 'S3',
        'D19-11974': 'S4', 'D19-11975': 'S5', 'D19-11973': 'S6', 'D19-11972': 'S7',
        'D19-11980': 'S8', 'D19-11979': 'S9', 'D19-9146':  'S10', 'D21-194103': 'S11',
        'D21-194106': 'S12', 'D21-194107': 'S12'
    }
    sample_responses = {
        'S1': 'NR', 'S2': 'R',  'S3': 'R', 'S4': 'NR', 'S5': 'R', 'S6': 'R',
        'S7': 'NR', 'S8': 'NR', 'S9': 'R', 'S10': 'NR', 'S11': 'R', 'S12': 'R'
    }

    X_rna_base.obs['joint_sample_id'] = X_rna_base.obs['sample_ID_long'].map(rna_to_joint).astype(str)
    X_rna_base.obs['lesion_response'] = X_rna_base.obs['joint_sample_id'].map(sample_responses)

    X_atac_base.obs['joint_sample_id'] = X_atac_base.obs['orig.ident'].map(atac_to_joint).astype(str)
    X_atac_base.obs['lesion_response'] = X_atac_base.obs['joint_sample_id'].map(sample_responses)

    valid_samples = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12']
    X_rna_base = X_rna_base[X_rna_base.obs['joint_sample_id'].isin(valid_samples)].copy()
    X_atac_base = X_atac_base[X_atac_base.obs['joint_sample_id'].isin(valid_samples)].copy()


    # Settings for Grid Search
    columns = [50000, 10000, 2000, 1000]
    feature_counts = [50000, 10000, 2000, 1000]

    feature_environments = [
        "4. Reduce Genes (RNA Only)",
        "5. Reduce Peaks (ATAC Only)",
        "6. Reduce Features (Both RNA & ATAC)"
    ]

    # Initialize Tracking Dataframe
    results_df = pd.DataFrame(index=feature_environments, columns=columns)
    
    # Baseline fallback parameters for unreduced modality 
    base_features = 2000


    for feat_num, col_name in zip(feature_counts, columns):
        print(f"\n--- Running Sweep for Data Scale Step: {col_name} features ---")

        # --- ENV 4: Scale RNA Features Only ---
        results_df.loc["4. Reduce Genes (RNA Only)", col_name] = evaluate_pipeline(X_rna_base, X_atac_base, num_genes=feat_num, num_peaks=base_features)

        # --- ENV 5: Scale ATAC Features Only ---
        results_df.loc["5. Reduce Peaks (ATAC Only)", col_name] = evaluate_pipeline(X_rna_base, X_atac_base, num_genes=base_features, num_peaks=feat_num)

        # --- ENV 6: Scale BOTH Features Simultaneously ---
        results_df.loc["6. Reduce Features (Both RNA & ATAC)", col_name] = evaluate_pipeline(X_rna_base, X_atac_base, num_genes=feat_num, num_peaks=feat_num)

    # Print Final Summary Matrix to Terminal
    print("\n======================= FINAL ACCURACY MATRIX =======================")
    print(results_df.to_string())
    print("=====================================================================")

    # Convert accuracy fractions (e.g., 1.0, 0.83) into percentages (100.0, 83.0)
    plot_df = results_df.astype(float) * 100

    # ----------------------------------------------------
    # PLOT: Feature Reduction Across Explicit Counts 
    # ----------------------------------------------------
    plt.figure(figsize=(7, 5), dpi=300)
    
    colors_feature = ['#d62728', '#9467bd', '#2ca02c']

    # Plot each feature reduction line
    for env, color in zip(feature_environments, colors_feature):
        plt.plot(feature_counts, plot_df.loc[env], marker='s', linestyle='--', linewidth=2.5, label=env, color=color)

    plt.title("Effect of Feature Reduction on CV Accuracy", fontsize=12, weight='bold', pad=12)
    plt.xlabel("Number of Retained Features", fontsize=10, weight='bold')
    plt.ylabel("Leave-One-Out CV Accuracy (%)", fontsize=10, weight='bold')
    plt.ylim(-5, 105)
    plt.xscale('log') #Uses logarithmic scale for x-axis to better visualize the range of feature counts
    plt.gca().invert_xaxis()  # Inverts axis so it goes 50000 -> 1000 (left to right)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower left', fontsize=9)
    plt.tight_layout()

    # Save to dedicated cell plot file
    plt.savefig("Plots/Feature_Reduction_Plot.png", bbox_inches='tight', dpi=300)
    plt.close()

    print ("Pipeline Complete! Saved 'Plots/Feature_Reduction_Plot.png'.")

if __name__ == "__main__":
    main()