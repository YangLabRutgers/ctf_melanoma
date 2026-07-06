import os
import sys
import numpy as np
import anndata
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

# Ensure Python can see modules in the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_import import convert_anndata_to_pseudobulk, convert_pseudobulk_to_tensor
from predict import run_coupled_tpls_classification

def main():
    print("--- Starting Clean Multi-Rank Evaluation Pipeline ---")
    
    # 1. LOAD AND PREPROCESS DATA
    print("Loading datasets...")
    X_rna = anndata.read_h5ad("/mnt/yang_lab/ar3023/ctf_melanoma/data/scrna_non_tumor.h5ad")
    X_rna_small = X_rna[:, :50].copy()

    X_atac = anndata.read_h5ad("/mnt/yang_lab/ar3023/ctf_melanoma/data/scatac_non_tumor_gene_activities.h5ad")
    X_atac_small = X_atac[:, :50].copy()

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

    X_rna_small.obs['joint_sample_id'] = X_rna_small.obs['sample_ID_long'].map(rna_to_joint).astype(str)
    X_rna_small.obs['lesion_response'] = X_rna_small.obs['joint_sample_id'].map(sample_responses)

    X_atac_small.obs['joint_sample_id'] = X_atac_small.obs['orig.ident'].map(atac_to_joint).astype(str)
    X_atac_small.obs['lesion_response'] = X_atac_small.obs['joint_sample_id'].map(sample_responses)

    valid_samples = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12']
    X_rna_small = X_rna_small[X_rna_small.obs['joint_sample_id'].isin(valid_samples)].copy()
    X_atac_small = X_atac_small[X_atac_small.obs['joint_sample_id'].isin(valid_samples)].copy()

    shared_cell_types = set(X_atac_small.obs['reannotated_predicted_id']).intersection(set(X_rna_small.obs['cell_type']))
    X_atac_small = X_atac_small[X_atac_small.obs['reannotated_predicted_id'].isin(shared_cell_types)].copy()
    X_rna_small = X_rna_small[X_rna_small.obs['cell_type'].isin(shared_cell_types)].copy()

    pseudobulk_atac_df = convert_anndata_to_pseudobulk(adata=X_atac_small, sample_col="joint_sample_id", cell_type_col="reannotated_predicted_id", outcome_col="lesion_response")
    pseudobulk_rna_df = convert_anndata_to_pseudobulk(adata=X_rna_small, sample_col="joint_sample_id", cell_type_col="cell_type", outcome_col="lesion_response")

    tensor_atac, _ = convert_pseudobulk_to_tensor(pseudobulk_df=pseudobulk_atac_df, outcome_col="lesion_response")
    tensor_rna, labels_rna = convert_pseudobulk_to_tensor(pseudobulk_df=pseudobulk_rna_df, outcome_col="lesion_response")

    y_raw = labels_rna["lesion_response"]
    y = np.array([0 if v == "R" else 1 for v in y_raw], dtype=int)
    tensors = [tensor_atac, tensor_rna]

    # 2. AUTOMATED LEAVE-ONE-OUT CROSS-VALIDATION
    ranks = [1, 2, 3, 4, 5]
    accuracies = []
    r2x_scores = []

    for rank in ranks:
        print(f"\n--- Running Cross-Validation Evaluation (Rank {rank}) ---")
        
        # Call the classification function directly on the whole dataset
        # The function handles cross-validation internally
        models, final_acc, cv_predictions, r2x_val = run_coupled_tpls_classification(
            tensors=tensors, 
            labels=y, 
            rank=rank, 
            return_proba=False
        )
        
        # Save metrics for plotting
        accuracies.append(final_acc)
        r2x_scores.append(r2x_val)
        
        print(f"-> Rank {rank} Finished | CV Accuracy: {final_acc*100:.2f}% | R2X: {r2x_val:.4f}")

    # 3. COMBINE PLOTS INTO ONE PYTHON FILE (MULTI-PANEL FIGURE)
    print("\n--- Saving Combined Performance Plots ---")
    os.makedirs("../Plots", exist_ok=True)

    # Creating a single 1 row, 2 column figure layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left Panel: Accuracy vs Rank
    ax1.plot(ranks, accuracies, marker='o', color='crimson', linewidth=2)
    ax1.set_title("Cross-Validation Accuracy vs. Model Rank")
    ax1.set_xlabel("ctPLS Rank")
    ax1.set_ylabel("Accuracy Score")
    ax1.set_xticks(ranks)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Right Panel: R2X Variance Explained vs Rank
    ax2.plot(ranks, r2x_scores, marker='s', color='navy', linewidth=2)
    ax2.set_title("Variance Explained (R2X) vs. Model Rank")
    ax2.set_xlabel("ctPLS Rank")
    ax2.set_ylabel("R2X Metric Value")
    ax2.set_xticks(ranks)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig("../Plots/model_evaluation_metrics.png", dpi=300)
    plt.close()

    print("Pipeline complete! Unified visualization saved to 'Plots/model_evaluation_metrics.png'.")

if __name__ == "__main__":
    main()