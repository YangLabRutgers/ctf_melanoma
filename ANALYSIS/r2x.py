import os
import sys
import pandas as pd
import numpy as np
import scanpy as sc
import anndata
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

# Ensure Python can see modules in the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_import import convert_anndata_to_pseudobulk, convert_pseudobulk_to_tensor
from .predict import run_coupled_tpls_classification

def evaluate_pipeline(X_rna_in, X_atac_in, num_genes, num_peaks, eval_rank=5):
    """Helper function to run the pipeline steps and return cross-validation accuracy."""
    X_rna_temp = X_rna_in.copy()
    X_atac_temp = X_atac_in.copy()
    
    # Identify highly variable features dynamically
    sc.pp.highly_variable_genes(X_rna_temp, n_top_genes=num_genes, flavor="seurat_v3", inplace=True)
    X_rna_small = X_rna_temp[:, X_rna_temp.var["highly_variable"]].copy()

    sc.pp.highly_variable_genes(X_atac_temp, n_top_genes=num_peaks, flavor="seurat_v3", inplace=True)
    X_atac_small = X_atac_temp[:, X_atac_temp.var["highly_variable"]].copy()

    # Shared Cell Types Intersection
    shared_cell_types = set(X_atac_small.obs['reannotated_predicted_id']).intersection(set(X_rna_small.obs['cell_type']))
    X_atac_small = X_atac_small[X_atac_small.obs['reannotated_predicted_id'].isin(shared_cell_types)].copy()
    X_rna_small = X_rna_small[X_rna_small.obs['cell_type'].isin(shared_cell_types)].copy()

    # Pseudobulk Conversion
    pseudobulk_atac_df = convert_anndata_to_pseudobulk(adata=X_atac_small, sample_col="joint_sample_id", cell_type_col="reannotated_predicted_id", outcome_col="lesion_response")
    pseudobulk_rna_df = convert_anndata_to_pseudobulk(adata=X_rna_small, sample_col="joint_sample_id", cell_type_col="cell_type", outcome_col="lesion_response")

    # Tensor Conversion
    tensor_atac, _ = convert_pseudobulk_to_tensor(pseudobulk_df=pseudobulk_atac_df, outcome_col="lesion_response")
    tensor_rna, labels_rna = convert_pseudobulk_to_tensor(pseudobulk_df=pseudobulk_rna_df, outcome_col="lesion_response")

    y_raw = labels_rna["lesion_response"]
    y = np.array([0 if v == "R" else 1 for v in y_raw], dtype=int)
    tensors = [tensor_atac, tensor_rna]

    # Run standard model evaluation at locked rank 5
    _, final_acc, _, _ = run_coupled_tpls_classification(
        tensors=tensors, 
        labels=y, 
        rank=eval_rank, 
        return_proba=False
    )
    return final_acc

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
    fractions = [1.0, 0.75, 0.50, 0.25, 0.05]
    columns = ["100%", "75%", "50%", "25%", "5%"]
    environments = [
        "1. Remove Cells (Both RNA & ATAC)",
        "2. Remove Cells (RNA Only)",
        "3. Remove Cells (ATAC Only)",
        "4. Reduce Genes (RNA Only)",
        "5. Reduce Peaks (ATAC Only)"
    ]

    # Initialize Tracking Dataframe
    results_df = pd.DataFrame(index=environments, columns=columns)
    
    # Baseline Sweet-spot parameters (2,000 features)
    base_genes = 2000
    base_peaks = 2000

    for frac, col_name in zip(fractions, columns):
        print(f"\n--- Running Sweep for Data Scale Step: {col_name} ---")
        
        # --- ENV 1: Downsample Cells for Both Modalities ---
        rna_env1 = sc.pp.subsample(X_rna_base, fraction=frac, copy=True) if frac < 1.0 else X_rna_base.copy()
        atac_env1 = sc.pp.subsample(X_atac_base, fraction=frac, copy=True) if frac < 1.0 else X_atac_base.copy()
        results_df.loc["1. Remove Cells (Both RNA & ATAC)", col_name] = evaluate_pipeline(rna_env1, atac_env1, base_genes, base_peaks)

        # --- ENV 2: Downsample Cells for RNA Only ---
        rna_env2 = sc.pp.subsample(X_rna_base, fraction=frac, copy=True) if frac < 1.0 else X_rna_base.copy()
        results_df.loc["2. Remove Cells (RNA Only)", col_name] = evaluate_pipeline(rna_env2, X_atac_base, base_genes, base_peaks)

        # --- ENV 3: Downsample Cells for ATAC Only ---
        atac_env3 = sc.pp.subsample(X_atac_base, fraction=frac, copy=True) if frac < 1.0 else X_atac_base.copy()
        results_df.loc["3. Remove Cells (ATAC Only)", col_name] = evaluate_pipeline(X_rna_base, atac_env3, base_genes, base_peaks)

        # --- ENV 4: Scale RNA Features Only ---
        current_genes = int(base_genes * frac) if int(base_genes * frac) >= 10 else 10
        results_df.loc["4. Reduce Genes (RNA Only)", col_name] = evaluate_pipeline(X_rna_base, X_atac_base, current_genes, base_peaks)

        # --- ENV 5: Scale ATAC Features Only ---
        current_peaks = int(base_peaks * frac) if int(base_peaks * frac) >= 10 else 10
        results_df.loc["5. Reduce Peaks (ATAC Only)", col_name] = evaluate_pipeline(X_rna_base, X_atac_base, base_genes, current_peaks)

    # Print Final Summary Matrix to Terminal
    print("\n======================= FINAL ACCURACY MATRIX =======================")
    print(results_df.to_string())
    print("=====================================================================")

    # Automatically Generate and Save Table Matrix Image
    fig, ax = plt.subplots(figsize=(10, 3.5), dpi=300)
    ax.axis('off')

    formatted_data = results_df.copy()
    for col in formatted_data.columns:
        formatted_data[col] = formatted_data[col].apply(lambda x: f"{x*100:.1f}%" if x in [0.666667, 0.583333, 0.416667] else f"{x*100:.0f}%")

    table = ax.table(
        cellText=formatted_data.values,
        rowLabels=formatted_data.index,
        colLabels=formatted_data.columns,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2c3e50')
        elif col == -1:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#ecf0f1')
        else:
            val = formatted_data.iloc[row-1, col]
            if any(drop in val for drop in ['41.6%', '58.3%', '66.7%']):
                cell.set_facecolor('#fadbd8')
            else:
                cell.set_facecolor('#f8f9f9')

    plt.title("Final Accuracy Matrix Across Stress Environments", fontsize=12, weight='bold', pad=20, color='#2c3e50')
    plt.savefig("Plots/final_accuracy_matrix_table.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    print("Pipeline complete! Visualizations saved to 'Plots/environment_stress_test_metrics.png' and 'Plots/final_accuracy_matrix_table.png'.")

if __name__ == "__main__":
    main()