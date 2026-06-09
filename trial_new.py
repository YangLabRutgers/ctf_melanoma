from data_import import convert_anndata_to_pseudobulk, convert_pseudobulk_to_tensor
import anndata
import numpy as np
from predict import run_coupled_tpls_classification

def main():
    X_atac = anndata.read_h5ad("/mnt/yang_lab/ar3023/ctf_melanoma/data/scatac_non_tumor_gene_activities.h5ad")
    X_rna = anndata.read_h5ad("/mnt/yang_lab/ar3023/ctf_melanoma/data/scrna_non_tumor.h5ad")
    
    print("--- Raw Input Objects ---")
    print(X_atac)
    print(X_rna)

    # 1. Map the 12 explicit RNA sample strings to simplified joint sample IDs (S1-S12)
    rna_to_joint = {
        'D19-11960': 'S1',  'D19-11966': 'S2',  'D19-11971': 'S3',
        'D19-11989': 'S4',  'D19-11990': 'S5',  'D19-11994': 'S6',
        'D19-11995': 'S7',  'D19-11997': 'S8',  'D19-11999': 'S9',
        'D19-9123':  'S10', 'D21-194004': 'S11', 'D21-194011': 'S12'
    }

    # 2. Map every individual ATAC run to its matching sample row (S1-S12)
    # Comma-separated runs from the spreadsheet are mapped to the same sample ID here!
    atac_to_joint = {
        'D19-11983': 'S1', 'D19-11984': 'S1',
        'D19-11985': 'S2',
        'D19-11988': 'S3',
        'D19-11974': 'S4',
        'D19-11975': 'S5',
        'D19-11973': 'S6',
        'D19-11972': 'S7',
        'D19-11980': 'S8',
        'D19-11979': 'S9',
        'D19-9146':  'S10',
        'D21-194103': 'S11',
        'D21-194106': 'S12', 'D21-194107': 'S12'
    }

    # 3. Associate clinical lesion responses directly to each of the 12 rows
    sample_responses = {
        'S1': 'NR', 'S2': 'R',  'S3': 'R', 'S4': 'NR', 'S5': 'R', 'S6': 'R',
        'S7': 'NR', 'S8': 'NR', 'S9': 'R', 'S10': 'NR', 'S11': 'R', 'S12': 'R'
    }

    # 4. Map metadata columns using our new 12-sample system
    X_rna = X_rna.copy()
    X_rna.obs['joint_sample_id'] = X_rna.obs['sample_ID_long'].map(rna_to_joint).astype(str)
    X_rna.obs['lesion_response'] = X_rna.obs['joint_sample_id'].map(sample_responses)

    X_atac = X_atac.copy()
    X_atac.obs['joint_sample_id'] = X_atac.obs['orig.ident'].map(atac_to_joint).astype(str)
    X_atac.obs['lesion_response'] = X_atac.obs['joint_sample_id'].map(sample_responses)

    # 5. Keep only rows that are valid matches (S1 through S12) in both datasets
    valid_samples = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12']
    X_rna = X_rna[X_rna.obs['joint_sample_id'].isin(valid_samples)].copy()
    X_atac = X_atac[X_atac.obs['joint_sample_id'].isin(valid_samples)].copy()

    # 6. Find and filter overlapping cell types between both datasets
    shared_cell_types = set(X_atac.obs['reannotated_predicted_id']).intersection(set(X_rna.obs['cell_type']))
    X_atac = X_atac[X_atac.obs['reannotated_predicted_id'].isin(shared_cell_types)].copy()
    X_rna = X_rna[X_rna.obs['cell_type'].isin(shared_cell_types)].copy()

    print("\n--- Filtered Check ---")
    print(f"Unique ATAC Sample IDs: {np.unique(X_atac.obs['joint_sample_id'])}")
    print(f"Unique RNA Sample IDs:  {np.unique(X_rna.obs['joint_sample_id'])}")

    # 7. Calculate pseudobulk datasets with perfectly aligned shapes
    pseudobulk_atac_df = convert_anndata_to_pseudobulk(
            adata=X_atac,
            sample_col="joint_sample_id",
            cell_type_col="reannotated_predicted_id",
            outcome_col="lesion_response"
        )

    pseudobulk_rna_df = convert_anndata_to_pseudobulk(
            adata=X_rna,
            sample_col="joint_sample_id",
            cell_type_col="cell_type",
            outcome_col="lesion_response"
        )

    # 8. Convert to 3D Tensors
    tensor_atac, labels_atac = convert_pseudobulk_to_tensor(
            pseudobulk_df=pseudobulk_atac_df,
            outcome_col="lesion_response",
        )
    print("\nFinal ATAC sample labels:", labels_atac["lesion_response"])

    tensor_rna, labels_rna = convert_pseudobulk_to_tensor(
            pseudobulk_df=pseudobulk_rna_df,
            outcome_col="lesion_response",
        )
    print("Final RNA sample labels:", labels_rna["lesion_response"])

    # 9. Run the Coupled-TPLS classification model
    y_raw = labels_rna["lesion_response"]
    y = np.array([0 if v == "R" else 1 for v in y_raw], dtype=int)

    (tpls, lr_model), tpls_acc, tpls_proba = run_coupled_tpls_classification(
            [tensor_atac, tensor_rna], y, rank=2, return_proba=True
        ) 

if __name__ == "__main__":
    main()