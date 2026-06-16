import anndata as ad
import numpy as np
import pandas as pd
from data_import import convert_anndata_to_pseudobulk, convert_pseudobulk_to_tensor
from predict import run_coupled_tpls_classification

def main():
    # Load AnnData objects
    X_atac = ad.read_h5ad("/mnt/yang_lab/ar3023/ctf_melanoma/data/scatac_non_tumor_gene_activities.h5ad")
    X_rna = ad.read_h5ad("/mnt/yang_lab/ar3023/ctf_melanoma/data/scrna_non_tumor.h5ad")
    
    print("Initial shapes:")
    print(f"ATAC shape: {X_atac.shape} | RNA shape: {X_rna.shape}")

    # Clinical response mapping (Patient ID -> Response)
    patient_responses = {
        '1224': 'NR', '1227': 'R', '509': 'R', '452': 'NR', 
        '1098': 'R', '776': 'NR', '856': 'NR', '1009': 'R', '620': 'R'
    }

    # ATAC Run ID -> Patient ID mapping
    atac_to_patient = {
        'D19-11983': '1224', 'D19-11984': '1224',
        'D19-11985': '1227', 'D19-11988': '509',
        'D19-11974': '452', 'D19-11975': '1098',
        'D19-11973': '509', 'D19-11972': '776',
        'D19-11980': '856', 'D19-11979': '1009',
        'D19-9146': '1098', 'D21-194103': '1098',
        'D21-194106': '620', 'D21-194107': '620'
    }

    # 1. Standardize Patient IDs across both modalities
    X_atac.obs['patient_id'] = X_atac.obs['orig.ident'].map(atac_to_patient).astype(str)
    X_rna.obs['patient_id'] = X_rna.obs['patient'].astype(str)

    # 2. Intersect on shared patients to ensure clean alignment
    overlapping_patients = set(X_atac.obs['patient_id']).intersection(set(X_rna.obs['patient_id']))
    X_atac = X_atac[X_atac.obs['patient_id'].isin(overlapping_patients)].copy()
    X_rna = X_rna[X_rna.obs['patient_id'].isin(overlapping_patients)].copy()

    # 3. Create the 'joint_sample_id' based on the mentor's logic
    # RNA drives the sample definition. We find which RNA sample belongs to which patient.
    rna_sample_map = dict(zip(X_rna.obs['patient_id'], X_rna.obs['sample_ID_long']))
    
    # Assign the shared sample ID tracking metric
    X_rna.obs['joint_sample_id'] = X_rna.obs['sample_ID_long'].astype(str)
    X_atac.obs['joint_sample_id'] = X_atac.obs['patient_id'].map(rna_sample_map).astype(str)

    # 4. Map clinical outcome tracking
    X_atac.obs['lesion_response'] = X_atac.obs['patient_id'].map(patient_responses)
    X_rna.obs['lesion_response'] = X_rna.obs['patient_id'].map(patient_responses)

    # 5. Filter for overlapping cell types between both objects
    shared_cell_types = set(X_atac.obs['reannotated_predicted_id']).intersection(set(X_rna.obs['cell_type']))
    X_atac = X_atac[X_atac.obs['reannotated_predicted_id'].isin(shared_cell_types)].copy()
    X_rna = X_rna[X_rna.obs['cell_type'].isin(shared_cell_types)].copy()

    print("\nAligned shapes (Post-pooling and cell-type filtering):")
    print(f"ATAC shape: {X_atac.shape} | RNA shape: {X_rna.shape}")
    print(f"Unique joint samples: {X_rna.obs['joint_sample_id'].nunique()}")

    # 6. Generate perfectly aligned pseudobulk dataframes
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

    # 7. Build Tensors (Sample x Feature x Cell Type)
    tensor_atac, labels_atac = convert_pseudobulk_to_tensor(
        pseudobulk_df=pseudobulk_atac_df,
        outcome_col="lesion_response",
    )

    tensor_rna, labels_rna = convert_pseudobulk_to_tensor(
        pseudobulk_df=pseudobulk_rna_df,
        outcome_col="lesion_response",
    )

    # Convert categorical outcomes into explicit integers (R=0, NR=1)
    y_raw = labels_rna["lesion_response"]
    y = np.array([0 if v == "R" else 1 for v in y_raw], dtype=int)

    print(f"\nFinal Tensors successfully built.")
    print(f"ATAC Tensor Shape: {tensor_atac.shape}")
    print(f"RNA Tensor Shape: {tensor_rna.shape}")
    print(f"Classification targets (y): {y}")

    # 8. Run Coupled Tensor Partial Least Squares Classification
    (tpls, lr_model), tpls_acc, tpls_proba = run_coupled_tpls_classification(
        [tensor_atac, tensor_rna], y, rank=2, return_proba=True
    ) 

    print(f"\nModel Execution Complete.")
    print(f"Coupled T-PLS Training Accuracy: {tpls_acc * 100:.2f}%")

if __name__ == "__main__":
    main()