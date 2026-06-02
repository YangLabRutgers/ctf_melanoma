from data_import import convert_anndata_to_pseudobulk, convert_pseudobulk_to_tensor
import anndata
import numpy as np
from predict import run_coupled_tpls_classification

def main():
    X_atac = anndata.read_h5ad("/mnt/yang_lab/ar3023/ctf_melanoma/data/scatac_non_tumor_gene_activities.h5ad")
    X_rna = anndata.read_h5ad("/mnt/yang_lab/ar3023/ctf_melanoma/data/scrna_non_tumor.h5ad")
    print(X_atac)
    print(X_rna)
    print(X_atac.obs['orig.ident'].unique())
    print(X_rna.obs['sample_ID_long'].unique())
    # print(X_atac.obs['orig.ident'].value_counts())
    print(X_rna.obs['patient'].unique())

    response_data = {
    'D19-11985': 'R',
    'D19-11988': 'R',
    'D19-11974': 'NR',
    'D19-11975': 'R',
    'D19-11973': 'R',
    'D19-11972': 'NR',
    'D19-11980': 'NR',
    'D19-11979': 'R',
    'D19-9146': 'NR',
    'D21-194103': 'R',
    'D21-194106': 'R',
    'D21-194107': 'R',
    'D19-11983': 'NR', 
    'D19-11984': 'NR',
}
    # 1. Create a helper map to turn ATAC text strings into numeric Patient IDs
    atac_to_patient = {
        'D19-11983': '1224', 'D19-11984': '1224',
        'D19-11985': '1227',
        'D19-11988': '509',
        'D19-11974': '452',
        'D19-11975': '1098',
        'D19-11973': '509',
        'D19-11972': '776',
        'D19-11980': '856',
        'D19-11979': '1009',
        'D19-9146': '1098',
        'D21-194103': '1098',
        'D21-194106': '620', 'D21-194107': '620'
    }

    # 2. Make a new 'patient_id' column in both datasets so they speak the same language
    # # 2. Make a new 'patient_id' column in both datasets so they speak the same language
    X_atac.obs['patient_id'] = X_atac.obs['orig.ident'].map(atac_to_patient).astype(str)
    X_rna.obs['patient_id'] = X_rna.obs['patient'].astype(str)

    # 3. Intersect on the shared numeric patient IDs
    overlapping_samples = set(X_atac.obs['patient_id']).intersection(set(X_rna.obs['patient_id']))

        # 4. Filter ATAC down to matching patients
    X_atac = X_atac[X_atac.obs['patient_id'].isin(overlapping_samples)]

    # Create a unique sample tracker column using the ATAC run IDs
    X_atac.obs['joint_sample_id'] = X_atac.obs['orig.ident'].astype(str)

    # Loop through the actual sample pairs, NOT individual cells
    X_rna_list = []
    unique_pairs = X_atac.obs[['patient_id', 'joint_sample_id']].drop_duplicates()

    for idx, row in unique_pairs.iterrows():
        pid = row['patient_id']
        sample_run = row['joint_sample_id']
        
        # Grab the RNA cells for this patient and tag them with the specific ATAC sample ID
        rna_subset = X_rna[X_rna.obs['patient_id'] == pid].copy()
        rna_subset.obs['joint_sample_id'] = sample_run
        
        X_rna_list.append(rna_subset)

    import anndata as ad
    X_rna = ad.concat(X_rna_list, axis=0)

    print(overlapping_samples)

        # Clean numeric clinical response mapping for the combined patient IDs
    patient_responses = {
            '1224': 'NR', '1227': 'R', '509': 'R', '452': 'NR', 
            '1098': 'R', '776': 'NR', '856': 'R', '1009': 'R', '620': 'R'
        }
        
    X_atac.obs['lesion_response'] = X_atac.obs['patient_id'].map(patient_responses)
    X_rna.obs['lesion_response'] = X_rna.obs['patient_id'].map(patient_responses)
    print(X_atac)

        # Find overlapping cell types between both datasets
    shared_cell_types = set(X_atac.obs['reannotated_predicted_id']).intersection(set(X_rna.obs['cell_type']))
        
        # Filter both objects down to only use those matching cell types
    X_atac = X_atac[X_atac.obs['reannotated_predicted_id'].isin(shared_cell_types)]
    X_rna = X_rna[X_rna.obs['cell_type'].isin(shared_cell_types)]

    print(X_atac)
    print(X_rna)
    print(np.unique(X_atac.obs['orig.ident']))
    print(np.unique(X_rna.obs['sample_ID_long']))

    print(np.unique(X_atac.obs['patient_id']))
    print(np.unique(X_rna.obs['patient_id']))

    print(np.unique(X_atac.obs['joint_sample_id']))
    print(np.unique(X_rna.obs['joint_sample_id']))

    

    #     # # Calculate pseudobulk datasets with perfectly aligned shapes
    # pseudobulk_atac_df = convert_anndata_to_pseudobulk(
    #         adata=X_atac,
    #         sample_col="joint_sample_id",
    #         cell_type_col="reannotated_predicted_id",
    #         outcome_col="lesion_response"
    #     )

    # pseudobulk_rna_df = convert_anndata_to_pseudobulk(
    #         adata=X_rna,
    #         sample_col="joint_sample_id",
    #         cell_type_col="cell_type",
    #         outcome_col="lesion_response"
    #     )

    # tensor_atac, labels = convert_pseudobulk_to_tensor(
    #         pseudobulk_df=pseudobulk_atac_df,
    #         outcome_col="lesion_response",
    #     )
    # print(labels["lesion_response"])


    # tensor_rna, labels = convert_pseudobulk_to_tensor(
    #     pseudobulk_df=pseudobulk_rna_df,
    #     outcome_col="lesion_response",
    #     )
    # print(labels["lesion_response"])


    # y_raw = labels["lesion_response"]
    # y = np.array([0 if v == "R" else 1 if v == "NR" else np.nan for v in y_raw], dtype=float)
    # y = y.astype(int)

    # (tpls, lr_model), tpls_acc, tpls_proba = run_coupled_tpls_classification(
    #         [tensor_atac, tensor_rna], y, rank=2,return_proba=True
    #     ) 




if __name__ == "__main__":
 main()