from data_import import convert_anndata_to_pseudobulk, convert_pseudobulk_to_tensor
import anndata
import numpy as np
from predict import run_coupled_tpls_classification

def main():
    X_atac = anndata.read_h5ad("data/scatac_non_tumor_gene_activities.h5ad")
    X_rna = anndata.read_h5ad("data/scrna_non_tumor.h5ad")
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
    # Only keep overlapping samples between the two datasets
    overlapping_samples = set(X_atac.obs['orig.ident']).intersection(set(X_rna.obs['sample_ID_long']))
    X_atac = X_atac[X_atac.obs['orig.ident'].isin(overlapping_samples)]
    X_rna = X_rna[X_rna.obs['sample_ID_long'].isin(overlapping_samples)]

    print(overlapping_samples)

    # Add lesion_response column to obs by mapping orig.ident
    X_atac.obs['lesion_response'] = X_atac.obs['orig.ident'].map(response_data)
    X_rna.obs['lesion_response'] = X_rna.obs['sample_ID_long'].map(response_data)
    print(X_atac)

    pseudobulk_atac_df = convert_anndata_to_pseudobulk(
        adata=X_atac,
        sample_col="orig.ident",
        cell_type_col="reannotated_predicted_id",
        outcome_col="lesion_response"
    )
    
    print(pseudobulk_atac_df)

    pseudobulk_rna_df = convert_anndata_to_pseudobulk(
        adata=X_rna,
        sample_col="sample_ID_long",
        cell_type_col="cell_type",
        outcome_col="lesion_response"
    )
    print(pseudobulk_rna_df)
    
    tensor_atac, labels = convert_pseudobulk_to_tensor(
        pseudobulk_df=pseudobulk_atac_df,
        outcome_col="lesion_response",
    )
    print(labels["lesion_response"])


    tensor_rna, labels = convert_pseudobulk_to_tensor(
        pseudobulk_df=pseudobulk_rna_df,
        outcome_col="lesion_response",
    )
    print(labels["lesion_response"])


    y_raw = labels["lesion_response"]
    y = np.array([0 if v == "R" else 1 if v == "NR" else np.nan for v in y_raw], dtype=float)
    y = y.astype(int)

    (tpls, lr_model), tpls_acc, tpls_proba = run_coupled_tpls_classification(
        [tensor_atac, tensor_rna], y, rank=2,return_proba=True
    )





    return



if __name__ == "__main__":
    main()