from data_import import convert_anndata_to_pseudobulk, convert_pseudobulk_to_tensor
import anndata
import numpy as np
from cmtf_pls.cmtf import ctPLS

def main():

    
    anndata_path = "data/scatac_non_tumor_peaks.h5ad"
    X_atac = anndata.read_h5ad(anndata_path)

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

    # Add lesion_response column to obs by mapping orig.ident
    X_atac.obs['lesion_response'] = X_atac.obs['orig.ident'].map(response_data)
    print(X_atac)
    pseudobulk_df = convert_anndata_to_pseudobulk(
        adata=X_atac,
        sample_col="orig.ident",
        cell_type_col="reannotated_predicted_id",
        outcome_col="lesion_response"
    )
    
    print(pseudobulk_df)

    tensor, labels = convert_pseudobulk_to_tensor(
        pseudobulk_df=pseudobulk_df,
        outcome_col="lesion_response",
        feature_cols=None,

    )
    print(labels["lesion_response"])

    # Randomize tensor to make a new one with the same shape but different values, for testing
    tensor2 = np.random.rand(*tensor.shape)

    tensors = [tensor, tensor2]

    y_raw = labels["lesion_response"]
    y = np.array([0 if v == "R" else 1 if v == "NR" else np.nan for v in y_raw], dtype=float)
    y = y.astype(int)
    tpls = ctPLS(n_components=2)
    tpls.fit(tensors, y)
    





    return



if __name__ == "__main__":
    main()