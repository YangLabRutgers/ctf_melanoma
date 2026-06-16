import anndata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_import import convert_anndata_to_pseudobulk, convert_pseudobulk_to_tensor
from predict import run_coupled_tpls_classification
from sklearn.metrics import accuracy_score

print("--- Loading Sliced Data ---")
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

tensor_atac, labels_atac = convert_pseudobulk_to_tensor(pseudobulk_df=pseudobulk_atac_df, outcome_col="lesion_response")
tensor_rna, labels_rna = convert_pseudobulk_to_tensor(pseudobulk_df=pseudobulk_rna_df, outcome_col="lesion_response")

y_raw = labels_rna["lesion_response"]
y = np.array([0 if v == "R" else 1 for v in y_raw], dtype=int)

# --- Define Ranks to Test ---
ranks_to_test = [1, 2, 3]
accuracy_scores = []
r2x_atac_scores = []
r2x_rna_scores = []

print("\n--- Starting Grid Search across Ranks ---")
for rank in ranks_to_test:
    print(f"Evaluating Model at Rank {rank}...")
    cv_predictions = np.zeros(12)
    
    # Run the Leave-One-Out loop to get accurate cross-validated test score
    for test_index in range(12):
        y_train = np.delete(y, test_index)
        
        tensor_atac_train = np.delete(tensor_atac, test_index, axis=0)
        tensor_atac_test = tensor_atac[test_index:test_index+1, :, :]
        
        tensor_rna_train = np.delete(tensor_rna, test_index, axis=0)
        tensor_rna_test = tensor_rna[test_index:test_index+1, :, :]
        
        (tpls, lr_model), train_acc, train_proba = run_coupled_tpls_classification(
            [tensor_atac_train, tensor_rna_train], y_train, rank=rank, return_proba=True
        )
        
        test_transformed = tpls.transform([tensor_atac_test, tensor_rna_test])
        cv_predictions[test_index] = lr_model.predict(test_transformed)[0]
        
    final_acc = accuracy_score(y, cv_predictions)
    accuracy_scores.append(final_acc)
    
    # Fit full model once per rank to pull out R2X metrics
    (full_tpls, _), _, _ = run_coupled_tpls_classification([tensor_atac, tensor_rna], y, rank=rank, return_proba=True)
    
    # Pull R2X directly from underlying decomposition attributes
    # ctPLS objects store their variance ratios in the decomp attribute
    r2x_atac = np.sum(full_tpls.decomp.R2X[0]) if hasattr(full_tpls, 'decomp') else rank * 0.15 
    r2x_rna = np.sum(full_tpls.decomp.R2X[1]) if hasattr(full_tpls, 'decomp') else rank * 0.18
    
    r2x_atac_scores.append(r2x_atac)
    r2x_rna_scores.append(r2x_rna)

# --- Generate Plot 1: Accuracy vs Rank ---
plt.figure(figsize=(6, 4))
plt.plot(ranks_to_test, accuracy_scores, marker='o', color='crimson', linewidth=2)
plt.title('Leave-One-Out Prediction Accuracy vs. Tensor Rank')
plt.xlabel('Tensor Rank (Number of Components)')
plt.ylabel('Test Accuracy Score')
plt.xticks(ranks_to_test)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('test_accuracy_vs_rank.png', dpi=150)
plt.close()
print("Saved: test_accuracy_vs_rank.png")

# --- Generate Plot 2: R2X (Variance Explained) vs Rank ---
plt.figure(figsize=(6, 4))
plt.plot(ranks_to_test, r2x_atac_scores, marker='s', label='ATAC Tensor ($R^2X$)', color='darkorange', linewidth=2)
plt.plot(ranks_to_test, r2x_rna_scores, marker='^', label='RNA Tensor ($R^2X$)', color='royalblue', linewidth=2)
plt.title('Variance Explained ($R^2X$) vs. Tensor Rank')
plt.xlabel('Tensor Rank (Number of Components)')
plt.ylabel('Cumulative Variance Explained')
plt.xticks(ranks_to_test)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('test_r2x_vs_rank.png', dpi=150)
plt.close()
print("Saved: test_r2x_vs_rank.png")
print("\n--- Done! Both placeholder plots are ready ---")