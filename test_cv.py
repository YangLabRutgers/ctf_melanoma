from data_import import convert_anndata_to_pseudobulk, convert_pseudobulk_to_tensor
import anndata

print("--- Starting our simple test ---")

# Load just the RNA file to see if it works
X_rna = anndata.read_h5ad("/mnt/yang_lab/ar3023/ctf_melanoma/data/scrna_non_tumor.h5ad")

print("Original RNA size is:", X_rna.shape)
X_rna_small = X_rna[:, :50].copy()
print("Sliced RNA size is now:", X_rna_small.shape)
# Load the ATAC file
X_atac = anndata.read_h5ad("/mnt/yang_lab/ar3023/ctf_melanoma/data/scatac_non_tumor_gene_activities.h5ad")
print("Original ATAC size is:", X_atac.shape)

# Slice it to the first 50 genes too
X_atac_small = X_atac[:, :50].copy()
print("Sliced ATAC size is now:", X_atac_small.shape)
# Mappings to clean up the sample names
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

print("Sample mappings successfully assigned to the sliced data.")
# Keep only rows that are S1 through S12
valid_samples = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12']
X_rna_small = X_rna_small[X_rna_small.obs['joint_sample_id'].isin(valid_samples)].copy()
X_atac_small = X_atac_small[X_atac_small.obs['joint_sample_id'].isin(valid_samples)].copy()

# Find and filter overlapping cell types
shared_cell_types = set(X_atac_small.obs['reannotated_predicted_id']).intersection(set(X_rna_small.obs['cell_type']))
X_atac_small = X_atac_small[X_atac_small.obs['reannotated_predicted_id'].isin(shared_cell_types)].copy()
X_rna_small = X_rna_small[X_rna_small.obs['cell_type'].isin(shared_cell_types)].copy()

print("Data filtering completed successfully.")
# Convert our sliced and filtered matrices into pseudobulk profiles
pseudobulk_atac_df = convert_anndata_to_pseudobulk(
    adata=X_atac_small,
    sample_col="joint_sample_id",
    cell_type_col="reannotated_predicted_id",
    outcome_col="lesion_response"
)

pseudobulk_rna_df = convert_anndata_to_pseudobulk(
    adata=X_rna_small,
    sample_col="joint_sample_id",
    cell_type_col="cell_type",
    outcome_col="lesion_response"
)

# Structure them into perfectly aligned 3D tensors
tensor_atac, labels_atac = convert_pseudobulk_to_tensor(
    pseudobulk_df=pseudobulk_atac_df,
    outcome_col="lesion_response"
)

tensor_rna, labels_rna = convert_pseudobulk_to_tensor(
    pseudobulk_df=pseudobulk_rna_df,
    outcome_col="lesion_response"
)

print(f"Pseudobulk tensor compilation complete.")
print(f"ATAC tensor shape: {tensor_atac.shape} | RNA tensor shape: {tensor_rna.shape}")
import numpy as np

# Create the 0 and 1 classification labels for our 12 samples
y_raw = labels_rna["lesion_response"]
y = np.array([0 if v == "R" else 1 for v in y_raw], dtype=int)

print("Target labels array (y) created successfully.")
print("Labels match sequence:", y)
cv_predictions = np.zeros(12)
print("\n--- Starting Leave-One-Out Cross-Validation Loop ---")

# We have 12 samples, so we loop from 0 to 11
for test_index in range(12):
    # 1. Identify which sample name is being left out for testing
    left_out_sample = labels_rna['samples'][test_index]
    
    # 2. Split the targets (y) into 11 training samples and 1 test sample
    y_train = np.delete(y, test_index)
    y_test = y[test_index]
    
    # 3. Split the ATAC tensor into 11 training samples and 1 test sample
    tensor_atac_train = np.delete(tensor_atac, test_index, axis=0)
    tensor_atac_test = tensor_atac[test_index:test_index+1, :, :]
    
    # 4. Split the RNA tensor into 11 training samples and 1 test sample
    tensor_rna_train = np.delete(tensor_rna, test_index, axis=0)
    tensor_rna_test = tensor_rna[test_index:test_index+1, :, :]
    
    print(f"Fold {test_index+1}: Left out {left_out_sample} | Training set size: {tensor_rna_train.shape[0]}")
    # --- Step 17: Run classification inside this validation fold ---
    # We import the function dynamically here to ensure it is isolated
    from predict import run_coupled_tpls_classification
    
    # We pass the 11 training samples into the TPLS function
    # Let's use rank=2 just like your trial script did
    (tpls, lr_model), train_acc, train_proba = run_coupled_tpls_classification(
        [tensor_atac_train, tensor_rna_train], 
        y_train, 
        rank=2, 
        return_proba=True
    )
    
    print(f"       -> Successfully completed Fold {test_index+1} training! Train Accuracy: {train_acc:.2f}")
    # Project the single left-out test sample into our trained model space
    test_transformed = tpls.transform([tensor_atac_test, tensor_rna_test])
    
    # Predict whether this left-out sample is a 0 or 1
    sample_pred = lr_model.predict(test_transformed)[0]
    
    # Save that prediction into our array
    cv_predictions[test_index] = sample_pred
    
    print(f"       -> Test Result | True Label: {y_test} | Predicted: {int(sample_pred)}")
# --- Calculate and print out the final cross-validation score ---
from sklearn.metrics import accuracy_score

final_cv_accuracy = accuracy_score(y, cv_predictions)
print("\n==============================================")
print(f"FINAL CROSS-VALIDATION ACCURACY: {final_cv_accuracy * 100:.2f}%")
print("==============================================")