
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def convert_anndata_to_pseudobulk(
    adata: ad.AnnData,
    sample_col: str,
    cell_type_col: str,
    outcome_col: str,
) -> pd.DataFrame:
    """Average expression by (sample, cell type)."""

    required_cols = [sample_col, cell_type_col, outcome_col]

    missing_cols = [col for col in required_cols if col not in adata.obs.columns]
    if missing_cols:
        raise ValueError(
            f"obs is missing required columns: {missing_cols}. "
            f"Available: {list(adata.obs.columns)}"
        )

    adata = adata[:, :1000].copy()  # limit to first 1000 features for testing

    x_df = adata.to_df()
    group_cols = [sample_col, cell_type_col]

    obs = adata.obs[[sample_col, cell_type_col, outcome_col]]
    sample_order = np.unique(obs[sample_col].to_numpy())
    cell_type_order = np.unique(obs[cell_type_col].to_numpy())
    obs[sample_col] = pd.Categorical(obs[sample_col], categories=sample_order, ordered=True)
    obs[cell_type_col] = pd.Categorical(obs[cell_type_col], categories=cell_type_order, ordered=True)

    pseudobulk = x_df.groupby([obs[col] for col in group_cols], observed=False, sort=False).mean()
    pseudobulk.index = pseudobulk.index.set_names(group_cols)

    # Fill NaN with 0 in feature columns
    pseudobulk = pseudobulk.fillna(0.0)

    outcome_nunique = obs.groupby(group_cols, observed=False, sort=False)[outcome_col].nunique()
    ambiguous_groups = outcome_nunique[outcome_nunique > 1]
    if not ambiguous_groups.empty:
        raise ValueError(
            f"Each ({sample_col}, {cell_type_col}) group must map to exactly one "
            f"'{outcome_col}' value. Ambiguous groups: {ambiguous_groups.index.tolist()}"
        )
    # Map outcome directly from sample index to ensure no NaN
    pseudobulk[outcome_col] = pseudobulk.index.get_level_values(sample_col).map(
        lambda s: obs[obs[sample_col] == s][outcome_col].iloc[0] if (obs[sample_col] == s).any() else None
    )
    # Extract unique categories from non-null outcome values
    outcome_categories = pd.Series(pseudobulk[outcome_col]).dropna().unique()
    pseudobulk[outcome_col] = pd.Categorical(
        pseudobulk[outcome_col],
        categories=outcome_categories,
        ordered=False
    )
    return pseudobulk


def convert_pseudobulk_to_tensor(
    pseudobulk_df: pd.DataFrame,
    outcome_col: str,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Convert one pseudobulk dataframe to one tensor + labels.

    Input must be indexed by (sample, cell_type).
    Output tensor shape: (n_samples, n_cell_types, n_features)
    """
    if not isinstance(pseudobulk_df.index, pd.MultiIndex) or pseudobulk_df.index.nlevels < 2:
        raise ValueError("pseudobulk_df index must be a MultiIndex with sample and cell_type.")

    sample_level = pseudobulk_df.index.names[0]
    cell_type_level = pseudobulk_df.index.names[1]

    samples = np.unique(pseudobulk_df.index.get_level_values(sample_level).to_numpy())
    cell_types = np.unique(pseudobulk_df.index.get_level_values(cell_type_level).to_numpy())

    if outcome_col not in pseudobulk_df.columns:
        raise ValueError(
            f"outcome_col '{outcome_col}' must be present as a column in pseudobulk_df. "
            f"Available columns: {list(pseudobulk_df.columns)}"
        )

    if feature_cols is None:
        data_df = pseudobulk_df.drop(columns=[outcome_col])
    else:
        if outcome_col in feature_cols:
            raise ValueError(f"feature_cols must not include outcome_col '{outcome_col}'.")
        missing_features = [f for f in feature_cols if f not in pseudobulk_df.columns]
        if missing_features:
            raise ValueError(f"Requested feature_cols not found: {missing_features}")
        data_df = pseudobulk_df[feature_cols]

    features = data_df.columns.to_numpy()

    sample_to_i = {s: i for i, s in enumerate(samples)}
    cell_to_j = {c: j for j, c in enumerate(cell_types)}

    tensor = np.full((len(samples), len(cell_types), len(features)), np.nan, dtype=np.float32)
    for index_values, row in data_df.iterrows():
        if not isinstance(index_values, tuple):
            index_values = (index_values,)
        sample = index_values[0]
        cell_type = index_values[1]
        tensor[sample_to_i[sample], cell_to_j[cell_type], :] = row.to_numpy(np.float32)

    sample_values = pseudobulk_df.index.get_level_values(sample_level).to_numpy()
    outcome_values = pseudobulk_df[outcome_col].to_numpy()
    sample_outcome_map: dict[object, object] = {}
    for sample, outcome in zip(sample_values, outcome_values, strict=False):
        if sample in sample_outcome_map and sample_outcome_map[sample] != outcome:
            raise ValueError(
                f"Each sample must map to one '{outcome_col}' value. "
                f"Sample with multiple values: {sample}"
            )
        sample_outcome_map[sample] = outcome

    labels = {
        "samples": samples,
        "cell_types": cell_types,
        "features": features,
        f"{outcome_col}": np.array([sample_outcome_map.get(sample) for sample in samples], dtype=object),
    }
    return tensor, labels

