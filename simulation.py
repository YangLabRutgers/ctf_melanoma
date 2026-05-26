import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch
from predict import run_coupled_tpls_classification


def build_axis_labels(n_items: int, prefix: str) -> list[str]:
    return [f"{prefix} {i + 1}" for i in range(n_items)]


def collect_unique_indices(index_groups: np.ndarray | list[np.ndarray]) -> np.ndarray:
    if isinstance(index_groups, np.ndarray):
        return np.unique(np.asarray(index_groups, dtype=int))

    unique_indices = sorted({int(index) for group in index_groups for index in np.asarray(group, dtype=int).tolist()})
    return np.asarray(unique_indices, dtype=int)


def build_active_row_colors(
    n_items: int,
    active_indices: np.ndarray | list[np.ndarray],
    active_color: str = "#d95f0e",
    inactive_color: str = "#d9d9d9",
) -> list[str]:
    active_set = set(collect_unique_indices(active_indices).tolist())
    return [active_color if index in active_set else inactive_color for index in range(n_items)]


def build_active_color_legend(
    active_color: str = "#d95f0e",
    inactive_color: str = "#d9d9d9",
    active_label: str = "Active",
    inactive_label: str = "Nonactive",
) -> dict[str, str]:
    return {
        active_label: active_color,
        inactive_label: inactive_color,
    }


def main():
    tensors, y, sample_categories, signal_metadata = create_dummy_coupled_dataset(
        tensor_a_noise_scale=.25,
        tensor_b_noise_scale=.25,
        category_1_strength=1.2,
        category_2_strength=0.8,
        feature_signal_jitter_a=0.35,
        feature_signal_jitter_b=0.35,
        cell_signal_jitter=0.1,
        variable_cell_importance=True,
        random_active_cells_per_sample=False,
        random_signal_features_per_sample=False,
        seed=0
    )

    cell_type_labels = signal_metadata["cell_type_labels"]
    feature_labels_a = signal_metadata["feature_labels_a"]
    feature_labels_b = signal_metadata["feature_labels_b"]

    active_cells = signal_metadata["global_active_cells"]
    active_features_a = signal_metadata["global_signal_features_a"]
    active_features_b = signal_metadata["global_signal_features_b"]

    cell_type_row_colors = build_active_row_colors(len(cell_type_labels), active_cells)
    feature_a_row_colors = build_active_row_colors(len(feature_labels_a), active_features_a)
    feature_b_row_colors = build_active_row_colors(len(feature_labels_b), active_features_b)

    cell_type_color_legend = build_active_color_legend()
    feature_a_color_legend = build_active_color_legend()
    feature_b_color_legend = build_active_color_legend()

    print("Active cell types:", signal_metadata["global_active_cells"])
    print("Tensor 1 signal features:", signal_metadata["global_signal_features_a"])
    print("Tensor 2 signal features:", signal_metadata["global_signal_features_b"])

    (tpls, lr_model), acc, predicted = run_coupled_tpls_classification(
        tensors=tensors,
        labels=y,
        rank=3,
        return_proba=True,
    )
    print(f"Classification accuracy: {acc:.2f}")

    sample_labels = [f"Patient_{i + 1:03d}" for i in range(len(y))]
    category_palette = {
        "Category_1": "#4c78a8",
        "Category_2": "#f58518",
    }
    sample_row_colors = [category_palette[str(cat)] for cat in sample_categories]

    save_clustermap(
        tpls.Xs_factors[0][0],
        "Shared Mode Patients/Samples",
        "shared_mode_patients_samples_clustermap.png",
        row_labels=sample_labels,
        row_colors=sample_row_colors,
        color_legend=category_palette,
    )

    # Use these helpers to mark active rows and add a legend for the active items.
    save_clustermap(
        tpls.Xs_factors[0][1],
        "Shared Mode Cell Type",
        "shared_mode_cell_type_clustermap.png",
        row_labels=cell_type_labels,
        row_colors=cell_type_row_colors,
        color_legend=cell_type_color_legend,
        legend_title="Activity",
    )
    save_clustermap(
        tpls.Xs_factors[0][2],
        "Tensor 1 Feature X",
        "tensor1_feature1_clustermap.png",
        row_labels=feature_labels_a,
        row_colors=feature_a_row_colors,
        color_legend=feature_a_color_legend,
        legend_title="Activity",
    )
    save_clustermap(
        tpls.Xs_factors[1][2],
        "Tensor 2 Feature Y",
        "tensor2_feature2_clustermap.png",
        row_labels=feature_labels_b,
        row_colors=feature_b_row_colors,
        color_legend=feature_b_color_legend,
        legend_title="Activity",
    )

    plot_logistic_weights(lr_model, "logistic_weights.png")

    # Combined R2X and accuracy plot
    plot_r2x_and_acc_over_components(tensors, y, 10, "r2x_and_accuracy_over_components.png")


    return




def save_clustermap(
    data: np.ndarray,
    title: str,
    filename: str,
    row_labels: list[str] | None = None,
    row_colors: list[str] | None = None,
    color_legend: dict[str, str] | None = None,
    legend_title: str = "Category",
) -> None:
    """Save a clustered heatmap for one factor matrix."""
    frame = pd.DataFrame(data)
    if row_labels is not None:
        frame.index = row_labels

    cluster_grid = sns.clustermap(
        frame,
        cmap=sns.diverging_palette(145, 300, s=60, as_cmap=True),
        center=0,
        row_cluster=True,
        col_cluster=True,
        row_colors=row_colors,
        figsize=(8, 8),
    )

    if color_legend:
        handles = [Patch(facecolor=color, edgecolor="none", label=label) for label, color in color_legend.items()]
        cluster_grid.ax_heatmap.legend(
            handles=handles,
            title=legend_title,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
        )

    out_dir = Path("plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / filename

    cluster_grid.fig.suptitle(title, y=1.02)
    cluster_grid.fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(cluster_grid.fig)


def plot_r2x_over_components(tensors, y, component, filename: str) -> None:
    """Plot cumulative R2X by component for both tensors and R2Y."""
    components = np.arange(1, component + 1)
    r2xs_tensor1 = pd.Series(0.0, index=components)
    r2xs_tensor2 = pd.Series(0.0, index=components)

    for i in components:
        print(f"Evaluating rank {i}...")
        (tpls, lr_model), acc, predicted = run_coupled_tpls_classification(
            tensors=tensors,
            labels=y,
            rank=i,
            return_proba=True,
        )

        r2xs_tensor1.loc[i] = tpls.R2Xs[0][-1]
        r2xs_tensor2.loc[i] = tpls.R2Xs[1][-1]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(r2xs_tensor1.index, r2xs_tensor1.values, marker="o", label="R2X (Tensor 1)")
    ax.plot(r2xs_tensor2.index, r2xs_tensor2.values, marker="o", label="R2X (Tensor 2)")
    ax.set_xlabel("Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("R2X")
    ax.set_xticks(components)
    ax.legend()
    fig.tight_layout()
    out_dir = Path("plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)



def plot_acc_over_components(tensors, y, component, filename: str) -> None:
    """Plot prediction accuracy across a small range of tPLS ranks."""
    ranks = np.arange(1, component + 1)
    accuracies = pd.Series(0.0, index=ranks, dtype=float)

    for rank in ranks:
        print(f"Evaluating rank {rank}...")
        (_, _), acc, _ = run_coupled_tpls_classification(
            tensors=tensors,
            labels=y,
            return_proba=True,
            rank=rank,
        )
        accuracies.loc[rank] = acc

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(accuracies.index, accuracies.values, marker="o")
    ax.set_xlabel("tPLS Components")
    ax.set_ylabel("Prediction Accuracy")
    ax.set_title("Rank vs Accuracy")
    out_dir = Path("plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_r2x_and_acc_over_components(tensors, y, max_component: int, filename: str) -> None:
    """Combine cumulative R2X (both tensors) and prediction accuracy in one figure.

    The metrics are shown as three stacked subplots so they stay visually separate
    while still saving to a single file.
    """
    components = np.arange(1, max_component + 1)
    r2x1 = pd.Series(0.0, index=components)
    r2x2 = pd.Series(0.0, index=components)
    accuracies = pd.Series(0.0, index=components, dtype=float)

    for k in components:
        print(f"Evaluating rank {k}...")
        (tpls, lr_model), acc, predicted = run_coupled_tpls_classification(
            tensors=tensors,
            labels=y,
            rank=int(k),
            return_proba=True,
        )

        r2x1.loc[k] = tpls.R2Xs[0][-1]
        r2x2.loc[k] = tpls.R2Xs[1][-1]
        accuracies.loc[k] = acc

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    axes[0].plot(components, r2x1.values, marker="o", color="#1f77b4")
    axes[0].set_ylabel("R2X")
    axes[0].set_title("R2X (Tensor 1)")
    axes[0].set_ylim(bottom=0)

    axes[1].plot(components, r2x2.values, marker="o", color="#ff7f0e")
    axes[1].set_ylabel("R2X")
    axes[1].set_title("R2X (Tensor 2)")
    axes[1].set_ylim(bottom=0)

    axes[2].plot(components, accuracies.values, marker="s", color="#2ca02c")
    axes[2].set_xlabel("tPLS Components")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_title("Prediction Accuracy")
    axes[2].set_ylim(bottom=0)

    for axis in axes:
        axis.set_xticks(components)
        axis.grid(True, alpha=0.25)

    fig.tight_layout()
    out_dir = Path("plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)




def plot_logistic_weights(model, filename: str) -> None:
    """Plot logistic regression coefficients for the model."""
    weights = np.asarray(model.coef_).ravel()
    components = np.arange(1, len(weights) + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#2b8cbe" if w >= 0 else "#d95f0e" for w in weights]
    ax.bar(components, weights, color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(components)
    ax.set_xlabel("Component")
    ax.set_ylabel("Logistic Regression Weight")
    fig.tight_layout()
    out_dir = Path("run")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)






def create_dummy_coupled_dataset(
    n_samples: int = 40,
    n_cell_types: int = 10,
    n_features_a: int = 30,
    n_features_b: int = 20,
    seed: int = 215,
    tensor_a_noise_scale: float = 1.0,
    tensor_b_noise_scale: float = 1.0,
    sample_noise_jitter: float = 0.35,
    feature_noise_jitter: float = 0.30,
    category_1_strength: float = -0.6,
    category_2_strength: float = 1.2,
    category_strength_jitter: float = 0.25,
    feature_signal_jitter_a: float = 0.35,
    feature_signal_jitter_b: float = 0.35,
    tensor_b_signal_scale: float = 1.0,
    cell_signal_jitter: float = 0.25,
    variable_cell_importance: bool = False,
    random_active_cells_per_sample: bool = True,
    random_signal_features_per_sample: bool = True,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, dict[str, object]]:
    """
    Create synthetic coupled tensors where specific subsets of cells 
    and features move together in relation to the class label.
    """
    if n_samples % 2 != 0:
        raise ValueError("n_samples must be even for balanced classes.")

    rng = np.random.default_rng(seed)
    
    # Generate balanced binary labels
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2), dtype=int)
    rng.shuffle(y)

    # Keep category labels binary so they match the binary class setup.
    sample_categories = np.where(y == 0, "Category_1", "Category_2").astype(object)

    if n_features_a < 1 or n_features_b < 1:
        raise ValueError("n_features_a and n_features_b must both be positive.")

    # Initialize tensors with heteroskedastic noise (sample- and feature-specific scales).
    feature_scale_a = rng.lognormal(mean=0.0, sigma=feature_noise_jitter, size=(1, 1, n_features_a))
    feature_scale_b = rng.lognormal(mean=0.0, sigma=feature_noise_jitter, size=(1, 1, n_features_b))
    sample_scale_a = rng.lognormal(mean=0.0, sigma=sample_noise_jitter, size=(n_samples, 1, 1))
    sample_scale_b = rng.lognormal(mean=0.0, sigma=sample_noise_jitter, size=(n_samples, 1, 1))
    tensor_a = (
        rng.normal(0.0, 1.0, size=(n_samples, n_cell_types, n_features_a))
        * tensor_a_noise_scale
        * feature_scale_a
        * sample_scale_a
    )
    tensor_b = (
        rng.normal(0.0, 1.0, size=(n_samples, n_cell_types, n_features_b))
        * tensor_b_noise_scale
        * feature_scale_b
        * sample_scale_b
    )

    # Define baseline active subsets for the signal.
    active_cells = rng.choice(n_cell_types, size=n_cell_types // 2, replace=False)
    
    # Choose independent feature sets for each tensor; there are no shared features.
    # The number of signal features is itself random so the structure is not fixed.
    n_signal_features_a = rng.integers(1, n_features_a + 1)
    n_signal_features_b = rng.integers(1, n_features_b + 1)
    coupled_features_a = rng.choice(n_features_a, size=n_signal_features_a, replace=False)
    coupled_features_b = rng.choice(n_features_b, size=n_signal_features_b, replace=False)

    # Feature-specific signal multipliers so strength is non-uniform across features.
    feature_signal_scale_a = rng.lognormal(mean=0.0, sigma=feature_signal_jitter_a, size=n_features_a)
    feature_signal_scale_b = rng.lognormal(mean=0.0, sigma=feature_signal_jitter_b, size=n_features_b)
    if variable_cell_importance:
        cell_signal_scale = rng.lognormal(mean=0.0, sigma=cell_signal_jitter, size=n_cell_types)
    else:
        cell_signal_scale = np.ones(n_cell_types)

    active_cells_by_sample: list[np.ndarray] = []
    signal_features_a_by_sample: list[np.ndarray] = []
    signal_features_b_by_sample: list[np.ndarray] = []

    # Inject coordinated signal with category separation and sample-level strength jitter.
    category_strength = {
        "Category_1": category_1_strength,
        "Category_2": category_2_strength,
    }

    for sample_idx in range(n_samples):
        base_strength = category_strength[str(sample_categories[sample_idx])]
        cat_strength = base_strength + rng.normal(0.0, category_strength_jitter)

        if random_active_cells_per_sample:
            n_active_sample = rng.integers(1, n_cell_types + 1)
            active_cells_sample = rng.choice(n_cell_types, size=n_active_sample, replace=False)
        else:
            active_cells_sample = active_cells
        
        active_cells_by_sample.append(np.array(active_cells_sample, copy=True))

        signal_features_a = coupled_features_a
        signal_features_b = coupled_features_b

        signal_features_a_by_sample.append(np.array(signal_features_a, copy=True))
        signal_features_b_by_sample.append(np.array(signal_features_b, copy=True))

        signal_values_a = cat_strength * feature_signal_scale_a[signal_features_a]
        signal_values_b = (
            cat_strength * tensor_b_signal_scale * feature_signal_scale_b[signal_features_b]
        )

        for cell_idx in active_cells_sample:
            tensor_a[sample_idx, cell_idx, signal_features_a] += signal_values_a * cell_signal_scale[cell_idx]
            tensor_b[sample_idx, cell_idx, signal_features_b] += signal_values_b * cell_signal_scale[cell_idx]

    signal_metadata = {
        "global_active_cells": np.array(active_cells, copy=True),
        "global_signal_features_a": np.array(coupled_features_a, copy=True),
        "global_signal_features_b": np.array(coupled_features_b, copy=True),
        "active_cells_by_sample": active_cells_by_sample,
        "signal_features_a_by_sample": signal_features_a_by_sample,
        "signal_features_b_by_sample": signal_features_b_by_sample,
        "cell_type_labels": build_axis_labels(n_cell_types, "Cell Type"),
        "feature_labels_a": build_axis_labels(n_features_a, "Tensor 1 Feature"),
        "feature_labels_b": build_axis_labels(n_features_b, "Tensor 2 Feature"),
    }

    return [tensor_a, tensor_b], y, sample_categories, signal_metadata



if __name__ == "__main__":
    main()





def plot_probability_heatmap(proba: pd.Series, labels: np.ndarray) -> None:
    """Plot predicted probabilities with true labels."""
    heat = pd.DataFrame(
        {
            "True label": labels,
            "Predicted P(class=1)": proba.to_numpy(dtype=float),
        }
    ).T

    plt.figure(figsize=(12, 2.8))
    sns.heatmap(heat, cmap="viridis", cbar=True)
    plt.title("Cross-Validated Probabilities")
    plt.xlabel("Sample index")
    plt.tight_layout()


def plot_prediction_heatmap(labels: np.ndarray, proba: pd.Series) -> None:
    """Plot hard predictions versus true labels."""
    pred = (proba.to_numpy(dtype=float) >= 0.5).astype(int)
    heat = pd.DataFrame({"True": labels, "Predicted": pred}).T

    plt.figure(figsize=(12, 2.5))
    sns.heatmap(heat, cmap="magma", cbar=True, vmin=0, vmax=1)
    plt.title("True vs Predicted Class")
    plt.xlabel("Sample index")
    plt.tight_layout()
