import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch
from predict import run_coupled_tpls_classification


def main():
    tensors, y, sample_categories = create_dummy_coupled_dataset(tensor_a_noise_scale=1, tensor_b_noise_scale=1, 
                                                                 category_1_strength=1.2, category_2_strength=0.8,
                                                                 feature_signal_jitter_a=0.45, feature_signal_jitter_b=0.45)

    (tpls, lr_model), acc, predicted = run_coupled_tpls_classification(
        tensors=tensors,
        labels=y,
        rank=2,
        return_proba=True,
    )

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
    save_clustermap(tpls.Xs_factors[0][1], "Shared Mode Cell Type", "shared_mode_cell_type_clustermap.png", row_labels= [f"Cell Type {i + 1}" for i in range(tpls.Xs_factors[0][1].shape[0])])
    save_clustermap(tpls.Xs_factors[0][2], "Tensor 1 Feature X", "tensor1_feature1_clustermap.png", row_labels=[f"Feature X {i + 1}" for i in range(tpls.Xs_factors[0][2].shape[0])])
    save_clustermap(tpls.Xs_factors[1][2], "Tensor 2 Feature Y", "tensor2_feature2_clustermap.png", row_labels=[f"Feature Y {i + 1}" for i in range(tpls.Xs_factors[1][2].shape[0])])
                    
    
    plot_r2x_over_components(tensors, y, 5, "r2x_over_components.png")
    plot_acc_over_components(tensors, y, 5, "accuracy_over_components.png")

    plot_logistic_weights(lr_model, "logistic_weights.png")

    return




def save_clustermap(
    data: np.ndarray,
    title: str,
    filename: str,
    row_labels: list[str] | None = None,
    row_colors: list[str] | None = None,
    color_legend: dict[str, str] | None = None,
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
            title="Category",
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
    random_active_cells_per_sample: bool = True,
    random_signal_features_per_sample: bool = True,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
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

        if random_signal_features_per_sample:
            n_signal_a_sample = rng.integers(1, n_features_a + 1)
            n_signal_b_sample = rng.integers(1, n_features_b + 1)
            signal_features_a = rng.choice(n_features_a, size=n_signal_a_sample, replace=False)
            signal_features_b = rng.choice(n_features_b, size=n_signal_b_sample, replace=False)
        else:
            signal_features_a = coupled_features_a
            signal_features_b = coupled_features_b

        signal_values_a = cat_strength * feature_signal_scale_a[signal_features_a]
        signal_values_b = (
            cat_strength * tensor_b_signal_scale * feature_signal_scale_b[signal_features_b]
        )

        for cell_idx in active_cells_sample:
            tensor_a[sample_idx, cell_idx, signal_features_a] += signal_values_a
            tensor_b[sample_idx, cell_idx, signal_features_b] += signal_values_b

    return [tensor_a, tensor_b], y, sample_categories



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
