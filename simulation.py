import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from predict import run_coupled_tpls_classification


def plot_clustermap(data: np.ndarray, title: str, filename: str, row_labels: list[str] | None = None) -> None:
    """Save a clustered heatmap for one factor matrix."""
    frame = pd.DataFrame(data)
    if row_labels is not None:
        frame.index = row_labels

    grid = sns.clustermap(
        frame,
        cmap=sns.diverging_palette(145, 300, s=60, as_cmap=True),
        figsize=(8, 8),
        method="average",
        metric="euclidean",
    )
    grid.fig.suptitle(title, y=1.02)
    grid.fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(grid.fig)


def plot_rank_vs_accuracy(tensors: list[np.ndarray], labels: np.ndarray, filename: str) -> None:
    """Plot prediction accuracy across a small range of tPLS ranks."""
    ranks = np.arange(1, 15)
    accuracies = pd.Series(0.0, index=ranks, dtype=float)

    for rank in ranks:
        (_, _), acc, _ = run_coupled_tpls_classification(
            tensors=tensors,
            labels=labels,
            return_proba=True,
            rank=rank,
        )
        accuracies.loc[rank] = acc

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(accuracies.index, accuracies.values, marker="o")
    # ax.set_ylim([0.5, 0.75])
    ax.set_xlabel("tPLS Components")
    ax.set_ylabel("Prediction Accuracy")
    ax.set_title("Rank vs Accuracy")
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    tensors, y = create_dummy_coupled_dataset()

    (tpls, _), _, _ = run_coupled_tpls_classification(
        tensors=tensors,
        labels=y,
        rank=2,
        return_proba=True,
    )

    plot_clustermap(tpls.Xs_factors[0][0], "Shared Patient Factors", "patient_factors.png")
    plot_clustermap(tpls.Xs_factors[0][1], "Shared Cell-Type Factors", "shared_mode_factors.png")
    plot_clustermap(tpls.Xs_factors[0][2], "Tensor A Specific Factors", "tensor_a_factors.png")
    plot_clustermap(tpls.Xs_factors[1][2], "Tensor B Specific Factors", "tensor_b_factors.png")
    plot_rank_vs_accuracy(tensors, y, "rank_vs_accuracy.png")

    return


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




import numpy as np

def create_dummy_coupled_dataset(
    n_samples: int = 40,
    n_cell_types: int = 10,
    n_features: int = 50,
    seed: int = 215,
) -> tuple[list[np.ndarray], np.ndarray]:
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

    # Initialize tensors with noise
    tensor_a = rng.normal(0.0, 1.0, size=(n_samples, n_cell_types, n_features))
    tensor_b = rng.normal(0.0, 1.0, size=(n_samples, n_cell_types, n_features))

    # Define the "Active" subsets for the signal
    # Half of Mode 1 (cell types)
    active_cells = rng.choice(n_cell_types, size=n_cell_types // 2, replace=False)
    
    # 20 features for Mode 2 (genes/features)
    # We select 20 specific indices that will be coupled across both tensors
    coupled_features = rng.choice(n_features, size=min(20, n_features), replace=False)

    # Inject the coordinated signal for the positive class
    signal_mask = (y == 1)
    
    for cell_idx in active_cells:
        # Increase intensity for coupled features in both tensors simultaneously
        # This creates the "moving together" relationship
        strength = 1.2
        tensor_a[np.ix_(signal_mask, [cell_idx], coupled_features)] += strength
        tensor_b[np.ix_(signal_mask, [cell_idx], coupled_features)] += (strength * 0.9)

    return [tensor_a, tensor_b], y



if __name__ == "__main__":
    main()

