import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from ctf import ctPLS

skf = StratifiedKFold(n_splits=5)



def run_coupled_tpls_classification(
    tensors: list[np.ndarray],
    labels: pd.Series | np.ndarray,
    rank: int,
    scoring: str = "accuracy",
    return_proba: bool = False,
    return_components: bool = False,
):
    """
    Fits coupled tPLS model to provided data and labels.

    Parameters:
        tensors (list of np.ndarray): coupled tensors
        labels (pd.Series): labels to regress data against
        rank (int): number of components to use in tPLS
        return_proba (bool, default:False): returns probability of each
            patient's classification
        return_components (bool, default:False): returns patient component
            factors

    Returns:
        models (tuple[tPLS, LR classifier]): tuple of trained tPLS and LR model
        acc (float): accuracy achieved over cross-validation
        pred (pd.Series, only returns if not return_proba): predicted value for
            each patient
        proba (pd.Series, only returns if return_proba): probability of positive
            classification for each patient
    """
    if return_proba and return_components:
        return_components = False

    np.random.seed(215)
    tpls = ctPLS(n_components=rank)
    tpls.fit(tensors, labels)

    predicted = pd.Series(0.0, index=np.arange(len(labels)), dtype=float)
    components = pd.DataFrame(
        0.0, index=np.arange(len(labels)), columns=np.arange(rank) + 1, dtype=float
    )

    model = LogisticRegressionCV(
        l1_ratios=[.1, .5, .7, .9, .95, .99, 1],
        solver="saga",
        use_legacy_attributes=False,
        n_jobs=-1,
        cv=skf,
        max_iter=100000,
        scoring=scoring,
    )
    model.fit(tpls.Xs_factors[0][0], labels)
    best_c = float(np.asarray(model.C_).ravel()[0])
    best_l1_ratio = float(np.asarray(model.l1_ratio_).ravel()[0])
    model = LogisticRegression(
        C=best_c,
        l1_ratio=best_l1_ratio,
        solver="saga",
        max_iter=100000,
    )
    for train_index, test_index in skf.split(labels, labels):
        train_data = [tensor[train_index, :, :] for tensor in tensors]
        test_data = [tensor[test_index, :, :] for tensor in tensors]
        train_labels = labels[train_index]
        tpls.fit(train_data, train_labels)

        train_transformed = tpls.transform(train_data)
        test_transformed = tpls.transform(test_data)
        model.fit(train_transformed, train_labels)

        if return_proba:
            predicted.iloc[test_index] = model.predict_proba(test_transformed)[
                :, 1
            ]
        elif return_components:
            components.iloc[test_index, :] = test_transformed
        else:
            predicted.iloc[test_index] = model.predict(test_transformed)

    if return_proba:
        acc = accuracy_score(labels, predicted.round().astype(int))
    else:
        acc = accuracy_score(labels, predicted)

    tpls.fit(tensors, labels)
    model.fit(tpls.transform(tensors), labels)

    # Calculate the R2X variance explained metric
    r2x_val = float(tpls.R2X(tensors))

    if return_proba:
        return (tpls, model), acc, predicted, r2x_val
    elif return_components:
        return (tpls, model), acc, components, r2x_val
    else:
        return (tpls, model), acc, predicted, r2x_val