from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score


def evaluate_pipeline(
    pipeline: Pipeline, X, y, *, additional: str | None = None, verbose: bool = False
) -> list:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = {"f1-macro": [], "f1-weighted": []}

    additional_list = []

    all_y_true = []
    all_y_pred = []

    le = LabelEncoder()
    y_encoded = np.array(le.fit_transform(y))

    for fold, (train_index, val_index) in enumerate(skf.split(X, y_encoded)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y_encoded[train_index], y_encoded[val_index]

        if verbose:
            print(f"{fold} fold: fitting")
        pipeline.fit(X_train, y_train)

        if verbose:
            print(f"{fold} fold: scoring")
        y_pred = pipeline.predict(X_val)

        f1_m = f1_score(y_val, y_pred, average="macro")
        scores["f1-macro"].append(f1_m)

        f1_w = f1_score(y_val, y_pred, average="weighted")
        scores["f1-weighted"].append(f1_w)

        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)

        if additional:
            additional_list.append(getattr(pipeline[-1], additional))

    f1_macro_scores = np.array(scores["f1-macro"])
    f1_weighted_scores = np.array(scores["f1-weighted"])

    print(f"Модель {pipeline[-1].__class__.__name__}")
    print(f"F1-macro: {f1_macro_scores.mean():0.3f} ± {f1_macro_scores.std():0.3f}")
    print(
        f"F1-weighted: {f1_weighted_scores.mean():0.3f} ± {f1_weighted_scores.std():0.3f}"
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    disp_raw = ConfusionMatrixDisplay.from_predictions(
        all_y_true, all_y_pred, display_labels=le.classes_, ax=ax, normalize="true"
    )
    plt.xticks(rotation=45, ha="right")
    plt.show()

    return additional_list
