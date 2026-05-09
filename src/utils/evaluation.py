from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


def evaluate_pipeline(pipeline, X, y, verbose=False):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = []

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
        score = pipeline.score(X_val, y_val)
        cv_scores.append(score)

        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)

    scores = np.array(cv_scores)
    print(f"Точность {pipeline[-1].__class__.__name__}: {scores.mean():0.3f} ± {scores.std():0.3f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    disp_raw = ConfusionMatrixDisplay.from_predictions(
        all_y_true, all_y_pred, display_labels=le.classes_, ax=ax, normalize="true"
    )
    plt.xticks(rotation=45, ha="right")
    plt.show()
