import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class Columns:
    categorical: list[str]
    numeric: list[str]
    binary: list[str]


def column_type_split(df: pd.DataFrame) -> Columns:
    cat_cols = [col for col in df.select_dtypes(include="string").columns]

    number_cols = df.select_dtypes(include="number").columns

    numeric_cols = [col for col in number_cols if len(df[col].unique()) > 2]
    binary_cols = [col for col in number_cols if len(df[col].unique()) == 2]

    return Columns(cat_cols, numeric_cols, binary_cols)


def feature_importance_df(feature_importance, names, *, use_abs = False) -> pd.DataFrame:
    feature_importance = np.array(feature_importance)
    if use_abs:
        feature_importance = np.abs(feature_importance)

    mean_importance = feature_importance.mean(axis=0)

    importance_df = pd.DataFrame(
        {"feature": names, "importance": mean_importance}
    ).sort_values("importance", ascending=False)

    return importance_df