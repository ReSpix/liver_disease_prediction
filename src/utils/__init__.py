import pandas as pd
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
