from typing import List
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Tuple


percent_columns = [
    "inflation",
    "key_rate",
    "deposit_1",
    "deposit_3",
    "deposit_6",
    "deposit_12",
    "fa_delta",
    "usd_delta",
    "IMOEX_delta",
    "RGBI_delta",
]
engineered_columns = [
    "deposit_spread",
    "usd_inverted",
    "fa_vs_usd",
    "diff_inflation",
    "year",
    "month",
    "quarter",
]
final_columns = percent_columns + engineered_columns


def preprocess_parquet(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df.columns = ["date", *df.columns[1:]]
    df["date"] = pd.to_datetime(df["date"])
    return df


def preprocess_context(
    df: pd.DataFrame, columns: List[str] = percent_columns
) -> pd.DataFrame:
    df = df.copy()
    df["context_data_from"] = pd.to_datetime(df["context_data_from"])
    df["context_data_to"] = pd.to_datetime(df["context_data_to"])
    df.dropna(inplace=True)

    df[columns] = df[columns].apply(lambda x: x.str.rstrip("%").astype(float))

    return df


def merge_frames(pq: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    pq = preprocess_parquet(pq)
    context = preprocess_context(context)

    intervals = pd.IntervalIndex.from_arrays(
        context["context_data_from"], context["context_data_to"], closed="both"
    )

    pq = pq[pq["date"] >= context["context_data_from"].min()].copy()
    pq["quarter_idx"] = intervals.get_indexer(pq["date"])
    pq = pq[pq["quarter_idx"] != -1]

    context = context.reset_index(drop=True)
    context["quarter_idx"] = context.index

    merged = pq.merge(context, on="quarter_idx", how="left")
    return merged


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter

    df["deposit_spread"] = df["deposit_12"] - df["deposit_1"]
    df["usd_inverted"] = -df["usd_delta"]
    df["fa_vs_usd"] = df["fa_delta"] - df["usd_delta"]

    df = df.sort_values(by=["quarter_idx", "date"])
    df["diff_inflation"] = df.groupby("quarter_idx")["inflation"].diff().fillna(0)

    return df


def prepare_features_and_target(
    df: pd.DataFrame, target_col: str = "cus_class"
) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    X = df[final_columns].values
    y = df[target_col].astype(int).values
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return X, y_encoded, le


def get_class_weights(y_encoded: np.ndarray) -> torch.Tensor:
    weights = compute_class_weight(
        "balanced", classes=np.unique(y_encoded), y=y_encoded
    )
    return torch.tensor(weights, dtype=torch.float32)


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    stratify_vals: np.ndarray,
    test_size: float = 0.3,
    val_size: float = 0.5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, temp_idx = next(sss.split(X, stratify_vals))

    X_train, y_train = X[train_idx], y[train_idx]
    X_temp, y_temp = X[temp_idx], y[temp_idx]
    stratify_temp = stratify_vals[temp_idx]

    sss_val = StratifiedShuffleSplit(
        n_splits=1, test_size=val_size, random_state=random_state
    )
    val_idx, test_idx = next(sss_val.split(X_temp, stratify_temp))

    X_val, y_val = X_temp[val_idx], y_temp[val_idx]
    X_test, y_test = X_temp[test_idx], y_temp[test_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test
