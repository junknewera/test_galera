from typing import List
import pandas as pd


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
