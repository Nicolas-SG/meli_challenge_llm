"""Compare MinHash LSH and semantic (embedding) near-duplicate detection:
which pairs each finds, how ``For_training`` differs, and how long each takes."""

from __future__ import annotations

import statistics
import time

import numpy as np
import pandas as pd

from rag_etl.flags import (
    DEFAULT_THRESHOLDS,
    build_minhashes,
    find_minhash_duplicate_pairs,
)
from rag_etl.semantic import embed_texts, find_semantic_duplicate_pairs


def _median_time(fn, repeats: int):
    times, result = [], None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times), result


def time_minhash(
    df: pd.DataFrame,
    repeats: int = 3,
    threshold: float | None = None,
    num_perm: int = 128,
    k: int = 5,
    seed: int = 1,
) -> dict:
    """Median wall time of the full MinHash path (signatures + LSH + best match)."""
    threshold = DEFAULT_THRESHOLDS["minhash"] if threshold is None else threshold

    def run():
        minhashes = build_minhashes(df["text"], num_perm=num_perm, k=k, seed=seed)
        pairs = find_minhash_duplicate_pairs(
            df, threshold=threshold, num_perm=num_perm, minhashes=minhashes
        )
        return minhashes, pairs

    seconds, (minhashes, pairs) = _median_time(run, repeats)
    return {"seconds_total": seconds, "pairs": pairs, "minhashes": minhashes}


def time_semantic(
    df: pd.DataFrame, model, repeats: int = 3, threshold: float | None = None, batch_size: int = 32
) -> dict:
    """Median wall time of encoding and of similarity search, measured
    separately. Model loading is not included (time it once, outside)."""
    threshold = DEFAULT_THRESHOLDS["semantic"] if threshold is None else threshold
    embed_texts(df["text"].iloc[:batch_size], model, batch_size=batch_size)  # warm-up (CUDA init)

    encode_s, embeddings = _median_time(lambda: embed_texts(df["text"], model, batch_size), repeats)
    search_s, pairs = _median_time(
        lambda: find_semantic_duplicate_pairs(df, threshold=threshold, embeddings=embeddings), repeats
    )
    return {
        "seconds_encode": encode_s,
        "seconds_search": search_s,
        "seconds_total": encode_s + search_s,
        "pairs": pairs,
        "embeddings": embeddings,
    }


def compare_pairs(
    df: pd.DataFrame,
    minhash_pairs: pd.DataFrame,
    semantic_pairs: pd.DataFrame,
    minhashes: pd.Series,
    embeddings: np.ndarray,
) -> pd.DataFrame:
    """Union of both pair sets, labelled ``both`` / ``only_minhash`` /
    ``only_semantic``, with both scores (the missing one is computed)."""
    key = ["left", "right"]
    merged = pd.merge(
        minhash_pairs.rename(columns={"similarity": "jaccard_est"}),
        semantic_pairs.rename(columns={"similarity": "cosine"}),
        on=key,
        how="outer",
        indicator=True,
    )
    merged["found_by"] = merged["_merge"].map(
        {"both": "both", "left_only": "only_minhash", "right_only": "only_semantic"}
    )
    position = pd.Series(np.arange(len(df)), index=df.index)
    merged["jaccard_est"] = [
        minhashes[l].jaccard(minhashes[r]) for l, r in zip(merged["left"], merged["right"])
    ]
    merged["cosine"] = [
        float(embeddings[position[l]] @ embeddings[position[r]])
        for l, r in zip(merged["left"], merged["right"])
    ]
    for side in ("left", "right"):
        merged[f"{side}_version"] = df.loc[merged[side], "version"].to_numpy()
        merged[f"{side}_section"] = df.loc[merged[side], "section"].to_numpy()
    merged["project_name"] = df.loc[merged["left"], "project_name"].to_numpy()
    return merged.drop(columns="_merge").sort_values(["found_by", "project_name", "left"]).reset_index(drop=True)


def compare_flags(minhash_df: pd.DataFrame, semantic_df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Crosstab of ``For_training`` between methods and the agreement rate."""
    table = pd.crosstab(
        minhash_df["For_training"].rename("minhash"),
        semantic_df["For_training"].rename("semantic"),
    )
    agreement = float((minhash_df["For_training"] == semantic_df["For_training"]).mean())
    return table, agreement
