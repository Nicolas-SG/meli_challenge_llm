"""Semantic near-duplicate detection with sentence embeddings.

Sections are embedded with a SentenceTransformer model (EmbeddingGemma by
default) and compared with cosine similarity. ``google/embeddinggemma-300m``
is gated on Hugging Face: accept its license and export ``HF_TOKEN`` (or run
``hf auth login``) before loading it.

``sentence_transformers`` is imported lazily so the rest of the pipeline does
not need torch unless the semantic method is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rag_etl.flags import PAIR_COLUMNS, select_best_matches

DEFAULT_MODEL = "google/embeddinggemma-300m"


def load_model(name: str = DEFAULT_MODEL, device: str | None = None):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device=device)


def embed_texts(texts, model, batch_size: int = 32) -> np.ndarray:
    """L2-normalized embeddings (so dot product == cosine similarity).

    Uses the model's ``STS`` prompt when it has one (EmbeddingGemma's prompt
    for symmetric text-to-text similarity)."""
    prompt_name = "STS" if "STS" in (getattr(model, "prompts", None) or {}) else None
    return model.encode(
        list(texts),
        prompt_name=prompt_name,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def cosine_candidates(df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    """Every cross-version pair within each project, scored by cosine.

    ``embeddings`` is aligned positionally with ``df`` rows."""
    position = pd.Series(np.arange(len(df)), index=df.index)
    frames = []
    for _, project in df.groupby("project_name", sort=False):
        if project["version"].nunique() < 2:
            continue
        pos = position[project.index].to_numpy()
        sims = embeddings[pos] @ embeddings[pos].T
        versions = project["version"].to_numpy()
        a, b = np.triu_indices(len(pos), k=1)
        cross = versions[a] != versions[b]
        a, b = a[cross], b[cross]
        frames.append(
            pd.DataFrame(
                {
                    "left": project.index[a],
                    "right": project.index[b],
                    "similarity": sims[a, b].astype(float),
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def find_semantic_duplicate_pairs(
    df: pd.DataFrame,
    threshold: float,
    embeddings: np.ndarray | None = None,
    model=None,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Cross-version near-duplicate pairs by embedding cosine similarity.

    Pass precomputed ``embeddings`` (aligned with ``df``) to skip encoding;
    otherwise ``model`` (or the default model) is used to embed ``df.text``."""
    if embeddings is None:
        model = model if model is not None else load_model()
        embeddings = embed_texts(df["text"], model, batch_size=batch_size)
    return select_best_matches(df, cosine_candidates(df, embeddings), threshold)
