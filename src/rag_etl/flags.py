"""Row-level flags for the sections DataFrame.

* ``is_TO_DO``      1 when a section's body is only a placeholder ("TO DO.",
                    "_TODO_", "TBD", ...), else 0.
* ``near_dup_group`` id shared by sections that are near-duplicates of each
                    other across versions of the same project (<NA> if none).
* ``For_training``  0 for near-duplicates that are not in the latest version
                    of their group, 1 otherwise.

Near-duplicates can be found with two methods (``flag_near_duplicates``):

* ``"minhash"``   MinHash + LSH over character 5-shingles (estimated Jaccard).
* ``"semantic"``  sentence-embedding cosine similarity (see ``rag_etl.semantic``).

Both produce candidate pairs that go through the same best-match selection
(:func:`select_best_matches`) and grouping (:func:`assign_near_dup_groups`).
"""

from __future__ import annotations

import re

import pandas as pd

# --------------------------------------------------------------------------
# TO DO placeholders
# --------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(
    r"^(?:to\s*-?\s*do|tbd|tbc|wip|work in progress|coming soon|pendiente|"
    r"por hacer|en construccion|en construcción)$",
    re.I,
)
_STRIP_RE = re.compile(
    r"!?\[[^\]]*\]\([^)]*\)"  # markdown links / images
    r"|<[^>]+>"  # html tags
    r"|[*_`~>#.:!\-\s]+"  # formatting and punctuation
)


def is_todo_text(text: str) -> bool:
    """True when the text, once formatting is stripped, is just a placeholder."""
    core = _STRIP_RE.sub(" ", text).strip()
    return bool(core) and bool(_PLACEHOLDER_RE.match(re.sub(r"\s+", " ", core)))


def flag_todo(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_TO_DO"] = out["text"].map(is_todo_text).astype("int8")
    return out


# --------------------------------------------------------------------------
# Versions
# --------------------------------------------------------------------------

_NUMERIC_VERSION_RE = re.compile(r"\d+(?:\.\d+)*")
LATEST_ALIASES = {"latest", "current", "stable", "main", "master"}


def version_sort_key(version: str) -> tuple:
    """Sortable key: alias folders like ``latest`` rank above any number,
    otherwise the leading numeric part is compared numerically
    (``0.0.10-x`` > ``0.0.6-y``; ``v0.0.2`` -> 0.0.2)."""
    if version.lower() in LATEST_ALIASES:
        return (1, (), version)
    match = _NUMERIC_VERSION_RE.search(version)
    numbers = tuple(int(n) for n in match.group().split(".")) if match else ()
    return (0, numbers, version)


# --------------------------------------------------------------------------
# Near duplicates: shared best-match selection and grouping
# --------------------------------------------------------------------------

PAIR_COLUMNS = ["left", "right", "similarity"]


def select_best_matches(
    df: pd.DataFrame, candidates: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    """Reduce scored candidate pairs to cross-version best matches.

    ``candidates`` has columns ``left``, ``right`` (row indexes of ``df``) and
    ``similarity``; it may be in either direction and contain duplicates.
    Only pairs from the *same project* and *different versions* are kept.
    Then, for every row and every other version, only its single most similar
    row in that version is kept, if it scores >= ``threshold``. Best-match
    avoids linking templated sibling sections (e.g. "Prod: db" vs
    "Stage: db") when the true counterpart exists.
    """
    if candidates.empty:
        return pd.DataFrame(columns=PAIR_COLUMNS)

    both_ways = pd.concat(
        [
            candidates[["left", "right", "similarity"]],
            candidates.rename(columns={"left": "right", "right": "left"}),
        ],
        ignore_index=True,
    )
    left, right = both_ways["left"].to_numpy(), both_ways["right"].to_numpy()
    project, version = df["project_name"], df["version"]
    both_ways = both_ways[
        (project.loc[left].to_numpy() == project.loc[right].to_numpy())
        & (version.loc[left].to_numpy() != version.loc[right].to_numpy())
        & (both_ways["similarity"].to_numpy() >= threshold)
    ].copy()
    if both_ways.empty:
        return pd.DataFrame(columns=PAIR_COLUMNS)

    # Best match of each row within each other version (ties -> lowest index).
    both_ways["right_version"] = version.loc[both_ways["right"]].to_numpy()
    both_ways = both_ways.sort_values(
        ["left", "right_version", "similarity", "right"],
        ascending=[True, True, False, True],
    )
    best = both_ways.drop_duplicates(["left", "right_version"])

    lo = best[["left", "right"]].min(axis=1)
    hi = best[["left", "right"]].max(axis=1)
    pairs = pd.DataFrame(
        {"left": lo, "right": hi, "similarity": best["similarity"]}
    ).drop_duplicates(["left", "right"])
    return pairs.sort_values(["left", "right"]).reset_index(drop=True)


def _connected_components(pairs: pd.DataFrame) -> dict:
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for left, right in zip(pairs["left"], pairs["right"]):
        parent[find(left)] = find(right)
    return {node: find(node) for node in parent}


def assign_near_dup_groups(df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Add ``near_dup_group`` and ``For_training`` from near-duplicate pairs.

    Pairs are linked into groups; within a group, rows from the latest
    version (see :func:`version_sort_key`) get ``For_training = 1`` and the
    rest 0. Rows without near-duplicates keep ``For_training = 1``.
    """
    out = df.copy()
    components = _connected_components(pairs)

    group_ids = {root: n for n, root in enumerate(sorted(set(components.values()), key=str))}
    out["near_dup_group"] = pd.Series(
        {idx: group_ids[root] for idx, root in components.items()}, dtype="Int64"
    ).reindex(out.index)

    version_key = out["version"].map(version_sort_key)
    out["For_training"] = 1
    for _, members in out[out["near_dup_group"].notna()].groupby("near_dup_group"):
        latest = max(version_key[i] for i in members.index)
        stale = [i for i in members.index if version_key[i] != latest]
        out.loc[stale, "For_training"] = 0
    out["For_training"] = out["For_training"].astype("int8")
    return out


# --------------------------------------------------------------------------
# MinHash + LSH
# --------------------------------------------------------------------------


def _shingles(text: str, k: int = 5) -> frozenset[str]:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    if len(normalized) <= k:
        return frozenset({normalized})
    return frozenset(normalized[i : i + k] for i in range(len(normalized) - k + 1))


def build_minhashes(
    texts: pd.Series, num_perm: int = 128, k: int = 5, seed: int = 1
) -> pd.Series:
    """One MinHash signature per text (char k-shingles), indexed like ``texts``."""
    from datasketch import MinHash

    shingle_sets = [[s.encode("utf-8") for s in _shingles(t, k)] for t in texts]
    return pd.Series(
        MinHash.bulk(shingle_sets, num_perm=num_perm, seed=seed), index=texts.index
    )


def find_minhash_duplicate_pairs(
    df: pd.DataFrame,
    threshold: float = 0.7,
    num_perm: int = 128,
    k: int = 5,
    seed: int = 1,
    minhashes: pd.Series | None = None,
) -> pd.DataFrame:
    """Cross-version near-duplicate pairs using MinHash + LSH.

    LSH (tuned for ``threshold``) proposes candidate pairs without comparing
    every pair; each candidate is scored with the MinHash Jaccard estimate and
    reduced with :func:`select_best_matches`.
    """
    from datasketch import MinHashLSH

    if minhashes is None:
        minhashes = build_minhashes(df["text"], num_perm=num_perm, k=k, seed=seed)
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for idx, mh in minhashes.items():
        lsh.insert(idx, mh)

    candidates = [
        (i, j, minhashes[i].jaccard(minhashes[j]))
        for i, mh in minhashes.items()
        for j in lsh.query(mh)
        if i < j
    ]
    candidates = pd.DataFrame(candidates, columns=PAIR_COLUMNS)
    return select_best_matches(df, candidates, threshold)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

DEFAULT_THRESHOLDS = {
    # Best-match Jaccard on docs_raw: true duplicates >= 0.80, closest
    # non-duplicate ~0.60.
    "minhash": 0.7,
    # Best-match cosine (EmbeddingGemma, STS prompt) on docs_raw: true
    # duplicates >= 0.968, closest non-duplicates 0.937 and 0.868.
    "semantic": 0.95,
}


def find_near_duplicate_pairs(
    df: pd.DataFrame, method: str = "minhash", threshold: float | None = None, **kwargs
) -> pd.DataFrame:
    if method not in DEFAULT_THRESHOLDS:
        raise ValueError(f"unknown method {method!r}; use one of {list(DEFAULT_THRESHOLDS)}")
    threshold = DEFAULT_THRESHOLDS[method] if threshold is None else threshold
    if method == "minhash":
        return find_minhash_duplicate_pairs(df, threshold=threshold, **kwargs)
    from rag_etl.semantic import find_semantic_duplicate_pairs

    return find_semantic_duplicate_pairs(df, threshold=threshold, **kwargs)


def flag_near_duplicates(
    df: pd.DataFrame, method: str = "minhash", threshold: float | None = None, **kwargs
) -> pd.DataFrame:
    """Add ``near_dup_group`` and ``For_training`` using ``method``
    (``"minhash"`` or ``"semantic"``). ``kwargs`` go to the pair finder."""
    pairs = find_near_duplicate_pairs(df, method=method, threshold=threshold, **kwargs)
    return assign_near_dup_groups(df, pairs)
