"""ANN candidate retrieval and lexical union for E3/E4.

E3 measures the attainable candidate-recall ceiling of multilingual-e5 ANN retrieval
(union with the lexical blocker); E4 turns the chosen operating point into production
candidate generation. This module provides the backend-agnostic primitives:

- :class:`AnnIndex` / :func:`build_ann_index` - inner-product ANN over entity embeddings.
  Uses faiss when available (``ivf_pq`` / ``ivf_flat`` / ``flat``) and falls back to an
  exact numpy inner-product search otherwise (tests, small corpora).
- :func:`retrieve` - top-K candidates per query (Source 1) row.
- :func:`union_with_lexical` - merge ANN candidates with the lexical blocker, dedup,
  and assign pass ids / scores.
- :func:`channel_attribution` - for truth pairs the lexical blocker missed, attribute
  recovery to retrieval channels (e5-name / e5-addr / e5-entity / char-gram / phonetic).

Embeddings are L2-normalised so inner product equals cosine similarity.
"""

import numpy as np
import pandas as pd

LEXICAL_PASS = 1
ANN_PASS_BASE = 11
DEFAULT_CHANNEL_PASS = {"name": 11, "addr": 12, "entity": 13, "char": 14, "phonetic": 15}
DEFAULT_CHANNEL_SCORE = {"name": 0.9, "addr": 0.85, "entity": 0.8, "char": 0.75, "phonetic": 0.7}


def l2_normalize(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    norm[norm == 0.0] = 1.0
    return matrix / norm


class AnnIndex:
    """Inner-product ANN index over entity embeddings.

    ``kind`` is one of ``flat``, ``ivf_flat``, ``ivf_pq``. When faiss is unavailable
    (or ``kind == "flat"``) search is exact via a dense matrix multiply.
    """

    def __init__(self, embeddings, ids, kind="flat", nlist=4096, m=48, nbits=8, seed=42):
        self.embeddings = l2_normalize(embeddings)
        self.ids = np.asarray(ids)
        self.kind = kind
        self.index = None
        if kind != "flat":
            faiss = _import_faiss()
            if faiss is not None:
                dim = self.embeddings.shape[1]
                quantizer = faiss.IndexFlatIP(dim)
                if kind == "ivf_pq":
                    index = faiss.IndexIVFPQ(quantizer, dim, int(nlist), int(m), int(nbits),
                                             faiss.METRIC_INNER_PRODUCT)
                elif kind == "ivf_flat":
                    index = faiss.IndexIVFFlat(quantizer, dim, int(nlist),
                                               faiss.METRIC_INNER_PRODUCT)
                else:
                    raise ValueError(f"unknown kind: {kind}")
                index.train(self.embeddings)
                index.add(self.embeddings)
                self.index = index
            else:
                self.kind = "flat"

    def search(self, queries, k, nprobe=32):
        queries = l2_normalize(queries)
        if self.index is None:
            if len(self.ids) == 0:
                empty = np.zeros((len(queries), 0), dtype=np.float32)
                return empty, np.empty((len(queries), 0), dtype=object)
            k = min(int(k), len(self.ids))
            sims = queries @ self.embeddings.T
            part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            scores = np.take_along_axis(sims, part, axis=1)
            order = np.argsort(-scores, axis=1, kind="stable")
            part = np.take_along_axis(part, order, axis=1)
            scores = np.take_along_axis(scores, order, axis=1)
        else:
            self.index.nprobe = int(nprobe)
            scores, part = self.index.search(queries, int(k))
        valid = part >= 0
        safe = np.where(valid, part, 0)
        neighbor_ids = self.ids[safe].astype(object)
        neighbor_ids[~valid] = None
        return scores.astype(np.float32), neighbor_ids


def _import_faiss():
    try:
        import faiss  # noqa: F401
    except Exception:
        return None
    import faiss

    return faiss


def build_ann_index(embeddings, ids, kind="ivf_pq", nlist=4096, m=48, nbits=8, seed=42):
    return AnnIndex(embeddings, ids, kind=kind, nlist=nlist, m=m, nbits=nbits, seed=seed)


def retrieve(index, query_embeddings, query_ids, k, channel="name", nprobe=32):
    """Return a long frame ``(s1_id, cand_id, score, rank, channel)`` per query."""
    scores, neighbors = index.search(query_embeddings, k, nprobe=nprobe)
    n_q, kk = scores.shape
    frame = pd.DataFrame(
        {
            "s1_id": np.repeat(np.asarray(query_ids), kk),
            "cand_id": neighbors.ravel(),
            "score": scores.ravel().astype(np.float32),
            "rank": np.tile(np.arange(kk, dtype=np.int32), n_q),
            "channel": channel,
        }
    )
    frame = frame[frame["cand_id"].notna()].copy()
    frame["cand_id"] = frame["cand_id"].astype(str)
    return frame


def union_with_lexical(ann_pairs, lexical_pairs=None, channel_pass=None, channel_score=None):
    """Union ANN candidates with lexical candidates.

    ``ann_pairs`` carries a ``channel`` column (or defaults to a single ``name`` channel);
    pass ids come from ``channel_pass`` (default 11-15) so the matcher can see which
    retriever fired. ``lexical_pairs`` (``s1_id, cand_id, pass_id, block_score``) is
    preserved; when a pair appears in both, the lexical (lower) pass id wins.
    """
    channel_pass = {**DEFAULT_CHANNEL_PASS, **(channel_pass or {})}
    channel_score = {**DEFAULT_CHANNEL_SCORE, **(channel_score or {})}
    ann = ann_pairs.copy()
    if "channel" not in ann.columns:
        ann["channel"] = "name"
    ann["pass_id"] = ann["channel"].map(channel_pass).fillna(ANN_PASS_BASE).astype("int16")
    if "score" in ann.columns:
        ann["block_score"] = ann["score"].astype(np.float32).fillna(0.5)
    else:
        ann["block_score"] = ann["channel"].map(channel_score).fillna(0.5).astype(np.float32)
    ann = ann[["s1_id", "cand_id", "pass_id", "block_score"]]

    frames = [ann]
    if lexical_pairs is not None and len(lexical_pairs):
        lex = lexical_pairs[["s1_id", "cand_id", "pass_id", "block_score"]].copy()
        lex["block_score"] = lex["block_score"].astype(np.float32)
        frames.append(lex)
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("pass_id", kind="stable").drop_duplicates(["s1_id", "cand_id"], keep="first")
    out["is_s2"] = out["cand_id"].astype(str).str.startswith("S2-")
    out["pass_id"] = out["pass_id"].astype("int16")
    out["block_score"] = out["block_score"].astype(np.float32)
    return out.reset_index(drop=True)


def channel_attribution(truth_pairs, lexical_pairs, channels):
    """Attribute truth pairs missed by the lexical blocker to retrieval channels.

    ``truth_pairs`` / ``lexical_pairs``: frames with ``s1_id, cand_id``.
    ``channels``: mapping name -> frame with ``s1_id, cand_id`` (e.g. each e5 field).

    Returns a dict with per-channel recovery counts, plus ``multiple`` and ``none``
    over the set of truth pairs the lexical blocker missed.
    """
    def keys(frame):
        return set(map(tuple, frame[["s1_id", "cand_id"]].to_numpy()))

    truth = keys(truth_pairs)
    lex = keys(lexical_pairs) if lexical_pairs is not None else set()
    missed = truth - lex
    channel_keys = {name: keys(frame) for name, frame in channels.items()}

    per_channel = {name: len(missed & ck) for name, ck in channel_keys.items()}
    recovered = set()
    multiple = 0
    for pair in missed:
        hits = [name for name, ck in channel_keys.items() if pair in ck]
        if len(hits) >= 1:
            recovered.add(pair)
        if len(hits) >= 2:
            multiple += 1
    return {
        "truth_pairs": len(truth),
        "lexical_found": len(truth & lex),
        "lexical_missed": len(missed),
        "per_channel": per_channel,
        "any_channel": len(recovered),
        "multiple_channels": multiple,
        "none": len(missed - recovered),
    }
