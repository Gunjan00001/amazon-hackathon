import numpy as np
import pandas as pd

MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_REVISION = "main"
VEC_DIM = 384


def quantize(v: np.ndarray) -> np.ndarray:
    return np.clip(np.round(np.asarray(v, dtype="float32") * 127.0), -127, 127).astype("int8")


def dequantize(v: np.ndarray) -> np.ndarray:
    return np.asarray(v, dtype="float32") / 127.0


def cosine_matrix_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    denom = np.clip(na * nb, 1e-9, None)
    return ((a * b).sum(axis=1) / denom).astype("float32")


def pair_vector_features(idx1, idx2, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    a = np.asarray(v1, dtype="float32")[idx1]
    b = np.asarray(v2, dtype="float32")[idx2]
    cos = cosine_matrix_rows(a, b)
    absdiff = np.abs(a - b).mean(axis=1).astype("float32")
    prod = (a * b).mean(axis=1).astype("float32")
    return np.column_stack([cos, absdiff, prod]).astype("float32")


def encode_texts(texts, model_id=MODEL_ID, revision=MODEL_REVISION, batch_size=256, prefix=""):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_id, revision=revision)
    texts = [prefix + t for t in texts]
    vecs = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=True)
    return vecs.astype("float16")


def record_text(name: str, address: str) -> str:
    return f"{name} | {address}".strip()


def load_vector_store(embed_dir, split):
    d = embed_dir
    ids = pd.read_parquet(d / f"{split}_ids.parquet")
    s1 = dequantize(np.load(d / f"{split}_s1.npy"))
    s2 = dequantize(np.load(d / f"{split}_s2.npy"))
    s3 = dequantize(np.load(d / f"{split}_s3.npy"))
    mid = np.concatenate([s2, s3], axis=0)
    return {"ids": ids, "s1": {"vec": s1}, "mid": {"vec": mid}, "dim": s1.shape[1]}
