import numpy as np

MODEL_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
MODEL_REVISION = "main"


def _part(name, address, country):
    return f"{name} | {address or 'none'} | {(country or '').strip().lower()}"


def serialize_pair(name1, addr1, country1, name2, addr2, country2) -> str:
    return f"{_part(name1, addr1, country1)} [SEP] {_part(name2, addr2, country2)}"


def score_texts(texts, model_id=MODEL_ID, revision=MODEL_REVISION, batch_size=256, fn=None) -> np.ndarray:
    if fn is not None:
        return np.asarray(fn(list(texts)), dtype="float32")
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(model_id, revision=revision, max_length=256)
    return np.asarray(model.predict(list(texts), batch_size=batch_size), dtype="float32")
