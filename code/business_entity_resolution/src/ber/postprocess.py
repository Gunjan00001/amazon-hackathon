import numpy as np
import pandas as pd


def one_to_one(probs: pd.Series, pairs: pd.DataFrame) -> pd.Series:
    frame = pairs.copy()
    frame["prob"] = np.asarray(probs, dtype=np.float32)
    order = frame.sort_values(["cand_id", "prob", "s1_id"], ascending=[True, False, True]).index
    kept = pd.Series(False, index=frame.index)
    seen = set()
    for idx in order:
        cand = frame.at[idx, "cand_id"]
        if cand not in seen:
            seen.add(cand)
            kept.at[idx] = True
    return kept
