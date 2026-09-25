import pandas as pd


def prune_top_k(df, k, score_col="score"):
    return (df.sort_values(score_col, ascending=False)
              .groupby("s1_idx", sort=False, as_index=False)
              .head(k)
              .reset_index(drop=True))


def matches_from_scores(df, threshold, score_col="score"):
    kept = df[df[score_col] >= threshold]
    out = {}
    for s1, mid in zip(kept["s1_idx"], kept["mid_idx"]):
        out.setdefault(int(s1), []).append(int(mid))
    return out


def one_to_one_assign(df, threshold, score_col="score"):
    kept = df[df[score_col] >= threshold].sort_values(score_col, ascending=False)
    used_mid = set()
    out = {}
    for s1, mid in zip(kept["s1_idx"], kept["mid_idx"]):
        s1, mid = int(s1), int(mid)
        if mid in used_mid:
            continue
        used_mid.add(mid)
        out.setdefault(s1, []).append(mid)
    return out


def to_entity_ids(matches, s1_ids, mid_ids):
    return {sid: [mid_ids[m] for m in matches.get(i, [])] for i, sid in enumerate(s1_ids)}
