from .io_tsv import split_id_list


def positive_pairs(s1_ids, mid_ids, labels):
    s1_map = {e: i for i, e in enumerate(s1_ids)}
    mid_map = {e: i for i, e in enumerate(mid_ids)}
    rows = []
    for sid, mids in zip(labels["source1_entity_id"], labels["matched_entity_ids"]):
        i = s1_map.get(sid)
        if i is None:
            continue
        for m in split_id_list(mids):
            j = mid_map.get(m)
            if j is not None:
                rows.append((i, j))
    return rows
