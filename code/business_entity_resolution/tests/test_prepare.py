import pandas as pd

from ber.prepare import prepare_frame


def test_prepare_frame_columns_and_values():
    df = pd.DataFrame(
        {
            "entity_id": ["S1-1"],
            "business_name": ["B+ Retail Inc"],
            "business_address": ["1795 Westchester Drive, High Point, NC"],
            "country": ["US"],
        }
    )
    out = prepare_frame(df)
    row = out.iloc[0]
    assert row["name_norm"] == "b retail inc"
    assert row["name_stripped"] == "b retail"
    assert row["suffix_class"] == "inc"
    assert row["name_script"] == "latin"
    assert row["house_no"] == "1795"
    assert row["state_key"] == "north carolina"
    assert row["addr_raw_missing"] in (False, 0)
    assert "b" in row["name_idf_tokens"]
