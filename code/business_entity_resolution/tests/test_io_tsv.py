from ber.io_tsv import format_id_list, split_id_list, write_matching_results


def test_id_list_roundtrip_and_empty():
    ids = ["S2-00047", "S3-00812", "S2-00193"]
    assert format_id_list(ids) == "S2-00047,S2-00193,S3-00812"
    assert format_id_list([]) == ""
    assert split_id_list("S2-00047,S3-00812") == ["S2-00047", "S3-00812"]
    assert split_id_list("") == []


def test_write_matching_results_exact_format(tmp_path):
    p = tmp_path / "matching_results.tsv"
    write_matching_results(p, ["S1-00002", "S1-00001"], {"S1-00001": ["S2-00047"], "S1-00002": []})
    text = p.read_text(encoding="utf-8")
    assert text == "source1_entity_id\tmatched_entity_ids\nS1-00002\t\nS1-00001\tS2-00047\n"
