from ber.translit import romanize


def test_romanize_identity_for_latin():
    assert romanize("Holloway Peak Inc") == "holloway peak inc"


def test_romanize_indic_is_latin():
    out = romanize("राम मार्केटिंग")
    assert out and all(ord(ch) < 128 for ch in out)


def test_romanize_mixed_keeps_latin():
    out = romanize("Sun पावर Provision")
    assert out.startswith("sun ")
    assert all(ord(ch) < 128 for ch in out)


def test_romanize_empty():
    assert romanize("") == ""
