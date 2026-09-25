from ber.normalize import detect_script, fold_name, name_tokens, normalize_name, strip_legal_suffix


def test_normalize_name_basic():
    assert normalize_name("  Orelee's   Barbershop!! ") == "orelee s barbershop"


def test_normalize_name_preserves_indic_letters():
    assert normalize_name("राम मार्केटिंग") == "राम मार्केटिंग"


def test_fold_name_strips_accents():
    assert fold_name("École primaire") == "ecole primaire"


def test_detect_script_latin_indic_mixed():
    assert detect_script("Holloway Peak Inc") == "latin"
    assert detect_script("राम मार्केटिंग") == "indic"
    assert detect_script("Sun पावर Provision") == "mixed"
    assert detect_script("") == "none"


def test_strip_legal_suffix_classes():
    assert strip_legal_suffix("B+ Retail Inc") == ("b retail", "inc")
    assert strip_legal_suffix("International South Consultants Private Ltd") == (
        "international south consultants",
        "ltd",
    )
    assert strip_legal_suffix("Béque") == ("beque", "")


def test_name_tokens_excludes_stopwords():
    tokens = name_tokens("The Best Bakery and Cafe")
    assert tokens == ["best", "bakery", "cafe"]
