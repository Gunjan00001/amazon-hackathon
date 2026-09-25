from ber.text import fold_ascii, normalize_address, normalize_name, phonetic_key, postal_key


def test_normalize_name_expands_legal_suffixes():
    assert normalize_name("Acme Pvt. Ltd.") == "acme private limited"


def test_normalize_name_handles_ampersand_and_accents():
    assert normalize_name("École & Fils") == "ecole and fils"


def test_fold_ascii_removes_non_ascii():
    out = fold_ascii("Sun पावर Provision")
    assert out.isascii()
    assert "sun" in out and "provision" in out


def test_normalize_address_keeps_digits():
    assert normalize_address("12, MG Rd., Bengaluru 560001") == "12 mg road bengaluru 560001"


def test_postal_key_detects_pin_and_zip():
    assert postal_key("12 MG Road Bengaluru 560001") == "560001"
    assert postal_key("1 Main St, Austin, TX 78701") == "78701"
    assert postal_key("no digits here") is None


def test_phonetic_key_is_ascii_and_stable():
    a = phonetic_key("Bengaluru")
    b = phonetic_key("Bengaluru")
    assert a and a == b and a.isascii()