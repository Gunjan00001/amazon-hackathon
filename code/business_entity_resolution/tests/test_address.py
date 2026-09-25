from ber.address import parse_address


def test_parse_us_address():
    p = parse_address("1795 Westchester Drive, High Point, NC", "US")
    assert p.house_no == "1795"
    assert "westchester" in p.street_tokens
    assert p.state_key == "north carolina"
    assert p.postal == ""
    assert not p.addr_missing


def test_parse_india_pin():
    p = parse_address("G-3/571, GULMOHAR COLONY, BHOPAL, Madhya Pradesh 462033", "India")
    assert p.postal == "462033"
    assert p.state_key == "madhya pradesh"


def test_parse_france_postal():
    p = parse_address("175 Boulevard du President Franklin Roosevelt, Bordeaux, 33000", "France")
    assert p.house_no == "175"
    assert p.postal == "33000"


def test_landmark_and_missing():
    p = parse_address("Near SBI ATM, MG Road", "India")
    assert p.landmark_flag is True
    assert parse_address("", "US").addr_missing is True
