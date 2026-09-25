import re
from dataclasses import dataclass

from ber.normalize import normalize_name

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada", "nh": "new hampshire",
    "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee",
    "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
}

IN_STATES = {
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh", "goa", "gujarat",
    "haryana", "himachal pradesh", "jharkhand", "karnataka", "kerala", "madhya pradesh",
    "maharashtra", "manipur", "meghalaya", "mizoram", "nagaland", "odisha", "punjab",
    "rajasthan", "sikkim", "tamil nadu", "telangana", "tripura", "uttar pradesh",
    "uttarakhand", "west bengal", "delhi", "jammu and kashmir", "ladakh", "puducherry",
    "chandigarh", "haryana", "haryana",
}

FR_REGIONS = {
    "auvergne rhone alpes", "bourgogne franche comte", "bretagne", "centre val de loire",
    "corse", "grand est", "hauts de france", "ile de france", "normandie",
    "nouvelle aquitaine", "occitanie", "pays de la loire", "provence alpes cote d azur",
    "bordeaux", "lille", "dunkerque", "la teste de buch",
}

STATE_NAMES = set(US_STATES.values()) | IN_STATES | FR_REGIONS

STREET_SUFFIXES = {
    "street", "st", "road", "rd", "avenue", "ave", "drive", "dr", "lane", "ln", "court",
    "ct", "boulevard", "blvd", "rue", "chemin", "route", "way", "plaza", "circle",
}

LANDMARK_WORDS = {"near", "opposite", "opp", "behind", "beside", "next"}


@dataclass(frozen=True)
class AddressParts:
    house_no: str
    street_tokens: tuple
    postal: str
    state_key: str
    city_tokens: tuple
    landmark_flag: bool
    addr_missing: bool


def _postal_re(country: str) -> re.Pattern:
    key = (country or "").strip().lower()
    if key == "us":
        return re.compile(r"\b\d{5}(?:-\d{4})?\b")
    return re.compile(r"\b\d{6}\b" if key == "india" else r"\b\d{5}\b")


def parse_address(raw: str, country: str) -> AddressParts:
    if not raw or not raw.strip():
        return AddressParts("", (), "", "", (), False, True)
    norm = normalize_name(raw)
    tokens = norm.split()
    house_no = tokens[0] if tokens and tokens[0][:1].isdigit() else ""
    postal_match = _postal_re(country).search(norm)
    postal = postal_match.group(0) if postal_match else ""
    state_key = ""
    for i in range(len(tokens) - 1):
        pair = f"{tokens[i]} {tokens[i + 1]}"
        if pair in STATE_NAMES:
            state_key = pair
            break
    if not state_key:
        for tok in tokens:
            if tok in STATE_NAMES or tok in US_STATES:
                state_key = US_STATES.get(tok, tok)
                break
    landmark = any(t in LANDMARK_WORDS for t in tokens)
    street = tuple(t for t in tokens if t not in STREET_SUFFIXES and t != house_no)[:12]
    return AddressParts(house_no, street, postal, state_key, tuple(tokens[-6:]), landmark, False)
