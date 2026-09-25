import re
import unicodedata

import jellyfish
from anyascii import anyascii

LEGAL_SUFFIXES = {
    "pvt": "private",
    "ltd": "limited",
    "co": "company",
    "corp": "corporation",
    "inc": "incorporated",
    "llp": "limited liability partnership",
    "llc": "limited liability company",
    "plc": "public limited company",
    "srl": "societe a responsabilite limitee",
    "sarl": "societe a responsabilite limitee",
}

ADDRESS_ABBREV = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "opp": "opposite",
    "nr": "near",
    "hse": "house",
    "bldg": "building",
    "flr": "floor",
    "apt": "apartment",
    "dist": "district",
    "pin": "postal",
    "po": "post office",
}

POSTAL_RE = re.compile(r"\b(\d{6}|\d{5}(?:-\d{4})?)\b")

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").casefold()
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("&", " and ")
    s = _PUNCT_RE.sub(" ", s)
    return _SPACE_RE.sub(" ", s).strip()


def fold_ascii(s: str) -> str:
    s = anyascii(s or "")
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if ord(c) < 128).lower()


def _expand(tokens, mapping):
    return [mapping.get(t, t) for t in tokens]


def normalize_name(s: str) -> str:
    return " ".join(_expand(normalize(s).split(), LEGAL_SUFFIXES))


def normalize_address(s: str) -> str:
    return " ".join(_expand(normalize(s).split(), ADDRESS_ABBREV))


def address_tokens(s: str) -> list:
    return [t for t in normalize_address(s).split() if not t.isdigit()]


def postal_key(s: str):
    m = POSTAL_RE.search(normalize(s))
    return m.group(1) if m else None


def phonetic_key(token: str) -> str:
    return jellyfish.metaphone(fold_ascii(token)) or ""
