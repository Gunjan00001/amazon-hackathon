import re
import unicodedata

_WS = re.compile(r"\s+")
_STOPWORDS = {"the", "and", "of", "for", "at", "in", "on", "a", "an", "to", "co"}


def _clean_chars(s: str) -> str:
    out = []
    for ch in s:
        category = unicodedata.category(ch)
        if ch.isspace() or ch.isalnum() or category[0] == "M":
            out.append(ch)
        else:
            out.append(" ")
    return _WS.sub(" ", "".join(out)).strip()

SUFFIX_CLASSES = {
    "pvt": "pvt",
    "private": "pvt",
    "ltd": "ltd",
    "limited": "ltd",
    "inc": "inc",
    "incorporated": "inc",
    "llc": "llc",
    "llp": "llp",
    "corp": "corp",
    "corporation": "corp",
    "sarl": "sarl",
    "sas": "sas",
    "sa": "sa",
    "gmbh": "gmbh",
}


def normalize_name(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").casefold()
    s = "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")
    return _clean_chars(s)


def fold_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").casefold()
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return _clean_chars(s)


def _script_of(ch: str) -> str:
    cp = ord(ch)
    if cp < 0x0250:
        return "latin"
    if 0x0900 <= cp <= 0x0D7F:
        return "indic"
    if 0x0400 <= cp <= 0x04FF:
        return "cyrillic"
    if 0x0600 <= cp <= 0x06FF:
        return "arabic"
    if 0x0590 <= cp <= 0x05FF:
        return "hebrew"
    if ch.isalpha():
        return "other"
    return ""


def detect_script(s: str) -> str:
    found = {_script_of(ch) for ch in (s or "") if ch.isalpha()}
    found.discard("")
    if not found:
        return "none"
    if found == {"latin"}:
        return "latin"
    if found <= {"indic", "latin"}:
        return "indic" if "indic" in found and "latin" not in found else "mixed"
    return "mixed"


def strip_legal_suffix(s: str) -> tuple:
    tokens = fold_name(s).split()
    suffix = ""
    while tokens and tokens[-1] in SUFFIX_CLASSES:
        if not suffix:
            suffix = SUFFIX_CLASSES[tokens[-1]]
        tokens.pop()
    return " ".join(tokens), suffix


def name_tokens(s: str) -> list:
    return [t for t in normalize_name(s).split() if t not in _STOPWORDS and len(t) > 1]
