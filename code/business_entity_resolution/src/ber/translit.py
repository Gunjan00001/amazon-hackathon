import functools

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

from ber.normalize import normalize_name

_SCHEMES = (
    (0x0900, sanscript.DEVANAGARI),
    (0x0980, sanscript.BENGALI),
    (0x0A00, sanscript.GURMUKHI),
    (0x0A80, sanscript.GUJARATI),
    (0x0B00, sanscript.ORIYA),
    (0x0B80, sanscript.TAMIL),
    (0x0C00, sanscript.TELUGU),
    (0x0C80, sanscript.KANNADA),
    (0x0D00, sanscript.MALAYALAM),
)


def _scheme_for(ch: str):
    cp = ord(ch)
    for start, scheme in _SCHEMES:
        if start <= cp < start + 0x80:
            return scheme
    return None


def _token_romanize(token: str) -> str:
    scheme = next((s for s in (_scheme_for(ch) for ch in token) if s is not None), None)
    if scheme is None:
        return token
    try:
        out = transliterate(token, scheme, sanscript.ITRANS)
    except Exception:
        return ""
    return " ".join("".join(ch if ord(ch) < 128 else " " for ch in out).split())


@functools.lru_cache(maxsize=200_000)
def _romanize_cached(s: str) -> str:
    return " ".join(filter(None, (_token_romanize(t) for t in s.split())))


def romanize(s: str) -> str:
    return _romanize_cached(normalize_name(s))
