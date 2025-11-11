from __future__ import annotations
import re
import math
from typing import Dict, List
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_WORD_RE = re.compile(r"[A-Za-z']+")
_URL_RE  = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)

def _words(text: str) -> List[str]:
    return _WORD_RE.findall(text or "")

def _sentences(text: str) -> List[str]:
    sents = re.split(r"[.!?]+", text or "")
    return [s.strip() for s in sents if s.strip()]

def _uppercase_word(w: str) -> bool:
    # Consider words with >= 3 letters and all letters uppercase
    letters = [c for c in w if c.isalpha()]
    return len(letters) >= 3 and all(c.isupper() for c in letters)

def style_features(text: str) -> Dict[str, float]:
    t = text or ""
    chars = len(t)
    ws = _words(t)
    sents = _sentences(t)

    word_count = len(ws)
    unique_words = len(set(w.lower() for w in ws))
    stop_count = sum(1 for w in ws if w.lower() in ENGLISH_STOP_WORDS)
    punct_count = len(re.findall(r"[^\w\s]", t))
    upper_char_count = sum(1 for c in t if c.isalpha() and c.isupper())
    upper_word_count = sum(1 for w in ws if _uppercase_word(w))
    digit_count = sum(1 for c in t if c.isdigit())
    exclam = t.count("!")
    quest  = t.count("?")
    url_ct = len(_URL_RE.findall(t))
    comma  = t.count(",")
    period = t.count(".")
    colon_semi = t.count(":") + t.count(";")
    quotes = t.count('"') + t.count("'")
    ellipses = t.count("...")  # naive but effective
    parens = t.count("(") + t.count(")")

    avg_word_len = (sum(len(w) for w in ws) / word_count) if word_count else 0.0
    ttr = (unique_words / word_count) if word_count else 0.0
    stop_ratio = (stop_count / word_count) if word_count else 0.0
    punct_ratio = (punct_count / chars) if chars else 0.0
    upper_char_ratio = (upper_char_count / chars) if chars else 0.0
    upper_word_ratio = (upper_word_count / word_count) if word_count else 0.0
    digit_ratio = (digit_count / chars) if chars else 0.0
    avg_sent_len_words = (word_count / len(sents)) if sents else 0.0

    return {
        "char_len": float(chars),
        "word_count": float(word_count),
        "avg_word_len": float(avg_word_len),
        "unique_word_ratio": float(ttr),
        "stopword_ratio": float(stop_ratio),
        "punct_ratio": float(punct_ratio),
        "uppercase_char_ratio": float(upper_char_ratio),
        "uppercase_word_ratio": float(upper_word_ratio),
        "digit_ratio": float(digit_ratio),
        "url_count": float(url_ct),
        "exclamation_count": float(exclam),
        "question_count": float(quest),
        "comma_count": float(comma),
        "period_count": float(period),
        "colon_semi_count": float(colon_semi),
        "quote_count": float(quotes),
        "ellipsis_count": float(ellipses),
        "parenthesis_count": float(parens),
        "avg_sentence_length_words": float(avg_sent_len_words),
    }

