from __future__ import annotations
import re
from typing import Dict, List

_WORD_RE = re.compile(r"[A-Za-z']+")
_SENT_RE = re.compile(r"[.!?]+")

def _words(text: str) -> List[str]:
    return _WORD_RE.findall(text or "")

def _sentences(text: str) -> int:
    return len([s for s in re.split(_SENT_RE, text or "") if s.strip()])

def _syllables_in_word(word: str) -> int:
    w = word.lower()
    if not w:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_is_vowel = False
    for ch in w:
        is_vowel = ch in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel
    # silent 'e'
    if w.endswith("e") and count > 1:
        count -= 1
    # 'le' ending (e.g., table)
    if w.endswith("le") and len(w) > 2 and w[-3] not in vowels:
        count += 1
    return max(count, 1)

def _syllable_count(text: str) -> int:
    return sum(_syllables_in_word(w) for w in _words(text))

def _complex_word_count(text: str) -> int:
    # Words with 3+ syllables (simple heuristic)
    return sum(1 for w in _words(text) if _syllables_in_word(w) >= 3)

def complexity_features(text: str) -> Dict[str, float]:
    words = _words(text)
    n_words = len(words)
    n_sents = _sentences(text)
    n_sents = n_sents if n_sents > 0 else 1  # avoid div by zero

    syllables = _syllable_count(text)
    complex_words = _complex_word_count(text)

    words_per_sent = n_words / n_sents if n_sents else 0.0
    syll_per_word = syllables / n_words if n_words else 0.0

    # Flesch Reading Ease
    fre = 206.835 - 1.015 * words_per_sent - 84.6 * syll_per_word
    # Flesch-Kincaid Grade Level
    fkgl = 0.39 * words_per_sent + 11.8 * syll_per_word - 15.59
    # Gunning Fog Index
    gfi = 0.4 * (words_per_sent + 100.0 * (complex_words / n_words if n_words else 0.0))

    return {
        "flesch_reading_ease": float(fre),
        "flesch_kincaid_grade": float(fkgl),
        "gunning_fog_index": float(gfi),
    }
