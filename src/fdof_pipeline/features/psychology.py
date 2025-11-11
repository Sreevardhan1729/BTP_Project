from __future__ import annotations
from typing import Dict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

def psychology_features(text: str) -> Dict[str, float]:
    scores = _analyzer.polarity_scores(text or "")
    # Ensure stable float outputs
    return {
        "vader_pos": float(scores.get("pos", 0.0)),
        "vader_neu": float(scores.get("neu", 0.0)),
        "vader_neg": float(scores.get("neg", 0.0)),
        "vader_compound": float(scores.get("compound", 0.0)),
    }
