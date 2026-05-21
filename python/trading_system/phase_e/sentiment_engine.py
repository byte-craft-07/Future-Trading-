# =============================================================
# FILE: trading_system/phase_e/sentiment_engine.py
# PURPOSE: Task 5 — News sentiment analysis layer
# =============================================================

import json
from datetime import datetime
from config_e import CONFIG, IST, LOGS_DIR_E

try:
    import feedparser
    FEED_AVAILABLE = True
except ImportError:
    FEED_AVAILABLE = False

try:
    from textblob import TextBlob
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False


def fetch_news_sentiment(config: dict) -> dict:
    """
    RSS news feeds se latest headlines fetch karo.
    TextBlob se sentiment score nikalo.

    Returns:
        Dict with score, signal, headlines list.
    """
    if not (FEED_AVAILABLE and TB_AVAILABLE):
        print("⚠️  feedparser/textblob not installed — neutral sentiment")
        return {
            "score": 0.0, "signal": "NEUTRAL",
            "confidence": 50.0, "headlines": [],
            "source": "fallback"
        }

    headlines = []
    scores    = []

    for url in config["news_sources"]:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:   # Top 5 per source
                title = entry.get("title", "")
                # Filter market-relevant keywords
                keywords = ["reliance", "nifty", "sensex",
                            "stock", "market", "bull", "bear",
                            "growth", "profit", "loss"]
                is_relevant = any(k in title.lower()
                                  for k in keywords)
                if is_relevant or True:    # Include all
                    blob  = TextBlob(title)
                    score = blob.sentiment.polarity   # -1 to +1
                    scores.append(score)
                    headlines.append({
                        "title": title[:80],
                        "score": round(score, 3)
                    })
        except Exception:
            continue

    if not scores:
        avg_score = 0.0
    else:
        # Weight recent news more
        weights   = list(range(len(scores), 0, -1))
        avg_score = float(
            sum(s * w for s, w in zip(scores, weights))
            / sum(weights)
        )

    # ── Convert to trading signal ──
    if   avg_score > 0.15 : signal = "BULLISH"
    elif avg_score > 0.05 : signal = "SLIGHTLY BULLISH"
    elif avg_score < -0.15: signal = "BEARISH"
    elif avg_score < -0.05: signal = "SLIGHTLY BEARISH"
    else                  : signal = "NEUTRAL"

    # Normalize to 0-1 (0.5 = neutral)
    confidence = (avg_score + 1) / 2 * 100

    result = {
        "score"     : round(avg_score, 4),
        "signal"    : signal,
        "confidence": round(confidence, 1),
        "headlines" : headlines[:5],
        "timestamp" : datetime.now(IST).strftime("%H:%M IST"),
        "source"    : "RSS feeds"
    }

    # Save sentiment log
    try:
        log_path = LOGS_DIR_E / "sentiment_log.json"
        with open(log_path, "w") as f:
            json.dump(result, f, indent=2)
    except Exception:
        pass

    print(f"\n💬 Sentiment: {signal} (score={avg_score:.3f}, "
          f"from {len(scores)} headlines)")

    return result

# EXPLANATION: Sentiment analysis news headlines se
# market ka mood samajhta hai. TextBlob polarity
# -1 (very negative) se +1 (very positive) tak.
# Yeh ensemble mein 15% weight deta hai.
# Agar koi news source down hai → neutral fallback.