import os
import re
import json
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any, Tuple

# ---------- FastAPI / DB ----------
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, conint
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

# ---------- NLP ----------
import nltk
from nltk.sentiment import vader
import spacy
from spacy.matcher import PhraseMatcher

# Optional: Transformers (set USE_TRANSFORMERS=true to enable if installed)
USE_TRANSFORMERS = os.getenv("USE_TRANSFORMERS", "false").lower() == "true"
if USE_TRANSFORMERS:
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    except Exception as e:
        print("[WARN] Transformers not available. Falling back to VADER.")
        USE_TRANSFORMERS = False

# ---------- Plotting ----------
import pandas as pd
import plotly.graph_objects as go

# ---------- Config ----------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "rural_tourism")

# For time bucketing in insights
DEFAULT_WINDOW_DAYS = int(os.getenv("WINDOW_DAYS", 60))
NEGATIVE_THRESHOLD_RATIO = float(os.getenv("NEGATIVE_THRESHOLD_RATIO", 0.3))  # 30%
MIN_MENTIONS_FOR_FLAG = int(os.getenv("MIN_MENTIONS_FOR_FLAG", 10))

# ---------- App ----------
app = FastAPI(title="Tourism Feedback & Sentiment API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- DB Client ----------
client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]
feedback_col = db["feedback"]

# ---------- Utilities ----------
# NOTE: Avoid custom ObjectId types in Pydantic models for OpenAPI stability.
# We will expose `_id` as a **string** instead.

# ---------- Pydantic Models ----------
class FeedbackIn(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    service_id: str = Field(..., description="Homestay or activity id")
    rating: conint(ge=1, le=5) = Field(..., description="1–5 stars")
    comment: str = Field(..., min_length=1)

class FeedbackOut(FeedbackIn):
    # Expose Mongo `_id` as a string to keep OpenAPI generation happy (Pydantic v2)
    id: str = Field(alias="_id")
    sentiment: str
    sentiment_score: float
    themes: List[str]
    timestamp: datetime

    model_config = {
        "populate_by_name": True,
    }

class InsightsOut(BaseModel):
    service_id: Optional[str]
    window_days: int
    total_feedback: int
    avg_rating: Optional[float]
    sentiment_distribution: Dict[str, int]
    daily_avg_rating: List[Dict[str, Any]]
    top_themes: List[Dict[str, Any]]
    common_keywords: List[Dict[str, Any]]
    flags: List[str]

# ---------- NLP Loaders ----------
_nltk_vader = None
_spacy_nlp = None
_matchers: Dict[str, PhraseMatcher] = {}
_transformer_pipe = None

THEME_SYNONYMS: Dict[str, List[str]] = {
    "cleanliness": ["clean", "cleanliness", "hygiene", "dirty", "dusty", "smelly", "spotless"],
    "hospitality": ["host", "hospitality", "welcoming", "friendly", "rude", "helpful", "staff"],
    "pricing": ["price", "pricing", "cost", "expensive", "overpriced", "cheap", "value"],
    "food": ["food", "meal", "breakfast", "lunch", "dinner", "taste", "cuisine", "restaurant"],
    "activities": ["activity", "trek", "hike", "tour", "boat", "craft", "dance", "workshop"],
    "location": ["location", "view", "scenic", "access", "near", "far", "remote"],
    "amenities": ["wifi", "internet", "ac", "air conditioning", "heater", "bathroom", "bed", "shower"],
    "safety": ["safe", "unsafe", "security", "danger", "theft", "night", "alone"],
}

STOPWORDS = set(
    """
    a an the and or but if while of on in at to for from by with about as into like through after over between out against during without before under around among
    i me my we our you your he him his she her it its they them their this that these those is are was were be been being have has had do does did will would can could should
    very more most less lot lots many much so such too not no yes ok okay great good bad
    """.split()
)

async def ensure_indexes():
    await feedback_col.create_index([("service_id", 1), ("timestamp", -1)])
    await feedback_col.create_index([("user_id", 1), ("service_id", 1)])


def _load_vader():
    global _nltk_vader
    if _nltk_vader is None:
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon')
        _nltk_vader = vader.SentimentIntensityAnalyzer()
    return _nltk_vader


def _load_spacy():
    global _spacy_nlp, _matchers
    if _spacy_nlp is None:
        try:
            _spacy_nlp = spacy.load("en_core_web_sm")
        except OSError:
            # try to download model on the fly
            from spacy.cli import download
            download("en_core_web_sm")
            _spacy_nlp = spacy.load("en_core_web_sm")

        # Build theme phrase matchers
        for theme, phrases in THEME_SYNONYMS.items():
            matcher = PhraseMatcher(_spacy_nlp.vocab, attr="LOWER")
            patterns = [ _spacy_nlp.make_doc(p) for p in phrases ]
            matcher.add(theme, patterns)
            _matchers[theme] = matcher
    return _spacy_nlp


def _load_transformer():
    global _transformer_pipe
    if USE_TRANSFORMERS and _transformer_pipe is None:
        # multilingual robust sentiment
        model_name = os.getenv("TRANSFORMERS_MODEL", "cardiffnlp/twitter-xlm-roberta-base-sentiment")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _transformer_pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, truncation=True)
    return _transformer_pipe


# ---------- NLP Core ----------

def analyze_sentiment(text: str) -> Tuple[str, float]:
    text = (text or "").strip()
    if not text:
        return ("neutral", 0.0)

    # Try transformers first if enabled
    pipe = _load_transformer()
    if pipe is not None:
        try:
            out = pipe(text[:512])[0]  # label e.g., 'positive' | 'neutral' | 'negative'
            label = out["label"].lower()
            score = float(out["score"]) if "score" in out else 0.0
            # Normalize labels e.g., some models return 'LABEL_0/1/2'
            if label.startswith("label_"):
                # Cardiff mapping: 0 -> negative, 1 -> neutral, 2 -> positive
                idx = int(label.split("_")[-1])
                label = ["negative", "neutral", "positive"][idx]
            return (label, score)
        except Exception as e:
            print("[WARN] Transformer inference failed:", e)

    # Fallback to VADER
    sia = _load_vader()
    scores = sia.polarity_scores(text)
    comp = scores.get("compound", 0.0)
    if comp >= 0.05:
        return ("positive", comp)
    elif comp <= -0.05:
        return ("negative", comp)
    else:
        return ("neutral", comp)


def extract_themes(text: str) -> List[str]:
    nlp = _load_spacy()
    doc = nlp(text.lower())
    found = set()
    for theme, matcher in _matchers.items():
        matches = matcher(doc)
        if matches:
            found.add(theme)
    # Fallback: simple keyword scan
    if not found:
        for theme, keys in THEME_SYNONYMS.items():
            for k in keys:
                if re.search(r"\b" + re.escape(k) + r"\b", text, flags=re.IGNORECASE):
                    found.add(theme)
                    break
    return sorted(found)


def extract_keywords(texts: List[str], top_k: int = 20) -> List[Tuple[str, int]]:
    from collections import Counter
    tokens = []
    for t in texts:
        for w in re.findall(r"[a-zA-Z]{3,}", (t or "").lower()):
            if w not in STOPWORDS:
                tokens.append(w)
    ctr = Counter(tokens)
    return ctr.most_common(top_k)


# ---------- Insert + Analyze ----------
async def upsert_feedback(doc: Dict[str, Any]) -> Dict[str, Any]:
    # Analyze
    sentiment, score = analyze_sentiment(doc.get("comment", ""))
    themes = extract_themes(doc.get("comment", ""))

    to_insert = {
        "user_id": doc["user_id"],
        "service_id": doc["service_id"],
        "rating": int(doc["rating"]),
        "comment": doc["comment"],
        "sentiment": sentiment,
        "sentiment_score": float(score),
        "themes": themes,
        "timestamp": doc.get("timestamp") or datetime.now(timezone.utc),
    }
    res = await feedback_col.insert_one(to_insert)
    to_insert["_id"] = str(res.inserted_id)
    return to_insert


# ---------- API Routes ----------
@app.on_event("startup")
async def on_startup():
    await ensure_indexes()
    _load_spacy()
    if USE_TRANSFORMERS:
        _load_transformer()
    else:
        _load_vader()


@app.post("/feedback", response_model=FeedbackOut)
async def create_feedback(item: FeedbackIn):
    try:
        saved = await upsert_feedback(item.dict())
        return saved
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feedback", response_model=List[FeedbackOut])
async def list_feedback(
    service_id: Optional[str] = None,
    user_id: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000)
):
    q: Dict[str, Any] = {}
    if service_id:
        q["service_id"] = service_id
    if user_id:
        q["user_id"] = user_id
    cursor = feedback_col.find(q).sort("timestamp", -1).limit(limit)
    docs = await cursor.to_list(length=limit)
    for d in docs:
        d["_id"] = str(d["_id"])  # ensure string for OpenAPI/JSON
    return docs


@app.get("/insights", response_model=InsightsOut)
async def insights(
    service_id: Optional[str] = None,
    window_days: int = Query(DEFAULT_WINDOW_DAYS, ge=7, le=365)
):
    since = datetime.now(timezone.utc) - timedelta(days=window_days)
    q: Dict[str, Any] = {"timestamp": {"$gte": since}}
    if service_id:
        q["service_id"] = service_id

    docs = await feedback_col.find(q).to_list(length=100000)
    total = len(docs)
    if total == 0:
        return InsightsOut(
            service_id=service_id,
            window_days=window_days,
            total_feedback=0,
            avg_rating=None,
            sentiment_distribution={"positive":0, "neutral":0, "negative":0},
            daily_avg_rating=[],
            top_themes=[],
            common_keywords=[],
            flags=[],
        )

    # Aggregate
    df = pd.DataFrame(docs)
    df['date'] = pd.to_datetime(df['timestamp']).dt.date

    avg_rating = round(float(df['rating'].mean()), 2)
    sent_counts = df['sentiment'].value_counts().to_dict()
    for s in ["positive", "neutral", "negative"]:
        sent_counts.setdefault(s, 0)

    daily = (
        df.groupby('date')['rating']
        .mean()
        .reset_index()
        .rename(columns={'rating':'avg_rating'})
    )

    # Themes
    theme_series = df['themes'].explode()
    theme_counts = theme_series.value_counts().head(15).reset_index()
    theme_counts.columns = ['theme', 'count']

    # Keywords
    common_kw = extract_keywords(df['comment'].tolist(), top_k=20)

    # Flags & Recommendations logic
    flags = []
    # For each theme, if negative ratio high -> flag
    for theme in THEME_SYNONYMS.keys():
        theme_mask = df['themes'].apply(lambda lst: theme in (lst or []))
        td = df[theme_mask]
        if len(td) >= MIN_MENTIONS_FOR_FLAG:
            neg_ratio = (td['sentiment'] == 'negative').mean() if len(td) else 0.0
            if neg_ratio >= NEGATIVE_THRESHOLD_RATIO:
                flags.append(
                    f"High negative sentiment about {theme} (ratio {neg_ratio:.0%} over {len(td)} mentions). Consider immediate improvements."
                )

    # Overall low rating
    if avg_rating < 3.5 and total >= 25:
        flags.append("Average rating below 3.5. Review pricing, cleanliness, and hospitality training.")

    # Build response
    return InsightsOut(
        service_id=service_id,
        window_days=window_days,
        total_feedback=total,
        avg_rating=avg_rating,
        sentiment_distribution={k:int(v) for k,v in sent_counts.items()},
        daily_avg_rating=[{"date": str(r['date']), "avg_rating": round(float(r['avg_rating']),2)} for _, r in daily.iterrows()],
        top_themes=[{"theme": str(r['theme']), "count": int(r['count'])} for _, r in theme_counts.iterrows()],
        common_keywords=[{"keyword": k, "count": int(v)} for k, v in common_kw],
        flags=flags,
    )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(service_id: Optional[str] = None, window_days: int = DEFAULT_WINDOW_DAYS):
    # Reuse /insights JSON and render Plotly
    data = await insights(service_id=service_id, window_days=window_days)
    if data.total_feedback == 0:
        return HTMLResponse("<h2>No feedback in the selected window.</h2>")

    # Build charts
    # Line: daily avg rating
    x_dates = [p['date'] for p in data.daily_avg_rating]
    y_avg = [p['avg_rating'] for p in data.daily_avg_rating]
    fig_rating = go.Figure()
    fig_rating.add_trace(go.Scatter(x=x_dates, y=y_avg, mode='lines+markers', name='Avg Rating'))
    fig_rating.update_layout(title='Daily Average Rating', xaxis_title='Date', yaxis_title='Avg Rating (1-5)', yaxis=dict(range=[1,5]))

    # Bar: sentiment distribution
    s_labels = list(data.sentiment_distribution.keys())
    s_values = list(data.sentiment_distribution.values())
    fig_sent = go.Figure()
    fig_sent.add_trace(go.Bar(x=s_labels, y=s_values, name='Sentiment'))
    fig_sent.update_layout(title='Sentiment Distribution')

    # Bar: top themes
    t_labels = [t['theme'] for t in data.top_themes]
    t_values = [t['count'] for t in data.top_themes]
    fig_theme = go.Figure()
    fig_theme.add_trace(go.Bar(x=t_labels, y=t_values, name='Themes'))
    fig_theme.update_layout(title='Top Themes')

    # Table: common keywords
    kw_tbl = go.Figure(data=[go.Table(
        header=dict(values=['Keyword', 'Count']),
        cells=dict(values=[[k['keyword'] for k in data.common_keywords], [k['count'] for k in data.common_keywords]])
    )])

    html = f"""
    <html>
      <head>
        <meta charset='utf-8'/>
        <title>Tourism Feedback Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
          body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial; margin: 24px; }}
          .grid {{ display: grid; grid-template-columns: 1fr; gap: 28px; }}
          @media (min-width: 1100px) {{ .grid {{ grid-template-columns: 1fr 1fr; }} }}
          .card {{ box-shadow: 0 8px 24px rgba(0,0,0,0.08); border-radius: 16px; padding: 16px; }}
          h1 {{ margin-bottom: 8px; }}
          .flags {{ color: #b00020; }}
        </style>
      </head>
      <body>
        <h1>Tourism Feedback Dashboard</h1>
        <p>Service: <b>{data.service_id or 'All services'}</b> • Window: last <b>{data.window_days}</b> days • Total feedback: <b>{data.total_feedback}</b> • Avg rating: <b>{data.avg_rating}</b></p>
        <div class="grid">
          <div class="card" id="rating"></div>
          <div class="card" id="sentiment"></div>
          <div class="card" id="themes"></div>
          <div class="card" id="keywords"></div>
        </div>
        <div class="card flags">
          <h3>Flags & Recommendations</h3>
          <ul>
            {''.join(f'<li>{f}</li>' for f in data.flags) or '<li>No flags. Keep up the good work!</li>'}
          </ul>
        </div>
        <script>
          var figRating = {json.dumps(go.Figure(fig_rating).to_plotly_json())};
          var figSent = {json.dumps(go.Figure(fig_sent).to_plotly_json())};
          var figTheme = {json.dumps(go.Figure(fig_theme).to_plotly_json())};
          var figKw = {json.dumps(kw_tbl.to_plotly_json())};
          Plotly.newPlot('rating', figRating.data, figRating.layout);
          Plotly.newPlot('sentiment', figSent.data, figSent.layout);
          Plotly.newPlot('themes', figTheme.data, figTheme.layout);
          Plotly.newPlot('keywords', figKw.data, figKw.layout);
        </script>
      </body>
    </html>
    """
    return HTMLResponse(html)


@app.get("/health")
async def health():
    return {"status": "ok", "transformers": USE_TRANSFORMERS}


# ---------- Maintenance: recompute sentiment/themes if logic changes ----------
@app.post("/recompute")
async def recompute(service_id: Optional[str] = None):
    q: Dict[str, Any] = {}
    if service_id:
        q["service_id"] = service_id
    cursor = feedback_col.find(q)
    updated = 0
    async for doc in cursor:
        s, sc = analyze_sentiment(doc.get("comment", ""))
        th = extract_themes(doc.get("comment", ""))
        await feedback_col.update_one({"_id": doc["_id"]}, {"$set": {"sentiment": s, "sentiment_score": sc, "themes": th}})
        updated += 1
    return {"updated": updated}


# ---------- Run (for local) ----------
# uvicorn app:app --reload --port 8080