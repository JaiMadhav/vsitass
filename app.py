import os
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, conint, constr
from pymongo import MongoClient
from bson import ObjectId

# --- Optional: Gemini only for human-friendly summary text (NOT for planning) ---
USE_SUMMARY_LLM = os.getenv("USE_SUMMARY_LLM", "false").lower() == "true"
try:
    import google.generativeai as genai
except Exception:
    genai = None
# If you enable USE_SUMMARY_LLM, set GOOGLE_API_KEY and optionally GEMINI_MODEL
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# ========= ENV & DB =========
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
if not MONGO_URI or not DB_NAME:
    raise RuntimeError("Please set MONGO_URI and DB_NAME in your .env")
mongo = MongoClient(MONGO_URI)
db = mongo[DB_NAME]

# ========= REQUEST DTO =========
class ItineraryRequest(BaseModel):
    village: constr(min_length=1)
    stateId: Optional[str] = None
    budgetTotalINR: float = Field(gt=0)
    people: conint(ge=1)
    startDate: constr(pattern=r"^\d{4}-\d{2}-\d{2}$")
    endDate: constr(pattern=r"^\d{4}-\d{2}-\d{2}$")
    interests: List[str] = Field(default_factory=list)

# ========= OUTPUT SCHEMA (server-enforced) =========
ITINERARY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "currency", "total_estimated_cost", "days"],
    "properties": {
        "summary": {"type": "string", "minLength": 1},
        "currency": {"type": "string", "minLength": 3, "maxLength": 3},
        "total_estimated_cost": {"type": "number", "minimum": 0},
        "days": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["date", "items", "day_estimated_cost"],
                "properties": {
                    "date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"},
                    "day_estimated_cost": {"type": "number", "minimum": 0},
                    "items": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "type",
                                "ref_id",
                                "name",
                                "start_time",
                                "end_time",
                                "notes",
                                "cost_breakdown",
                            ],
                            "properties": {
                                "type": {"type": "string", "enum": ["activity", "event", "package", "free_time"]},
                                "ref_id": {"type": ["string", "null"]},
                                "name": {"type": "string"},
                                "start_time": {"type": "string", "pattern": r"^[0-2]\d:[0-5]\d$"},
                                "end_time": {"type": "string", "pattern": r"^[0-2]\d:[0-5]\d$"},
                                "notes": {"type": "string"},
                                "cost_breakdown": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": ["basis", "quantity", "unit_price", "subtotal"],
                                    "properties": {
                                        "basis": {"type": "string", "enum": ["per_person", "per_group", "free", "package"]},
                                        "quantity": {"type": "number", "minimum": 0},
                                        "unit_price": {"type": "number", "minimum": 0},
                                        "subtotal": {"type": "number", "minimum": 0},
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    },
}

# ========= UTILS =========
def to_date(iso: str) -> datetime:
    return datetime.strptime(iso, "%Y-%m-%d")

def fmt_date_yymmdd(val) -> str:
    if isinstance(val, datetime):
        return val.strftime("%Y-%m-%d")
    if isinstance(val, str):
        try:
            return datetime.fromisoformat(val[:19]).strftime("%Y-%m-%d")
        except Exception:
            return val
    return str(val)

def as_money(x) -> float:
    try: return float(x)
    except: return 0.0

def time_to_min(t: str) -> int:
    # "HH:MM" -> minutes
    h, m = t.split(":")
    return int(h)*60 + int(m)

def overlaps(a_start: str, a_end: str, b_start: str, b_end: str) -> bool:
    a1, a2 = time_to_min(a_start), time_to_min(a_end)
    b1, b2 = time_to_min(b_start), time_to_min(b_end)
    return not (a2 <= b1 or b2 <= a1)

def _free_block() -> Dict[str, Any]:
    return {
        "type": "free_time",
        "ref_id": None,
        "name": "Free time",
        "start_time": "16:00",
        "end_time": "18:00",
        "notes": "Relax & explore locally.",
        "cost_breakdown": {"basis": "free", "quantity": 0, "unit_price": 0, "subtotal": 0},
    }

def price_for(basis: str, unit_price: float, people: int) -> float:
    if basis == "per_person": return unit_price * people
    if basis in ("per_group", "package"): return unit_price
    return 0.0

def recompute_totals(itinerary: dict) -> dict:
    total = 0.0
    for day in itinerary["days"]:
        day_total = 0.0
        for item in day["items"]:
            cb = item["cost_breakdown"]
            if cb["basis"] == "per_person":
                cb["quantity"] = max(cb.get("quantity", 0) or 0, 1)  # ensure >=1
            sub = float(cb.get("quantity", 0) or 0) * float(cb.get("unit_price", 0) or 0)
            cb["subtotal"] = round(sub, 2)
            day_total += cb["subtotal"]
        day["day_estimated_cost"] = round(day_total, 2)
        total += day_total
    itinerary["total_estimated_cost"] = round(total, 2)
    return itinerary

# ========= ALGORITHMIC SCHEDULER =========
# Strategy:
# 1) Optional package coverage: pick ONE package (best days covered per INR) if it fits budget and window.
# 2) Coverage-first greedy: For each day, place at least one item (prefer free or cheapest event; else free/cheap activity).
# 3) Round-robin fill to 3 items/day without overlaps (prefer higher popularity, then cheaper), while budget remains.
def build_algorithmic_itinerary(
    people: int,
    start: datetime,
    end: datetime,
    budget: float,
    events_block: List[dict],
    activities_block: List[dict],
    packages_block: List[dict],
    minimize_free_days: bool = True,
    allow_flex_over_budget: bool = False,  # set True if you want ≤ budget * 1.10 instead
) -> dict:

    day_list = []
    d = start
    while d.date() <= end.date():
        day_list.append(d.date().isoformat())
        d += timedelta(days=1)

    # Map events to date coverage
    events_by_day: Dict[str, List[dict]] = {ds: [] for ds in day_list}
    for e in events_block:
        s, t = e["start_date"], e["end_date"]
        for ds in day_list:
            if s <= ds <= t:
                ev = {
                    "type": "event",
                    "ref_id": e["_id"],
                    "name": e["title"],
                    "start_time": "10:00",
                    "end_time": "12:00",
                    "notes": "Scheduled event block.",
                    "basis": "per_group" if e.get("price", 0) > 0 else "free",
                    "unit_price": float(e.get("price", 0) or 0),
                    "pop": 50.0,  # synthetic pop for events
                }
                events_by_day[ds].append(ev)

    # Clean & normalize activities
    acts: List[dict] = []
    for a in activities_block:
        basis = a.get("price_type", "free")
        acts.append({
            "type": "activity",
            "ref_id": a["_id"],
            "name": a["name"],
            "start_time": a.get("start_time", "09:00"),
            "end_time": a.get("end_time", "11:00"),
            "notes": "Popular activity.",
            "basis": basis,
            "unit_price": float(a.get("price", 0) or 0),
            "pop": float(a.get("popularity_score", 0) or 0),
        })

    # Optional: choose one package for multi-day coverage
    chosen_package = None
    if packages_block:
        # rank by "days_covered / (price+1)" to maximize coverage per INR
        candidates = sorted(
            packages_block,
            key=lambda p: (-(int(p.get("duration_days", 1) or 1) / (float(p.get("price", 0) or 0) + 1.0)),
                           float(p.get("price", 0) or 0))
        )
        for p in candidates:
            price = float(p.get("price", 0) or 0)
            if price <= budget:  # must fit hard budget
                chosen_package = p
                budget -= price
                # schedule for first K days (or all if fewer)
                k = min(int(p.get("duration_days", 1) or 1), len(day_list))
                for i in range(k):
                    ds = day_list[i]
                    # whole-day package blocks other items; we'll still allow extra evening slot if needed
                    events_by_day[ds].insert(0, {
                        "type": "package",
                        "ref_id": p["_id"],
                        "name": p["package_name"],
                        "start_time": "09:00",
                        "end_time": "18:00",
                        "notes": f"Day {i+1} of {k} for package.",
                        "basis": "package",
                        "unit_price": price if i == 0 else 0.0,  # charge once
                        "pop": 80.0,  # high priority
                    })
                break  # choose only one package

    # Build empty skeleton
    days_out: List[dict] = [{"date": ds, "day_estimated_cost": 0, "items": []} for ds in day_list]

    # ---- 1) Coverage-first greedy (ensure >=1 item/day when possible) ----
    for idx, ds in enumerate(day_list):
        day_items = days_out[idx]["items"]
        remaining = budget

        # If package already consumes day with 09-18 block, keep it as first item
        pkg_item = next((x for x in events_by_day[ds] if x["type"] == "package"), None)
        if pkg_item:
            cost = price_for(pkg_item["basis"], pkg_item["unit_price"], people)
            if cost <= budget:
                budget -= cost
                day_items.append(_mk_output_item(pkg_item, people))
                continue  # package day already covered (optionally filled later with evening slot)

        # Prefer free or cheapest event on that day
        day_events = sorted(events_by_day[ds], key=lambda x: (price_for(x["basis"], x["unit_price"], people), -x["pop"]))
        placed = False
        for ev in day_events:
            cost = price_for(ev["basis"], ev["unit_price"], people)
            if cost <= budget and not _conflict(day_items, ev):
                budget -= cost
                day_items.append(_mk_output_item(ev, people))
                placed = True
                break

        if placed:
            continue

        # Else take a free/high-pop or cheap activity
        for act in sorted(acts, key=lambda x: (-x["pop"], price_for(x["basis"], x["unit_price"], people))):
            cost = price_for(act["basis"], act["unit_price"], people)
            if cost <= budget and not _conflict(day_items, act):
                budget -= cost
                day_items.append(_mk_output_item(act, people))
                placed = True
                break

        if not placed and minimize_free_days:
            # Try any free activity
            for act in sorted(acts, key=lambda x: (-x["pop"], x["unit_price"])):
                if act["basis"] == "free" and not _conflict(day_items, act):
                    day_items.append(_mk_output_item(act, people))
                    placed = True
                    break

        if not placed:
            day_items.append(_free_block())

    # ---- 2) Round-robin fill up to 3 items/day without overlap, while budget remains ----
    day_index = 0
    # Build a global priority list of candidates to try (activities first, then events),
    # favoring popularity then low cost to stay within budget.
    all_candidates = (
        sorted(acts, key=lambda x: (-x["pop"], price_for(x["basis"], x["unit_price"], people)))
        +
        sorted([e for ds in day_list for e in events_by_day[ds] if e["type"] != "package"],
               key=lambda x: (-x["pop"], price_for(x["basis"], x["unit_price"], people)))
    )

    while day_index < len(day_list):
        if budget <= 0:
            break
        ds = day_list[day_index]
        day = next(d for d in days_out if d["date"] == ds)
        if len(day["items"]) >= 3:
            day_index += 1
            continue

        placed = False
        for cand in all_candidates:
            # Avoid placing the same named item twice in a day
            if any(cand["name"] == it["name"] for it in day["items"] if it["type"] != "free_time"):
                continue
            # Avoid placing day-specific events on the wrong day
            if cand["type"] == "event":
                if cand not in events_by_day[ds]:
                    continue
            # Respect overlaps
            if _conflict(day["items"], cand):
                continue
            cost = price_for(cand["basis"], cand["unit_price"], people)
            # Strict budget: DO NOT exceed
            if cost <= budget:
                budget -= cost
                day["items"].append(_mk_output_item(cand, people))
                placed = True
                if len(day["items"]) >= 3:
                    break

        # If could not place anything and day still has <1 real item, ensure at least a free block exists
        if not placed and len(day["items"]) == 0:
            day["items"].append(_free_block())

        # Move to next day in round-robin
        day_index += 1

    # Replace leftover empty lists with a free block (shouldn’t happen often)
    for day in days_out:
        if not day["items"]:
            day["items"].append(_free_block())

    itinerary = {
        "summary": "",  # fill next
        "currency": "INR",
        "total_estimated_cost": 0,
        "days": days_out
    }
    recompute_totals(itinerary)

    # ---- 3) Human summary (no LLM) ----
    itinerary["summary"] = summarize_human(itinerary, people, len(day_list), chosen_package)

    # ---- 4) Optional: nicer summary via Gemini (content-only, no numbers changed) ----
    if USE_SUMMARY_LLM and genai and os.getenv("GOOGLE_API_KEY"):
        try:
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            model = genai.GenerativeModel(GEMINI_MODEL)
            brief = json.dumps({
                "village_days": [d["date"] for d in days_out],
                "people": people,
                "total_cost": itinerary["total_estimated_cost"],
                "items_by_day": [
                    [{"type": it["type"], "name": it["name"], "time": f"{it['start_time']}-{it['end_time']}",
                      "cost": it["cost_breakdown"]["subtotal"]} for it in day["items"]]
                    for day in days_out
                ]
            })
            prompt = (
                "Write a concise, friendly trip summary (3–5 sentences) for a rural India itinerary. "
                "Emphasize that every day is utilized with minimal free time, and that the plan fits the budget. "
                "Keep it plain text, no bullets. Input JSON:\n" + brief
            )
            resp = model.generate_content(prompt)
            if resp and (resp.text or (resp.candidates and resp.candidates[0].content.parts)):
                text = resp.text or resp.candidates[0].content.parts[0].text
                itinerary["summary"] = text.strip()
        except Exception:
            pass

    return itinerary


def _mk_output_item(src: dict, people: int) -> dict:
    # Build output item with cost_breakdown consistent with schema
    basis = src["basis"]
    unit = float(src.get("unit_price", 0) or 0)
    qty = people if basis == "per_person" else (1 if basis in ("per_group", "package") else 0)
    return {
        "type": src["type"],
        "ref_id": src.get("ref_id"),
        "name": src["name"],
        "start_time": src.get("start_time", "09:00"),
        "end_time": src.get("end_time", "11:00"),
        "notes": src.get("notes", ""),
        "cost_breakdown": {
            "basis": basis,
            "quantity": qty,
            "unit_price": unit,
            "subtotal": round(qty * unit, 2),
        },
    }

def _conflict(day_items: List[dict], cand: dict) -> bool:
    # free_time never conflicts (we still keep to <=3 items logic)
    if cand["type"] == "free_time":
        return False
    for it in day_items:
        if it["type"] == "free_time":
            # allow replacing a free block; but do not “overlap” with it
            # treat free_time as occupying 16:00–18:00; avoid overlap if cand overlaps that
            if overlaps("16:00", "18:00", cand["start_time"], cand["end_time"]):
                return True
            continue
        if overlaps(it["start_time"], it["end_time"], cand["start_time"], cand["end_time"]):
            return True
    return False

def summarize_human(it: dict, people: int, days_count: int, chosen_package: Optional[dict]) -> str:
    items = [x for d in it["days"] for x in d["items"] if x["type"] != "free_time"]
    events = [x for x in items if x["type"] == "event"]
    acts   = [x for x in items if x["type"] == "activity"]
    pkgs   = [x for x in items if x["type"] == "package"]
    used_days = sum(1 for d in it["days"] if any(x["type"] != "free_time" for x in d["items"]))
    free_days = days_count - used_days
    msg = [
        f"This plan uses all {days_count} day(s) with minimal idle time.",
        f"You’ll experience {len(events)} event(s), {len(acts)} activity(ies){' and 1 package' if pkgs else ''}.",
        f"Designed for {people} traveler(s), total spend is ₹{int(it['total_estimated_cost']):,} within your budget.",
    ]
    if free_days > 0:
        msg.append(f"Only {free_days} day(s) keep a short free block for rest or local wandering.")
    if chosen_package:
        msg.append(f"The '{chosen_package['package_name']}' package anchors the start of your trip.")
    return " ".join(msg)

# ========= CORE PLANNING PIPELINE =========
def plan_itinerary(payload: ItineraryRequest) -> dict:
    # 1) resolve state (optional)
    state = None
    if payload.stateId:
        try:
            state = db.states.find_one({"_id": ObjectId(payload.stateId)})
            if not state:
                raise HTTPException(status_code=404, detail="State not found")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid stateId")

    # 2) resolve village by name (+ state filter if present)
    village_query = {"village_name": {"$regex": payload.village, "$options": "i"}}
    if state:
        village_query["state_id"] = state["_id"]
    village = db.villages.find_one(village_query)
    if not village:
        raise HTTPException(status_code=404, detail="Village not found")

    # 3) parse window
    start = to_date(payload.startDate)
    end = to_date(payload.endDate)
    if end < start:
        raise HTTPException(status_code=400, detail="endDate must be after startDate")

    # 4) fetch supply (only from DB)
    events = list(
        db.events.find(
            {"village_id": village["_id"], "start_date": {"$lte": end}, "end_date": {"$gte": start}}
        ).sort("start_date", 1).limit(100)
    )
    activities = list(
        db.activities.find(
            {"village_id": village["_id"], "$or": [
                {"category": {"$in": payload.interests}},
                {"tags": {"$in": payload.interests}},
            ]} if payload.interests else {"village_id": village["_id"]}
        ).sort([("popularity_score", -1)]).limit(200)
    )
    packages = list(
        db.packages.find({"village_id": village["_id"]}).sort("price", 1).limit(50)
    )

    # 5) slim blocks
    events_block = [
        {"_id": str(e["_id"]), "title": e["title"],
         "start_date": fmt_date_yymmdd(e.get("start_date")),
         "end_date": fmt_date_yymmdd(e.get("end_date")),
         "price": as_money(e.get("price", 0))}
        for e in events
    ]
    activities_block = [
        {"_id": str(a["_id"]), "name": a["name"], "start_time": a.get("start_time", "09:00"),
         "end_time": a.get("end_time", "11:00"),
         "price": as_money(a.get("price", 0)), "price_type": a.get("price_type", "free"),
         "category": a.get("category"), "tags": a.get("tags", []),
         "popularity_score": as_money(a.get("popularity_score", 0))}
        for a in activities
    ]
    packages_block = [
        {"_id": str(p["_id"]), "package_name": p["package_name"], "price": as_money(p.get("price", 0)),
         "duration_days": int(p.get("duration", 1) or 1)}
        for p in packages
    ]

    # 6) algorithmic plan (no LLM for schedule or cost)
    itinerary = build_algorithmic_itinerary(
        people=payload.people,
        start=start,
        end=end,
        budget=payload.budgetTotalINR,
        events_block=events_block,
        activities_block=activities_block,
        packages_block=packages_block,
        minimize_free_days=True,
        allow_flex_over_budget=False,   # set True if you ever want to allow +10%
    )

    # final math guard (already respected)
    recompute_totals(itinerary)
    if itinerary["total_estimated_cost"] > payload.budgetTotalINR:
        # As a hard guard, if somehow exceeded, drop last paid items until within budget.
        # (Shouldn’t trigger, but safe.)
        for day in reversed(itinerary["days"]):
            for i in range(len(day["items"]) - 1, -1, -1):
                cb = day["items"][i]["cost_breakdown"]
                if cb["subtotal"] > 0:
                    del day["items"][i]
                    recompute_totals(itinerary)
                    if itinerary["total_estimated_cost"] <= payload.budgetTotalINR:
                        break
            if itinerary["total_estimated_cost"] <= payload.budgetTotalINR:
                break
        # ensure day not empty
        for day in itinerary["days"]:
            if not day["items"]:
                day["items"].append(_free_block())
        recompute_totals(itinerary)

    # 7) Respond
    state_name = None
    if village.get("state_id"):
        st = db.states.find_one({"_id": village["state_id"]})
        if st: state_name = st.get("state_name")
    elif state:
        state_name = state.get("state_name")

    return {
        "village": village["village_name"],
        "state": state_name,
        "params": payload.model_dump(),
        "itinerary": itinerary,
    }

# ========= HTML VIEW =========
def _esc(s: str) -> str:
    return (s or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def rs(x) -> str:
    try:
        n = float(x); return f"₹{n:,.0f}"
    except: return "₹0"

def render_itinerary_html(result: dict) -> str:
    it = result["itinerary"]; p = result["params"]
    header = f"""
    <header class="header">
      <h1>Itinerary for {_esc(result['village'])} {f"({_esc(result['state'])})" if result.get('state') else ""}</h1>
      <p><strong>Dates:</strong> {p['startDate']} → {p['endDate']} &nbsp;|&nbsp;
         <strong>People:</strong> {p['people']} &nbsp;|&nbsp;
         <strong>Budget:</strong> {rs(p['budgetTotalINR'])}</p>
      <p><strong>Interests:</strong> {_esc(", ".join(p.get("interests", [])) or "(none)")}</p>
    </header>
    """

    days_html = ""
    for day in it["days"]:
        items_li = ""
        for item in day["items"]:
            cb = item["cost_breakdown"]
            items_li += f"""
              <li class="it">
                <div class="it-title"><span class="badge">{_esc(item['type'].title())}</span> {_esc(item['name'])}</div>
                <div class="it-meta">{_esc(item['start_time'])}–{_esc(item['end_time'])}
                &nbsp;•&nbsp; <em>{_esc(cb['basis'])}</em>
                &nbsp;•&nbsp; qty {cb.get('quantity',0)} × {rs(cb.get('unit_price',0))} = <strong>{rs(cb.get('subtotal',0))}</strong></div>
              </li>
            """
        days_html += f"""
          <section class="day">
            <div class="day-head">
              <h3>📅 {_esc(day['date'])}</h3>
              <div class="day-total">Day total: <strong>{rs(day['day_estimated_cost'])}</strong></div>
            </div>
            <ul class="items">{items_li}</ul>
          </section>
        """

    html = f"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Rural Trip Planner</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  :root {{ --bg:#0b1020; --card:#121933; --ink:#e9eefc; --muted:#9db2f9; --pill:#2b3570; --accent:#7aa2ff; }}
  body {{ margin:0; background:var(--bg); color:var(--ink); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; }}
  .wrap {{ max-width: 980px; margin: 24px auto 64px; padding: 0 16px; }}
  .card {{ background: var(--card); border:1px solid #1d2857; border-radius: 16px; padding: 20px; }}
  .header h1 {{ margin: 0 0 6px; font-size: 28px; }}
  .header p {{ margin:4px 0; color: var(--muted); }}
  .summary {{ white-space: pre-wrap; line-height: 1.5; margin: 8px 0 16px; }}
  .day {{ margin-top: 18px; }}
  .day-head {{ display:flex; justify-content: space-between; align-items: center; gap:16px; padding-bottom:6px; border-bottom: 1px dashed #2a387c; }}
  .items {{ list-style: none; padding:0; margin: 10px 0 0; display: grid; gap: 10px; }}
  .it {{ background:#0f1630; border:1px solid #22327a; border-radius:12px; padding:10px 12px; }}
  .it-title {{ font-weight: 600; margin-bottom: 4px; }}
  .badge {{ background: var(--pill); color: var(--ink); padding: 2px 8px; border-radius: 999px; font-size: 12px; margin-right: 6px; }}
  .it-meta {{ color: var(--muted); font-size: 14px; }}
  .grand {{ margin-top: 18px; display:flex; justify-content: space-between; align-items:center; gap:12px; }}
  .grand-total {{ font-size: 18px; background:#0f1630; border:1px solid #22327a; padding:10px 14px; border-radius: 10px; }}
  .btn {{ display:inline-block; background:#1d2857; border:1px solid #2a387c; color:#e9eefc; padding:10px 14px; border-radius:10px; text-decoration:none; }}
  .btn:hover {{ background:#22327a; }}
</style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      {header}
      <h2>Summary</h2>
      <div class="summary">{_esc(it['summary'])}</div>
      {days_html}
      <section class="grand">
        <div class="grand-total">Trip total: <strong>{rs(it['total_estimated_cost'])}</strong> ({_esc(it['currency'])})</div>
        <div><a class="btn" href="/plan">Plan another</a> &nbsp; <a class="btn" href="/docs" target="_blank">API</a></div>
      </section>
    </div>
  </div>
</body>
</html>
"""
    return html

def form_html(error: Optional[str] = None) -> str:
    err = f'<p style="color:#ff9e9e;background:#381919;padding:8px 12px;border:1px solid #7a2a2a;border-radius:8px;">{_esc(error)}</p>' if error else ""
    today = datetime.now().date().isoformat()
    later = (datetime.now().date() + timedelta(days=3)).isoformat()
    return f"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Rural Trip Planner – Form</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  body {{ background:#0b1020; color:#e9eefc; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; }}
  .wrap {{ max-width: 720px; margin: 36px auto; padding: 0 16px; }}
  .card {{ background:#121933; border:1px solid #1d2857; border-radius:16px; padding:20px; }}
  h1 {{ margin:0 0 14px; }}
  .row {{ display:grid; grid-template-columns: 1fr 1fr; gap:12px; }}
  label {{ font-size:14px; color:#9db2f9; display:block; margin-bottom:4px; }}
  input {{ width:100%; padding:10px; border-radius:10px; border:1px solid #2a387c; background:#0f1630; color:#e9eefc; }}
  .help {{ font-size:12px; color:#9db2f9; }}
  .actions {{ margin-top:16px; }}
  button {{ background:#1d2857; border:1px solid #2a387c; color:#e9eefc; padding:10px 14px; border-radius:10px; cursor:pointer; }}
  button:hover {{ background:#22327a; }}
  a {{ color:#7aa2ff; text-decoration:none; }}
</style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Plan your rural trip</h1>
      {err}
      <form method="post" action="/plan">
        <div class="row">
          <div>
            <label>Village *</label>
            <input name="village" placeholder="e.g., Hodka" required />
          </div>
          <div>
            <label>StateId (optional, Mongo _id)</label>
            <input name="stateId" placeholder="5f... (leave blank usually)" />
          </div>
        </div>

        <div class="row" style="margin-top:10px;">
          <div>
            <label>Budget (INR) *</label>
            <input type="number" name="budgetTotalINR" value="20000" min="0" step="100" required />
          </div>
          <div>
            <label>People *</label>
            <input type="number" name="people" value="2" min="1" required />
          </div>
        </div>

        <div class="row" style="margin-top:10px;">
          <div>
            <label>Start date *</label>
            <input type="date" name="startDate" value="{today}" required />
          </div>
          <div>
            <label>End date *</label>
            <input type="date" name="endDate" value="{later}" required />
          </div>
        </div>

        <div style="margin-top:10px;">
          <label>Interests (comma-separated)</label>
          <input name="interests" placeholder="weaving, music, festival" />
          <div class="help">Used to pick activities (matches activity.category or tags).</div>
        </div>

        <div class="actions">
          <button type="submit">Build itinerary</button>
          &nbsp; <a href="/docs" target="_blank">API Docs</a>
        </div>
      </form>
    </div>
  </div>
</body>
</html>
"""

# ========= FASTAPI =========
app = FastAPI(title="Rural Planner (Algorithmic)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=RedirectResponse)
def index():
    return RedirectResponse(url="/plan", status_code=307)

@app.get("/plan", response_class=HTMLResponse)
def get_form():
    return HTMLResponse(content=form_html(), status_code=200)

@app.post("/plan", response_class=HTMLResponse)
def post_form(
    village: str = Form(...),
    budgetTotalINR: float = Form(...),
    people: int = Form(...),
    startDate: str = Form(...),
    endDate: str = Form(...),
    interests: str = Form(""),
    stateId: Optional[str] = Form(None),
):
    try:
        interest_list = [s.strip() for s in interests.split(",") if s.strip()] if interests else []
        payload = ItineraryRequest(
            village=village,
            stateId=stateId or None,
            budgetTotalINR=budgetTotalINR,
            people=people,
            startDate=startDate,
            endDate=endDate,
            interests=interest_list,
        )
        result = plan_itinerary(payload)
        return HTMLResponse(content=render_itinerary_html(result), status_code=200)
    except HTTPException as he:
        return HTMLResponse(content=form_html(f"{he.status_code}: {he.detail}"), status_code=he.status_code)
    except Exception as e:
        return HTMLResponse(content=form_html(f"Unexpected error: {e}"), status_code=500)

@app.post("/api/itinerary")
def api_itinerary(payload: ItineraryRequest):
    return plan_itinerary(payload)

# Run locally
if __name__ == "__main__":
    import uvicorn
    # Optional: init Gemini for summary if enabled
    if USE_SUMMARY_LLM and genai and os.getenv("GOOGLE_API_KEY"):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
