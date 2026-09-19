#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 XOMAT AI PRO v5.0 — Flask Web Edition
 Ivory Theme + Fixed API + 3-Period Prediction + Win/Loss Tracking
==============================================================================
 pip install -r requirements.txt
 python app.py  →  http://localhost:5000
==============================================================================
"""

import os, json, time, math, threading
from datetime import datetime, timezone
import requests
import numpy as np

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from flask import Flask, jsonify, render_template_string
from flask_cors import CORS

# ==============================================================================
# CONFIG
# ==============================================================================
API_ENDPOINTS = [
    # Primary with version fix (KEY FIX!)
    "https://draw.ar-lottery01.com/WinGo/WinGo_1M/GetHistoryIssuePage.json?v=3.6",
    "https://draw.ar-lottery01.com/WinGo/WinGo_1M/GetHistoryIssuePage.json?v=3.31",
    "https://draw.ar-lottery01.com/WinGo/WinGo_1M/GetHistoryIssuePage.json",
]

API_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://draw.ar-lottery01.com/",
    "Origin": "https://draw.ar-lottery01.com",
    "Prefer": "apiversion=2.1",
}

DATA_DIR  = "/data" if os.path.exists("/data") else os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(DATA_DIR, "wingo.json")
POLL_SEC  = 5
TIMEOUT   = 12
HISTORY_LIMIT = 5000

app = Flask(__name__)
CORS(app)

# ==============================================================================
# WIN GO HELPERS
# ==============================================================================
def wingo_colors(n):
    if n == 0: return ["RED", "VIOLET"]
    if n == 5: return ["GREEN", "VIOLET"]
    if n in (1, 3, 7, 9): return ["GREEN"]
    return ["RED"]

def wingo_size(n): return "BIG" if n >= 5 else "SMALL"

def parse_api_color(c):
    c = (c or "").upper()
    out = []
    if "GREEN"  in c: out.append("GREEN")
    if "RED"    in c: out.append("RED")
    if "VIOLET" in c: out.append("VIOLET")
    return out or ["RED"]

def next_issue(iss):
    try: return str(int(iss) + 1)
    except: return iss

# ==============================================================================
# STORE
# ==============================================================================
_LOCK = threading.Lock()

def empty_stats():
    return {"total": 0, "wins": 0, "losses": 0,
            "numberWins": 0, "bsWins": 0, "colorWins": 0,
            "lastResult": None}

def load_store():
    if not os.path.exists(DATA_FILE):
        return {"lastIssue": None, "lastPrediction": None,
                "pendingPeriods": [], "records": [], "stats": empty_stats()}
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, list):
            return {"lastIssue": None, "lastPrediction": None,
                    "pendingPeriods": [], "records": d, "stats": empty_stats()}
        for k, v in [("records", []), ("stats", empty_stats()),
                     ("pendingPeriods", []), ("lastPrediction", None)]:
            d.setdefault(k, v)
        return d
    except Exception as e:
        print(f"[store] load error: {e}")
        return {"lastIssue": None, "lastPrediction": None,
                "pendingPeriods": [], "records": [], "stats": empty_stats()}

def save_store(store):
    try:
        os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(store, f, indent=2)
    except Exception as e:
        print(f"[store] save error: {e}")

# ==============================================================================
# API FETCH (with retry + version fallback)
# ==============================================================================
def fetch_api():
    """Try multiple endpoints + version params until one works."""
    for url in API_ENDPOINTS:
        try:
            r = requests.get(url, timeout=TIMEOUT, headers=API_HEADERS)
            print(f"[api] {r.status_code} ← {url[:80]}")
            if r.status_code != 200:
                continue
            data = r.json()
            lst = data.get("data", {}).get("list", [])
            if lst:
                print(f"[api] ✅ Got {len(lst)} records")
                return lst
            else:
                print(f"[api] ⚠️ Empty list")
        except Exception as e:
            print(f"[api] ❌ {url[:80]}: {e}")
            continue
    return None

# ==============================================================================
# PROFESSIONAL RNG ENGINE
# ==============================================================================
class XomatEngine:
    def __init__(self, records):
        self.records = [r for r in records if isinstance(r.get("number"), int)
                        and 0 <= r["number"] <= 9]
        self.n = len(self.records)
        self._precompute()

    def _precompute(self):
        n = self.n
        self.freq = np.zeros(10)
        self.red = self.green = self.violet = 0

        for r in self.records:
            self.freq[r["number"]] += 1
            for c in (r.get("color") or "").split("/"):
                c = c.strip().upper()
                if c == "RED":    self.red += 1
                elif c == "GREEN":  self.green += 1
                elif c == "VIOLET": self.violet += 1

        last15 = [r["number"] for r in self.records[:min(15, n)]]
        self.missing = [i for i in range(10) if i not in last15]

        self.v1 = np.zeros(10); self.v2 = np.zeros(10)
        for r in self.records[:25]:     self.v1[r["number"]] += 1
        for r in self.records[25:50]:   self.v2[r["number"]] += 1

        self.current = self.records[0]["number"] if n > 0 else 0

        self.m1 = np.zeros(10); self.m1t = 0
        for i in range(1, n):
            if self.records[i]["number"] == self.current:
                self.m1[self.records[i-1]["number"]] += 1
                self.m1t += 1

        self.m2 = np.zeros(10); self.m2t = 0
        if n >= 3:
            a, b = self.current, self.records[1]["number"]
            for i in range(2, n - 1):
                if (self.records[i]["number"] == a and
                    self.records[i+1]["number"] == b):
                    self.m2[self.records[i-1]["number"]] += 1
                    self.m2t += 1

        self.hot = np.zeros(10)
        for r in self.records[:30]: self.hot[r["number"]] += 1
        self.maxHot = max(self.hot) or 1

        if n > 0:
            first_bs = wingo_size(self.records[0]["number"])
            self.bs_streak = 0
            for r in self.records:
                if wingo_size(r["number"]) == first_bs:
                    self.bs_streak += 1
                else: break
            self.bs_streak_type = first_bs

            first_c = wingo_colors(self.records[0]["number"])[0]
            self.color_streak = 0
            for r in self.records:
                if wingo_colors(r["number"])[0] == first_c:
                    self.color_streak += 1
                else: break
            self.color_streak_type = first_c
        else:
            self.bs_streak = self.color_streak = 0
            self.bs_streak_type = self.color_streak_type = "BIG"

    def chi_square_test(self):
        if self.n < 30:
            return {"chi2": 0, "p_value": 1.0, "biased": False, "verdict": "insufficient data"}
        expected = np.full(10, self.n / 10.0)
        chi2 = float(np.sum((self.freq - expected) ** 2 / expected))
        if HAS_SCIPY:
            p = float(1 - scipy_stats.chi2.cdf(chi2, df=9))
        else:
            p = 0.5
        return {"chi2": round(chi2, 4), "p_value": round(p, 4),
                "biased": p < 0.05,
                "verdict": "BIASED / exploitable" if p < 0.05 else "uniform / fair"}

    def runs_test(self):
        if self.n < 20:
            return {"z": 0.0, "p_value": 1.0, "runs": 0, "verdict": "insufficient data"}
        seq = [1 if r["number"] >= 5 else 0 for r in self.records][::-1]
        n1 = sum(seq); n0 = len(seq) - n1
        if n1 == 0 or n0 == 0:
            return {"z": 0.0, "p_value": 1.0, "runs": 1, "verdict": "single type"}
        runs = 1
        for i in range(1, len(seq)):
            if seq[i] != seq[i-1]: runs += 1
        mu = 2 * n1 * n0 / (n1 + n0) + 1
        var = (2 * n1 * n0 * (2 * n1 * n0 - n1 - n0)) / (((n1 + n0) ** 2) * (n1 + n0 - 1))
        if var <= 0:
            return {"z": 0.0, "p_value": 1.0, "runs": runs, "verdict": "degenerate"}
        z = (runs - mu) / math.sqrt(var)
        if HAS_SCIPY:
            p = 2 * (1 - scipy_stats.norm.cdf(abs(z)))
        else:
            p = 0.5
        if p < 0.05:
            verdict = "PATTERN DETECTED" if z < 0 else "TRENDING"
        else:
            verdict = "random"
        return {"z": round(z, 3), "p_value": round(p, 4), "runs": runs, "verdict": verdict}

    def bayesian_posterior(self, alpha=1.0):
        return ((self.freq + alpha) / (self.n + 10 * alpha)).tolist()

    def wma(self, window=50, decay=0.9):
        if self.n == 0: return [0.1] * 10
        weights = np.array([decay ** i for i in range(min(window, self.n))])
        weights /= weights.sum()
        recent = np.array([r["number"] for r in self.records[:len(weights)]])
        counts = np.zeros(10)
        for num, w in zip(recent, weights): counts[num] += w
        return counts.tolist()

    def transition_matrix(self):
        M = np.zeros((10, 10))
        for i in range(1, self.n):
            M[self.records[i]["number"]][self.records[i-1]["number"]] += 1
        rs = M.sum(axis=1, keepdims=True); rs[rs == 0] = 1
        return (M / rs).tolist()

    def predict(self, periods=3):
        if self.n < 5:
            return self._empty(periods)

        bayes = np.array(self.bayesian_posterior())
        wma   = np.array(self.wma())
        m1n   = self.m1 / self.m1t if self.m1t else np.full(10, 0.1)
        m2n   = self.m2 / self.m2t if self.m2t else np.full(10, 0.1)

        miss_mask = np.array([1.0 if i in self.missing else 0.0 for i in range(10)])
        vel = self.v1 - self.v2
        vmax = max(1, abs(vel).max())
        veln = (vel + vmax) / (2 * vmax)
        hotn = self.hot / self.maxHot
        coldn = 1 - hotn

        avg = self.n / 10.0
        balance = np.maximum(0, 1 - self.freq / avg)

        anti = np.ones(10); anti[self.current] = 0.0

        streak_boost = np.zeros(10)
        if self.bs_streak >= 4:
            want = "SMALL" if self.bs_streak_type == "BIG" else "BIG"
            for i in range(10):
                if wingo_size(i) == want: streak_boost[i] = 1.0

        W = {"bayes": 0.18, "wma": 0.12, "m1": 0.15, "m2": 0.10,
             "miss": 0.12, "vel": 0.10, "hot": 0.06, "cold": 0.05,
             "bal": 0.07, "anti": 0.03, "str": 0.02}

        composite = (bayes * W["bayes"] * 10 +
                     wma * W["wma"] * 10 +
                     m1n * W["m1"] +
                     m2n * W["m2"] +
                     miss_mask * W["miss"] +
                     veln * W["vel"] +
                     hotn * W["hot"] +
                     coldn * W["cold"] +
                     balance * W["bal"] +
                     anti * W["anti"] * 0.1 +
                     streak_boost * W["str"])

        ranked = list(np.argsort(-composite))
        total_score = composite.sum() or 1
        conf_base = 55 + min(30, self.n / 40)

        confs = []
        for k in range(periods):
            share = composite[ranked[k]] / total_score
            conf = conf_base + share * 250 - k * 4
            confs.append(round(min(99.0, max(40.0, conf)), 2))

        chi = self.chi_square_test()
        runs = self.runs_test()

        if self.bs_streak >= 5:              pattern = "DRAGON STREAK — BREAK IMMINENT"
        elif chi["biased"]:                  pattern = f"RNG BIAS DETECTED (p={chi['p_value']})"
        elif runs["verdict"] == "PATTERN DETECTED": pattern = "AUTOCORRELATION PATTERN"
        elif len(self.missing) >= 4:         pattern = "MULTI-MISSING CYCLE ACTIVE"
        elif self.m1t and (self.m1[ranked[0]] / self.m1t) > 0.18: pattern = "MARKOV STRONG SIGNAL"
        elif max(self.v1) - min(self.v1) > 3: pattern = "HIGH VELOCITY MOMENTUM"
        elif self.n >= 500:                  pattern = "DEEP NEURAL MATCH"
        else:                                pattern = "PATTERN CONVERGENCE"

        period_list = []
        for k in range(periods):
            num = int(ranked[k])
            cols = wingo_colors(num)
            period_list.append({
                "rank": k + 1, "number": num,
                "bs": wingo_size(num),
                "color": cols[0], "altColors": cols[1:],
                "confidence": confs[k],
                "score": round(float(composite[num]), 4),
                "probability": round(float(bayes[num]), 4),
            })

        return {
            "number": int(ranked[0]), "number2": int(ranked[1]),
            "number3": int(ranked[2]) if len(ranked) > 2 else 0,
            "bs": wingo_size(int(ranked[0])), "bs2": wingo_size(int(ranked[1])),
            "bs3": wingo_size(int(ranked[2])) if len(ranked) > 2 else "BIG",
            "color": wingo_colors(int(ranked[0]))[0],
            "color2": wingo_colors(int(ranked[1]))[0],
            "color3": wingo_colors(int(ranked[2]))[0] if len(ranked) > 2 else "RED",
            "confidence": confs[0], "confidence2": confs[1],
            "confidence3": confs[2] if len(confs) > 2 else 0,
            "pattern": pattern,
            "top": [int(x) for x in ranked[:3]],
            "freq": [int(x) for x in self.freq.tolist()],
            "missing": self.missing,
            "score": [round(float(x), 4) for x in composite.tolist()],
            "probability": [round(x, 4) for x in bayes.tolist()],
            "transition": self.transition_matrix(),
            "chi": chi, "runs": runs,
            "red": self.red, "green": self.green, "violet": self.violet,
            "bsStreak": self.bs_streak, "bsStreakType": self.bs_streak_type,
            "colorStreak": self.color_streak, "colorStreakType": self.color_streak_type,
            "ready": self.n >= 50, "total": self.n, "periods": period_list,
        }

    def _empty(self, periods):
        return {
            "number": 0, "number2": 0, "number3": 0,
            "bs": "BIG", "bs2": "BIG", "bs3": "BIG",
            "color": "RED", "color2": "RED", "color3": "RED",
            "confidence": 0, "confidence2": 0, "confidence3": 0,
            "pattern": "INSUFFICIENT DATA",
            "top": [0, 0, 0], "freq": [0]*10, "missing": [],
            "score": [0.0]*10, "probability": [0.1]*10,
            "transition": [[0.1]*10 for _ in range(10)],
            "chi": {"chi2": 0, "p_value": 1, "biased": False, "verdict": "n/a"},
            "runs": {"z": 0, "p_value": 1, "runs": 0, "verdict": "n/a"},
            "red": 0, "green": 0, "violet": 0,
            "bsStreak": 0, "bsStreakType": "BIG",
            "colorStreak": 0, "colorStreakType": "RED",
            "ready": False, "total": 0, "periods": [],
        }

# ==============================================================================
# SYNC
# ==============================================================================
def sync_store(store, api_list):
    if not api_list: return 0
    seen = {str(r["issue"]) for r in store["records"]}
    pending = store.get("pendingPeriods") or []
    pending_by_issue = {str(p.get("forIssue")): p for p in pending}
    added = 0

    for item in reversed(api_list):
        iss = str(item.get("issueNumber", ""))
        if not iss or iss in seen: continue
        num = int(item.get("number", 0))
        size = wingo_size(num)
        cols = parse_api_color(item.get("color", ""))

        rec = {"issue": iss, "number": num, "size": size,
               "color": "/".join(cols), "ts": int(time.time())}

        pred = pending_by_issue.get(iss)
        if pred:
            pn, pb, pc = pred["number"], pred["bs"], pred["color"]
            num_hit = (pn == num); bs_hit = (pb == size); col_hit = pc in cols
            rec.update({
                "predictedNumber": pn, "predictedBS": pb, "predictedColor": pc,
                "numberWin": num_hit, "bsWin": bs_hit, "colorWin": col_hit,
                "win": num_hit or bs_hit or col_hit,
                "rank": pred.get("rank", 1),
            })
        else:
            rec["win"] = None

        store["records"].insert(0, rec)
        seen.add(iss)
        added += 1

    if len(store["records"]) > HISTORY_LIMIT:
        store["records"] = store["records"][:HISTORY_LIMIT]

    s = empty_stats()
    for r in store["records"]:
        w = r.get("win")
        if w is None: continue
        s["total"] += 1
        if w: s["wins"] += 1
        else: s["losses"] += 1
        if r.get("numberWin"): s["numberWins"] += 1
        if r.get("bsWin"):     s["bsWins"] += 1
        if r.get("colorWin"):  s["colorWins"] += 1
    for r in store["records"]:
        if r.get("win") is not None:
            s["lastResult"] = r; break
    store["stats"] = s
    store["lastIssue"] = str(api_list[0].get("issueNumber", store.get("lastIssue") or ""))
    return added

# ==============================================================================
# BACKGROUND POLLER
# ==============================================================================
_store_lock = threading.Lock()
_cache = {"analysis": None, "stats": None, "lastIssue": "--", "total": 0,
          "online": False, "lastSync": 0, "apiStatus": "connecting"}

def poll_loop():
    store = load_store()
    last_seen = store.get("lastIssue")
    fails = 0

    while True:
        try:
            api_list = fetch_api()

            if api_list is None:
                fails += 1
                with _store_lock:
                    _cache["online"] = False
                    _cache["apiStatus"] = f"offline ({fails} fails)"
                time.sleep(POLL_SEC)
                continue

            fails = 0
            with _store_lock:
                _cache["online"] = True
                _cache["apiStatus"] = f"live ({len(api_list)} items)"

                newest = str(api_list[0].get("issueNumber", ""))
                if newest != last_seen:
                    added = sync_store(store, api_list)
                    last_seen = store.get("lastIssue")
                    if added > 0 or not store.get("pendingPeriods"):
                        engine = XomatEngine(store["records"])
                        analysis = engine.predict(periods=3)
                        preds = []
                        iss = store.get("lastIssue") or "0"
                        for p in analysis["periods"]:
                            iss = next_issue(iss)
                            preds.append({
                                "forIssue": iss, "number": p["number"],
                                "bs": p["bs"], "color": p["color"],
                                "rank": p["rank"],
                            })
                        store["pendingPeriods"] = preds
                        store["lastPrediction"] = preds[0] if preds else None
                        save_store(store)
                else:
                    engine = XomatEngine(store["records"])
                    analysis = engine.predict(periods=3)

                _cache["analysis"]  = analysis
                _cache["stats"]     = store["stats"]
                _cache["lastIssue"] = store.get("lastIssue") or "--"
                _cache["total"]     = len(store["records"])
                _cache["lastSync"]  = time.time()
        except Exception as e:
            print(f"[poll] error: {e}")
        time.sleep(POLL_SEC)

# ==============================================================================
# ROUTES
# ==============================================================================
@app.route("/api/analysis")
def api_analysis():
    with _store_lock:
        a = _cache["analysis"]
        if a is None:
            return jsonify({"ok": False, "msg": "loading", "apiStatus": _cache["apiStatus"]})
        return jsonify({
            "ok": True, "analysis": a, "stats": _cache["stats"],
            "lastIssue": _cache["lastIssue"],
            "nextIssue": next_issue(_cache["lastIssue"]),
            "total": _cache["total"], "online": _cache["online"],
            "lastSync": _cache["lastSync"], "serverTime": time.time(),
            "apiStatus": _cache["apiStatus"],
        })

@app.route("/api/history")
def api_history():
    store = load_store()
    return jsonify({"ok": True, "records": store["records"][:100]})

@app.route("/api/stats")
def api_stats():
    store = load_store()
    return jsonify({"ok": True, "stats": store["stats"],
                    "total": len(store["records"])})

# ==============================================================================
# FRONTEND — Ivory Theme Edition
# ==============================================================================
INDEX_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>XOMAT AI PRO v5 — Ivory Edition</title>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&family=Orbitron:wght@600;700;900&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
  :root{
    --ivory:#F5EFE4;
    --ivory-2:#EDE4D2;
    --ivory-3:#E5D9BE;
    --ivory-4:#D9CBA8;
    --ivory-dark:#B8A582;
    --gold:#C9A227;
    --gold-bright:#E0BC3F;
    --gold-deep:#8F7211;
    --maroon:#6B0F1A;
    --maroon-2:#8B1420;
    --maroon-3:#4A0812;
    --crimson:#B8213A;
    --neon:#FF2A3B;
    --green:#0F9D58;
    --green-2:#13B36A;
    --purple:#7C3AED;
    --ink:#2A1F12;
    --ink-2:#4A3B24;
    --ink-soft:#6B5A3E;
    --shadow:rgba(107,15,26,0.15);
  }
  *{box-sizing:border-box;margin:0;padding:0}
  html,body{margin:0;padding:0;overflow-x:hidden}
  body{
    background:
      radial-gradient(ellipse at 15% 8%, #FFF9EC 0%, transparent 45%),
      radial-gradient(ellipse at 85% 92%, #F0E0C8 0%, transparent 50%),
      linear-gradient(180deg, var(--ivory) 0%, var(--ivory-2) 100%);
    background-attachment:fixed;
    color:var(--ink);
    font-family:'Plus Jakarta Sans',sans-serif;
    min-height:100vh;
    position:relative;
  }
  /* Subtle paper texture */
  body::before{
    content:'';position:fixed;inset:0;pointer-events:none;z-index:0;opacity:.35;
    background-image:
      radial-gradient(circle at 20% 30%, rgba(201,162,39,.06) 0%, transparent 25%),
      radial-gradient(circle at 80% 70%, rgba(139,20,32,.05) 0%, transparent 25%),
      repeating-linear-gradient(45deg, rgba(184,165,130,.03) 0px, rgba(184,165,130,.03) 2px, transparent 2px, transparent 8px);
  }
  #particles{position:fixed;inset:0;z-index:1;pointer-events:none;opacity:.4}

  .wrap{position:relative;z-index:2;max-width:1500px;margin:0 auto;padding:16px}

  /* ═══════════ HEADER ═══════════ */
  header{
    display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;
    gap:14px;padding:18px 24px;
    background:linear-gradient(135deg, #FFFCF4 0%, var(--ivory-2) 50%, var(--ivory-3) 100%);
    border:2px solid var(--gold);
    border-radius:22px;
    box-shadow:
      0 12px 40px var(--shadow),
      0 4px 0 var(--gold-deep),
      inset 0 1px 0 rgba(255,255,255,.9),
      inset 0 -2px 12px rgba(201,162,39,.15);
    position:relative;overflow:hidden;
  }
  header::after{
    content:'';position:absolute;left:0;right:0;top:0;height:3px;
    background:linear-gradient(90deg,var(--gold-deep),var(--gold-bright),var(--gold-deep));
    background-size:200% 100%;animation:flow 4s linear infinite;
  }
  @keyframes flow{0%{background-position:0% 0%}100%{background-position:200% 0%}}
  header::before{
    content:'';position:absolute;top:-50%;right:-10%;width:280px;height:280px;
    background:radial-gradient(circle, rgba(201,162,39,.18) 0%, transparent 70%);
    border-radius:50%;pointer-events:none;
  }

  .brand{display:flex;align-items:center;gap:14px;position:relative;z-index:2}
  .brain{
    width:56px;height:56px;border-radius:50%;
    display:flex;align-items:center;justify-content:center;
    background:radial-gradient(circle at 30% 30%, var(--maroon-2), var(--maroon-3));
    border:3px solid var(--gold);
    box-shadow:
      0 0 0 3px var(--ivory),
      0 0 25px rgba(139,20,32,.4),
      0 6px 15px rgba(0,0,0,.2),
      inset 0 0 15px rgba(0,0,0,.4);
    animation:pulse 2.5s infinite ease-in-out;
  }
  @keyframes pulse{
    0%,100%{box-shadow:0 0 0 3px var(--ivory), 0 0 25px rgba(139,20,32,.4), 0 6px 15px rgba(0,0,0,.2), inset 0 0 15px rgba(0,0,0,.4)}
    50%{box-shadow:0 0 0 3px var(--ivory), 0 0 45px rgba(201,162,39,.7), 0 6px 15px rgba(0,0,0,.2), inset 0 0 15px rgba(0,0,0,.4)}
  }
  .brain i{font-size:28px;color:var(--gold-bright);text-shadow:0 0 12px rgba(224,188,63,.9)}

  .brand h1{
    font-family:'Orbitron',sans-serif;font-weight:900;letter-spacing:2px;font-size:26px;
    background:linear-gradient(135deg, var(--maroon-2) 0%, var(--gold-deep) 50%, var(--maroon-2) 100%);
    -webkit-background-clip:text;background-clip:text;color:transparent;
    text-shadow:0 2px 4px rgba(0,0,0,.08);
  }
  .brand p{font-size:11px;color:var(--ink-soft);font-family:'JetBrains Mono',monospace;letter-spacing:.5px}

  .status-pills{display:flex;flex-wrap:wrap;gap:9px;font-size:12px;position:relative;z-index:2}
  .pill{
    display:flex;align-items:center;gap:8px;padding:9px 14px;border-radius:12px;
    background:linear-gradient(180deg, #FFFDF8, var(--ivory));
    border:1.5px solid var(--gold);
    box-shadow:0 3px 0 var(--ivory-4), inset 0 1px 0 #fff;
    font-family:'JetBrains Mono',monospace;font-size:11px;
    color:var(--ink-2);
  }
  .pill i{color:var(--maroon-2);font-size:13px}
  .pill b{color:var(--maroon-3);font-weight:700;margin-left:2px}
  .pill.live{
    background:linear-gradient(180deg, #F0FDF4, #DCFCE7);
    border-color:var(--green);
  }
  .pill.live i,.pill.live b{color:var(--green)}
  .pill.offline{
    background:linear-gradient(180deg, #FEF2F2, #FEE2E2);
    border-color:var(--crimson);
  }
  .pill.offline i,.pill.offline b{color:var(--crimson)}
  .live-dot{
    width:9px;height:9px;border-radius:50%;background:var(--green);
    box-shadow:0 0 0 0 rgba(15,157,88,.7);animation:livePulse 1.8s infinite;
  }
  @keyframes livePulse{
    0%{box-shadow:0 0 0 0 rgba(15,157,88,.7)}
    70%{box-shadow:0 0 0 12px rgba(15,157,88,0)}
    100%{box-shadow:0 0 0 0 rgba(15,157,88,0)}
  }

  /* ═══════════ HERO — 3 Predictions ═══════════ */
  .hero{
    margin-top:18px;padding:26px;border-radius:24px;
    background:linear-gradient(135deg, #FFFCF3 0%, var(--ivory-2) 40%, var(--ivory-3) 100%);
    border:2px solid var(--gold);
    box-shadow:
      0 15px 50px var(--shadow),
      0 5px 0 var(--gold-deep),
      inset 0 1px 0 #fff,
      inset 0 0 80px rgba(201,162,39,.06);
    position:relative;overflow:hidden;
  }
  .hero::before{
    content:'';position:absolute;top:-100px;right:-100px;width:400px;height:400px;
    background:radial-gradient(circle, rgba(139,20,32,.08) 0%, transparent 70%);
    border-radius:50%;pointer-events:none;
  }
  .hero::after{
    content:'';position:absolute;inset:0;pointer-events:none;
    background-image:
      linear-gradient(rgba(201,162,39,.05) 1px, transparent 1px),
      linear-gradient(90deg, rgba(201,162,39,.05) 1px, transparent 1px);
    background-size:24px 24px;
    animation:gridMove 12s linear infinite;
  }
  @keyframes gridMove{0%{background-position:0 0}100%{background-position:24px 24px}}

  .hero-title{
    display:flex;align-items:center;gap:10px;flex-wrap:wrap;
    margin-bottom:18px;position:relative;z-index:2;
  }
  .tag{
    background:linear-gradient(135deg, var(--maroon-2) 0%, var(--maroon) 100%);
    color:#fff;font-size:11px;font-weight:700;letter-spacing:1.5px;
    padding:7px 14px;border-radius:99px;text-transform:uppercase;
    font-family:'JetBrains Mono',monospace;
    box-shadow:0 3px 10px rgba(139,20,32,.3), inset 0 1px 0 rgba(255,255,255,.15);
    border:1px solid rgba(201,162,39,.4);
  }
  .tag.gold{
    background:linear-gradient(135deg, var(--gold-deep) 0%, var(--gold) 50%, var(--gold-deep) 100%);
    color:var(--maroon-3);
    box-shadow:0 3px 10px rgba(143,114,17,.4), inset 0 1px 0 rgba(255,255,255,.4);
  }

  .issue-info{
    display:flex;flex-wrap:wrap;align-items:center;gap:22px;
    font-family:'JetBrains Mono',monospace;font-size:12px;
    padding:14px 18px;margin-bottom:20px;
    background:linear-gradient(180deg, #FFFDF8, var(--ivory));
    border:1px solid var(--ivory-4);border-radius:14px;
    box-shadow:inset 0 1px 4px rgba(184,165,130,.15);
    position:relative;z-index:2;
  }
  .issue-info .lbl{color:var(--ink-soft);font-size:10px;letter-spacing:1.2px;font-weight:600}
  .issue-info .val{color:var(--maroon-3);font-weight:700;font-size:13px}
  .issue-info .val.gold{color:var(--gold-deep);font-size:14px}
  .issue-info .val.purple{color:var(--purple)}

  .cards3{
    display:grid;grid-template-columns:repeat(auto-fit, minmax(280px, 1fr));
    gap:18px;margin-top:8px;position:relative;z-index:2;
  }

  .pred-card{
    padding:22px;border-radius:20px;text-align:center;
    background:linear-gradient(180deg, #FFFDF8 0%, var(--ivory) 100%);
    border:2px solid var(--ivory-4);
    box-shadow:
      0 8px 24px rgba(42,31,18,.08),
      0 3px 0 var(--ivory-dark),
      inset 0 1px 0 #fff;
    position:relative;overflow:hidden;
    transition:transform .35s cubic-bezier(.34,1.56,.64,1), box-shadow .35s;
  }
  .pred-card:hover{
    transform:translateY(-6px) scale(1.02);
    box-shadow:0 20px 45px rgba(42,31,18,.15), 0 3px 0 var(--ivory-dark), inset 0 1px 0 #fff;
  }
  .pred-card::before{
    content:'';position:absolute;top:0;left:0;right:0;height:5px;
    background:linear-gradient(90deg, var(--gold-deep), var(--gold-bright), var(--gold-deep));
  }
  .pred-card.r1::before{background:linear-gradient(90deg, var(--gold-deep), #FFD966, var(--gold-deep))}
  .pred-card.r2::before{background:linear-gradient(90deg, var(--maroon-2), var(--crimson), var(--maroon-2))}
  .pred-card.r3::before{background:linear-gradient(90deg, #5B21B6, var(--purple), #5B21B6)}

  .pred-card.r1{
    border-color:var(--gold);
    box-shadow:0 10px 30px rgba(201,162,39,.25), 0 3px 0 var(--gold-deep), inset 0 1px 0 #fff;
  }
  .pred-card.r2{
    border-color:var(--crimson);
    box-shadow:0 10px 30px rgba(184,33,58,.22), 0 3px 0 #8B1420, inset 0 1px 0 #fff;
  }
  .pred-card.r3{
    border-color:var(--purple);
    box-shadow:0 10px 30px rgba(124,58,237,.22), 0 3px 0 #5B21B6, inset 0 1px 0 #fff;
  }
  /* Medal badge */
  .medal{
    position:absolute;top:12px;right:12px;
    width:36px;height:36px;border-radius:50%;
    display:flex;align-items:center;justify-content:center;
    font-family:'Orbitron',sans-serif;font-weight:900;font-size:14px;
    box-shadow:0 4px 10px rgba(0,0,0,.15), inset 0 1px 0 rgba(255,255,255,.6);
  }
  .pred-card.r1 .medal{background:linear-gradient(135deg, #FFD966, var(--gold-deep));color:#4A3300}
  .pred-card.r2 .medal{background:linear-gradient(135deg, #E0C8A0, #A08866);color:#3A2810}
  .pred-card.r3 .medal{background:linear-gradient(135deg, #C4B5FD, #7C3AED);color:#2E1065}

  .rank{
    font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:2px;
    margin-bottom:14px;font-weight:700;
  }
  .rank.r1{color:var(--gold-deep)}
  .rank.r2{color:var(--maroon-2)}
  .rank.r3{color:var(--purple)}

  .digit{
    font-family:'Orbitron',sans-serif;font-weight:900;font-size:84px;line-height:1;
    margin:10px 0 16px;position:relative;
    display:inline-block;
  }
  .pred-card.r1 .digit{
    color:var(--gold-deep);
    text-shadow:
      0 0 20px rgba(201,162,39,.5),
      0 4px 0 var(--gold),
      0 8px 12px rgba(42,31,18,.15);
    animation:digitGlow 2s ease-in-out infinite;
  }
  .pred-card.r2 .digit{
    color:var(--maroon-2);
    text-shadow:
      0 0 20px rgba(184,33,58,.4),
      0 4px 0 var(--crimson),
      0 8px 12px rgba(42,31,18,.15);
  }
  .pred-card.r3 .digit{
    color:var(--purple);
    text-shadow:
      0 0 20px rgba(124,58,237,.4),
      0 4px 0 #5B21B6,
      0 8px 12px rgba(42,31,18,.15);
  }
  @keyframes digitGlow{
    0%,100%{filter:brightness(1)}
    50%{filter:brightness(1.25) drop-shadow(0 0 15px rgba(201,162,39,.7))}
  }

  .bsc{
    display:flex;justify-content:center;gap:9px;flex-wrap:wrap;
    font-family:'JetBrains Mono',monospace;font-size:12px;
    margin-bottom:14px;
  }
  .chip{
    padding:6px 14px;border-radius:10px;font-weight:800;letter-spacing:.6px;
    box-shadow:0 2px 0 rgba(0,0,0,.15), inset 0 1px 0 rgba(255,255,255,.4);
    text-shadow:0 1px 1px rgba(0,0,0,.1);
  }
  .chip.big{background:linear-gradient(180deg, var(--crimson), var(--maroon-2));color:#fff}
  .chip.small{background:linear-gradient(180deg, var(--green-2), var(--green));color:#fff}
  .chip.red{background:linear-gradient(180deg, var(--crimson), var(--maroon-2));color:#fff}
  .chip.green{background:linear-gradient(180deg, var(--green-2), var(--green));color:#fff}
  .chip.violet{background:linear-gradient(180deg, #A78BFA, var(--purple));color:#fff}

  .conf-bar{
    height:10px;border-radius:8px;
    background:linear-gradient(180deg, #E8DCC0, var(--ivory-4));
    box-shadow:inset 0 2px 4px rgba(42,31,18,.15);
    overflow:hidden;margin-top:12px;position:relative;
  }
  .conf-fill{
    height:100%;border-radius:8px;
    background:linear-gradient(90deg, var(--maroon-2), var(--gold-bright), var(--maroon-2));
    background-size:200% 100%;
    animation:confShine 3s linear infinite;
    transition:width 1.2s cubic-bezier(.4,0,.2,1);
    box-shadow:0 0 12px rgba(201,162,39,.5), inset 0 1px 0 rgba(255,255,255,.4);
  }
  @keyframes confShine{0%{background-position:0% 0%}100%{background-position:200% 0%}}
  .conf-txt{
    font-family:'JetBrains Mono',monospace;font-size:11px;
    color:var(--ink-2);margin-top:8px;font-weight:600;
  }

  /* ═══════════ STATS GRID ═══════════ */
  .grid-stat{
    display:grid;grid-template-columns:repeat(auto-fit, minmax(190px, 1fr));
    gap:14px;margin-top:22px;
  }
  .stat{
    padding:20px;border-radius:18px;
    background:linear-gradient(180deg, #FFFDF8, var(--ivory));
    border:1.5px solid var(--ivory-4);
    box-shadow:
      0 6px 18px rgba(42,31,18,.07),
      0 3px 0 var(--ivory-dark),
      inset 0 1px 0 #fff;
    transition:transform .3s, box-shadow .3s;
    position:relative;overflow:hidden;
  }
  .stat::before{
    content:'';position:absolute;top:0;left:0;width:4px;height:100%;
    background:linear-gradient(180deg, var(--gold), var(--maroon-2));
  }
  .stat:hover{
    transform:translateY(-3px);
    box-shadow:0 12px 28px rgba(42,31,18,.12), 0 3px 0 var(--ivory-dark), inset 0 1px 0 #fff;
  }
  .stat-lbl{
    font-size:11px;color:var(--ink-soft);font-family:'JetBrains Mono',monospace;
    letter-spacing:1.2px;font-weight:700;margin-bottom:6px;
  }
  .stat-val{
    font-size:28px;font-weight:900;font-family:'Orbitron',sans-serif;
    color:var(--maroon-3);line-height:1.1;
  }
  .stat-val.gold{color:var(--gold-deep)}
  .stat-val.green{color:var(--green)}
  .stat-val.red{color:var(--crimson)}
  .stat-sub{
    font-size:10px;color:var(--ink-soft);font-family:'JetBrains Mono',monospace;
    margin-top:5px;opacity:.75;
  }

  /* ═══════════ PANELS ═══════════ */
  .row2{
    display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:22px;
  }
  @media(max-width:900px){.row2{grid-template-columns:1fr}}

  .panel{
    padding:22px;border-radius:20px;
    background:linear-gradient(180deg, #FFFDF8 0%, var(--ivory) 100%);
    border:1.5px solid var(--ivory-4);
    box-shadow:
      0 8px 24px rgba(42,31,18,.08),
      0 3px 0 var(--ivory-dark),
      inset 0 1px 0 #fff;
    position:relative;
  }
  .panel-h{
    display:flex;align-items:center;gap:11px;margin-bottom:16px;
    padding-bottom:14px;border-bottom:2px solid var(--ivory-3);
    position:relative;
  }
  .panel-h::after{
    content:'';position:absolute;bottom:-2px;left:0;width:60px;height:2px;
    background:linear-gradient(90deg, var(--gold), transparent);
  }
  .panel-h i{
    color:var(--maroon-2);font-size:18px;
    padding:8px;border-radius:10px;
    background:linear-gradient(180deg, var(--ivory-2), var(--ivory-3));
    box-shadow:0 2px 0 var(--ivory-4), inset 0 1px 0 #fff;
  }
  .panel-h h3{
    font-family:'Orbitron',sans-serif;font-size:13px;letter-spacing:1.5px;
    color:var(--maroon-3);text-transform:uppercase;font-weight:700;
  }

  /* ═══════════ FREQUENCY ═══════════ */
  .freq-list{display:flex;flex-direction:column;gap:7px}
  .freq-row{
    display:flex;align-items:center;gap:10px;
    font-family:'JetBrains Mono',monospace;font-size:11px;
    padding:4px 0;
  }
  .freq-row .num{
    width:22px;height:22px;line-height:22px;text-align:center;
    color:#fff;font-weight:800;border-radius:6px;
    background:linear-gradient(180deg, var(--maroon-2), var(--maroon-3));
    box-shadow:0 2px 0 rgba(0,0,0,.2);
    font-size:12px;
  }
  .freq-row .marker{
    width:16px;text-align:center;color:var(--gold-deep);
    font-size:14px;font-weight:700;
  }
  .freq-row .bar{
    flex:1;height:18px;border-radius:9px;
    background:linear-gradient(180deg, #E8DCC0, var(--ivory-4));
    box-shadow:inset 0 2px 4px rgba(42,31,18,.15);
    overflow:hidden;position:relative;
  }
  .freq-row .fill{
    height:100%;border-radius:9px;
    background:linear-gradient(90deg, var(--maroon-3), var(--maroon-2), var(--crimson));
    transition:width .9s cubic-bezier(.4,0,.2,1);
    box-shadow:inset 0 1px 0 rgba(255,255,255,.25);
    position:relative;
  }
  .freq-row.hi .fill{
    background:linear-gradient(90deg, var(--gold-deep), var(--gold-bright), var(--gold-deep));
    box-shadow:inset 0 1px 0 rgba(255,255,255,.4), 0 0 12px rgba(201,162,39,.5);
  }
  .freq-row .count{
    width:34px;text-align:right;color:var(--ink-2);font-weight:700;
  }

  /* ═══════════ COLORS ═══════════ */
  .color-break{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}
  .cb{
    padding:18px 12px;border-radius:14px;text-align:center;
    font-family:'JetBrains Mono',monospace;
    border:2px solid;
    position:relative;overflow:hidden;
    box-shadow:0 4px 12px rgba(42,31,18,.08), inset 0 1px 0 rgba(255,255,255,.6);
  }
  .cb::before{
    content:'';position:absolute;top:0;left:0;right:0;height:3px;
  }
  .cb.red{
    background:linear-gradient(180deg, #FEF2F2, #FECACA);
    border-color:#DC2626;color:#991B1B;
  }
  .cb.red::before{background:linear-gradient(90deg, #7F1D1D, #DC2626, #7F1D1D)}
  .cb.green{
    background:linear-gradient(180deg, #F0FDF4, #BBF7D0);
    border-color:var(--green);color:#065F46;
  }
  .cb.green::before{background:linear-gradient(90deg, #064E3B, var(--green), #064E3B)}
  .cb.violet{
    background:linear-gradient(180deg, #FAF5FF, #E9D5FF);
    border-color:var(--purple);color:#5B21B6;
  }
  .cb.violet::before{background:linear-gradient(90deg, #4C1D95, var(--purple), #4C1D95)}
  .cb .pct{font-size:24px;font-weight:900;font-family:'Orbitron',sans-serif;color:inherit;margin:4px 0}
  .cb .lbl{font-size:10px;letter-spacing:2px;margin-bottom:4px;font-weight:800}
  .cb .cnt{font-size:11px;opacity:.8;font-weight:600}

  /* ═══════════ STATISTICAL TESTS ═══════════ */
  .tests{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:8px}
  .test{
    padding:16px;border-radius:12px;
    background:linear-gradient(180deg, #FFFDF8, var(--ivory));
    border:1.5px solid var(--ivory-4);
    box-shadow:0 3px 0 var(--ivory-dark), inset 0 1px 0 #fff;
    position:relative;
  }
  .test .name{
    font-size:10px;color:var(--ink-soft);
    font-family:'JetBrains Mono',monospace;
    margin-bottom:8px;letter-spacing:1.2px;font-weight:700;
  }
  .test .result{font-size:15px;font-weight:800;font-family:'Orbitron',sans-serif;letter-spacing:.5px}
  .test .meta{
    font-size:10px;color:var(--ink-soft);font-family:'JetBrains Mono',monospace;
    margin-top:5px;opacity:.8;
  }
  .test.bad{
    background:linear-gradient(180deg, #FEF2F2, #FEE2E2);
    border-color:var(--crimson);
  }
  .test.bad .result{color:var(--crimson)}
  .test.good{
    background:linear-gradient(180deg, #F0FDF4, #DCFCE7);
    border-color:var(--green);
  }
  .test.good .result{color:var(--green)}

  /* ═══════════ WIN/LOSS ═══════════ */
  .wl-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:6px}
  .wl{
    padding:18px 12px;border-radius:14px;text-align:center;
    border:2px solid;
    box-shadow:0 4px 12px rgba(42,31,18,.08), inset 0 1px 0 rgba(255,255,255,.6);
    position:relative;overflow:hidden;
  }
  .wl::before{
    content:'';position:absolute;top:0;left:0;right:0;height:3px;
  }
  .wl.win{
    background:linear-gradient(180deg, #F0FDF4, #BBF7D0);
    border-color:var(--green);
  }
  .wl.win::before{background:linear-gradient(90deg, #064E3B, var(--green), #064E3B)}
  .wl.loss{
    background:linear-gradient(180deg, #FEF2F2, #FECACA);
    border-color:var(--crimson);
  }
  .wl.loss::before{background:linear-gradient(90deg, #7F1D1D, var(--crimson), #7F1D1D)}
  .wl .big{font-size:36px;font-weight:900;font-family:'Orbitron',sans-serif;line-height:1}
  .wl.win .big{color:#047857}
  .wl.loss .big{color:#B91C1C}
  .wl .lbl{
    font-size:10px;letter-spacing:2.5px;
    color:var(--ink-soft);font-family:'JetBrains Mono',monospace;
    font-weight:800;margin-top:4px;
  }

  /* ═══════════ HISTORY ═══════════ */
  .hist{max-height:400px;overflow-y:auto;margin-top:10px;padding-right:6px}
  .hist::-webkit-scrollbar{width:8px}
  .hist::-webkit-scrollbar-track{background:var(--ivory-2);border-radius:8px}
  .hist::-webkit-scrollbar-thumb{
    background:linear-gradient(180deg, var(--maroon-2), var(--maroon-3));
    border-radius:8px;border:2px solid var(--ivory-2);
  }
  .hist-row{
    display:grid;grid-template-columns:1fr 55px 70px 70px 68px;gap:8px;
    padding:11px 12px;border-radius:10px;
    font-family:'JetBrains Mono',monospace;font-size:11px;align-items:center;
    border-bottom:1px solid var(--ivory-3);
    transition:background .2s;
  }
  .hist-row:hover{background:rgba(201,162,39,.06)}
  .hist-row .iss{
    color:var(--ink-soft);font-size:10px;font-weight:600;
    overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
  }
  .hist-row .n{
    font-weight:900;color:var(--maroon-3);text-align:center;font-size:16px;
    font-family:'Orbitron',sans-serif;
  }
  .hist-row .badge{
    padding:4px 8px;border-radius:7px;font-size:10px;text-align:center;
    font-weight:800;letter-spacing:.5px;text-transform:uppercase;
    box-shadow:0 1px 3px rgba(0,0,0,.1), inset 0 1px 0 rgba(255,255,255,.4);
  }
  .badge.big{background:linear-gradient(180deg, var(--crimson), var(--maroon-2));color:#fff}
  .badge.small{background:linear-gradient(180deg, var(--green-2), var(--green));color:#fff}
  .badge.red{background:linear-gradient(180deg, var(--crimson), var(--maroon-2));color:#fff}
  .badge.green{background:linear-gradient(180deg, var(--green-2), var(--green));color:#fff}
  .badge.violet{background:linear-gradient(180deg, #A78BFA, var(--purple));color:#fff}
  .badge.win{background:linear-gradient(180deg, var(--green-2), #047857);color:#fff}
  .badge.loss{background:linear-gradient(180deg, var(--crimson), #7F1D1D);color:#fff}
  .badge.pend{
    background:linear-gradient(180deg, var(--gold-bright), var(--gold-deep));
    color:#3A2810;
  }

  /* ═══════════ FOOTER ═══════════ */
  footer{
    margin-top:26px;padding:22px;text-align:center;
    font-family:'JetBrains Mono',monospace;font-size:11px;
    color:var(--ink-soft);font-weight:600;letter-spacing:.5px;
    background:linear-gradient(180deg, transparent, rgba(201,162,39,.08));
    border-top:2px solid var(--ivory-4);
    position:relative;
  }
  footer::before{
    content:'';position:absolute;top:-2px;left:50%;transform:translateX(-50%);
    width:120px;height:2px;
    background:linear-gradient(90deg, transparent, var(--gold), transparent);
  }

  /* ═══════════ FLASH ═══════════ */
  .flash{animation:flashBg .9s ease-out}
  @keyframes flashBg{
    0%{background-color:rgba(201,162,39,.28); transform:scale(1.01)}
    100%{background-color:transparent; transform:scale(1)}
  }

  /* ═══════════ API STATUS BADGE ═══════════ */
  .api-badge{
    display:inline-flex;align-items:center;gap:6px;
    padding:5px 10px;border-radius:8px;
    font-family:'JetBrains Mono',monospace;font-size:10px;
    font-weight:700;letter-spacing:.5px;
  }
  .api-badge.live{
    background:linear-gradient(180deg, #F0FDF4, #DCFCE7);
    color:#047857;border:1px solid var(--green);
  }
  .api-badge.offline{
    background:linear-gradient(180deg, #FEF2F2, #FEE2E2);
    color:#991B1B;border:1px solid var(--crimson);
  }

  /* ═══════════ LOADER ═══════════ */
  .loader{
    display:inline-block;width:14px;height:14px;border-radius:50%;
    border:2px solid var(--ivory-4);border-top-color:var(--maroon-2);
    animation:spin 1s linear infinite;
  }
  @keyframes spin{0%{transform:rotate(0)}100%{transform:rotate(360deg)}}

  /* ═══════════ SHIMMER for loading ═══════════ */
  .shimmer{
    background:linear-gradient(90deg, transparent 0%, rgba(201,162,39,.25) 50%, transparent 100%);
    background-size:200% 100%;
    animation:shimmer 1.6s linear infinite;
  }
  @keyframes shimmer{0%{background-position:-200% 0}100%{background-position:200% 0}}

  /* Responsive */
  @media(max-width:640px){
    .brand h1{font-size:20px}
    .brand p{font-size:10px}
    .brain{width:48px;height:48px}
    .brain i{font-size:22px}
    .digit{font-size:68px}
    .stat-val{font-size:22px}
    .hero{padding:18px}
    .panel{padding:16px}
  }
</style>
</head>
<body>
<canvas id="particles"></canvas>

<div class="wrap">
  <header>
    <div class="brand">
      <div class="brain"><i class="fa-solid fa-brain"></i></div>
      <div>
        <h1>XOMAT AI <span style="color:var(--maroon-2)">PRO</span></h1>
        <p>v5.0 · Ivory Edition · Live RNG Analysis</p>
      </div>
    </div>
    <div class="status-pills">
      <div class="pill" id="pillApi">
        <i class="fa-solid fa-satellite-dish"></i>
        <span>API</span><b id="apiStatus">connecting…</b>
      </div>
      <div class="pill">
        <i class="fa-regular fa-clock"></i>
        <b id="pillClock">--:--:-- UTC</b>
      </div>
      <div class="pill">
        <i class="fa-solid fa-hourglass-half"></i>
        <span>NEXT</span><b id="pillCountdown">--s</b>
      </div>
      <div class="pill live" id="pillLive">
        <span class="live-dot"></span>
        <b id="liveText">syncing</b>
      </div>
    </div>
  </header>

  <!-- HERO -->
  <section class="hero">
    <div class="hero-title">
      <span class="tag"><i class="fa-solid fa-crown"></i> Best 3 Periods — Next 3 Draws</span>
      <span class="tag gold"><i class="fa-solid fa-chart-line"></i> Multi-Model Ensemble</span>
    </div>
    <div class="issue-info">
      <div><div class="lbl">CURRENT ISSUE</div><div class="val" id="curIssue">—</div></div>
      <div><div class="lbl">NEXT PERIOD</div><div class="val gold" id="nextIssue">—</div></div>
      <div><div class="lbl">PATTERN</div><div class="val purple" id="patternTag">—</div></div>
    </div>
    <div class="cards3" id="predCards">
      <div class="pred-card r1">
        <div class="medal">1</div>
        <div class="rank r1">★ RANK #1 — PRIMARY</div>
        <div class="digit">-</div>
        <div class="bsc"><div class="chip big">BIG</div><div class="chip red">RED</div></div>
        <div class="conf-bar"><div class="conf-fill" style="width:0%"></div></div>
        <div class="conf-txt">0% confidence</div>
      </div>
      <div class="pred-card r2">
        <div class="medal">2</div>
        <div class="rank r2">★ RANK #2 — SECONDARY</div>
        <div class="digit">-</div>
        <div class="bsc"><div class="chip big">BIG</div><div class="chip red">RED</div></div>
        <div class="conf-bar"><div class="conf-fill" style="width:0%"></div></div>
        <div class="conf-txt">0% confidence</div>
      </div>
      <div class="pred-card r3">
        <div class="medal">3</div>
        <div class="rank r3">★ RANK #3 — TERTIARY</div>
        <div class="digit">-</div>
        <div class="bsc"><div class="chip big">BIG</div><div class="chip red">RED</div></div>
        <div class="conf-bar"><div class="conf-fill" style="width:0%"></div></div>
        <div class="conf-txt">0% confidence</div>
      </div>
    </div>
  </section>

  <!-- STATS -->
  <div class="grid-stat">
    <div class="stat"><div class="stat-lbl">TOTAL RECORDS</div>
      <div class="stat-val" id="statTotal">0</div>
      <div class="stat-sub">accumulated from API</div></div>
    <div class="stat"><div class="stat-lbl">WIN RATE</div>
      <div class="stat-val green" id="statWinRate">0%</div>
      <div class="stat-sub">all-time accuracy</div></div>
    <div class="stat"><div class="stat-lbl">WINS</div>
      <div class="stat-val green" id="statWins">0</div>
      <div class="stat-sub">successful predictions</div></div>
    <div class="stat"><div class="stat-lbl">LOSSES</div>
      <div class="stat-val red" id="statLosses">0</div>
      <div class="stat-sub">failed predictions</div></div>
    <div class="stat"><div class="stat-lbl">TOP CONFIDENCE</div>
      <div class="stat-val gold" id="statConf">0%</div>
      <div class="stat-sub">rank #1 signal strength</div></div>
  </div>

  <!-- ROW 2: Frequency + Colors/Tests -->
  <div class="row2">
    <div class="panel">
      <div class="panel-h"><i class="fa-solid fa-chart-simple"></i><h3>Frequency 0–9</h3></div>
      <div class="freq-list" id="freqList"></div>
    </div>
    <div class="panel">
      <div class="panel-h"><i class="fa-solid fa-palette"></i><h3>Color Distribution</h3></div>
      <div class="color-break">
        <div class="cb red"><div class="lbl">RED</div><div class="pct" id="cbRedPct">0%</div><div class="cnt" id="cbRedCnt">0</div></div>
        <div class="cb green"><div class="lbl">GREEN</div><div class="pct" id="cbGreenPct">0%</div><div class="cnt" id="cbGreenCnt">0</div></div>
        <div class="cb violet"><div class="lbl">VIOLET</div><div class="pct" id="cbVioletPct">0%</div><div class="cnt" id="cbVioletCnt">0</div></div>
      </div>
      <div class="panel-h" style="margin-top:22px"><i class="fa-solid fa-flask"></i><h3>Statistical Tests</h3></div>
      <div class="tests">
        <div class="test" id="testChi">
          <div class="name">CHI-SQUARE (uniformity)</div>
          <div class="result">—</div>
          <div class="meta">p-value: —</div>
        </div>
        <div class="test" id="testRuns">
          <div class="name">RUNS TEST (autocorrelation)</div>
          <div class="result">—</div>
          <div class="meta">z-score: —</div>
        </div>
      </div>
    </div>
  </div>

  <!-- ROW 3: Win/Loss + History -->
  <div class="row2">
    <div class="panel">
      <div class="panel-h"><i class="fa-solid fa-trophy"></i><h3>Win / Loss Tracker</h3></div>
      <div class="wl-grid">
        <div class="wl win"><div class="big" id="wlWins">0</div><div class="lbl">WINS</div></div>
        <div class="wl loss"><div class="big" id="wlLosses">0</div><div class="lbl">LOSSES</div></div>
      </div>
      <div style="margin-top:16px;font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--ink-2)">
        <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid var(--ivory-3)">
          <span style="font-weight:700">Number Hits</span><b id="hitNum" style="color:var(--maroon-2);font-size:13px">0</b></div>
        <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid var(--ivory-3)">
          <span style="font-weight:700">Big/Small Hits</span><b id="hitBS" style="color:var(--maroon-2);font-size:13px">0</b></div>
        <div style="display:flex;justify-content:space-between;padding:8px 0">
          <span style="font-weight:700">Color Hits</span><b id="hitCol" style="color:var(--maroon-2);font-size:13px">0</b></div>
      </div>
    </div>
    <div class="panel">
      <div class="panel-h"><i class="fa-solid fa-list"></i><h3>Recent Draws</h3></div>
      <div class="hist">
        <div class="hist-row" style="color:var(--ink-soft);font-size:10px;border-bottom:2px solid var(--ivory-4);font-weight:800">
          <span>ISSUE</span><span style="text-align:center">NUM</span>
          <span style="text-align:center">SIZE</span><span style="text-align:center">COLOR</span>
          <span style="text-align:center">RESULT</span>
        </div>
        <div id="histBody"></div>
      </div>
    </div>
  </div>

  <footer>
    © 2026 XOMAT AI PRO · Ivory Edition v5.0 · Real Statistical Engine (Bayesian · Markov · Chi² · Runs Test) · Auto-Sync 5s
  </footer>
</div>

<script>
/* ═══════════ PARTICLES (Gold + Maroon on Ivory) ═══════════ */
(function(){
  const c=document.getElementById('particles'),ctx=c.getContext('2d');
  let W,H,P=[];
  function rs(){W=c.width=innerWidth;H=c.height=innerHeight}
  rs();addEventListener('resize',rs);
  const N=Math.min(70,Math.floor(innerWidth/22));
  for(let i=0;i<N;i++)P.push({
    x:Math.random()*W,y:Math.random()*H,
    vx:(Math.random()-.5)*.35,vy:(Math.random()-.5)*.35,
    r:Math.random()*1.8+.8,
    hue:Math.random()>.55?42:355,
    a:Math.random()*.5+.25
  });
  (function s(){
    ctx.clearRect(0,0,W,H);
    for(let i=0;i<P.length;i++)for(let j=i+1;j<P.length;j++){
      const dx=P[i].x-P[j].x,dy=P[i].y-P[j].y,d2=dx*dx+dy*dy;
      if(d2<18000){
        ctx.strokeStyle=`hsla(${P[i].hue},60%,45%,${(1-d2/18000)*.18})`;
        ctx.lineWidth=.7;
        ctx.beginPath();
        ctx.moveTo(P[i].x,P[i].y);ctx.lineTo(P[j].x,P[j].y);ctx.stroke();
      }
    }
    P.forEach(p=>{
      p.x+=p.vx;p.y+=p.vy;
      if(p.x<0||p.x>W)p.vx*=-1;if(p.y<0||p.y>H)p.vy*=-1;
      const g=ctx.createRadialGradient(p.x,p.y,0,p.x,p.y,p.r*8);
      g.addColorStop(0,`hsla(${p.hue},80%,55%,${p.a})`);
      g.addColorStop(1,`hsla(${p.hue},80%,50%,0)`);
      ctx.fillStyle=g;
      ctx.beginPath();ctx.arc(p.x,p.y,p.r*8,0,Math.PI*2);ctx.fill();
    });
    requestAnimationFrame(s);
  })();
})();

/* ═══════════ HELPERS ═══════════ */
const $ = id => document.getElementById(id);
let LAST_ISSUE = null;

function chipClass(v){
  v = (v||'').toUpperCase();
  if(v==='BIG') return 'chip big';
  if(v==='SMALL') return 'chip small';
  if(v==='RED') return 'chip red';
  if(v==='GREEN') return 'chip green';
  if(v==='VIOLET') return 'chip violet';
  return 'chip';
}

/* ═══════════ RENDER ═══════════ */
function render(data){
  if(!data.ok || !data.analysis) return;
  const a = data.analysis, s = data.stats;

  // API badge
  const pillApi = $('pillApi');
  const apiStatus = $('apiStatus');
  if(data.online){
    pillApi.classList.remove('offline');
    pillApi.classList.add('live');
    apiStatus.textContent = data.apiStatus || 'LIVE';
  } else {
    pillApi.classList.remove('live');
    pillApi.classList.add('offline');
    apiStatus.textContent = data.apiStatus || 'OFFLINE';
  }

  $('curIssue').textContent = data.lastIssue;
  $('nextIssue').textContent = data.nextIssue;
  $('patternTag').textContent = a.pattern;

  $('statTotal').textContent = data.total;
  $('statWinRate').textContent = s.total ? ((s.wins/s.total*100).toFixed(1)+'%') : '0%';
  $('statWins').textContent = s.wins;
  $('statLosses').textContent = s.losses;
  $('statConf').textContent = (a.confidence||0).toFixed(1)+'%';

  // 3 Prediction cards
  const cards = $('predCards').children;
  const periods = a.periods || [];
  for(let i=0;i<3;i++){
    const card = cards[i];
    const p = periods[i];
    if(!p) continue;
    card.querySelector('.digit').textContent = p.number;
    const bsc = card.querySelector('.bsc');
    bsc.innerHTML = `<div class="${chipClass(p.bs)}">${p.bs}</div>
                     <div class="${chipClass(p.color)}">${p.color}</div>`;
    const fill = card.querySelector('.conf-fill');
    fill.style.width = p.confidence + '%';
    card.querySelector('.conf-txt').textContent = p.confidence.toFixed(1) + '% confidence';
  }

  // Frequency
  const freq = a.freq;
  const maxFreq = Math.max(...freq) || 1;
  const top1 = a.top[0], top2 = a.top[1], top3 = a.top[2];
  let fh = '';
  for(let i=0;i<10;i++){
    const w = (freq[i]/maxFreq*100).toFixed(0);
    const marker = i===top1?'★':(i===top2?'☆':(i===top3?'◆':''));
    const cls = (i===top1||i===top2||i===top3)?'freq-row hi':'freq-row';
    fh += `<div class="${cls}">
      <span class="marker">${marker}</span>
      <span class="num">${i}</span>
      <span class="bar"><span class="fill" style="width:${w}%"></span></span>
      <span class="count">${freq[i]}</span>
    </div>`;
  }
  $('freqList').innerHTML = fh;

  // Colors
  const total = (a.red+a.green+a.violet) || 1;
  $('cbRedPct').textContent = ((a.red/total)*100).toFixed(1)+'%';
  $('cbGreenPct').textContent = ((a.green/total)*100).toFixed(1)+'%';
  $('cbVioletPct').textContent = ((a.violet/total)*100).toFixed(1)+'%';
  $('cbRedCnt').textContent = a.red;
  $('cbGreenCnt').textContent = a.green;
  $('cbVioletCnt').textContent = a.violet;

  // Tests
  const chi = a.chi, runs = a.runs;
  const tChi = $('testChi');
  tChi.className = 'test ' + (chi.biased ? 'bad' : 'good');
  tChi.querySelector('.result').textContent = chi.biased ? '⚠ BIASED' : '✓ UNIFORM';
  tChi.querySelector('.meta').textContent = `χ²=${chi.chi2}, p=${chi.p_value}`;

  const tRuns = $('testRuns');
  const runsBad = runs.verdict.includes('PATTERN') || runs.verdict.includes('TREND');
  tRuns.className = 'test ' + (runsBad ? 'bad' : 'good');
  tRuns.querySelector('.result').textContent = runs.verdict.toUpperCase();
  tRuns.querySelector('.meta').textContent = `z=${runs.z}, p=${runs.p_value}`;

  // Win/Loss
  $('wlWins').textContent = s.wins;
  $('wlLosses').textContent = s.losses;
  $('hitNum').textContent = s.numberWins;
  $('hitBS').textContent = s.bsWins;
  $('hitCol').textContent = s.colorWins;

  // Live text
  $('liveText').textContent = data.online ? 'live' : 'offline';

  // Flash + refresh history on new issue
  if(data.lastIssue !== LAST_ISSUE){
    LAST_ISSUE = data.lastIssue;
    loadHistory();
    document.querySelectorAll('.pred-card').forEach(c=>{
      c.classList.add('flash');
      setTimeout(()=>c.classList.remove('flash'), 900);
    });
  }
}

/* ═══════════ HISTORY ═══════════ */
async function loadHistory(){
  try{
    const r = await fetch('/api/history');
    const d = await r.json();
    if(!d.ok) return;
    const rows = d.records.slice(0, 40);
    $('histBody').innerHTML = rows.map(rec=>{
      const sizeCls = rec.size==='BIG'?'badge big':'badge small';
      let colorCls = 'badge ';
      const c = (rec.color||'').split('/')[0];
      if(c==='RED') colorCls+='red';
      else if(c==='GREEN') colorCls+='green';
      else colorCls+='violet';
      let res = '<span class="badge pend">—</span>';
      if(rec.win === true)  res = '<span class="badge win">WIN</span>';
      if(rec.win === false) res = '<span class="badge loss">LOSS</span>';
      return `<div class="hist-row">
        <span class="iss">${rec.issue}</span>
        <span class="n">${rec.number}</span>
        <span class="${sizeCls}">${rec.size}</span>
        <span class="${colorCls}">${c||'-'}</span>
        ${res}
      </div>`;
    }).join('');
  }catch(e){ console.error(e); }
}

/* ═══════════ POLL ═══════════ */
async function poll(){
  try{
    const r = await fetch('/api/analysis');
    const d = await r.json();
    if(d.ok) render(d);
  }catch(e){
    $('pillApi').classList.remove('live');
    $('pillApi').classList.add('offline');
    $('apiStatus').textContent = 'OFFLINE';
  }
}

/* ═══════════ CLOCK ═══════════ */
function tick(){
  const now = new Date();
  const u = now.toUTCString().split(' ');
  $('pillClock').textContent = u[4] + ' UTC';
  const rem = 60 - now.getUTCSeconds();
  $('pillCountdown').textContent = rem + 's';
}
setInterval(tick,1000); tick();

/* ═══════════ BOOT ═══════════ */
setInterval(poll, 3000); poll();
loadHistory();
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

# ==============================================================================
# START
# ==============================================================================
if __name__ == "__main__":
    t = threading.Thread(target=poll_loop, daemon=True)
    t.start()
    print("\n" + "═"*70)
    print("  XOMAT AI PRO v5.0 — Ivory Edition")
    print("  ➜  http://localhost:5000")
    print("  ➜  API: /api/analysis  /api/history  /api/stats")
    print("═"*70 + "\n")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
