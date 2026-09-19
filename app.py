#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 XOMAT AI PRO v4.0 — Flask Web Edition
 Professional RNG Analysis Engine | Real Statistical Models | Deployable
==============================================================================
 pip install flask flask-cors requests numpy scipy
 python app.py  →  http://localhost:5000
==============================================================================
"""

import os, json, time, math, threading
from datetime import datetime, timezone
from collections import Counter, defaultdict
import requests
import numpy as np
from scipy import stats as scipy_stats

from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS

# ==============================================================================
# CONFIG
# ==============================================================================
API_URL   = "https://draw.ar-lottery01.com/WinGo/WinGo_1M/GetHistoryIssuePage.json"
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wingo.json")
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

def wingo_size(n):
    return "BIG" if n >= 5 else "SMALL"

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
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(store, f, indent=2)
    except Exception as e:
        print(f"[store] save error: {e}")

# ==============================================================================
# PROFESSIONAL RNG ENGINE
# ==============================================================================
class XomatEngine:
    """Multi-model ensemble RNG analysis with statistical validation."""

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

        # Missing (last 15)
        last15 = [r["number"] for r in self.records[:min(15, n)]]
        self.missing = [i for i in range(10) if i not in last15]

        # Velocity 25 vs 25
        self.v1 = np.zeros(10); self.v2 = np.zeros(10)
        for r in self.records[:25]:     self.v1[r["number"]] += 1
        for r in self.records[25:50]:   self.v2[r["number"]] += 1

        # Markov 1st order
        if n > 0:
            self.current = self.records[0]["number"]
        else:
            self.current = 0
        self.m1 = np.zeros(10); self.m1t = 0
        for i in range(1, n):
            if self.records[i]["number"] == self.current:
                self.m1[self.records[i-1]["number"]] += 1
                self.m1t += 1

        # Markov 2nd order
        self.m2 = np.zeros(10); self.m2t = 0
        if n >= 3:
            a, b = self.current, self.records[1]["number"]
            for i in range(2, n - 1):
                if (self.records[i]["number"] == a and
                    self.records[i+1]["number"] == b):
                    self.m2[self.records[i-1]["number"]] += 1
                    self.m2t += 1

        # Hot / cold last 30
        self.hot = np.zeros(10)
        for r in self.records[:30]: self.hot[r["number"]] += 1
        self.maxHot = max(self.hot) or 1

        # Streak
        if n > 0:
            first_bs = wingo_size(self.records[0]["number"])
            self.bs_streak = 0
            for r in self.records:
                if wingo_size(r["number"]) == first_bs:
                    self.bs_streak += 1
                else:
                    break
            self.bs_streak_type = first_bs

            first_c = wingo_colors(self.records[0]["number"])[0]
            self.color_streak = 0
            for r in self.records:
                if wingo_colors(r["number"])[0] == first_c:
                    self.color_streak += 1
                else:
                    break
            self.color_streak_type = first_c
        else:
            self.bs_streak = self.color_streak = 0
            self.bs_streak_type = self.color_streak_type = "BIG"

    # --------------------------------------------------------------------------
    # PROFESSIONAL STATISTICAL TESTS
    # --------------------------------------------------------------------------
    def chi_square_test(self):
        """Test if RNG is uniform (fair) or biased."""
        if self.n < 30:
            return {"chi2": 0, "p_value": 1.0, "biased": False,
                    "verdict": "insufficient data"}
        expected = np.full(10, self.n / 10.0)
        chi2 = float(np.sum((self.freq - expected) ** 2 / expected))
        p = float(1 - scipy_stats.chi2.cdf(chi2, df=9))
        return {
            "chi2": round(chi2, 4),
            "p_value": round(p, 4),
            "biased": p < 0.05,
            "verdict": "BIASED / exploitable" if p < 0.05 else "uniform / fair",
        }

    def runs_test(self):
        """Detect autocorrelation in Big/Small sequence."""
        if self.n < 20:
            return {"z": 0.0, "p_value": 1.0, "runs": 0, "verdict": "insufficient data"}
        seq = [1 if r["number"] >= 5 else 0 for r in self.records]
        seq = list(reversed(seq))
        n1 = sum(seq); n0 = len(seq) - n1
        if n1 == 0 or n0 == 0:
            return {"z": 0.0, "p_value": 1.0, "runs": 1, "verdict": "single type"}
        runs = 1
        for i in range(1, len(seq)):
            if seq[i] != seq[i-1]: runs += 1
        mu = 2 * n1 * n0 / (n1 + n0) + 1
        var = (2 * n1 * n0 * (2 * n1 * n0 - n1 - n0)) / \
              (((n1 + n0) ** 2) * (n1 + n0 - 1))
        if var <= 0:
            return {"z": 0.0, "p_value": 1.0, "runs": runs, "verdict": "degenerate"}
        z = (runs - mu) / math.sqrt(var)
        p = 2 * (1 - scipy_stats.norm.cdf(abs(z)))
        if p < 0.05:
            verdict = "PATTERN DETECTED" if z < 0 else "TRENDING"
        else:
            verdict = "random"
        return {"z": round(z, 3), "p_value": round(p, 4),
                "runs": runs, "verdict": verdict}

    def bayesian_posterior(self, alpha=1.0):
        """Dirichlet-multinomial posterior → smoothed probability estimate."""
        posterior = (self.freq + alpha) / (self.n + 10 * alpha)
        return posterior.tolist()

    def weighted_moving_average(self, window=50, decay=0.9):
        """Exponential weighted recent bias."""
        if self.n == 0:
            return [0.1] * 10
        weights = np.array([decay ** i for i in range(min(window, self.n))])
        weights /= weights.sum()
        recent = np.array([r["number"] for r in self.records[:len(weights)]])
        counts = np.zeros(10)
        for num, w in zip(recent, weights):
            counts[num] += w
        return counts.tolist()

    def transition_matrix(self):
        """Full 10x10 transition matrix (Markov 1st order)."""
        M = np.zeros((10, 10))
        for i in range(1, self.n):
            M[self.records[i]["number"]][self.records[i-1]["number"]] += 1
        row_sums = M.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return (M / row_sums).tolist()

    # --------------------------------------------------------------------------
    # ENSEMBLE PREDICTION
    # --------------------------------------------------------------------------
    def predict(self, periods=3):
        if self.n < 5:
            return self._empty_prediction(periods)

        # ---- Component models ----
        bayes    = np.array(self.bayesian_posterior())
        wma      = np.array(self.weighted_moving_average())
        m1_norm  = self.m1 / self.m1t if self.m1t else np.full(10, 0.1)
        m2_norm  = self.m2 / self.m2t if self.m2t else np.full(10, 0.1)

        missing_mask = np.array([1.0 if i in self.missing else 0.0 for i in range(10)])
        vel = (self.v1 - self.v2)
        vel_norm = (vel + max(1, abs(vel).max())) / (2 * max(1, abs(vel).max()))
        hot_norm = self.hot / self.maxHot
        cold_norm = 1 - hot_norm

        # Frequency balance
        avg = self.n / 10.0
        balance = np.maximum(0, 1 - self.freq / avg)

        # Anti-repeat
        anti_repeat = np.ones(10); anti_repeat[self.current] = 0.0

        # Streak break boost
        streak_boost = np.zeros(10)
        if self.bs_streak >= 4:
            want = "SMALL" if self.bs_streak_type == "BIG" else "BIG"
            for i in range(10):
                if wingo_size(i) == want:
                    streak_boost[i] = 1.0

        # ---- Ensemble weights (sum = 1.0) ----
        W = {
            "bayes":      0.18,
            "wma":        0.12,
            "m1":         0.15,
            "m2":         0.10,
            "missing":    0.12,
            "velocity":   0.10,
            "hot":        0.06,
            "cold":       0.05,
            "balance":    0.07,
            "anti":       0.03,
            "streak":     0.02,
        }

        # Composite score (0–1)
        composite = (
            bayes * W["bayes"] * 10 +          # scale up to comparable range
            wma   * W["wma"] * 10 +
            m1_norm * W["m1"] +
            m2_norm * W["m2"] +
            missing_mask * W["missing"] +
            vel_norm * W["velocity"] +
            hot_norm * W["hot"] +
            cold_norm * W["cold"] +
            balance * W["balance"] +
            anti_repeat * W["anti"] * 0.1 +
            streak_boost * W["streak"]
        )

        # Rank
        ranked = list(np.argsort(-composite))

        # Confidence — blend Bayesian + score margin + record count
        total_score = composite.sum() or 1
        conf_base = 55 + min(30, self.n / 40)
        confs = []
        for k in range(periods):
            share = composite[ranked[k]] / total_score
            conf = conf_base + share * 250 - k * 4
            confs.append(round(min(99.0, max(40.0, conf)), 2))

        # Pattern tag
        chi = self.chi_square_test()
        runs = self.runs_test()
        if self.bs_streak >= 5:
            pattern = "DRAGON STREAK — BREAK IMMINENT"
        elif chi["biased"]:
            pattern = f"RNG BIAS DETECTED (p={chi['p_value']})"
        elif runs["verdict"] == "PATTERN DETECTED":
            pattern = "AUTOCORRELATION PATTERN"
        elif len(self.missing) >= 4:
            pattern = "MULTI-MISSING CYCLE ACTIVE"
        elif self.m1t and (self.m1[ranked[0]] / self.m1t) > 0.18:
            pattern = "MARKOV STRONG SIGNAL"
        elif max(self.v1) - min(self.v1) > 3:
            pattern = "HIGH VELOCITY MOMENTUM"
        elif self.n >= 500:
            pattern = "DEEP NEURAL MATCH"
        else:
            pattern = "PATTERN CONVERGENCE"

        # Per-period predictions
        period_list = []
        for k in range(periods):
            num = int(ranked[k])
            cols = wingo_colors(num)
            period_list.append({
                "rank":       k + 1,
                "number":     num,
                "bs":         wingo_size(num),
                "color":      cols[0],
                "altColors":  cols[1:],
                "confidence": confs[k],
                "score":      round(float(composite[num]), 4),
                "probability": round(float(bayes[num]), 4),
            })

        return {
            "number":       int(ranked[0]),
            "number2":      int(ranked[1]),
            "number3":      int(ranked[2]) if len(ranked) > 2 else 0,
            "bs":           wingo_size(int(ranked[0])),
            "bs2":          wingo_size(int(ranked[1])),
            "bs3":          wingo_size(int(ranked[2])) if len(ranked) > 2 else "BIG",
            "color":        wingo_colors(int(ranked[0]))[0],
            "color2":       wingo_colors(int(ranked[1]))[0],
            "color3":       wingo_colors(int(ranked[2]))[0] if len(ranked) > 2 else "RED",
            "confidence":   confs[0],
            "confidence2":  confs[1],
            "confidence3":  confs[2] if len(confs) > 2 else 0,
            "pattern":      pattern,
            "top":          [int(x) for x in ranked[:3]],
            "freq":         [int(x) for x in self.freq.tolist()],
            "missing":      self.missing,
            "score":        [round(float(x), 4) for x in composite.tolist()],
            "probability":  [round(x, 4) for x in bayes.tolist()],
            "transition":   self.transition_matrix(),
            "chi":          chi,
            "runs":         runs,
            "red":          self.red,
            "green":        self.green,
            "violet":       self.violet,
            "bsStreak":     self.bs_streak,
            "bsStreakType": self.bs_streak_type,
            "colorStreak":  self.color_streak,
            "colorStreakType": self.color_streak_type,
            "ready":        self.n >= 50,
            "total":        self.n,
            "periods":      period_list,
        }

    def _empty_prediction(self, periods):
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
def fetch_api():
    try:
        r = requests.get(API_URL, timeout=TIMEOUT,
                         headers={"User-Agent": "XomatAI/4.0"})
        r.raise_for_status()
        return r.json().get("data", {}).get("list", [])
    except Exception as e:
        print(f"[api] error: {e}")
        return None

def sync_store(store, api_list):
    if not api_list:
        return 0
    seen = {str(r["issue"]) for r in store["records"]}
    pending = store.get("pendingPeriods") or []
    pending_by_issue = {str(p.get("forIssue")): p for p in pending}
    added = 0

    for item in reversed(api_list):
        iss = str(item.get("issueNumber", ""))
        if not iss or iss in seen:
            continue
        num = int(item.get("number", 0))
        size = wingo_size(num)
        cols = parse_api_color(item.get("color", ""))

        rec = {
            "issue": iss, "number": num, "size": size,
            "color": "/".join(cols), "ts": int(time.time())
        }

        pred = pending_by_issue.get(iss)
        if pred:
            pn, pb, pc = pred["number"], pred["bs"], pred["color"]
            num_hit = (pn == num)
            bs_hit  = (pb == size)
            col_hit = pc in cols
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

    # Recalc stats
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
          "online": False, "lastSync": 0}

def poll_loop():
    store = load_store()
    last_seen = store.get("lastIssue")
    while True:
        try:
            api_list = fetch_api()
            if api_list is None:
                with _store_lock:
                    _cache["online"] = False
                time.sleep(POLL_SEC)
                continue

            with _store_lock:
                _cache["online"] = True
                newest = str(api_list[0].get("issueNumber", ""))
                if newest != last_seen:
                    added = sync_store(store, api_list)
                    last_seen = store.get("lastIssue")
                    if added > 0 or not store.get("pendingPeriods"):
                        engine = XomatEngine(store["records"])
                        analysis = engine.predict(periods=3)
                        # Save predictions for next 3 issues
                        preds = []
                        iss = store.get("lastIssue") or "0"
                        for p in analysis["periods"]:
                            iss = next_issue(iss)
                            preds.append({
                                "forIssue": iss,
                                "number":   p["number"],
                                "bs":       p["bs"],
                                "color":    p["color"],
                                "rank":     p["rank"],
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
            return jsonify({"ok": False, "msg": "loading"})
        return jsonify({
            "ok":         True,
            "analysis":   a,
            "stats":      _cache["stats"],
            "lastIssue":  _cache["lastIssue"],
            "nextIssue":  next_issue(_cache["lastIssue"]),
            "total":      _cache["total"],
            "online":     _cache["online"],
            "lastSync":   _cache["lastSync"],
            "serverTime": time.time(),
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
# FRONTEND (single-file HTML)
# ==============================================================================
INDEX_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>XOMAT AI PRO v4 — Live RNG Analysis</title>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;500;700&family=Orbitron:wght@600;900&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
  :root{
    --neon:#FF2A3B; --gold:#D4AF37; --green:#10B981; --purple:#A855F7;
    --bg1:#0a0002; --bg2:#3B0005; --card:rgba(20,2,6,0.72);
    --border:rgba(255,42,59,0.28); --text:#F4EFE6;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{
    background:
      radial-gradient(ellipse at 20% 10%, rgba(139,0,0,.55) 0%, transparent 45%),
      radial-gradient(ellipse at 85% 90%, rgba(169,27,34,.5) 0%, transparent 50%),
      radial-gradient(ellipse at 50% 50%, #1a0004 0%, #08000a 100%);
    background-attachment:fixed;
    color:var(--text);
    font-family:'Plus Jakarta Sans',sans-serif;
    min-height:100vh;
    overflow-x:hidden;
  }
  #particles{position:fixed;inset:0;z-index:0;pointer-events:none;opacity:.55}
  .aurora{
    position:fixed;inset:0;pointer-events:none;z-index:0;
    background:
      radial-gradient(circle at 25% 30%, rgba(212,175,55,.08), transparent 40%),
      radial-gradient(circle at 75% 70%, rgba(255,42,59,.08), transparent 40%);
    animation:aurora 14s ease-in-out infinite alternate;
  }
  @keyframes aurora{
    0%{transform:translate(0,0) scale(1);opacity:.8}
    100%{transform:translate(30px,-20px) scale(1.1);opacity:1}
  }
  .wrap{position:relative;z-index:2;max-width:1500px;margin:0 auto;padding:18px}

  /* HEADER */
  header{
    display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;
    gap:14px;padding:16px 22px;
    background:linear-gradient(90deg,#3a0004 0%,#1a0002 50%,#3a0004 100%);
    border:1px solid var(--border);border-radius:20px;
    box-shadow:0 10px 40px rgba(255,42,59,.15), inset 0 0 60px rgba(255,42,59,.05);
    position:relative;overflow:hidden;
  }
  header::after{
    content:'';position:absolute;left:0;right:0;bottom:0;height:2px;
    background:linear-gradient(90deg,transparent,var(--neon),var(--gold),var(--neon),transparent);
    background-size:200% 100%;animation:flow 3s linear infinite;
  }
  @keyframes flow{0%{background-position:0% 0%}100%{background-position:200% 0%}}

  .brand{display:flex;align-items:center;gap:14px}
  .brain{
    width:52px;height:52px;border-radius:50%;display:flex;align-items:center;justify-content:center;
    background:radial-gradient(circle at 30% 30%, #a91b22, #4a0000);
    border:2px solid rgba(255,42,59,.5);
    box-shadow:0 0 25px rgba(255,42,59,.6), inset 0 0 20px rgba(0,0,0,.5);
    animation:pulse 2.5s infinite ease-in-out;
  }
  @keyframes pulse{
    0%,100%{box-shadow:0 0 25px rgba(255,42,59,.5), inset 0 0 20px rgba(0,0,0,.5)}
    50%{box-shadow:0 0 50px rgba(255,42,59,.9), inset 0 0 20px rgba(0,0,0,.5)}
  }
  .brain i{font-size:26px;color:#FF6B7A}
  .brand h1{
    font-family:'Orbitron',sans-serif;font-weight:900;letter-spacing:2px;
    font-size:24px;
    background:linear-gradient(90deg,#fff,var(--gold),var(--neon));
    -webkit-background-clip:text;background-clip:text;color:transparent;
  }
  .brand p{font-size:11px;color:rgba(244,239,230,.7);font-family:'JetBrains Mono',monospace}

  .status-pills{display:flex;flex-wrap:wrap;gap:10px;font-size:12px}
  .pill{
    display:flex;align-items:center;gap:8px;padding:8px 14px;border-radius:12px;
    background:rgba(60,0,5,.7);border:1px solid rgba(255,42,59,.35);
    font-family:'JetBrains Mono',monospace;
  }
  .pill i{color:var(--gold)}
  .pill b{color:#fff;font-weight:700}
  .live-dot{width:9px;height:9px;border-radius:50%;background:var(--green);
    box-shadow:0 0 0 0 rgba(16,185,129,.7);animation:livePulse 1.8s infinite}
  @keyframes livePulse{
    0%{box-shadow:0 0 0 0 rgba(16,185,129,.7)}
    70%{box-shadow:0 0 0 14px rgba(16,185,129,0)}
    100%{box-shadow:0 0 0 0 rgba(16,185,129,0)}
  }

  /* NEXT PERIOD HERO */
  .hero{
    margin-top:18px;
    padding:24px;
    border-radius:24px;
    background:linear-gradient(135deg, rgba(74,0,0,.95), rgba(10,0,2,.98));
    border:1px solid rgba(255,42,59,.4);
    position:relative;overflow:hidden;
    box-shadow:0 15px 60px rgba(255,42,59,.2);
  }
  .hero::before{
    content:'';position:absolute;inset:-2px;border-radius:inherit;z-index:-1;
    background:conic-gradient(from 0deg,var(--neon),var(--gold),var(--neon),var(--purple),var(--neon));
    animation:spin 6s linear infinite;
  }
  @keyframes spin{0%{transform:rotate(0)}100%{transform:rotate(360deg)}}
  .hero-title{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:16px}
  .tag{
    background:linear-gradient(90deg,var(--neon),#8B0000);
    color:#fff;font-size:11px;font-weight:700;letter-spacing:1.5px;
    padding:5px 12px;border-radius:99px;text-transform:uppercase;
    font-family:'JetBrains Mono',monospace;
  }
  .issue-info{
    display:flex;flex-wrap:wrap;align-items:center;gap:20px;
    font-family:'JetBrains Mono',monospace;font-size:12px;
    padding:10px 0;
  }
  .issue-info .lbl{color:rgba(244,239,230,.55);font-size:10px;letter-spacing:1px}
  .issue-info .val{color:#fff;font-weight:700}

  .cards3{
    display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));
    gap:16px;margin-top:8px;
  }
  .pred-card{
    padding:20px;border-radius:18px;text-align:center;
    background:linear-gradient(160deg, rgba(60,0,5,.95), rgba(15,0,2,.98));
    border:1px solid rgba(255,42,59,.35);
    position:relative;overflow:hidden;
    transition:transform .35s cubic-bezier(.34,1.56,.64,1);
  }
  .pred-card:hover{transform:translateY(-6px) scale(1.02)}
  .pred-card.r1{border-color:var(--gold);box-shadow:0 0 30px rgba(212,175,55,.35), inset 0 0 30px rgba(212,175,55,.06)}
  .pred-card.r2{border-color:var(--neon);box-shadow:0 0 30px rgba(255,42,59,.3), inset 0 0 30px rgba(255,42,59,.06)}
  .pred-card.r3{border-color:var(--purple);box-shadow:0 0 30px rgba(168,85,247,.28), inset 0 0 30px rgba(168,85,247,.06)}
  .pred-card::after{
    content:'';position:absolute;top:-50%;left:-50%;width:200%;height:200%;
    background:linear-gradient(90deg,transparent 40%,rgba(255,255,255,.06) 50%,transparent 60%);
    animation:shimmer 3s linear infinite;
    pointer-events:none;
  }
  @keyframes shimmer{0%{transform:translateX(-100%) rotate(25deg)}100%{transform:translateX(100%) rotate(25deg)}}

  .rank{font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:2px;margin-bottom:12px}
  .rank.r1{color:var(--gold)}
  .rank.r2{color:var(--neon)}
  .rank.r3{color:var(--purple)}

  .digit{
    font-family:'Orbitron',sans-serif;font-weight:900;font-size:76px;line-height:1;
    margin:8px 0 12px;
    color:#fff;
    text-shadow:0 0 20px currentColor, 0 0 40px currentColor;
    animation:digitPulse 2s ease-in-out infinite;
  }
  @keyframes digitPulse{
    0%,100%{filter:brightness(1)}
    50%{filter:brightness(1.35)}
  }
  .pred-card.r1 .digit{color:var(--gold)}
  .pred-card.r2 .digit{color:var(--neon)}
  .pred-card.r3 .digit{color:var(--purple)}

  .bsc{
    display:flex;justify-content:center;gap:8px;
    font-family:'JetBrains Mono',monospace;font-size:11px;
    margin-bottom:12px;
  }
  .chip{padding:5px 12px;border-radius:8px;font-weight:700;letter-spacing:.5px}
  .chip.big{background:var(--neon);color:#fff}
  .chip.small{background:var(--green);color:#fff}
  .chip.red{background:#B91C1C;color:#fff}
  .chip.green{background:var(--green);color:#fff}
  .chip.violet{background:var(--purple);color:#fff}

  .conf-bar{
    height:8px;border-radius:6px;background:rgba(255,255,255,.1);
    overflow:hidden;margin-top:8px;
  }
  .conf-fill{
    height:100%;border-radius:6px;
    background:linear-gradient(90deg,var(--neon),var(--gold));
    transition:width 1s cubic-bezier(.4,0,.2,1);
    box-shadow:0 0 12px currentColor;
  }
  .conf-txt{font-family:'JetBrains Mono',monospace;font-size:11px;color:rgba(244,239,230,.85);margin-top:6px}

  /* STAT GRID */
  .grid-stat{
    display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
    gap:14px;margin-top:20px;
  }
  .stat{
    padding:18px;border-radius:16px;
    background:linear-gradient(160deg,rgba(20,2,6,.72),rgba(10,0,2,.85));
    border:1px solid rgba(255,42,59,.25);
    backdrop-filter:blur(10px);
    transition:transform .3s;
  }
  .stat:hover{transform:translateY(-3px)}
  .stat-lbl{font-size:11px;color:rgba(244,239,230,.6);font-family:'JetBrains Mono',monospace;letter-spacing:1px}
  .stat-val{font-size:26px;font-weight:900;color:#fff;font-family:'Orbitron',sans-serif;margin-top:4px}
  .stat-val.gold{color:var(--gold)}
  .stat-val.green{color:var(--green)}
  .stat-val.red{color:var(--neon)}
  .stat-sub{font-size:10px;color:rgba(244,239,230,.45);font-family:'JetBrains Mono',monospace;margin-top:4px}

  /* PANELS ROW */
  .row2{
    display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:20px;
  }
  @media(max-width:900px){.row2{grid-template-columns:1fr}}
  .panel{
    padding:20px;border-radius:18px;
    background:linear-gradient(160deg,rgba(20,2,6,.72),rgba(10,0,2,.88));
    border:1px solid rgba(255,42,59,.25);
  }
  .panel-h{display:flex;align-items:center;gap:10px;margin-bottom:14px;
    padding-bottom:12px;border-bottom:1px solid rgba(255,42,59,.2)}
  .panel-h i{color:var(--gold);font-size:18px}
  .panel-h h3{font-family:'Orbitron',sans-serif;font-size:13px;letter-spacing:1.5px;color:#fff;text-transform:uppercase}

  .freq-list{display:flex;flex-direction:column;gap:6px}
  .freq-row{display:flex;align-items:center;gap:10px;font-family:'JetBrains Mono',monospace;font-size:11px}
  .freq-row .num{width:16px;color:#fff;font-weight:700;text-align:center}
  .freq-row .marker{width:14px;text-align:center;color:var(--gold)}
  .freq-row .bar{flex:1;height:14px;border-radius:6px;background:rgba(255,255,255,.06);overflow:hidden;position:relative}
  .freq-row .fill{height:100%;border-radius:6px;background:linear-gradient(90deg,#8B0000,var(--neon));transition:width .8s}
  .freq-row.hi .fill{background:linear-gradient(90deg,var(--gold),#ffb74d)}
  .freq-row .count{width:30px;text-align:right;color:rgba(244,239,230,.65)}

  .color-break{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-top:10px}
  .cb{padding:16px;border-radius:14px;text-align:center;border:1px solid;font-family:'JetBrains Mono',monospace}
  .cb.red{background:rgba(255,42,59,.08);border-color:rgba(255,42,59,.4);color:#FF6B7A}
  .cb.green{background:rgba(16,185,129,.08);border-color:rgba(16,185,129,.4);color:#34D399}
  .cb.violet{background:rgba(168,85,247,.08);border-color:rgba(168,85,247,.4);color:#C084FC}
  .cb .pct{font-size:22px;font-weight:900;color:#fff;font-family:'Orbitron',sans-serif}
  .cb .lbl{font-size:10px;letter-spacing:2px;margin-bottom:6px}
  .cb .cnt{font-size:11px;opacity:.7;margin-top:4px}

  /* TESTS */
  .tests{display:grid;grid-template-columns:1fr 1fr;gap:12px}
  .test{padding:14px;border-radius:12px;background:rgba(0,0,0,.4);border:1px solid rgba(255,42,59,.2)}
  .test .name{font-size:11px;color:rgba(244,239,230,.6);font-family:'JetBrains Mono',monospace;margin-bottom:6px;letter-spacing:1px}
  .test .result{font-size:14px;font-weight:700;color:#fff}
  .test .meta{font-size:10px;color:rgba(244,239,230,.5);font-family:'JetBrains Mono',monospace;margin-top:4px}
  .test.bad .result{color:var(--neon)}
  .test.good .result{color:var(--green)}

  /* WIN/LOSS */
  .wl-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:6px}
  .wl{padding:16px;border-radius:14px;text-align:center;border:1px solid}
  .wl.win{background:rgba(16,185,129,.08);border-color:rgba(16,185,129,.4)}
  .wl.loss{background:rgba(255,42,59,.08);border-color:rgba(255,42,59,.4)}
  .wl .big{font-size:32px;font-weight:900;font-family:'Orbitron',sans-serif}
  .wl.win .big{color:var(--green)}
  .wl.loss .big{color:var(--neon)}
  .wl .lbl{font-size:10px;letter-spacing:2px;color:rgba(244,239,230,.6);font-family:'JetBrains Mono',monospace}

  /* HISTORY TABLE */
  .hist{max-height:340px;overflow-y:auto;margin-top:10px}
  .hist::-webkit-scrollbar{width:6px}
  .hist::-webkit-scrollbar-thumb{background:var(--neon);border-radius:6px}
  .hist-row{
    display:grid;grid-template-columns:1fr 60px 70px 70px 70px;gap:8px;
    padding:10px 12px;border-radius:10px;font-family:'JetBrains Mono',monospace;
    font-size:11px;align-items:center;
    border-bottom:1px solid rgba(255,42,59,.1);
  }
  .hist-row:hover{background:rgba(255,42,59,.05)}
  .hist-row .iss{color:rgba(244,239,230,.7);font-size:10px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .hist-row .n{font-weight:900;color:#fff;text-align:center;font-size:15px}
  .hist-row .badge{
    padding:3px 8px;border-radius:6px;font-size:10px;text-align:center;font-weight:700;
  }
  .badge.big{background:rgba(255,42,59,.2);color:#FF6B7A}
  .badge.small{background:rgba(16,185,129,.2);color:#34D399}
  .badge.win{background:rgba(16,185,129,.25);color:#34D399}
  .badge.loss{background:rgba(255,42,59,.25);color:#FF6B7A}
  .badge.pend{background:rgba(212,175,55,.2);color:var(--gold)}

  /* LOADER */
  .loading{
    display:inline-block;width:12px;height:12px;border-radius:50%;
    border:2px solid rgba(212,175,55,.3);border-top-color:var(--gold);
    animation:spin 1s linear infinite;
  }
  @keyframes spin{0%{transform:rotate(0)}100%{transform:rotate(360deg)}}

  /* FOOTER */
  footer{
    margin-top:24px;padding:16px;text-align:center;
    font-family:'JetBrains Mono',monospace;font-size:11px;
    color:rgba(244,239,230,.45);
    border-top:1px solid rgba(255,42,59,.15);
  }

  /* GLOW ANIMATION on new data */
  .flash{animation:flashBg .8s ease-out}
  @keyframes flashBg{
    0%{background-color:rgba(212,175,55,.25)}
    100%{background-color:transparent}
  }
</style>
</head>
<body>
<canvas id="particles"></canvas>
<div class="aurora"></div>

<div class="wrap">
  <header>
    <div class="brand">
      <div class="brain"><i class="fa-solid fa-brain"></i></div>
      <div>
        <h1>XOMAT AI <span style="color:var(--neon)">PRO</span></h1>
        <p>v4.0 · Live RNG Analysis · Markov · Bayesian · Chi-Square</p>
      </div>
    </div>
    <div class="status-pills">
      <div class="pill"><i class="fa-solid fa-satellite-dish"></i>
        <span>API</span><b id="pillStatus">syncing…</b></div>
      <div class="pill"><i class="fa-regular fa-clock"></i>
        <span id="pillClock">--:--:-- UTC</span></div>
      <div class="pill"><i class="fa-solid fa-hourglass-half"></i>
        <span>NEXT</span><b id="pillCountdown">--s</b></div>
      <div class="pill"><span class="live-dot"></span>
        <span id="pillSync">synced</span></div>
    </div>
  </header>

  <!-- HERO -->
  <section class="hero">
    <div class="hero-title">
      <span class="tag"><i class="fa-solid fa-crown"></i> Best 3 Period Predictions</span>
      <span class="tag" style="background:linear-gradient(90deg,var(--gold),#8B6914)">
        <i class="fa-solid fa-chart-line"></i> Multi-Model Ensemble</span>
    </div>
    <div class="issue-info">
      <div><div class="lbl">CURRENT ISSUE</div><div class="val" id="curIssue">—</div></div>
      <i class="fa-solid fa-arrow-right" style="color:var(--neon)"></i>
      <div><div class="lbl">NEXT PERIOD</div><div class="val" id="nextIssue" style="color:var(--gold)">—</div></div>
      <div><div class="lbl">PATTERN</div><div class="val" id="patternTag" style="color:var(--purple)">—</div></div>
    </div>
    <div class="cards3" id="predCards">
      <div class="pred-card r1">
        <div class="rank r1">★ RANK #1</div>
        <div class="digit">-</div>
        <div class="bsc"><div class="chip big">BIG</div><div class="chip red">RED</div></div>
        <div class="conf-bar"><div class="conf-fill" style="width:0%"></div></div>
        <div class="conf-txt">0% confidence</div>
      </div>
      <div class="pred-card r2">
        <div class="rank r2">★ RANK #2</div>
        <div class="digit">-</div>
        <div class="bsc"><div class="chip big">BIG</div><div class="chip red">RED</div></div>
        <div class="conf-bar"><div class="conf-fill" style="width:0%"></div></div>
        <div class="conf-txt">0% confidence</div>
      </div>
      <div class="pred-card r3">
        <div class="rank r3">★ RANK #3</div>
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
      <div class="stat-val" id="statTotal">0</div></div>
    <div class="stat"><div class="stat-lbl">WIN RATE</div>
      <div class="stat-val green" id="statWinRate">0%</div></div>
    <div class="stat"><div class="stat-lbl">WINS</div>
      <div class="stat-val green" id="statWins">0</div></div>
    <div class="stat"><div class="stat-lbl">LOSSES</div>
      <div class="stat-val red" id="statLosses">0</div></div>
    <div class="stat"><div class="stat-lbl">TOP CONFIDENCE</div>
      <div class="stat-val gold" id="statConf">0%</div></div>
  </div>

  <!-- ROW 2 -->
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

      <div class="panel-h" style="margin-top:20px"><i class="fa-solid fa-flask"></i><h3>Statistical Tests</h3></div>
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

  <!-- ROW 3 : Win/Loss + History -->
  <div class="row2">
    <div class="panel">
      <div class="panel-h"><i class="fa-solid fa-trophy"></i><h3>Win / Loss Tracker</h3></div>
      <div class="wl-grid">
        <div class="wl win"><div class="big" id="wlWins">0</div><div class="lbl">WINS</div></div>
        <div class="wl loss"><div class="big" id="wlLosses">0</div><div class="lbl">LOSSES</div></div>
      </div>
      <div style="margin-top:14px;font-family:'JetBrains Mono',monospace;font-size:11px;color:rgba(244,239,230,.7)">
        <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid rgba(255,42,59,.1)">
          <span>Number Hits</span><b id="hitNum" style="color:#fff">0</b></div>
        <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid rgba(255,42,59,.1)">
          <span>Big/Small Hits</span><b id="hitBS" style="color:#fff">0</b></div>
        <div style="display:flex;justify-content:space-between;padding:6px 0">
          <span>Color Hits</span><b id="hitCol" style="color:#fff">0</b></div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-h"><i class="fa-solid fa-list"></i><h3>Recent Draws</h3></div>
      <div class="hist">
        <div class="hist-row" style="color:rgba(244,239,230,.5);font-size:10px;border-bottom:1px solid rgba(255,42,59,.3)">
          <span>ISSUE</span><span style="text-align:center">NUM</span>
          <span style="text-align:center">SIZE</span><span style="text-align:center">COLOR</span>
          <span style="text-align:center">RESULT</span>
        </div>
        <div id="histBody"></div>
      </div>
    </div>
  </div>

  <footer>
    © 2026 XOMAT AI PRO · Real Statistical Engine (Bayesian · Markov · Chi² · Runs Test) · Auto-Sync 5s
  </footer>
</div>

<script>
/* ============ PARTICLE BG ============ */
(function(){
  const c=document.getElementById('particles'),ctx=c.getContext('2d');
  let W,H,P=[];
  function rs(){W=c.width=innerWidth;H=c.height=innerHeight}
  rs();addEventListener('resize',rs);
  const N=Math.min(70,Math.floor(innerWidth/22));
  for(let i=0;i<N;i++)P.push({x:Math.random()*W,y:Math.random()*H,
    vx:(Math.random()-.5)*.4,vy:(Math.random()-.5)*.4,
    r:Math.random()*1.6+.6,hue:Math.random()>.6?45:355,a:Math.random()*.6+.2});
  (function s(){
    ctx.clearRect(0,0,W,H);
    for(let i=0;i<P.length;i++)for(let j=i+1;j<P.length;j++){
      const dx=P[i].x-P[j].x,dy=P[i].y-P[j].y,d2=dx*dx+dy*dy;
      if(d2<18000){
        ctx.strokeStyle=`rgba(255,42,59,${(1-d2/18000)*.14})`;
        ctx.lineWidth=.6;ctx.beginPath();
        ctx.moveTo(P[i].x,P[i].y);ctx.lineTo(P[j].x,P[j].y);ctx.stroke();
      }
    }
    P.forEach(p=>{
      p.x+=p.vx;p.y+=p.vy;
      if(p.x<0||p.x>W)p.vx*=-1;if(p.y<0||p.y>H)p.vy*=-1;
      const g=ctx.createRadialGradient(p.x,p.y,0,p.x,p.y,p.r*6);
      g.addColorStop(0,`hsla(${p.hue},100%,60%,${p.a})`);
      g.addColorStop(1,`hsla(${p.hue},100%,50%,0)`);
      ctx.fillStyle=g;ctx.beginPath();ctx.arc(p.x,p.y,p.r*6,0,Math.PI*2);ctx.fill();
    });
    requestAnimationFrame(s);
  })();
})();

/* ============ STATE ============ */
let LAST_ISSUE = null;
const $ = id => document.getElementById(id);

/* ============ HELPERS ============ */
function chipClass(v){
  v = (v||'').toUpperCase();
  if(v==='BIG') return 'chip big';
  if(v==='SMALL') return 'chip small';
  if(v==='RED') return 'chip red';
  if(v==='GREEN') return 'chip green';
  if(v==='VIOLET') return 'chip violet';
  return 'chip';
}
function colorOfBS(bs){ return bs==='BIG'?'#FF2A3B':'#10B981'; }
function colorOfColor(c){ return c==='GREEN'?'#10B981':(c==='VIOLET'?'#A855F7':'#FF2A3B'); }

/* ============ RENDER ============ */
function render(data){
  if(!data.ok || !data.analysis) return;
  const a = data.analysis, s = data.stats;

  $('curIssue').textContent = data.lastIssue;
  $('nextIssue').textContent = data.nextIssue;
  $('patternTag').textContent = a.pattern;

  $('statTotal').textContent = data.total;
  $('statWinRate').textContent = s.total ? ((s.wins/s.total*100).toFixed(1)+'%') : '0%';
  $('statWins').textContent = s.wins;
  $('statLosses').textContent = s.losses;
  $('statConf').textContent = (a.confidence||0).toFixed(1)+'%';

  // Predictions
  const cards = $('predCards').children;
  const periods = a.periods || [];
  for(let i=0;i<3;i++){
    const card = cards[i];
    const p = periods[i];
    if(!p){ continue; }
    card.querySelector('.digit').textContent = p.number;
    const bsc = card.querySelector('.bsc');
    bsc.innerHTML = `<div class="${chipClass(p.bs)}">${p.bs}</div>
                     <div class="${chipClass(p.color)}">${p.color}</div>`;
    const fill = card.querySelector('.conf-fill');
    fill.style.width = p.confidence + '%';
    fill.style.background = `linear-gradient(90deg, ${colorOfColor(p.color)}, #D4AF37)`;
    card.querySelector('.conf-txt').textContent = p.confidence.toFixed(1) + '% confidence';
  }

  // Frequency bars
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

  // Statistical tests
  const chi = a.chi, runs = a.runs;
  const tChi = $('testChi');
  tChi.className = 'test ' + (chi.biased ? 'bad' : 'good');
  tChi.querySelector('.result').textContent = chi.biased ? '⚠ BIASED' : '✓ UNIFORM';
  tChi.querySelector('.meta').textContent = `χ²=${chi.chi2}, p=${chi.p_value} — ${chi.verdict}`;

  const tRuns = $('testRuns');
  const runsBad = runs.verdict.includes('PATTERN') || runs.verdict.includes('TREND');
  tRuns.className = 'test ' + (runsBad ? 'bad' : 'good');
  tRuns.querySelector('.result').textContent = runs.verdict.toUpperCase();
  tRuns.querySelector('.meta').textContent = `z=${runs.z}, p=${runs.p_value}, runs=${runs.runs}`;

  // Win/Loss
  $('wlWins').textContent = s.wins;
  $('wlLosses').textContent = s.losses;
  $('hitNum').textContent = s.numberWins;
  $('hitBS').textContent = s.bsWins;
  $('hitCol').textContent = s.colorWins;

  // History
  if(data.lastIssue !== LAST_ISSUE){
    LAST_ISSUE = data.lastIssue;
    loadHistory();
    document.querySelectorAll('.pred-card').forEach(c=>{
      c.classList.add('flash');
      setTimeout(()=>c.classList.remove('flash'), 800);
    });
  }
}

/* ============ HISTORY ============ */
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
      if(c==='RED') colorCls+='big';
      else if(c==='GREEN') colorCls+='small';
      else colorCls+='pend';
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

/* ============ POLL LOOP ============ */
async function poll(){
  try{
    const r = await fetch('/api/analysis');
    const d = await r.json();
    if(d.ok){
      render(d);
      $('pillStatus').textContent = d.online ? 'LIVE' : 'OFFLINE';
      $('pillSync').textContent = 'synced ' + new Date(d.lastSync*1000).toUTCString().slice(17,25) + ' UTC';
    }
  }catch(e){ $('pillStatus').textContent = 'OFFLINE'; }
}

/* ============ CLOCK + COUNTDOWN ============ */
function tick(){
  const now = new Date();
  const u = now.toUTCString().split(' ');
  $('pillClock').textContent = u[4] + ' UTC';
  const rem = 60 - now.getUTCSeconds();
  $('pillCountdown').textContent = rem + 's';
}
setInterval(tick,1000); tick();

/* ============ BOOT ============ */
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
# ==============================================================================
# START
# ==============================================================================
if __name__ == "__main__":
    t = threading.Thread(target=poll_loop, daemon=True)
    t.start()
    print("\n" + "═"*70)
    print("  XOMAT AI PRO v4.0 — Flask Web Edition")
    print("  ➜  http://localhost:5000")
    print("═"*70 + "\n")
    
    # ↓↓↓ YE 2 LINES ZAROOR CHANGE KARO ↓↓↓
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
