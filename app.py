#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 XOMAT AI PRO v7.0 — APScheduler + External Cron + AI Training Pipeline
 Full Background Data Collection | Gunicorn Safe | 24/7 via cron-job.org
==============================================================================
"""
import os, json, time, math, threading, csv
from datetime import datetime, timezone
import requests
import numpy as np

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.interval import IntervalTrigger
    HAS_APSCHEDULER = True
except ImportError:
    HAS_APSCHEDULER = False
    print("[xomat] ⚠ apscheduler not installed — falling back to threading", flush=True)

from flask import Flask, jsonify, render_template_string
from flask_cors import CORS

# ==============================================================================
# CONFIG
# ==============================================================================
API_ENDPOINTS = [
    "https://draw.ar-lottery01.com/WinGo/WinGo_1M/GetHistoryIssuePage.json?v=3.6",
    "https://draw.ar-lottery01.com/WinGo/WinGo_1M/GetHistoryIssuePage.json?v=3.31",
    "https://draw.ar-lottery01.com/WinGo/WinGo_1M/GetHistoryIssuePage.json?v=3.7",
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

# Render Disk mounted at /data? Use it, else local dir
DATA_DIR  = "/data" if os.path.exists("/data") else os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(DATA_DIR, "wingo.json")
CSV_FILE  = os.path.join(DATA_DIR, "training_data.csv")
LOG_FILE  = os.path.join(DATA_DIR, "cron.log")

POLL_SEC  = 5           # internal scheduler interval
TIMEOUT   = 12
HISTORY_LIMIT = 50000   # 50k max records in JSON (CSV unlimited)

app = Flask(__name__)
CORS(app)

# ==============================================================================
# HELPERS
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
    if "GREEN" in c: out.append("GREEN")
    if "RED" in c: out.append("RED")
    if "VIOLET" in c: out.append("VIOLET")
    return out or ["RED"]

def next_issue(iss):
    try: return str(int(iss) + 1)
    except: return iss

def log(msg):
    line = f"[xomat {datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    # Also append to log file (max 500 lines)
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f:
                lines = f.readlines()[-499:]
        else:
            lines = []
        lines.append(line + "\n")
        with open(LOG_FILE, "w") as f:
            f.writelines(lines)
    except:
        pass

# ==============================================================================
# STORE
# ==============================================================================
def empty_stats():
    return {"total": 0, "wins": 0, "losses": 0,
            "numberWins": 0, "bsWins": 0, "colorWins": 0, "lastResult": None}

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
        log(f"load error: {e}")
        return {"lastIssue": None, "lastPrediction": None,
                "pendingPeriods": [], "records": [], "stats": empty_stats()}

def save_store(store):
    try:
        os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(store, f, indent=2)
    except Exception as e:
        log(f"save error: {e}")

def append_to_csv(records):
    """AI training ke liye CSV me append karo."""
    try:
        new_file = not os.path.exists(CSV_FILE)
        with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["issue", "number", "size", "color", "ts", "iso_time",
                            "win", "predictedNumber", "predictedBS", "predictedColor"])
            for r in records:
                w.writerow([
                    r.get("issue", ""),
                    r.get("number", ""),
                    r.get("size", ""),
                    r.get("color", ""),
                    r.get("ts", ""),
                    datetime.fromtimestamp(r.get("ts", 0), timezone.utc).isoformat() if r.get("ts") else "",
                    r.get("win") if r.get("win") is not None else "",
                    r.get("predictedNumber", ""),
                    r.get("predictedBS", ""),
                    r.get("predictedColor", ""),
                ])
    except Exception as e:
        log(f"CSV append error: {e}")

# ==============================================================================
# API FETCH
# ==============================================================================
def fetch_api():
    for url in API_ENDPOINTS:
        try:
            r = requests.get(url, timeout=TIMEOUT, headers=API_HEADERS)
            if r.status_code != 200:
                continue
            data = r.json()
            lst = data.get("data", {}).get("list", [])
            if lst:
                return lst
        except Exception as e:
            continue
    return None

# ==============================================================================
# ENGINE (multi-factor scoring)
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
                if c == "RED": self.red += 1
                elif c == "GREEN": self.green += 1
                elif c == "VIOLET": self.violet += 1

        last15 = [r["number"] for r in self.records[:min(15, n)]]
        self.missing = [i for i in range(10) if i not in last15]

        self.v1 = np.zeros(10); self.v2 = np.zeros(10)
        for r in self.records[:25]: self.v1[r["number"]] += 1
        for r in self.records[25:50]: self.v2[r["number"]] += 1

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
            fbs = wingo_size(self.records[0]["number"])
            self.bs_streak = 0
            for r in self.records:
                if wingo_size(r["number"]) == fbs: self.bs_streak += 1
                else: break
            self.bs_streak_type = fbs

            fc = wingo_colors(self.records[0]["number"])[0]
            self.color_streak = 0
            for r in self.records:
                if wingo_colors(r["number"])[0] == fc: self.color_streak += 1
                else: break
            self.color_streak_type = fc
        else:
            self.bs_streak = self.color_streak = 0
            self.bs_streak_type = self.color_streak_type = "BIG"

    def chi_square_test(self):
        if self.n < 30:
            return {"chi2": 0, "p_value": 1.0, "biased": False, "verdict": "insufficient"}
        expected = np.full(10, self.n / 10.0)
        chi2 = float(np.sum((self.freq - expected) ** 2 / expected))
        p = float(1 - scipy_stats.chi2.cdf(chi2, df=9)) if HAS_SCIPY else 0.5
        return {"chi2": round(chi2, 4), "p_value": round(p, 4),
                "biased": p < 0.05,
                "verdict": "BIASED" if p < 0.05 else "uniform"}

    def runs_test(self):
        if self.n < 20:
            return {"z": 0.0, "p_value": 1.0, "runs": 0, "verdict": "insufficient"}
        seq = [1 if r["number"] >= 5 else 0 for r in self.records][::-1]
        n1 = sum(seq); n0 = len(seq) - n1
        if n1 == 0 or n0 == 0:
            return {"z": 0.0, "p_value": 1.0, "runs": 1, "verdict": "single"}
        runs = 1
        for i in range(1, len(seq)):
            if seq[i] != seq[i-1]: runs += 1
        mu = 2 * n1 * n0 / (n1 + n0) + 1
        var = (2 * n1 * n0 * (2 * n1 * n0 - n1 - n0)) / (((n1 + n0) ** 2) * (n1 + n0 - 1))
        if var <= 0: return {"z": 0.0, "p_value": 1.0, "runs": runs, "verdict": "degenerate"}
        z = (runs - mu) / math.sqrt(var)
        p = 2 * (1 - scipy_stats.norm.cdf(abs(z))) if HAS_SCIPY else 0.5
        verdict = "PATTERN" if p < 0.05 else "random"
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

    def predict(self, periods=3):
        if self.n < 5: return self._empty(periods)

        bayes = np.array(self.bayesian_posterior())
        wma = np.array(self.wma())
        m1n = self.m1 / self.m1t if self.m1t else np.full(10, 0.1)
        m2n = self.m2 / self.m2t if self.m2t else np.full(10, 0.1)
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

        composite = (bayes * W["bayes"] * 10 + wma * W["wma"] * 10 +
                     m1n * W["m1"] + m2n * W["m2"] + miss_mask * W["miss"] +
                     veln * W["vel"] + hotn * W["hot"] + coldn * W["cold"] +
                     balance * W["bal"] + anti * W["anti"] * 0.1 +
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

        if self.bs_streak >= 5: pattern = "DRAGON STREAK — BREAK IMMINENT"
        elif chi["biased"]: pattern = "RNG BIAS DETECTED"
        elif runs["verdict"] == "PATTERN": pattern = "AUTOCORRELATION PATTERN"
        elif len(self.missing) >= 4: pattern = "MULTI-MISSING CYCLE"
        elif self.m1t and (self.m1[ranked[0]] / self.m1t) > 0.18: pattern = "MARKOV STRONG SIGNAL"
        elif max(self.v1) - min(self.v1) > 3: pattern = "HIGH VELOCITY MOMENTUM"
        elif self.n >= 500: pattern = "DEEP NEURAL MATCH"
        else: pattern = "PATTERN CONVERGENCE"

        period_list = []
        for k in range(periods):
            num = int(ranked[k])
            cols = wingo_colors(num)
            period_list.append({
                "rank": k + 1, "number": num,
                "bs": wingo_size(num), "color": cols[0], "altColors": cols[1:],
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
            "top": [0,0,0], "freq": [0]*10, "missing": [],
            "score": [0.0]*10, "probability": [0.1]*10,
            "chi": {"chi2":0,"p_value":1,"biased":False,"verdict":"n/a"},
            "runs": {"z":0,"p_value":1,"runs":0,"verdict":"n/a"},
            "red":0,"green":0,"violet":0,
            "bsStreak":0,"bsStreakType":"BIG",
            "colorStreak":0,"colorStreakType":"RED",
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
    new_records = []

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
            rec.update({"predictedNumber": pn, "predictedBS": pb, "predictedColor": pc,
                        "numberWin": num_hit, "bsWin": bs_hit, "colorWin": col_hit,
                        "win": num_hit or bs_hit or col_hit, "rank": pred.get("rank", 1)})
        else:
            rec["win"] = None
        store["records"].insert(0, rec)
        new_records.append(rec)
        seen.add(iss)
        added += 1

    if len(store["records"]) > HISTORY_LIMIT:
        store["records"] = store["records"][:HISTORY_LIMIT]

    # Save new records to CSV for AI training
    if new_records:
        append_to_csv(new_records)

    s = empty_stats()
    for r in store["records"]:
        w = r.get("win")
        if w is None: continue
        s["total"] += 1
        if w: s["wins"] += 1
        else: s["losses"] += 1
        if r.get("numberWin"): s["numberWins"] += 1
        if r.get("bsWin"): s["bsWins"] += 1
        if r.get("colorWin"): s["colorWins"] += 1
    for r in store["records"]:
        if r.get("win") is not None:
            s["lastResult"] = r; break
    store["stats"] = s
    store["lastIssue"] = str(api_list[0].get("issueNumber", store.get("lastIssue") or ""))
    return added

# ==============================================================================
# CACHE
# ==============================================================================
_store_lock = threading.Lock()
_cache = {"analysis": None, "stats": empty_stats(),
          "lastIssue": "--", "total": 0, "online": False,
          "lastSync": 0, "apiStatus": "connecting", "newPrediction": False,
          "cronTicks": 0, "lastCronResult": None}

# ==============================================================================
# CORE CRON JOB (both APScheduler and external cron call this)
# ==============================================================================
def cron_fetch_and_save(trigger="internal"):
    """Main cron job — fetch API, save to JSON + CSV, update cache."""
    try:
        with _store_lock:
            _cache["cronTicks"] += 1
        api_list = fetch_api()
        if not api_list:
            with _store_lock:
                _cache["online"] = False
                _cache["apiStatus"] = "offline — API no data"
                _cache["lastCronResult"] = f"{trigger}: no data at {datetime.now(timezone.utc).strftime('%H:%M:%S')}"
            return {"ok": False, "reason": "no data"}

        store = load_store()
        newest = str(api_list[0].get("issueNumber", ""))
        last_seen = store.get("lastIssue")
        added = 0

        if newest != last_seen:
            added = sync_store(store, api_list)
            log(f"🆕 New issue: {newest} | Added {added} records | Trigger: {trigger}")
            last_seen = store.get("lastIssue")

            # Recompute predictions
            tmp_engine = XomatEngine(store["records"])
            tmp_analysis = tmp_engine.predict(periods=3)
            preds = []
            iss = store.get("lastIssue") or "0"
            for p in tmp_analysis["periods"]:
                iss = next_issue(iss)
                preds.append({
                    "forIssue": iss, "number": p["number"],
                    "bs": p["bs"], "color": p["color"], "rank": p["rank"]
                })
            store["pendingPeriods"] = preds
            store["lastPrediction"] = preds[0] if preds else None
            save_store(store)
            with _store_lock:
                _cache["newPrediction"] = True
        else:
            log(f"⏰ No new issue ({newest}) | Trigger: {trigger}")

        # Always refresh analysis in cache
        with _store_lock:
            engine = XomatEngine(store["records"])
            _cache["analysis"] = engine.predict(periods=3)
            _cache["stats"] = store["stats"]
            _cache["lastIssue"] = store.get("lastIssue") or "--"
            _cache["total"] = len(store["records"])
            _cache["lastSync"] = time.time()
            _cache["online"] = True
            _cache["apiStatus"] = f"live ({len(api_list)})"
            _cache["lastCronResult"] = f"{trigger}: ok | +{added} | {newest}"

        return {"ok": True, "added": added, "newest": newest, "total": len(store["records"])}

    except Exception as e:
        log(f"❌ Cron error ({trigger}): {e}")
        with _store_lock:
            _cache["online"] = False
            _cache["apiStatus"] = f"error: {str(e)[:50]}"
            _cache["lastCronResult"] = f"{trigger}: error {str(e)[:80]}"
        return {"ok": False, "error": str(e)}

# ==============================================================================
# INTERNAL SCHEDULER (APScheduler) — fork-safe
# ==============================================================================
_scheduler = None
_scheduler_lock = threading.Lock()

def start_scheduler():
    """Start APScheduler once per process (safe for gunicorn fork)."""
    global _scheduler
    if not HAS_APSCHEDULER:
        # Fallback: use threading loop
        return _start_thread_fallback()
    
    with _scheduler_lock:
        if _scheduler is not None and _scheduler.running:
            return
        try:
            _scheduler = BackgroundScheduler(
                daemon=True,
                job_defaults={
                    "coalesce": True,
                    "max_instances": 1,
                    "misfire_grace_time": 30,
                }
            )
            _scheduler.add_job(
                cron_fetch_and_save,
                trigger=IntervalTrigger(seconds=POLL_SEC),
                id="xomat_fetch",
                name="Fetch WinGo data",
                replace_existing=True,
                kwargs={"trigger": "APScheduler"},
            )
            _scheduler.start()
            log(f"✅ APScheduler started — every {POLL_SEC}s")
        except Exception as e:
            log(f"❌ Scheduler start failed: {e}")

# Fallback if APScheduler not available
_fallback_thread = None
_fallback_lock = threading.Lock()

def _start_thread_fallback():
    global _fallback_thread
    with _fallback_lock:
        if _fallback_thread is not None and _fallback_thread.is_alive():
            return
        def _loop():
            log("🚀 Fallback thread poller started")
            while True:
                cron_fetch_and_save(trigger="fallback-thread")
                time.sleep(POLL_SEC)
        _fallback_thread = threading.Thread(target=_loop, daemon=True, name="xomat-fallback")
        _fallback_thread.start()
        log("✅ Fallback thread started")

# Try at import time
try:
    start_scheduler()
except Exception as e:
    log(f"Initial scheduler failed: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# GUNICORN FORK SAFETY — restart scheduler on first request after fork
# ═══════════════════════════════════════════════════════════════════════════
@app.before_request
def _ensure_scheduler():
    """Har request pe check karo — agar scheduler band hai to restart."""
    global _scheduler
    if HAS_APSCHEDULER:
        if _scheduler is None or not _scheduler.running:
            start_scheduler()
    else:
        _start_thread_fallback()

# ==============================================================================
# ROUTES
# ==============================================================================
@app.route("/api/analysis")
def api_analysis():
    with _store_lock:
        a = _cache["analysis"]
        is_new = _cache["newPrediction"]
        _cache["newPrediction"] = False
        if a is None:
            return jsonify({"ok": False, "msg": "loading", "apiStatus": _cache["apiStatus"]})
        return jsonify({
            "ok": True, "analysis": a, "stats": _cache["stats"],
            "lastIssue": _cache["lastIssue"],
            "nextIssue": next_issue(_cache["lastIssue"]),
            "total": _cache["total"], "online": _cache["online"],
            "lastSync": _cache["lastSync"], "serverTime": time.time(),
            "apiStatus": _cache["apiStatus"],
            "isNewPrediction": is_new,
        })

@app.route("/api/history")
def api_history():
    store = load_store()
    return jsonify({"ok": True, "records": store["records"][:100]})

@app.route("/api/stats")
def api_stats():
    store = load_store()
    return jsonify({"ok": True, "stats": store["stats"], "total": len(store["records"])})

@app.route("/api/cron-tick")
def api_cron_tick():
    """External cron (cron-job.org) ye endpoint hit karega → fetch + save."""
    result = cron_fetch_and_save(trigger="external-cron")
    with _store_lock:
        return jsonify({
            "ok": result.get("ok", False),
            "trigger": "external-cron",
            "result": result,
            "apiStatus": _cache.get("apiStatus"),
            "total": _cache.get("total"),
            "lastIssue": _cache.get("lastIssue"),
            "serverTime": time.time(),
        })

@app.route("/api/debug")
def api_debug():
    with _store_lock:
        sched_status = "not-started"
        jobs_info = []
        if HAS_APSCHEDULER and _scheduler is not None:
            sched_status = "running" if _scheduler.running else "stopped"
            try:
                for job in _scheduler.get_jobs():
                    jobs_info.append({
                        "id": job.id,
                        "next_run": str(job.next_run_time) if job.next_run_time else None,
                        "trigger": str(job.trigger),
                    })
            except: pass
        elif _fallback_thread is not None:
            sched_status = "fallback-thread" if _fallback_thread.is_alive() else "dead"

        return jsonify({
            "ok": True,
            "scheduler": sched_status,
            "schedulerJobs": jobs_info,
            "apschedulerAvailable": HAS_APSCHEDULER,
            "apiStatus": _cache.get("apiStatus"),
            "online": _cache.get("online"),
            "lastSync": _cache.get("lastSync"),
            "lastSyncHuman": datetime.fromtimestamp(_cache.get("lastSync", 0), timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC') if _cache.get("lastSync") else None,
            "total": _cache.get("total"),
            "hasAnalysis": _cache.get("analysis") is not None,
            "lastIssue": _cache.get("lastIssue"),
            "cronTicks": _cache.get("cronTicks", 0),
            "lastCronResult": _cache.get("lastCronResult"),
            "wingoFileExists": os.path.exists(DATA_FILE),
            "wingoFileSize": os.path.getsize(DATA_FILE) if os.path.exists(DATA_FILE) else 0,
            "csvFileExists": os.path.exists(CSV_FILE),
            "csvFileSize": os.path.getsize(CSV_FILE) if os.path.exists(CSV_FILE) else 0,
            "dataDir": DATA_DIR,
            "usingRenderDisk": DATA_DIR == "/data",
        })

@app.route("/api/daily-report")
def api_daily_report():
    """Aaj kitne records aaye."""
    store = load_store()
    today = datetime.now(timezone.utc).date()
    today_records = [
        r for r in store["records"]
        if r.get("ts") and datetime.fromtimestamp(r["ts"], timezone.utc).date() == today
    ]
    return jsonify({
        "ok": True,
        "date": str(today),
        "recordsToday": len(today_records),
        "totalRecords": len(store["records"]),
        "csvExists": os.path.exists(CSV_FILE),
        "csvSizeMB": round(os.path.getsize(CSV_FILE) / 1024 / 1024, 2) if os.path.exists(CSV_FILE) else 0,
    })

@app.route("/api/export-csv")
def api_export_csv():
    """CSV file download — AI training ke liye."""
    if not os.path.exists(CSV_FILE):
        return jsonify({"ok": False, "msg": "CSV not created yet"}), 404
    from flask import send_file
    return send_file(CSV_FILE, mimetype="text/csv",
                     as_attachment=True, download_name="wingo_training_data.csv")

# ==============================================================================
# FRONTEND
# ==============================================================================
INDEX_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>XOMAT AI PRO v7 — ULTRA VFX + TTS + Auto Cron</title>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&family=Orbitron:wght@600;700;900&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
:root{
  --ivory:#F5EFE4; --ivory-2:#EDE4D2; --ivory-3:#E5D9BE; --ivory-4:#D9CBA8;
  --ivory-dark:#B8A582; --gold:#C9A227; --gold-bright:#E0BC3F; --gold-deep:#8F7211;
  --maroon:#6B0F1A; --maroon-2:#8B1420; --maroon-3:#4A0812;
  --crimson:#B8213A; --neon:#FF2A3B; --green:#0F9D58; --green-2:#13B36A;
  --purple:#7C3AED; --ink:#2A1F12; --ink-2:#4A3B24; --ink-soft:#6B5A3E;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{margin:0;padding:0;overflow-x:hidden}
body{
  background:
    radial-gradient(ellipse at 15% 8%, #FFF9EC 0%, transparent 45%),
    radial-gradient(ellipse at 85% 92%, #F0E0C8 0%, transparent 50%),
    linear-gradient(180deg, var(--ivory) 0%, var(--ivory-2) 100%);
  background-attachment:fixed;color:var(--ink);
  font-family:'Plus Jakarta Sans',sans-serif;min-height:100vh;position:relative;overflow-x:hidden;
}
#aurora{position:fixed;inset:0;pointer-events:none;z-index:0;opacity:.55;
  background:
    conic-gradient(from 0deg at 20% 30%, transparent 0deg, rgba(201,162,39,.15) 60deg, transparent 120deg),
    conic-gradient(from 180deg at 80% 70%, transparent 0deg, rgba(184,33,58,.12) 60deg, transparent 120deg);
  animation:auroraSpin 30s linear infinite;mix-blend-mode:multiply}
@keyframes auroraSpin{0%{transform:rotate(0deg) scale(1.2)}100%{transform:rotate(360deg) scale(1.2)}}
#plasma{position:fixed;inset:0;pointer-events:none;z-index:0;opacity:.35;
  background-image:linear-gradient(rgba(201,162,39,.08) 1px, transparent 1px),linear-gradient(90deg, rgba(201,162,39,.08) 1px, transparent 1px);
  background-size:40px 40px;animation:plasmaMove 20s linear infinite;
  mask-image:radial-gradient(ellipse at center, black 30%, transparent 80%);
  -webkit-mask-image:radial-gradient(ellipse at center, black 30%, transparent 80%)}
@keyframes plasmaMove{0%{background-position:0 0}100%{background-position:40px 40px}}
#particles{position:fixed;inset:0;z-index:1;pointer-events:none;opacity:.65}
body::after{content:'';position:fixed;inset:0;pointer-events:none;z-index:9998;
  background:repeating-linear-gradient(0deg, transparent 0px, transparent 2px, rgba(42,31,18,.015) 2px, rgba(42,31,18,.015) 3px);
  mix-blend-mode:multiply}
.wrap{position:relative;z-index:10;max-width:1500px;margin:0 auto;padding:16px}
header{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;
  gap:14px;padding:20px 26px;
  background:linear-gradient(135deg, #FFFCF4 0%, var(--ivory-2) 50%, var(--ivory-3) 100%);
  border:2px solid var(--gold);border-radius:24px;
  box-shadow:0 14px 48px rgba(107,15,26,.18), 0 5px 0 var(--gold-deep), inset 0 1px 0 rgba(255,255,255,.95), inset 0 -3px 20px rgba(201,162,39,.18);
  position:relative;overflow:hidden;animation:headerGlow 4s ease-in-out infinite}
@keyframes headerGlow{
  0%,100%{box-shadow:0 14px 48px rgba(107,15,26,.18), 0 5px 0 var(--gold-deep), inset 0 1px 0 rgba(255,255,255,.95), inset 0 -3px 20px rgba(201,162,39,.18)}
  50%{box-shadow:0 14px 58px rgba(201,162,39,.35), 0 5px 0 var(--gold-deep), inset 0 1px 0 rgba(255,255,255,.95), inset 0 -3px 30px rgba(224,188,63,.28)}}
header::before{content:'';position:absolute;inset:0;
  background:linear-gradient(120deg, transparent 20%, rgba(255,255,255,.5) 50%, transparent 80%);
  transform:translateX(-100%);animation:holographic 6s ease-in-out infinite}
@keyframes holographic{0%{transform:translateX(-100%)}60%,100%{transform:translateX(100%)}}
header::after{content:'';position:absolute;left:0;right:0;top:0;height:3px;
  background:linear-gradient(90deg, var(--gold-deep), var(--gold-bright), var(--crimson), var(--gold-bright), var(--gold-deep));
  background-size:300% 100%;animation:flow 5s linear infinite}
@keyframes flow{0%{background-position:0% 0%}100%{background-position:300% 0%}}
.brand{display:flex;align-items:center;gap:16px;position:relative;z-index:2}
.brain{width:62px;height:62px;border-radius:50%;display:flex;align-items:center;justify-content:center;
  background:radial-gradient(circle at 30% 30%, var(--maroon-2), var(--maroon-3));
  border:3px solid var(--gold);
  box-shadow:0 0 0 3px var(--ivory), 0 0 30px rgba(184,33,58,.5), 0 0 60px rgba(201,162,39,.3), 0 8px 20px rgba(0,0,0,.25), inset 0 0 20px rgba(0,0,0,.5);
  animation:brainPulse 2.5s infinite ease-in-out;position:relative}
@keyframes brainPulse{
  0%,100%{box-shadow:0 0 0 3px var(--ivory), 0 0 30px rgba(184,33,58,.5), 0 0 60px rgba(201,162,39,.3), 0 8px 20px rgba(0,0,0,.25), inset 0 0 20px rgba(0,0,0,.5);transform:scale(1)}
  50%{box-shadow:0 0 0 3px var(--ivory), 0 0 50px rgba(184,33,58,.7), 0 0 90px rgba(224,188,63,.5), 0 8px 20px rgba(0,0,0,.25), inset 0 0 20px rgba(0,0,0,.5);transform:scale(1.05)}}
.brain::before{content:'';position:absolute;inset:-8px;border-radius:50%;
  border:2px dashed rgba(201,162,39,.5);animation:rotateSlow 12s linear infinite}
@keyframes rotateSlow{0%{transform:rotate(0)}100%{transform:rotate(360deg)}}
.brain i{font-size:30px;color:var(--gold-bright);
  text-shadow:0 0 15px rgba(224,188,63,1), 0 0 30px rgba(224,188,63,.7);
  animation:iconPulse 1.5s ease-in-out infinite}
@keyframes iconPulse{0%,100%{transform:scale(1)}50%{transform:scale(1.15)}}
.brand h1{font-family:'Orbitron',sans-serif;font-weight:900;letter-spacing:2.5px;font-size:28px;
  background:linear-gradient(135deg, var(--maroon-2) 0%, var(--gold-deep) 50%, var(--maroon-2) 100%);
  background-size:200% 200%;-webkit-background-clip:text;background-clip:text;color:transparent;
  animation:titleShine 4s ease-in-out infinite}
@keyframes titleShine{0%,100%{background-position:0% 50%}50%{background-position:100% 50%}}
.brand p{font-size:11px;color:var(--ink-soft);font-family:'JetBrains Mono',monospace;letter-spacing:.8px;margin-top:2px}
.status-pills{display:flex;flex-wrap:wrap;gap:10px;font-size:12px;position:relative;z-index:2;align-items:center}
.pill{display:flex;align-items:center;gap:8px;padding:10px 15px;border-radius:12px;
  background:linear-gradient(180deg, #FFFDF8, var(--ivory));border:1.5px solid var(--gold);
  box-shadow:0 3px 0 var(--ivory-4), inset 0 1px 0 #fff, 0 4px 12px rgba(201,162,39,.15);
  font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--ink-2);transition:all .3s}
.pill:hover{transform:translateY(-2px);box-shadow:0 6px 0 var(--ivory-4), inset 0 1px 0 #fff, 0 8px 20px rgba(201,162,39,.25)}
.pill i{color:var(--maroon-2);font-size:13px}
.pill b{color:var(--maroon-3);font-weight:700;margin-left:2px}
.pill.live{background:linear-gradient(180deg, #F0FDF4, #DCFCE7);border-color:var(--green);animation:liveGlow 2s infinite}
@keyframes liveGlow{0%,100%{box-shadow:0 3px 0 #86EFAC, inset 0 1px 0 #fff, 0 4px 12px rgba(15,157,88,.2)}50%{box-shadow:0 3px 0 #86EFAC, inset 0 1px 0 #fff, 0 4px 24px rgba(15,157,88,.5)}}
.pill.live i,.pill.live b{color:var(--green)}
.pill.offline{background:linear-gradient(180deg, #FEF2F2, #FEE2E2);border-color:var(--crimson)}
.pill.offline i,.pill.offline b{color:var(--crimson)}
.live-dot{width:9px;height:9px;border-radius:50%;background:var(--green);
  box-shadow:0 0 0 0 rgba(15,157,88,.7);animation:livePulse 1.8s infinite}
@keyframes livePulse{0%{box-shadow:0 0 0 0 rgba(15,157,88,.7)}70%{box-shadow:0 0 0 12px rgba(15,157,88,0)}100%{box-shadow:0 0 0 0 rgba(15,157,88,0)}}
.tts-btn{display:flex;align-items:center;gap:8px;padding:10px 16px;border-radius:12px;
  background:linear-gradient(135deg, var(--maroon-2), var(--maroon-3));color:#fff;border:2px solid var(--gold);
  cursor:pointer;font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:700;
  letter-spacing:1px;text-transform:uppercase;
  box-shadow:0 4px 0 var(--maroon-3), inset 0 1px 0 rgba(255,255,255,.2), 0 6px 20px rgba(139,20,32,.3);
  transition:all .25s cubic-bezier(.34,1.56,.64,1);position:relative;overflow:hidden}
.tts-btn:hover{transform:translateY(-3px)}
.tts-btn.on{background:linear-gradient(135deg, var(--green-2), var(--green));border-color:#86EFAC}
.hero{margin-top:20px;padding:28px;border-radius:26px;
  background:linear-gradient(135deg, #FFFCF3 0%, var(--ivory-2) 40%, var(--ivory-3) 100%);
  border:2px solid var(--gold);
  box-shadow:0 18px 55px rgba(107,15,26,.18), 0 6px 0 var(--gold-deep), inset 0 1px 0 #fff, inset 0 0 100px rgba(201,162,39,.08);
  position:relative;overflow:hidden}
.hero::before{content:'';position:absolute;inset:0;pointer-events:none;
  background-image:linear-gradient(rgba(201,162,39,.06) 1px, transparent 1px),linear-gradient(90deg, rgba(201,162,39,.06) 1px, transparent 1px);
  background-size:28px 28px;animation:gridMove 15s linear infinite;
  mask-image:radial-gradient(ellipse at top, black 20%, transparent 75%);
  -webkit-mask-image:radial-gradient(ellipse at top, black 20%, transparent 75%)}
@keyframes gridMove{0%{background-position:0 0}100%{background-position:28px 28px}}
.hero-title{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:20px;position:relative;z-index:2}
.tag{background:linear-gradient(135deg, var(--maroon-2) 0%, var(--maroon) 100%);
  color:#fff;font-size:11px;font-weight:800;letter-spacing:1.8px;
  padding:8px 16px;border-radius:99px;text-transform:uppercase;font-family:'JetBrains Mono',monospace;
  box-shadow:0 4px 14px rgba(139,20,32,.4), inset 0 1px 0 rgba(255,255,255,.2);
  border:1px solid rgba(201,162,39,.5);transition:all .3s}
.tag.gold{background:linear-gradient(135deg, var(--gold-deep) 0%, var(--gold) 50%, var(--gold-deep) 100%);color:var(--maroon-3)}
.issue-info{display:flex;flex-wrap:wrap;align-items:center;gap:26px;
  font-family:'JetBrains Mono',monospace;font-size:12px;
  padding:16px 22px;margin-bottom:22px;
  background:linear-gradient(180deg, #FFFDF8, var(--ivory));
  border:1.5px solid var(--ivory-4);border-radius:16px;
  box-shadow:inset 0 2px 6px rgba(184,165,130,.15), 0 4px 14px rgba(42,31,18,.05);
  position:relative;z-index:2}
.issue-info .lbl{color:var(--ink-soft);font-size:10px;letter-spacing:1.5px;font-weight:800;margin-bottom:3px}
.issue-info .val{color:var(--maroon-3);font-weight:800;font-size:13px}
.issue-info .val.gold{color:var(--gold-deep);font-size:14px}
.issue-info .val.purple{color:var(--purple);font-size:13px}
.cards3{display:grid;grid-template-columns:repeat(auto-fit, minmax(290px, 1fr));gap:20px;margin-top:10px;position:relative;z-index:2}
.pred-card{padding:24px;border-radius:22px;text-align:center;
  background:linear-gradient(180deg, #FFFDF8 0%, var(--ivory) 100%);
  border:2px solid var(--ivory-4);
  box-shadow:0 10px 28px rgba(42,31,18,.1), 0 4px 0 var(--ivory-dark), inset 0 1px 0 #fff;
  position:relative;overflow:hidden;transition:transform .45s cubic-bezier(.34,1.56,.64,1), box-shadow .45s}
.pred-card:hover{transform:translateY(-8px) scale(1.03)}
.pred-card::before{content:'';position:absolute;top:0;left:0;right:0;height:5px;
  background:linear-gradient(90deg, var(--gold-deep), var(--gold-bright), var(--gold-deep));
  background-size:200% 100%;animation:flow 3s linear infinite}
.pred-card.r1{border-color:var(--gold);box-shadow:0 12px 35px rgba(201,162,39,.3), 0 4px 0 var(--gold-deep)}
.pred-card.r2{border-color:var(--crimson);box-shadow:0 12px 35px rgba(184,33,58,.28), 0 4px 0 #8B1420}
.pred-card.r3{border-color:var(--purple);box-shadow:0 12px 35px rgba(124,58,237,.26), 0 4px 0 #5B21B6}
.medal{position:absolute;top:14px;right:14px;width:42px;height:42px;border-radius:50%;
  display:flex;align-items:center;justify-content:center;
  font-family:'Orbitron',sans-serif;font-weight:900;font-size:16px;z-index:3}
.pred-card.r1 .medal{background:linear-gradient(135deg, #FFD966, var(--gold-deep));color:#4A3300}
.pred-card.r2 .medal{background:linear-gradient(135deg, #E0C8A0, #A08866);color:#3A2810}
.pred-card.r3 .medal{background:linear-gradient(135deg, #C4B5FD, #7C3AED);color:#2E1065}
.rank{font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:2.5px;margin-bottom:16px;font-weight:800;text-transform:uppercase}
.rank.r1{color:var(--gold-deep)}.rank.r2{color:var(--maroon-2)}.rank.r3{color:var(--purple)}
.digit{font-family:'Orbitron',sans-serif;font-weight:900;font-size:96px;line-height:1;margin:14px 0 20px;position:relative;display:inline-block;letter-spacing:-3px}
.pred-card.r1 .digit{color:var(--gold-deep);text-shadow:0 0 25px rgba(201,162,39,.6), 0 5px 0 var(--gold), 0 9px 15px rgba(42,31,18,.2);animation:digitGlow 2s ease-in-out infinite}
@keyframes digitGlow{0%,100%{filter:brightness(1) drop-shadow(0 0 15px rgba(201,162,39,.5))}50%{filter:brightness(1.25) drop-shadow(0 0 25px rgba(224,188,63,.9))}}
.pred-card.r2 .digit{color:var(--maroon-2);text-shadow:0 0 25px rgba(184,33,58,.5), 0 5px 0 var(--crimson)}
.pred-card.r3 .digit{color:var(--purple);text-shadow:0 0 25px rgba(124,58,237,.5), 0 5px 0 #5B21B6}
.digit.flash-glitch{animation:glitch 0.6s ease-out}
@keyframes glitch{0%{transform:translate(0);filter:brightness(1)}15%{transform:translate(-3px,2px);filter:brightness(1.8)}30%{transform:translate(3px,-2px);filter:brightness(1.5)}45%{transform:translate(-2px,1px);filter:brightness(1.7)}100%{transform:translate(0);filter:brightness(1)}}
.bsc{display:flex;justify-content:center;gap:10px;flex-wrap:wrap;font-family:'JetBrains Mono',monospace;font-size:12px;margin-bottom:16px}
.chip{padding:7px 16px;border-radius:10px;font-weight:900;letter-spacing:.8px;box-shadow:0 3px 0 rgba(0,0,0,.2), inset 0 1px 0 rgba(255,255,255,.4)}
.chip.big{background:linear-gradient(180deg, var(--crimson), var(--maroon-2));color:#fff}
.chip.small{background:linear-gradient(180deg, var(--green-2), var(--green));color:#fff}
.chip.red{background:linear-gradient(180deg, var(--crimson), var(--maroon-2));color:#fff}
.chip.green{background:linear-gradient(180deg, var(--green-2), var(--green));color:#fff}
.chip.violet{background:linear-gradient(180deg, #A78BFA, var(--purple));color:#fff}
.conf-bar{height:12px;border-radius:8px;background:linear-gradient(180deg, #E8DCC0, var(--ivory-4));box-shadow:inset 0 2px 5px rgba(42,31,18,.18);overflow:hidden;margin-top:14px;position:relative}
.conf-fill{height:100%;border-radius:8px;background:linear-gradient(90deg, var(--maroon-2), var(--gold-bright), var(--crimson), var(--gold-bright), var(--maroon-2));background-size:300% 100%;animation:confShine 3s linear infinite;transition:width 1.4s cubic-bezier(.4,0,.2,1);box-shadow:0 0 18px rgba(201,162,39,.6);position:relative;overflow:hidden}
.conf-fill::after{content:'';position:absolute;inset:0;background:linear-gradient(90deg, transparent, rgba(255,255,255,.5), transparent);animation:liquidFlow 2s linear infinite}
@keyframes confShine{0%{background-position:0% 0%}100%{background-position:300% 0%}}
@keyframes liquidFlow{0%{transform:translateX(-100%)}100%{transform:translateX(100%)}}
.conf-txt{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--ink-2);margin-top:10px;font-weight:800}
.grid-stat{display:grid;grid-template-columns:repeat(auto-fit, minmax(200px, 1fr));gap:16px;margin-top:26px}
.stat{padding:22px;border-radius:20px;background:linear-gradient(180deg, #FFFDF8, var(--ivory));border:1.5px solid var(--ivory-4);box-shadow:0 8px 22px rgba(42,31,18,.08), 0 4px 0 var(--ivory-dark), inset 0 1px 0 #fff;transition:all .35s;position:relative;overflow:hidden}
.stat::before{content:'';position:absolute;top:0;left:0;width:5px;height:100%;background:linear-gradient(180deg, var(--gold), var(--maroon-2))}
.stat:hover{transform:translateY(-5px);box-shadow:0 16px 40px rgba(42,31,18,.15), 0 4px 0 var(--ivory-dark)}
.stat-lbl{font-size:11px;color:var(--ink-soft);font-family:'JetBrains Mono',monospace;letter-spacing:1.5px;font-weight:800;margin-bottom:8px}
.stat-val{font-size:30px;font-weight:900;font-family:'Orbitron',sans-serif;color:var(--maroon-3);line-height:1.1}
.stat-val.gold{color:var(--gold-deep)}.stat-val.green{color:var(--green)}.stat-val.red{color:var(--crimson)}
.stat-sub{font-size:10px;color:var(--ink-soft);font-family:'JetBrains Mono',monospace;margin-top:6px}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:26px}
@media(max-width:900px){.row2{grid-template-columns:1fr}}
.panel{padding:24px;border-radius:22px;background:linear-gradient(180deg, #FFFDF8 0%, var(--ivory) 100%);border:1.5px solid var(--ivory-4);box-shadow:0 10px 26px rgba(42,31,18,.08), 0 4px 0 var(--ivory-dark);position:relative}
.panel-h{display:flex;align-items:center;gap:12px;margin-bottom:18px;padding-bottom:16px;border-bottom:2px solid var(--ivory-3)}
.panel-h i{color:var(--maroon-2);font-size:18px;padding:9px;border-radius:11px;background:linear-gradient(180deg, var(--ivory-2), var(--ivory-3));box-shadow:0 3px 0 var(--ivory-4)}
.panel-h h3{font-family:'Orbitron',sans-serif;font-size:13px;letter-spacing:1.8px;color:var(--maroon-3);text-transform:uppercase;font-weight:800}
.freq-list{display:flex;flex-direction:column;gap:8px}
.freq-row{display:flex;align-items:center;gap:10px;font-family:'JetBrains Mono',monospace;font-size:11px;padding:4px 0}
.freq-row .num{width:24px;height:24px;line-height:24px;text-align:center;color:#fff;font-weight:900;border-radius:7px;background:linear-gradient(180deg, var(--maroon-2), var(--maroon-3));font-size:12px}
.freq-row.hi .num{background:linear-gradient(180deg, var(--gold-bright), var(--gold-deep));color:#3A2810}
.freq-row .marker{width:18px;text-align:center;color:var(--gold-deep);font-size:16px;font-weight:900}
.freq-row .bar{flex:1;height:20px;border-radius:10px;background:linear-gradient(180deg, #E8DCC0, var(--ivory-4));box-shadow:inset 0 2px 5px rgba(42,31,18,.18);overflow:hidden}
.freq-row .fill{height:100%;border-radius:10px;background:linear-gradient(90deg, var(--maroon-3), var(--maroon-2), var(--crimson));transition:width 1s}
.freq-row.hi .fill{background:linear-gradient(90deg, var(--gold-deep), var(--gold-bright))}
.freq-row .count{width:38px;text-align:right;color:var(--ink-2);font-weight:800}
.color-break{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px}
.cb{padding:20px 14px;border-radius:16px;text-align:center;font-family:'JetBrains Mono',monospace;border:2px solid;position:relative;overflow:hidden}
.cb.red{background:linear-gradient(180deg, #FEF2F2, #FECACA);border-color:#DC2626;color:#991B1B}
.cb.green{background:linear-gradient(180deg, #F0FDF4, #BBF7D0);border-color:var(--green);color:#065F46}
.cb.violet{background:linear-gradient(180deg, #FAF5FF, #E9D5FF);border-color:var(--purple);color:#5B21B6}
.cb .pct{font-size:26px;font-weight:900;font-family:'Orbitron',sans-serif;color:inherit;margin:6px 0}
.cb .lbl{font-size:10px;letter-spacing:2.5px;margin-bottom:5px;font-weight:900}
.cb .cnt{font-size:12px;opacity:.85;font-weight:700}
.tests{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:10px}
.test{padding:18px;border-radius:14px;background:linear-gradient(180deg, #FFFDF8, var(--ivory));border:1.5px solid var(--ivory-4);box-shadow:0 4px 0 var(--ivory-dark)}
.test .name{font-size:10px;color:var(--ink-soft);font-family:'JetBrains Mono',monospace;margin-bottom:9px;letter-spacing:1.3px;font-weight:800}
.test .result{font-size:15px;font-weight:900;font-family:'Orbitron',sans-serif}
.test .meta{font-size:10px;color:var(--ink-soft);font-family:'JetBrains Mono',monospace;margin-top:6px}
.test.bad{background:linear-gradient(180deg, #FEF2F2, #FEE2E2);border-color:var(--crimson)}.test.bad .result{color:var(--crimson)}
.test.good{background:linear-gradient(180deg, #F0FDF4, #DCFCE7);border-color:var(--green)}.test.good .result{color:var(--green)}
.wl-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:8px}
.wl{padding:22px 14px;border-radius:16px;text-align:center;border:2px solid}
.wl.win{background:linear-gradient(180deg, #F0FDF4, #BBF7D0);border-color:var(--green)}
.wl.loss{background:linear-gradient(180deg, #FEF2F2, #FECACA);border-color:var(--crimson)}
.wl .big{font-size:40px;font-weight:900;font-family:'Orbitron',sans-serif;line-height:1}
.wl.win .big{color:#047857}.wl.loss .big{color:#B91C1C}
.wl .lbl{font-size:10px;letter-spacing:2.5px;color:var(--ink-soft);font-family:'JetBrains Mono',monospace;font-weight:900;margin-top:6px}
.hist{max-height:420px;overflow-y:auto;margin-top:12px;padding-right:8px}
.hist::-webkit-scrollbar{width:8px}
.hist::-webkit-scrollbar-thumb{background:linear-gradient(180deg, var(--maroon-2), var(--maroon-3));border-radius:8px}
.hist-row{display:grid;grid-template-columns:1fr 58px 72px 72px 72px;gap:9px;padding:12px 14px;border-radius:11px;font-family:'JetBrains Mono',monospace;font-size:11px;align-items:center;border-bottom:1px solid var(--ivory-3)}
.hist-row:hover{background:rgba(201,162,39,.08)}
.hist-row .iss{color:var(--ink-soft);font-size:10px;font-weight:700;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.hist-row .n{font-weight:900;color:var(--maroon-3);text-align:center;font-size:17px;font-family:'Orbitron',sans-serif}
.hist-row .badge{padding:5px 9px;border-radius:8px;font-size:10px;text-align:center;font-weight:900;letter-spacing:.6px}
.badge.big{background:linear-gradient(180deg, var(--crimson), var(--maroon-2));color:#fff}
.badge.small{background:linear-gradient(180deg, var(--green-2), var(--green));color:#fff}
.badge.red{background:linear-gradient(180deg, var(--crimson), var(--maroon-2));color:#fff}
.badge.green{background:linear-gradient(180deg, var(--green-2), var(--green));color:#fff}
.badge.violet{background:linear-gradient(180deg, #A78BFA, var(--purple));color:#fff}
.badge.win{background:linear-gradient(180deg, var(--green-2), #047857);color:#fff}
.badge.loss{background:linear-gradient(180deg, var(--crimson), #7F1D1D);color:#fff}
.badge.pend{background:linear-gradient(180deg, var(--gold-bright), var(--gold-deep));color:#3A2810}
footer{margin-top:30px;padding:26px;text-align:center;font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--ink-soft);font-weight:700;border-top:2px solid var(--ivory-4)}
.flash{animation:flashBg 1s ease-out}
@keyframes flashBg{0%{background-color:rgba(201,162,39,.35);transform:scale(1.015)}100%{background-color:transparent;transform:scale(1)}}
.toast{position:fixed;bottom:24px;left:50%;transform:translateX(-50%) translateY(120%);padding:14px 24px;border-radius:14px;font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:700;z-index:99999;background:linear-gradient(135deg, var(--maroon-2), var(--maroon-3));color:#fff;border:2px solid var(--gold);box-shadow:0 10px 40px rgba(139,20,32,.5);transition:transform .4s cubic-bezier(.34,1.56,.64,1);max-width:90vw;text-align:center}
.toast.show{transform:translateX(-50%) translateY(0)}
@media(max-width:640px){.brand h1{font-size:22px}.digit{font-size:76px}.stat-val{font-size:22px}.hero{padding:20px}.panel{padding:18px}}
</style>
</head>
<body>
<div id="aurora"></div>
<div id="plasma"></div>
<canvas id="particles"></canvas>

<div class="wrap">
  <header>
    <div class="brand">
      <div class="brain"><i class="fa-solid fa-brain"></i></div>
      <div>
        <h1>XOMAT AI <span style="color:var(--maroon-2)">PRO</span></h1>
        <p>v7.0 · APScheduler + External Cron · AI Training</p>
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
      <button class="tts-btn" id="ttsBtn" onclick="toggleTTS()">
        <i class="fa-solid fa-volume-xmark" id="ttsIcon"></i>
        <span id="ttsText">Voice OFF</span>
      </button>
    </div>
  </header>

  <section class="hero">
    <div class="hero-title">
      <span class="tag"><i class="fa-solid fa-crown"></i> Best 3 Periods</span>
      <span class="tag gold"><i class="fa-solid fa-chart-line"></i> Multi-Model Ensemble</span>
      <span class="tag" style="background:linear-gradient(135deg,var(--green),#047857)">
        <i class="fa-solid fa-clock"></i> AUTO CRON
      </span>
    </div>
    <div class="issue-info">
      <div><div class="lbl">CURRENT ISSUE</div><div class="val" id="curIssue">—</div></div>
      <div><div class="lbl">NEXT PERIOD</div><div class="val gold" id="nextIssue">—</div></div>
      <div><div class="lbl">PATTERN</div><div class="val purple" id="patternTag">—</div></div>
      <div><div class="lbl">CRON TICKS</div><div class="val" id="cronTicks">0</div></div>
    </div>
    <div class="cards3" id="predCards">
      <div class="pred-card r1"><div class="medal">1</div><div class="rank r1">★ RANK #1</div><div class="digit">-</div><div class="bsc"><div class="chip big">BIG</div><div class="chip red">RED</div></div><div class="conf-bar"><div class="conf-fill" style="width:0%"></div></div><div class="conf-txt">0% confidence</div></div>
      <div class="pred-card r2"><div class="medal">2</div><div class="rank r2">★ RANK #2</div><div class="digit">-</div><div class="bsc"><div class="chip big">BIG</div><div class="chip red">RED</div></div><div class="conf-bar"><div class="conf-fill" style="width:0%"></div></div><div class="conf-txt">0% confidence</div></div>
      <div class="pred-card r3"><div class="medal">3</div><div class="rank r3">★ RANK #3</div><div class="digit">-</div><div class="bsc"><div class="chip big">BIG</div><div class="chip red">RED</div></div><div class="conf-bar"><div class="conf-fill" style="width:0%"></div></div><div class="conf-txt">0% confidence</div></div>
    </div>
  </section>

  <div class="grid-stat">
    <div class="stat"><div class="stat-lbl">TOTAL RECORDS</div><div class="stat-val" id="statTotal">0</div><div class="stat-sub">JSON + CSV</div></div>
    <div class="stat"><div class="stat-lbl">WIN RATE</div><div class="stat-val green" id="statWinRate">0%</div><div class="stat-sub">all-time accuracy</div></div>
    <div class="stat"><div class="stat-lbl">WINS</div><div class="stat-val green" id="statWins">0</div><div class="stat-sub">successful predictions</div></div>
    <div class="stat"><div class="stat-lbl">LOSSES</div><div class="stat-val red" id="statLosses">0</div><div class="stat-sub">failed predictions</div></div>
    <div class="stat"><div class="stat-lbl">TOP CONFIDENCE</div><div class="stat-val gold" id="statConf">0%</div><div class="stat-sub">rank #1 signal</div></div>
  </div>

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
      <div class="panel-h" style="margin-top:24px"><i class="fa-solid fa-flask"></i><h3>Statistical Tests</h3></div>
      <div class="tests">
        <div class="test" id="testChi"><div class="name">CHI-SQUARE</div><div class="result">—</div><div class="meta">p-value: —</div></div>
        <div class="test" id="testRuns"><div class="name">RUNS TEST</div><div class="result">—</div><div class="meta">z-score: —</div></div>
      </div>
    </div>
  </div>

  <div class="row2">
    <div class="panel">
      <div class="panel-h"><i class="fa-solid fa-trophy"></i><h3>Win / Loss Tracker</h3></div>
      <div class="wl-grid">
        <div class="wl win"><div class="big" id="wlWins">0</div><div class="lbl">WINS</div></div>
        <div class="wl loss"><div class="big" id="wlLosses">0</div><div class="lbl">LOSSES</div></div>
      </div>
      <div style="margin-top:18px;font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--ink-2)">
        <div style="display:flex;justify-content:space-between;padding:9px 0;border-bottom:1px solid var(--ivory-3)"><span style="font-weight:700">Number Hits</span><b id="hitNum" style="color:var(--maroon-2)">0</b></div>
        <div style="display:flex;justify-content:space-between;padding:9px 0;border-bottom:1px solid var(--ivory-3)"><span style="font-weight:700">Big/Small Hits</span><b id="hitBS" style="color:var(--maroon-2)">0</b></div>
        <div style="display:flex;justify-content:space-between;padding:9px 0"><span style="font-weight:700">Color Hits</span><b id="hitCol" style="color:var(--maroon-2)">0</b></div>
      </div>
      <div style="margin-top:14px;padding:12px;background:linear-gradient(180deg,#F0FDF4,#DCFCE7);border:1.5px solid var(--green);border-radius:10px;font-family:'JetBrains Mono',monospace;font-size:10px;color:#065F46">
        <div style="font-weight:900;margin-bottom:4px">📊 AI TRAINING DATA</div>
        <div style="display:flex;justify-content:space-between;padding:2px 0"><span>CSV exists:</span><b id="csvExists">—</b></div>
        <div style="display:flex;justify-content:space-between;padding:2px 0"><span>CSV size:</span><b id="csvSize">— MB</b></div>
        <a href="/api/export-csv" style="display:block;margin-top:6px;padding:6px;background:#0F9D58;color:#fff;text-align:center;border-radius:6px;font-weight:900;text-decoration:none">⬇ Download CSV</a>
      </div>
    </div>
    <div class="panel">
      <div class="panel-h"><i class="fa-solid fa-list"></i><h3>Recent Draws</h3></div>
      <div class="hist">
        <div class="hist-row" style="color:var(--ink-soft);font-size:10px;border-bottom:2px solid var(--ivory-4);font-weight:800"><span>ISSUE</span><span style="text-align:center">NUM</span><span style="text-align:center">SIZE</span><span style="text-align:center">COLOR</span><span style="text-align:center">RESULT</span></div>
        <div id="histBody"></div>
      </div>
    </div>
  </div>

  <footer>
    © 2026 XOMAT AI PRO v7.0 · APScheduler + External Cron · AI Training CSV · Auto-Sync
  </footer>
</div>

<div class="toast" id="toast"><span id="toastText">Notification</span></div>

<script>
(function(){
  const c=document.getElementById('particles'),ctx=c.getContext('2d');
  let W,H,P=[];
  function rs(){W=c.width=innerWidth;H=c.height=innerHeight}
  rs();addEventListener('resize',rs);
  const N=Math.min(90,Math.floor(innerWidth/18));
  for(let i=0;i<N;i++)P.push({x:Math.random()*W,y:Math.random()*H,vx:(Math.random()-.5)*.4,vy:(Math.random()-.5)*.4,r:Math.random()*2+.8,hue:Math.random()>.5?42:355,a:Math.random()*.6+.3});
  let mx=0,my=0;
  addEventListener('mousemove',e=>{mx=e.clientX;my=e.clientY});
  (function s(){
    ctx.clearRect(0,0,W,H);
    for(let i=0;i<P.length;i++){
      for(let j=i+1;j<P.length;j++){
        const dx=P[i].x-P[j].x,dy=P[i].y-P[j].y,d2=dx*dx+dy*dy;
        if(d2<22000){ctx.strokeStyle=`hsla(${P[i].hue},70%,45%,${(1-d2/22000)*.22})`;ctx.lineWidth=.8;ctx.beginPath();ctx.moveTo(P[i].x,P[i].y);ctx.lineTo(P[j].x,P[j].y);ctx.stroke();}
      }
      const mdx=P[i].x-mx,mdy=P[i].y-my,md2=mdx*mdx+mdy*mdy;
      if(md2<30000){ctx.strokeStyle=`hsla(${P[i].hue},90%,60%,${(1-md2/30000)*.5})`;ctx.lineWidth=1.2;ctx.beginPath();ctx.moveTo(P[i].x,P[i].y);ctx.lineTo(mx,my);ctx.stroke();}
    }
    P.forEach(p=>{
      p.x+=p.vx;p.y+=p.vy;
      if(p.x<0||p.x>W)p.vx*=-1;if(p.y<0||p.y>H)p.vy*=-1;
      const g=ctx.createRadialGradient(p.x,p.y,0,p.x,p.y,p.r*8);
      g.addColorStop(0,`hsla(${p.hue},85%,60%,${p.a})`);
      g.addColorStop(1,`hsla(${p.hue},85%,50%,0)`);
      ctx.fillStyle=g;ctx.beginPath();ctx.arc(p.x,p.y,p.r*8,0,Math.PI*2);ctx.fill();
    });
    requestAnimationFrame(s);
  })();
})();

let ttsEnabled=false,lastSpokenIssue=null,preferredVoice=null;
function initVoices(){
  const voices=speechSynthesis.getVoices();
  if(!voices.length)return;
  preferredVoice=voices.find(v=>v.lang==='en-IN')||voices.find(v=>v.lang==='en-US'&&v.name.toLowerCase().includes('female'))||voices.find(v=>v.lang==='en-US')||voices.find(v=>v.lang.startsWith('en'))||voices[0];
}
speechSynthesis.onvoiceschanged=initVoices;initVoices();
function toggleTTS(){
  ttsEnabled=!ttsEnabled;
  const btn=document.getElementById('ttsBtn'),icon=document.getElementById('ttsIcon'),txt=document.getElementById('ttsText');
  if(ttsEnabled){btn.classList.add('on');icon.className='fa-solid fa-volume-high';txt.textContent='Voice ON';speak("Voice announcements enabled.");showToast("🔊 Voice ON");}
  else{btn.classList.remove('on');icon.className='fa-solid fa-volume-xmark';txt.textContent='Voice OFF';speechSynthesis.cancel();showToast("🔇 Voice OFF");}
}
function speak(text){
  if(!ttsEnabled||!text)return;
  speechSynthesis.cancel();
  const u=new SpeechSynthesisUtterance(text);
  u.lang='en-IN';u.rate=1.0;u.pitch=1.05;
  if(preferredVoice)u.voice=preferredVoice;
  speechSynthesis.speak(u);
}
function announcePrediction(a){
  const p1=a.periods?.[0],p2=a.periods?.[1],p3=a.periods?.[2];
  if(!p1)return;
  let t=`Next period prediction. Primary: ${p1.number}, ${p1.bs}, color ${p1.color}, ${Math.round(p1.confidence)} percent. `;
  if(p2)t+=`Secondary: ${p2.number}, ${p2.bs}. `;
  if(p3)t+=`Tertiary: ${p3.number}. `;
  t+=`Pattern: ${a.pattern}.`;
  speak(t);
}
function showToast(msg){
  const t=document.getElementById('toast');
  document.getElementById('toastText').textContent=msg;
  t.classList.add('show');
  clearTimeout(t._tid);
  t._tid=setTimeout(()=>t.classList.remove('show'),3000);
}
const $=id=>document.getElementById(id);
let LAST_ISSUE=null;
function chipClass(v){
  v=(v||'').toUpperCase();
  if(v==='BIG')return'chip big';if(v==='SMALL')return'chip small';
  if(v==='RED')return'chip red';if(v==='GREEN')return'chip green';
  if(v==='VIOLET')return'chip violet';return'chip';
}
function render(data){
  if(!data.ok||!data.analysis)return;
  const a=data.analysis,s=data.stats;
  const pillApi=$('pillApi');
  if(data.online){pillApi.classList.remove('offline');pillApi.classList.add('live');$('apiStatus').textContent=data.apiStatus||'LIVE';}
  else{pillApi.classList.remove('live');pillApi.classList.add('offline');$('apiStatus').textContent=data.apiStatus||'OFFLINE';}
  $('curIssue').textContent=data.lastIssue;
  $('nextIssue').textContent=data.nextIssue;
  $('patternTag').textContent=a.pattern;
  $('statTotal').textContent=data.total;
  $('statWinRate').textContent=s.total?((s.wins/s.total*100).toFixed(1)+'%'):'0%';
  $('statWins').textContent=s.wins;
  $('statLosses').textContent=s.losses;
  $('statConf').textContent=(a.confidence||0).toFixed(1)+'%';
  const cards=$('predCards').children;
  const periods=a.periods||[];
  for(let i=0;i<3;i++){
    const card=cards[i],p=periods[i];
    if(!p)continue;
    const dEl=card.querySelector('.digit');
    dEl.textContent=p.number;
    const bsc=card.querySelector('.bsc');
    bsc.innerHTML=`<div class="${chipClass(p.bs)}">${p.bs}</div><div class="${chipClass(p.color)}">${p.color}</div>`;
    card.querySelector('.conf-fill').style.width=p.confidence+'%';
    card.querySelector('.conf-txt').textContent=p.confidence.toFixed(1)+'% confidence';
  }
  const freq=a.freq,maxFreq=Math.max(...freq)||1;
  const top1=a.top[0],top2=a.top[1],top3=a.top[2];
  let fh='';
  for(let i=0;i<10;i++){
    const w=(freq[i]/maxFreq*100).toFixed(0);
    const marker=i===top1?'★':(i===top2?'☆':(i===top3?'◆':''));
    const cls=(i===top1||i===top2||i===top3)?'freq-row hi':'freq-row';
    fh+=`<div class="${cls}"><span class="marker">${marker}</span><span class="num">${i}</span><span class="bar"><span class="fill" style="width:${w}%"></span></span><span class="count">${freq[i]}</span></div>`;
  }
  $('freqList').innerHTML=fh;
  const total=(a.red+a.green+a.violet)||1;
  $('cbRedPct').textContent=((a.red/total)*100).toFixed(1)+'%';
  $('cbGreenPct').textContent=((a.green/total)*100).toFixed(1)+'%';
  $('cbVioletPct').textContent=((a.violet/total)*100).toFixed(1)+'%';
  $('cbRedCnt').textContent=a.red;$('cbGreenCnt').textContent=a.green;$('cbVioletCnt').textContent=a.violet;
  const chi=a.chi,runs=a.runs;
  const tChi=$('testChi');
  tChi.className='test '+(chi.biased?'bad':'good');
  tChi.querySelector('.result').textContent=chi.biased?'⚠ BIASED':'✓ UNIFORM';
  tChi.querySelector('.meta').textContent=`χ²=${chi.chi2}, p=${chi.p_value}`;
  const tRuns=$('testRuns');
  const runsBad=runs.verdict.includes('PATTERN')||runs.verdict.includes('TREND');
  tRuns.className='test '+(runsBad?'bad':'good');
  tRuns.querySelector('.result').textContent=runs.verdict.toUpperCase();
  tRuns.querySelector('.meta').textContent=`z=${runs.z}, p=${runs.p_value}`;
  $('wlWins').textContent=s.wins;$('wlLosses').textContent=s.losses;
  $('hitNum').textContent=s.numberWins;$('hitBS').textContent=s.bsWins;$('hitCol').textContent=s.colorWins;
  $('liveText').textContent=data.online?'live':'offline';
  if(data.lastIssue!==LAST_ISSUE){
    LAST_ISSUE=data.lastIssue;
    loadHistory();
    document.querySelectorAll('.pred-card').forEach(c=>{c.classList.add('flash');setTimeout(()=>c.classList.remove('flash'),1000);});
    document.querySelectorAll('.digit').forEach(d=>{d.classList.add('flash-glitch');setTimeout(()=>d.classList.remove('flash-glitch'),700);});
    if(ttsEnabled&&lastSpokenIssue!==data.nextIssue){lastSpokenIssue=data.nextIssue;setTimeout(()=>announcePrediction(a),500);}
    showToast(`🆕 New issue: ${data.lastIssue}`);
  }
}
async function loadHistory(){
  try{
    const r=await fetch('/api/history');const d=await r.json();
    if(!d.ok)return;
    $('histBody').innerHTML=d.records.slice(0,40).map(rec=>{
      const sizeCls=rec.size==='BIG'?'badge big':'badge small';
      let colorCls='badge ';
      const c=(rec.color||'').split('/')[0];
      if(c==='RED')colorCls+='red';else if(c==='GREEN')colorCls+='green';else colorCls+='violet';
      let res='<span class="badge pend">—</span>';
      if(rec.win===true)res='<span class="badge win">WIN</span>';
      if(rec.win===false)res='<span class="badge loss">LOSS</span>';
      return `<div class="hist-row"><span class="iss">${rec.issue}</span><span class="n">${rec.number}</span><span class="${sizeCls}">${rec.size}</span><span class="${colorCls}">${c||'-'}</span>${res}</div>`;
    }).join('');
  }catch(e){}
}
async function loadDebug(){
  try{
    const r=await fetch('/api/debug');const d=await r.json();
    $('cronTicks').textContent=d.cronTicks||0;
    $('csvExists').textContent=d.csvFileExists?'✅ Yes':'❌ No';
    $('csvSize').textContent=(d.csvFileSize?(d.csvFileSize/1024/1024).toFixed(2):'0.00')+' MB';
  }catch(e){}
}
async function poll(){
  try{
    const r=await fetch('/api/analysis');const d=await r.json();
    if(d.ok)render(d);
  }catch(e){
    $('pillApi').classList.remove('live');$('pillApi').classList.add('offline');$('apiStatus').textContent='OFFLINE';
  }
}
function tick(){
  const now=new Date();
  const u=now.toUTCString().split(' ');
  $('pillClock').textContent=u[4]+' UTC';
  const rem=60-now.getUTCSeconds();
  $('pillCountdown').textContent=rem+'s';
}
setInterval(tick,1000);tick();
setInterval(poll,3000);poll();
setInterval(loadDebug,5000);loadDebug();
loadHistory();
setTimeout(()=>showToast("🚀 XOMAT v7.0 · Auto Cron running · Tap Voice to enable TTS"),1200);
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

# ==============================================================================
# LOCAL DEV
# ==============================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "═"*70, flush=True)
    print("  XOMAT AI PRO v7.0 — APScheduler + External Cron", flush=True)
    print(f"  ➜  http://localhost:{port}", flush=True)
    print("═"*70 + "\n", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
