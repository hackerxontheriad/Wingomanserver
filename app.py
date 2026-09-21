#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 XOMAT AI PRO v8.2 — Pattern Matcher + Upload + 4-Day Training Target
 - JSON Upload with timestamped path
 - Time Sequence table (issue, sum, size, color, number)
 - CSV auto-export (AI training)
 - Target: 5760 records (4 days × 1440) for full training
==============================================================================
"""
import os, json, time, math, threading, csv, base64
from datetime import datetime, timezone
from collections import Counter
import requests
import numpy as np

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    HAS_APSCHEDULER = True
except ImportError:
    HAS_APSCHEDULER = False

from flask import Flask, jsonify, render_template_string, send_file, request
from flask_cors import CORS

# ==============================================================================
# CONFIG
# ==============================================================================
API_ENDPOINTS = [
    "https://draw.ar-lottery01.com/WinGo/WinGo_1M/GetHistoryIssuePage.json?v=3.6",
    "https://draw.ar-lottery01.com/WinGo/WinGo_1M/GetHistoryIssuePage.json?v=3.31",
    "https://draw.ar-lottery01.com/WinGo/WinGo_1M/GetHistoryIssuePage.json",
]
API_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://draw.ar-lottery01.com/",
    "Origin": "https://draw.ar-lottery01.com",
}

DATA_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_FILE    = os.path.join(DATA_DIR, "wingo.json")
UPLOAD_DIR   = os.path.join(DATA_DIR, "uploads")
CSV_FILE     = os.path.join(DATA_DIR, "training_data.csv")
POLL_SEC     = 60
TIMEOUT      = 12
HISTORY_LIMIT = 20000

# 4-day training target
TRAINING_TARGET = 1440 * 4   # = 5760 records

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
    print(f"[xomat {datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)

def calc_sum(records, idx, window=5):
    """Last N numbers ka sum (from idx toward older)."""
    end = min(idx + window, len(records))
    return sum(r["number"] for r in records[idx:end])

# ==============================================================================
# STORE
# ==============================================================================
def empty_stats():
    return {"total": 0, "wins": 0, "losses": 0,
            "numberWins": 0, "bsWins": 0, "colorWins": 0, "lastResult": None}

def load_store():
    if not os.path.exists(DATA_FILE):
        return {"lastIssue": None, "pending": [], "records": [], "stats": empty_stats()}
    try:
        with open(DATA_FILE) as f:
            d = json.load(f)
        if isinstance(d, list):
            return {"lastIssue": None, "pending": [], "records": d, "stats": empty_stats()}
        d.setdefault("records", []); d.setdefault("stats", empty_stats()); d.setdefault("pending", [])
        return d
    except Exception as e:
        log(f"load error: {e}")
        return {"lastIssue": None, "pending": [], "records": [], "stats": empty_stats()}

def save_store(store):
    try:
        with open(DATA_FILE, "w") as f:
            json.dump(store, f, indent=2)
    except Exception as e:
        log(f"save error: {e}")

def append_to_csv(records):
    """Append records to CSV for AI training."""
    try:
        new_file = not os.path.exists(CSV_FILE)
        with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["issue", "number", "size", "color", "sum5", "ts", "iso_time", "win"])
            for r in records:
                w.writerow([
                    r.get("issue", ""), r.get("number", ""),
                    r.get("size", ""), r.get("color", ""),
                    r.get("sum5", ""), r.get("ts", ""),
                    datetime.fromtimestamp(r.get("ts", 0), timezone.utc).isoformat() if r.get("ts") else "",
                    r.get("win") if r.get("win") is not None else "",
                ])
    except Exception as e:
        log(f"CSV error: {e}")

def regenerate_full_csv():
    """Rebuild the entire CSV from store (after bulk upload)."""
    try:
        store = load_store()
        records = store["records"]
        try:
            records = sorted(records,
                             key=lambda r: int(r["issue"]) if str(r["issue"]).isdigit() else 0)
        except: pass

        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["issue", "number", "size", "color", "sum5", "ts", "iso_time", "win"])
            for i, r in enumerate(records):
                s5 = calc_sum(records, max(0, len(records) - 1 - i), 5)
                w.writerow([
                    r.get("issue", ""), r.get("number", ""),
                    r.get("size", ""), r.get("color", ""),
                    s5, r.get("ts", ""),
                    datetime.fromtimestamp(r.get("ts", 0), timezone.utc).isoformat() if r.get("ts") else "",
                    r.get("win") if r.get("win") is not None else "",
                ])
        log(f"♻ Regenerated CSV with {len(records)} records")
    except Exception as e:
        log(f"CSV regen error: {e}")

# ==============================================================================
# API FETCH
# ==============================================================================
def fetch_api():
    for url in API_ENDPOINTS:
        try:
            r = requests.get(url, timeout=TIMEOUT, headers=API_HEADERS)
            if r.status_code != 200: continue
            lst = r.json().get("data", {}).get("list", [])
            if lst:
                log(f"API ✅ {len(lst)} records")
                return lst
        except Exception as e:
            log(f"API ❌ {str(e)[:60]}")
    return None

# ==============================================================================
# 🧠 ADVANCED PATTERN ENGINE
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

        self.current = self.records[0]["number"] if n else 0
        self.m1 = np.zeros(10); self.m1t = 0
        for i in range(1, n):
            if self.records[i]["number"] == self.current:
                self.m1[self.records[i-1]["number"]] += 1
                self.m1t += 1

        self.m2 = np.zeros(10); self.m2t = 0
        if n >= 3:
            a, b = self.current, self.records[1]["number"]
            for i in range(2, n - 1):
                if self.records[i]["number"] == a and self.records[i+1]["number"] == b:
                    self.m2[self.records[i-1]["number"]] += 1
                    self.m2t += 1

        self.hot = np.zeros(10)
        for r in self.records[:30]: self.hot[r["number"]] += 1
        self.maxHot = max(self.hot) or 1

        if n:
            first_bs = wingo_size(self.records[0]["number"])
            self.bs_streak = 0
            for r in self.records:
                if wingo_size(r["number"]) == first_bs: self.bs_streak += 1
                else: break
            self.bs_streak_type = first_bs

            first_c = wingo_colors(self.records[0]["number"])[0]
            self.color_streak = 0
            for r in self.records:
                if wingo_colors(r["number"])[0] == first_c: self.color_streak += 1
                else: break
            self.color_streak_type = first_c
        else:
            self.bs_streak = self.color_streak = 0
            self.bs_streak_type = self.color_streak_type = "BIG"

    def pattern_match(self, length=4):
        if self.n < length + 5:
            return None
        key = [r["number"] for r in self.records[:length]]
        key_rev = tuple(key[::-1])
        outcomes_bs = Counter(); outcomes_color = Counter(); outcomes_num = Counter()
        samples = 0

        for i in range(length, self.n):
            window = tuple(self.records[i - j]["number"] for j in range(length - 1, -1, -1))
            if window == key_rev:
                idx_next = i - length
                if idx_next < 0: continue
                nxt = self.records[idx_next]
                samples += 1
                outcomes_bs[nxt["size"]] += 1
                for c in (nxt.get("color") or "").split("/"):
                    outcomes_color[c.strip().upper()] += 1
                outcomes_num[nxt["number"]] += 1

        if samples == 0:
            return {"pattern": "-".join(map(str, key_rev)), "sample": 0,
                    "dominantBS": None, "dominantColor": None,
                    "topNums": [], "bsBreak": {}, "colorBreak": {}}

        total_bs = sum(outcomes_bs.values()) or 1
        total_col = sum(outcomes_color.values()) or 1
        dom_bs = outcomes_bs.most_common(1)[0][0]
        dom_color = outcomes_color.most_common(1)[0][0]
        top_nums = [n for n, _ in outcomes_num.most_common(3)]

        return {
            "pattern": "-".join(map(str, key_rev)),
            "sample": samples,
            "dominantBS": dom_bs,
            "bsBreak": {k: round(v/total_bs, 3) for k, v in outcomes_bs.items()},
            "dominantColor": dom_color,
            "colorBreak": {k: round(v/total_col, 3) for k, v in outcomes_color.items()},
            "topNums": top_nums,
        }

    def predict(self, periods=3):
        if self.n < 5:
            return self._empty()

        bayes = (self.freq + 1) / (self.n + 10)
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

        pat4 = self.pattern_match(4)
        pat5 = self.pattern_match(5)
        pat6 = self.pattern_match(6)

        pat_score = np.zeros(10)
        pattern_boost = np.zeros(10)
        for pat, w in [(pat4, 0.06), (pat5, 0.09), (pat6, 0.13)]:
            if pat and pat["sample"] >= 3:
                for num, cnt in Counter(pat.get("topNums", [])).items():
                    if 0 <= num <= 9:
                        pat_score[num] += w * cnt
                if pat["dominantBS"]:
                    for i in range(10):
                        if wingo_size(i) == pat["dominantBS"]:
                            pattern_boost[i] += w * 0.8
                if pat["dominantColor"]:
                    for i in range(10):
                        if pat["dominantColor"] in wingo_colors(i):
                            pattern_boost[i] += w * 0.6

        streak_boost = np.zeros(10)
        if self.bs_streak >= 4:
            want = "SMALL" if self.bs_streak_type == "BIG" else "BIG"
            for i in range(10):
                if wingo_size(i) == want: streak_boost[i] += 0.10
        if self.color_streak >= 4:
            want = "GREEN" if self.color_streak_type == "RED" else "RED"
            for i in range(10):
                if want in wingo_colors(i): streak_boost[i] += 0.08

        composite = (
            bayes * 0.16 * 10 + m1n * 0.14 + m2n * 0.10 +
            miss_mask * 0.11 + veln * 0.09 + hotn * 0.05 + coldn * 0.05 +
            balance * 0.08 + anti * 0.02 +
            pat_score + pattern_boost + streak_boost
        )

        ranked = list(np.argsort(-composite))
        total_score = composite.sum() or 1
        conf_base = 55 + min(30, self.n / 40)

        if self.bs_streak >= 4 and self.bs_streak > self.color_streak:
            primary_signal = "SIZE"
        elif self.color_streak >= 4 and self.color_streak > self.bs_streak:
            primary_signal = "COLOR"
        elif self.bs_streak >= self.color_streak and self.bs_streak >= 3:
            primary_signal = "SIZE"
        elif self.color_streak > self.bs_streak and self.color_streak >= 3:
            primary_signal = "COLOR"
        else:
            primary_signal = "BALANCED"

        periods_out = []
        for k in range(min(periods, 10)):
            num = int(ranked[k])
            cols = wingo_colors(num)
            share = composite[num] / total_score
            conf = conf_base + share * 250 - k * 4
            conf = round(min(99.0, max(40.0, conf)), 2)
            periods_out.append({
                "rank": k + 1, "number": num,
                "bs": wingo_size(num), "color": cols[0],
                "altColors": cols[1:], "confidence": conf,
            })

        top1, top2 = periods_out[0], periods_out[1]

        if self.bs_streak >= 5: pattern_tag = "DRAGON STREAK"
        elif self.color_streak >= 5: pattern_tag = "COLOR STREAK"
        elif pat6 and pat6["sample"] >= 5: pattern_tag = "6-LENGTH PATTERN MATCH"
        elif pat5 and pat5["sample"] >= 4: pattern_tag = "5-LENGTH PATTERN MATCH"
        elif pat4 and pat4["sample"] >= 3: pattern_tag = "4-LENGTH PATTERN MATCH"
        elif self.m1t > 5 and (self.m1[ranked[0]] / self.m1t) > 0.18: pattern_tag = "MARKOV STRONG"
        elif max(self.v1) - min(self.v1) > 3: pattern_tag = "VELOCITY MOMENTUM"
        else: pattern_tag = "PATTERN CONVERGENCE"

        return {
            "number": top1["number"], "number2": top2["number"],
            "bs": top1["bs"], "bs2": top2["bs"],
            "color": top1["color"], "color2": top2["color"],
            "confidence": top1["confidence"], "confidence2": top2["confidence"],
            "primarySignal": primary_signal,
            "sizeStreak": self.bs_streak, "sizeStreakType": self.bs_streak_type,
            "colorStreak": self.color_streak, "colorStreakType": self.color_streak_type,
            "patternTag": pattern_tag,
            "patterns": {"len4": pat4, "len5": pat5, "len6": pat6},
            "top": [int(x) for x in ranked[:3]],
            "freq": [int(x) for x in self.freq.tolist()],
            "red": self.red, "green": self.green, "violet": self.violet,
            "missing": self.missing, "total": self.n,
            "ready": self.n >= 30, "periods": periods_out,
        }

    def _empty(self):
        return {
            "number": 0, "number2": 0, "bs": "BIG", "bs2": "BIG",
            "color": "RED", "color2": "RED",
            "confidence": 0, "confidence2": 0,
            "primarySignal": "BALANCED",
            "sizeStreak": 0, "sizeStreakType": "BIG",
            "colorStreak": 0, "colorStreakType": "RED",
            "patternTag": "LEARNING",
            "patterns": {"len4": None, "len5": None, "len6": None},
            "top": [0, 0, 0], "freq": [0]*10,
            "red": 0, "green": 0, "violet": 0,
            "missing": [], "total": 0, "ready": False, "periods": [],
        }

# ==============================================================================
# SYNC
# ==============================================================================
def sync_store(store, api_list):
    if not api_list: return 0
    seen = {str(r["issue"]) for r in store["records"]}
    pending = store.get("pending") or []
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
            num_hit = (pred["number"] == num)
            bs_hit = (pred["bs"] == size)
            col_hit = pred["color"] in cols
            rec.update({
                "predictedNumber": pred["number"], "predictedBS": pred["bs"],
                "predictedColor": pred["color"],
                "numberWin": num_hit, "bsWin": bs_hit, "colorWin": col_hit,
                "win": num_hit or bs_hit or col_hit,
            })
        else:
            rec["win"] = None

        store["records"].insert(0, rec)
        new_records.append(rec)
        seen.add(iss)
        added += 1

    if len(store["records"]) > HISTORY_LIMIT:
        store["records"] = store["records"][:HISTORY_LIMIT]

    if new_records:
        append_to_csv(new_records)

    s = empty_stats()
    for r in store["records"]:
        if r.get("win") is None: continue
        s["total"] += 1
        if r["win"]: s["wins"] += 1
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
_lock = threading.Lock()
_cache = {"analysis": None, "stats": empty_stats(), "lastIssue": "--",
          "total": 0, "online": False, "lastSync": 0, "apiStatus": "connecting",
          "newPrediction": False}

# ==============================================================================
# CRON JOB
# ==============================================================================
def cron_fetch(trigger="internal"):
    try:
        api_list = fetch_api()
        if not api_list:
            with _lock:
                _cache["online"] = False
                _cache["apiStatus"] = "offline"
            return
        store = load_store()
        newest = str(api_list[0].get("issueNumber", ""))
        last = store.get("lastIssue")

        if newest != last:
            added = sync_store(store, api_list)
            log(f"🆕 {newest} | +{added} | {trigger}")

            eng = XomatEngine(store["records"])
            an = eng.predict(periods=3)
            pending = []
            iss = store.get("lastIssue") or "0"
            for p in an["periods"]:
                iss = next_issue(iss)
                pending.append({"forIssue": iss, "number": p["number"],
                                "bs": p["bs"], "color": p["color"], "rank": p["rank"]})
            store["pending"] = pending
            save_store(store)
            with _lock:
                _cache["newPrediction"] = True

        eng = XomatEngine(store["records"])
        with _lock:
            _cache["analysis"] = eng.predict(periods=3)
            _cache["stats"] = store["stats"]
            _cache["lastIssue"] = store.get("lastIssue") or "--"
            _cache["total"] = len(store["records"])
            _cache["lastSync"] = time.time()
            _cache["online"] = True
            _cache["apiStatus"] = f"live ({len(api_list)})"
    except Exception as e:
        log(f"cron error: {e}")

# ==============================================================================
# SCHEDULER
# ==============================================================================
_scheduler = None
_sched_lock = threading.Lock()

def start_scheduler():
    global _scheduler
    if not HAS_APSCHEDULER:
        def loop():
            log("🚀 fallback thread")
            while True:
                cron_fetch("fallback")
                time.sleep(POLL_SEC)
        threading.Thread(target=loop, daemon=True).start()
        return
    with _sched_lock:
        if _scheduler and _scheduler.running: return
        _scheduler = BackgroundScheduler(daemon=True,
            job_defaults={"coalesce": True, "max_instances": 1})
        _scheduler.add_job(cron_fetch, CronTrigger(second=5),
                           id="fetch", replace_existing=True,
                           kwargs={"trigger": "scheduler"})
        _scheduler.start()
        log("✅ APScheduler started — fetch at :05s every minute")

try: start_scheduler()
except Exception as e: log(f"sched init: {e}")

@app.before_request
def _ensure_sched():
    if _scheduler is None or (HAS_APSCHEDULER and not _scheduler.running):
        start_scheduler()

# ==============================================================================
# ROUTES
# ==============================================================================
@app.route("/api/analysis")
def api_analysis():
    with _lock:
        a = _cache["analysis"]
        is_new = _cache["newPrediction"]
        _cache["newPrediction"] = False
        if a is None:
            return jsonify({"ok": False, "msg": "loading", "apiStatus": _cache["apiStatus"]})
        total = _cache["total"]
        progress = min(100, round(total / TRAINING_TARGET * 100, 1))
        return jsonify({
            "ok": True, "analysis": a, "stats": _cache["stats"],
            "lastIssue": _cache["lastIssue"],
            "nextIssue": next_issue(_cache["lastIssue"]),
            "total": total, "online": _cache["online"],
            "lastSync": _cache["lastSync"], "serverTime": time.time(),
            "apiStatus": _cache["apiStatus"],
            "isNewPrediction": is_new,
            "trainingTarget": TRAINING_TARGET,
            "trainingProgress": progress,
            "trainingReady": total >= TRAINING_TARGET,
        })

@app.route("/api/history")
def api_history():
    return jsonify({"ok": True, "records": load_store()["records"][:100]})

@app.route("/api/numbers")
def api_numbers():
    store = load_store()
    eng = XomatEngine(store["records"])
    return jsonify({
        "ok": True,
        "freq": [int(x) for x in eng.freq.tolist()],
        "colorByNum": {i: wingo_colors(i) for i in range(10)},
        "total": eng.n,
    })

@app.route("/api/debug")
def api_debug():
    with _lock:
        return jsonify({
            "ok": True, "online": _cache["online"],
            "apiStatus": _cache["apiStatus"], "total": _cache["total"],
            "lastIssue": _cache["lastIssue"], "lastSync": _cache["lastSync"],
            "hasAnalysis": _cache["analysis"] is not None,
            "scheduler": "running" if (_scheduler and getattr(_scheduler, "running", False)) else "not-running",
            "fileSize": os.path.getsize(DATA_FILE) if os.path.exists(DATA_FILE) else 0,
            "csvExists": os.path.exists(CSV_FILE),
            "csvSize": os.path.getsize(CSV_FILE) if os.path.exists(CSV_FILE) else 0,
            "trainingTarget": TRAINING_TARGET,
            "trainingProgress": min(100, round(_cache["total"] / TRAINING_TARGET * 100, 1)),
        })

@app.route("/api/backup")
def api_backup():
    if not os.path.exists(DATA_FILE):
        return jsonify({"ok": False}), 404
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return send_file(DATA_FILE, mimetype="application/json",
                     as_attachment=True, download_name=f"xomat_{ts}.json")

@app.route("/api/export-csv")
def api_export_csv():
    """CSV export for AI training."""
    if not os.path.exists(CSV_FILE):
        # Generate on-the-fly if missing
        regenerate_full_csv()
    if not os.path.exists(CSV_FILE):
        return jsonify({"ok": False, "msg": "CSV not available"}), 404
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    return send_file(CSV_FILE, mimetype="text/csv",
                     as_attachment=True, download_name=f"xomat_training_{ts}.csv")

# ==============================================================================
# JSON UPLOAD + TIME SEQUENCE
# ==============================================================================
@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Upload JSON file → save to timestamped path + merge into store."""
    try:
        if 'file' not in request.files:
            return jsonify({"ok": False, "msg": "No file provided"}), 400

        f = request.files['file']
        if not f.filename.lower().endswith('.json'):
            return jsonify({"ok": False, "msg": "Only .json files allowed"}), 400

        raw = f.read().decode('utf-8', errors='replace')
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            return jsonify({"ok": False, "msg": f"Invalid JSON: {e}"}), 400

        # Accept multiple structures
        if isinstance(data, dict):
            records = data.get("records") or data.get("data") or data.get("list") or []
            if isinstance(data.get("data"), dict):
                records = data["data"].get("list", records)
        elif isinstance(data, list):
            records = data
        else:
            return jsonify({"ok": False, "msg": "Invalid structure"}), 400

        if not isinstance(records, list) or not records:
            return jsonify({"ok": False, "msg": "No records found"}), 400

        # Clean + validate
        clean = []
        for r in records:
            if not isinstance(r, dict): continue
            try: num = int(r.get("number", -1))
            except: continue
            if not (0 <= num <= 9): continue
            iss = str(r.get("issue") or r.get("issueNumber") or r.get("period") or "")
            if not iss: continue
            cols = parse_api_color(r.get("color", "")) if r.get("color") else wingo_colors(num)
            clean.append({
                "issue": iss, "number": num,
                "size": wingo_size(num), "color": "/".join(cols),
                "ts": int(r.get("ts") or time.time()),
                "uploaded": True,
            })

        if not clean:
            return jsonify({"ok": False, "msg": "No valid records"}), 400

        # Save to timestamped path
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        fname = f"{ts_str}_upload.json"
        fpath = os.path.join(UPLOAD_DIR, fname)
        with open(fpath, "w", encoding="utf-8") as fh:
            json.dump({
                "uploadedAt": datetime.now(timezone.utc).isoformat(),
                "source": f.filename,
                "count": len(clean),
                "records": clean,
            }, fh, indent=2)

        # Merge (skip duplicates by issue)
        store = load_store()
        seen = {str(r["issue"]) for r in store["records"]}
        added = 0
        for rec in clean:
            if rec["issue"] not in seen:
                store["records"].append(rec)
                seen.add(rec["issue"])
                added += 1

        # Sort newest-first
        try:
            store["records"].sort(
                key=lambda r: int(r["issue"]) if str(r["issue"]).isdigit() else 0,
                reverse=True)
        except: pass

        if store["records"]:
            store["lastIssue"] = str(store["records"][0]["issue"])
        save_store(store)

        # Regenerate full CSV with new data
        regenerate_full_csv()

        # Refresh cache
        eng = XomatEngine(store["records"])
        with _lock:
            _cache["analysis"] = eng.predict(periods=3)
            _cache["total"] = len(store["records"])
            _cache["lastIssue"] = store["records"][0]["issue"] if store["records"] else "--"

        log(f"📤 Upload: {f.filename} → {fname} | +{added} new / {len(clean)} parsed")

        return jsonify({
            "ok": True, "fileName": fname, "path": fpath,
            "parsed": len(clean), "added": added,
            "total": len(store["records"]),
        })
    except Exception as e:
        log(f"upload error: {e}")
        return jsonify({"ok": False, "msg": str(e)}), 500


@app.route("/api/uploads")
def api_uploads():
    """List all uploaded JSON files."""
    if not os.path.exists(UPLOAD_DIR):
        return jsonify({"ok": True, "files": []})
    files = []
    for fn in sorted(os.listdir(UPLOAD_DIR), reverse=True):
        if fn.lower().endswith(".json"):
            fp = os.path.join(UPLOAD_DIR, fn)
            try:
                with open(fp) as f: d = json.load(f)
                cnt = d.get("count", len(d.get("records", [])))
            except: cnt = 0
            files.append({
                "name": fn, "size": os.path.getsize(fp),
                "count": cnt,
                "modified": datetime.fromtimestamp(os.path.getmtime(fp), timezone.utc).isoformat(),
            })
    return jsonify({"ok": True, "files": files[:50]})


@app.route("/api/sequence")
def api_sequence():
    """Time-sequenced data: issue, sum, size, color, number."""
    store = load_store()
    records = store["records"]
    try:
        records = sorted(records,
                         key=lambda r: int(r["issue"]) if str(r["issue"]).isdigit() else 0,
                         reverse=True)
    except: pass

    limit = int(request.args.get("limit", 60))
    out = []
    for i, r in enumerate(records[:limit]):
        out.append({
            "issue": r["issue"],
            "number": r["number"],
            "sum": calc_sum(records, i, 5),
            "size": r["size"],
            "color": r["color"],
            "ts": r.get("ts"),
        })
    return jsonify({"ok": True, "count": len(out), "records": out})


@app.route("/api/training-status")
def api_training_status():
    """Training dataset status."""
    store = load_store()
    total = len(store["records"])
    csv_rows = 0
    if os.path.exists(CSV_FILE):
        try:
            with open(CSV_FILE, "rb") as f:
                csv_rows = sum(1 for _ in f) - 1   # minus header
        except: pass
    return jsonify({
        "ok": True,
        "jsonRecords": total,
        "csvRows": csv_rows,
        "target": TRAINING_TARGET,
        "progress": min(100, round(total / TRAINING_TARGET * 100, 2)),
        "ready": total >= TRAINING_TARGET,
        "daysOfData": round(total / 1440, 2),
        "uploads": len([f for f in (os.listdir(UPLOAD_DIR) if os.path.exists(UPLOAD_DIR) else []) if f.endswith(".json")]),
    })

# ==============================================================================
# HTML FRONTEND
# ==============================================================================
INDEX_HTML = r"""<!DOCTYPE html>
<html class="dark" lang="en"><head>
<meta charset="utf-8"/>
<meta content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" name="viewport"/>
<title>XOMAT AI - RNG Analysis Engine</title>
<script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
<script>
tailwind.config = {
  darkMode: 'class',
  theme: { extend: {
    colors: {
      crimson: {950:'#070002',900:'#140106',850:'#20020a',800:'#380613',700:'#5c0a1f',600:'#8a0f2e',500:'#c41541',400:'#e52656'},
      champagne: {50:'#FCF9F5',100:'#F7EFE5',200:'#EFE2D2',300:'#DFCDB8',800:'#4A3E31',900:'#261F17'},
      ruby:'#E61E4D', gold:'#D4AF37'
    },
    fontFamily: {
      sans:['-apple-system','BlinkMacSystemFont','Segoe UI','Roboto','sans-serif'],
      mono:['SF Mono','ui-monospace','Menlo','Monaco','monospace'],
    }
  }}
}
</script>
<style>
@keyframes orbPulse {
  0%,100% { box-shadow: 0 0 35px 8px rgba(229,38,86,.45), inset 0 0 30px 10px rgba(255,60,100,.5), 0 0 70px 18px rgba(180,10,40,.25); transform:scale(1); }
  50%     { box-shadow: 0 0 55px 16px rgba(229,38,86,.7), inset 0 0 45px 15px rgba(255,90,130,.7), 0 0 95px 28px rgba(220,20,60,.4); transform:scale(1.025); }
}
@keyframes spinCW { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }
@keyframes spinCCW { from{transform:rotate(360deg)} to{transform:rotate(0deg)} }
@keyframes radarWave { 0%{transform:scale(.9);opacity:.8} 100%{transform:scale(1.6);opacity:0} }
@keyframes digitRoll { 0%{transform:translateY(-20%);opacity:0} 100%{transform:translateY(0);opacity:1} }
.ai-orb-core { animation: orbPulse 3.5s ease-in-out infinite; }
.ai-orbit-1  { animation: spinCW 12s linear infinite; }
.ai-orbit-2  { animation: spinCCW 18s linear infinite; }
.radar-ping  { animation: radarWave 2.2s cubic-bezier(0,0,.2,1) infinite; }
.glass-dark  { background:rgba(18,2,7,.78); backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px); }
.digit-roll  { animation: digitRoll .5s cubic-bezier(.4,0,.2,1); }
.pulse-live  { animation: pulseLive 1.4s ease-in-out infinite; }
@keyframes pulseLive { 0%,100%{opacity:1} 50%{opacity:.4} }
</style>
</head>
<body class="bg-[#080104] text-neutral-100 min-h-screen font-sans antialiased selection:bg-crimson-600 selection:text-white pb-12 overflow-x-hidden">

<header class="sticky top-0 z-40 bg-[#0c0106]/95 backdrop-blur-md border-b border-crimson-900/60 px-4 py-2.5 shadow-lg">
  <div class="flex items-center justify-between max-w-md mx-auto">
    <div class="flex items-center space-x-2.5">
      <div class="w-8 h-8 rounded-full bg-gradient-to-tr from-crimson-600 via-rose-500 to-amber-300 p-0.5 shadow-md">
        <div class="w-full h-full bg-[#140106] rounded-full flex items-center justify-center">
          <svg class="w-4 h-4 text-rose-400" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>
        </div>
      </div>
      <div>
        <div class="flex items-center space-x-1 leading-none">
          <span class="text-base font-black tracking-tight text-white">XOMAT</span>
          <span class="text-xs px-1.5 py-0.5 rounded bg-crimson-600/60 text-rose-200 font-bold tracking-wide">AI</span>
        </div>
        <p class="text-[9px] text-neutral-400 tracking-tighter">v8.2 • Upload • Train</p>
      </div>
    </div>
    <div class="flex items-center space-x-2">
      <div class="flex items-center space-x-1.5 px-2 py-1 rounded-full bg-emerald-950/70 border border-emerald-500/40 text-[11px] text-emerald-300">
        <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 pulse-live"></span>
        <span class="font-medium text-[10px]" id="online-tag">Online</span>
      </div>
      <div class="px-2 py-1 rounded-lg bg-crimson-950 border border-amber-500/30 text-right">
        <div class="text-[10px] font-bold text-amber-200 leading-none">WinGo 1M</div>
        <div class="text-[9px] font-mono text-neutral-400 leading-none mt-0.5" id="live-clock">--:--:-- UTC</div>
      </div>
    </div>
  </div>
</header>

<main class="max-w-md mx-auto px-3.5 pt-3.5 space-y-4">

  <!-- TRAINING PROGRESS (top priority) -->
  <section class="rounded-2xl p-3 border bg-gradient-to-r from-neutral-900 via-crimson-950 to-neutral-900 border-amber-500/40 text-white shadow-xl">
    <div class="flex items-center justify-between mb-1">
      <span class="text-[10px] uppercase tracking-widest text-amber-300 font-bold">🎯 Training Dataset Progress</span>
      <span class="text-[10px] font-mono text-amber-200" id="train-pct">0%</span>
    </div>
    <div class="w-full bg-black/60 h-2 rounded-full overflow-hidden">
      <div id="train-bar" class="h-full bg-gradient-to-r from-amber-500 to-rose-500 transition-all" style="width:0%"></div>
    </div>
    <div class="flex justify-between text-[10px] font-mono mt-1">
      <span class="text-neutral-300"><b id="train-current">0</b> records</span>
      <span class="text-neutral-400">target: <b class="text-amber-300">5760</b> (4 days)</span>
      <span class="text-neutral-300"><b id="train-days">0</b> days</span>
    </div>
  </section>

  <!-- TOP STATS -->
  <section class="grid grid-cols-2 gap-2">
    <div class="bg-champagne-100 rounded-xl p-2.5 border border-champagne-300 shadow-md text-neutral-900 flex items-center space-x-2.5">
      <div class="w-8 h-8 rounded-lg bg-neutral-900 text-amber-300 flex items-center justify-center shrink-0 shadow">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/></svg>
      </div>
      <div class="min-w-0">
        <div class="text-base font-black tracking-tight leading-none text-neutral-950" id="stat-total">0</div>
        <div class="text-[10px] text-neutral-600 font-medium truncate mt-0.5">Total Records</div>
      </div>
    </div>
    <div class="bg-champagne-100 rounded-xl p-2.5 border border-champagne-300 shadow-md text-neutral-900 flex items-center space-x-2.5">
      <div class="w-8 h-8 rounded-lg bg-neutral-900 text-rose-400 flex items-center justify-center shrink-0 shadow">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/></svg>
      </div>
      <div class="min-w-0">
        <div class="text-base font-black tracking-tight leading-none text-neutral-950" id="stat-issue-short">--</div>
        <div class="text-[10px] text-neutral-600 font-medium truncate mt-0.5">Latest Issue</div>
      </div>
    </div>
    <div class="bg-champagne-100 rounded-xl p-2.5 border border-champagne-300 shadow-md text-neutral-900 flex items-center space-x-2.5">
      <div class="w-8 h-8 rounded-lg bg-neutral-900 text-emerald-400 flex items-center justify-center shrink-0 shadow">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9" stroke-width="2"/><path d="M12 7v5l3 2" stroke-linecap="round" stroke-width="2"/></svg>
      </div>
      <div class="min-w-0">
        <div class="text-[11px] font-bold text-neutral-950 leading-tight" id="stat-update">--</div>
        <div class="text-[9px] font-mono text-neutral-600" id="stat-update-time">--:--:--</div>
      </div>
    </div>
    <div class="bg-champagne-100 rounded-xl p-2.5 border border-champagne-300 shadow-md text-neutral-900 flex items-center space-x-2.5">
      <div class="w-8 h-8 rounded-lg bg-emerald-900/90 text-emerald-300 flex items-center justify-center shrink-0 shadow">
        <span class="w-2.5 h-2.5 rounded-full bg-emerald-400 pulse-live"></span>
      </div>
      <div class="min-w-0">
        <div class="text-[11px] font-black text-emerald-800 leading-tight" id="stat-status">Live</div>
        <div class="text-[9px] text-neutral-600 font-medium">AI Status</div>
        <div class="text-[8px] text-neutral-500">Conf: <span id="stat-conf">0%</span></div>
      </div>
    </div>
  </section>

  <!-- UPLOAD JSON -->
  <section class="bg-champagne-100 rounded-2xl p-4 border border-champagne-300 shadow-lg text-neutral-900">
    <div class="flex items-center justify-between pb-2 border-b border-champagne-300 mb-3">
      <div class="flex items-center space-x-2">
        <svg class="w-4 h-4 text-neutral-800" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M7 16a4 4 0 01-.88-7.9A5 5 0 1115.9 6h.1a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" stroke-linecap="round" stroke-width="2"/></svg>
        <span class="text-xs font-black uppercase tracking-wider">Upload JSON Data</span>
      </div>
      <span class="text-[10px] text-neutral-500 font-mono">.json</span>
    </div>

    <label id="upload-zone" class="block border-2 border-dashed border-champagne-300 rounded-xl p-5 text-center cursor-pointer hover:border-rose-500 hover:bg-rose-50/40 transition-colors">
      <input id="file-input" type="file" accept=".json,application/json" class="hidden">
      <div class="w-10 h-10 mx-auto rounded-full bg-neutral-900 text-amber-300 flex items-center justify-center mb-2">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M12 4v16m8-8H4" stroke-linecap="round" stroke-width="2"/></svg>
      </div>
      <div class="text-xs font-bold text-neutral-800">Tap to select JSON file</div>
      <div class="text-[10px] text-neutral-500 mt-0.5">Path: uploads/YYYY-MM-DD_HH-MM-SS_upload.json</div>
      <div class="text-[10px] text-rose-600 font-mono mt-1" id="upload-status">Ready</div>
    </label>

    <div id="upload-result" class="hidden mt-3 p-2 rounded-lg bg-emerald-50 border border-emerald-300 text-[11px] text-emerald-800 font-mono"></div>

    <div class="mt-3 pt-2 border-t border-champagne-300">
      <div class="text-[10px] font-bold text-neutral-600 uppercase tracking-wider mb-1.5">Saved Files (uploads/)</div>
      <div id="uploads-list" class="space-y-1 max-h-32 overflow-y-auto">
        <div class="text-[10px] text-neutral-500 italic">No uploads yet</div>
      </div>
    </div>
  </section>

  <!-- AI ORB -->
  <section class="relative rounded-2xl overflow-hidden p-0.5 shadow-2xl bg-gradient-to-b from-rose-700/60 via-crimson-900 to-[#120106]">
    <div class="relative bg-gradient-to-b from-[#190209] via-[#0e0105] to-[#080003] rounded-[15px] p-4 text-center overflow-hidden">
      <div class="absolute -top-12 -left-12 w-48 h-48 bg-crimson-600/30 rounded-full blur-3xl pointer-events-none"></div>
      <div class="absolute -bottom-16 -right-16 w-56 h-56 bg-amber-600/20 rounded-full blur-3xl pointer-events-none"></div>
      <div class="relative my-4 flex items-center justify-center">
        <div class="ai-orbit-1 absolute w-56 h-56 rounded-full border border-rose-500/25 border-dashed pointer-events-none"></div>
        <div class="ai-orbit-2 absolute w-48 h-48 rounded-full border border-amber-400/20 pointer-events-none flex items-center justify-between px-1">
          <span class="w-1.5 h-1.5 rounded-full bg-amber-400 shadow-sm"></span>
          <span class="w-1.5 h-1.5 rounded-full bg-rose-400 shadow-sm"></span>
        </div>
        <div class="ai-orb-core relative w-36 h-36 rounded-full bg-gradient-to-br from-[#8a0928] via-[#e61e4d] to-[#3a030f] flex flex-col items-center justify-center p-3 text-white border-2 border-rose-400/60 select-none">
          <div class="absolute top-1.5 left-5 w-14 h-6 rounded-full bg-white/30 blur-[2px] transform -rotate-[25deg] pointer-events-none"></div>
          <div class="relative z-10 w-9 h-9 rounded-full bg-black/40 backdrop-blur-sm border border-rose-300/50 flex items-center justify-center mb-1">
            <svg class="w-5 h-5 text-rose-200" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M12 3v3M12 18v3M3 12h3M18 12h3"/><circle cx="12" cy="12" r="7" stroke-dasharray="2 2"/></svg>
          </div>
          <div class="relative z-10 text-[13px] font-black tracking-wider uppercase text-white drop-shadow">XOMAT AI</div>
          <div class="relative z-10 text-[7px] tracking-widest uppercase text-rose-200/90 font-medium">RNG ENGINE</div>
          <div class="absolute inset-0 rounded-full border border-rose-300/40 animate-ping opacity-25 pointer-events-none"></div>
        </div>
      </div>
      <div class="mt-2 inline-block px-4 py-1.5 rounded-full bg-crimson-950/90 border border-crimson-700/50 shadow-inner">
        <div class="flex items-center space-x-2 text-[11px] font-medium text-rose-200">
          <span class="inline-block w-2 h-2 rounded-full bg-amber-400 pulse-live"></span>
          <span id="learning-status">Analyzing patterns...</span>
        </div>
      </div>
      <p class="text-[10px] text-neutral-400 mt-1">Next update in <span id="countdown" class="font-mono text-amber-300">--s</span></p>
      <div class="grid grid-cols-3 gap-1.5 mt-4 pt-3 border-t border-crimson-900/70">
        <div class="bg-crimson-950/70 border border-crimson-800/60 rounded-xl p-2 text-center">
          <div class="w-6 h-6 mx-auto rounded-full bg-rose-900/60 border border-rose-500/50 text-rose-300 flex items-center justify-center mb-1">
            <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M4 6h16M4 10h16M4 14h16M4 18h16" stroke-linecap="round" stroke-width="2"/></svg>
          </div>
          <div class="text-[10px] font-bold text-neutral-100">Number</div>
          <div class="text-[8px] text-neutral-400">Freq &amp; Velocity</div>
        </div>
        <div class="bg-crimson-950/70 border border-crimson-800/60 rounded-xl p-2 text-center">
          <div class="w-6 h-6 mx-auto rounded-full bg-purple-900/60 border border-purple-500/50 text-purple-300 flex items-center justify-center mb-1">
            <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" stroke-linecap="round" stroke-width="2"/></svg>
          </div>
          <div class="text-[10px] font-bold text-neutral-100">Size</div>
          <div class="text-[8px] text-neutral-400" id="size-streak-info">-- x0</div>
        </div>
        <div class="bg-crimson-950/70 border border-crimson-800/60 rounded-xl p-2 text-center">
          <div class="w-6 h-6 mx-auto rounded-full bg-emerald-900/60 border border-emerald-500/50 text-emerald-300 flex items-center justify-center mb-1">
            <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><circle cx="12" cy="12" r="8" stroke-width="2"/></svg>
          </div>
          <div class="text-[10px] font-bold text-neutral-100">Color</div>
          <div class="text-[8px] text-neutral-400" id="color-streak-info">-- x0</div>
        </div>
      </div>
    </div>
  </section>

  <!-- PREDICTION CARD -->
  <section class="bg-champagne-100 rounded-2xl p-4 border border-champagne-300 shadow-lg text-neutral-900 relative">
    <div class="flex items-center justify-between mb-3">
      <div class="flex items-center space-x-2">
        <div class="w-6 h-6 rounded-md bg-neutral-900 text-amber-300 flex items-center justify-center">
          <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="m4.93 4.93 4.24 4.24M14.83 9.17l4.24-4.24M14.83 14.83l4.24 4.24M9.17 14.83l-4.24 4.24"/></svg>
        </div>
        <h2 class="text-xs font-black uppercase tracking-wider text-neutral-900">AI PREDICTION</h2>
      </div>
      <span class="px-2 py-0.5 rounded-full bg-emerald-600 text-white font-bold text-[10px] tracking-wide flex items-center gap-1">
        <span class="w-1.5 h-1.5 rounded-full bg-white pulse-live"></span> Live
      </span>
    </div>

    <div class="mb-3 p-2 rounded-xl bg-neutral-900 text-white text-center">
      <div class="text-[9px] uppercase tracking-widest text-amber-300">Primary Signal</div>
      <div class="text-sm font-black" id="primary-signal">BALANCED</div>
    </div>

    <div class="grid grid-cols-2 gap-2">
      <div class="rounded-xl p-3 bg-gradient-to-br from-rose-600 to-crimson-700 text-white shadow-lg">
        <div class="text-[9px] uppercase tracking-widest opacity-80">Rank #1</div>
        <div class="text-4xl font-black leading-none my-1" id="p1-number">--</div>
        <div class="flex gap-1 mt-1.5">
          <span class="text-[10px] px-2 py-0.5 rounded font-bold bg-black/30" id="p1-bs">--</span>
          <span class="text-[10px] px-2 py-0.5 rounded font-bold bg-black/30" id="p1-color">--</span>
        </div>
        <div class="text-[9px] font-mono mt-1 opacity-90">Conf: <span id="p1-conf">0%</span></div>
      </div>
      <div class="rounded-xl p-3 bg-gradient-to-br from-amber-500 to-orange-600 text-white shadow-lg">
        <div class="text-[9px] uppercase tracking-widest opacity-80">Rank #2</div>
        <div class="text-4xl font-black leading-none my-1" id="p2-number">--</div>
        <div class="flex gap-1 mt-1.5">
          <span class="text-[10px] px-2 py-0.5 rounded font-bold bg-black/30" id="p2-bs">--</span>
          <span class="text-[10px] px-2 py-0.5 rounded font-bold bg-black/30" id="p2-color">--</span>
        </div>
        <div class="text-[9px] font-mono mt-1 opacity-90">Conf: <span id="p2-conf">0%</span></div>
      </div>
    </div>

    <div class="mt-3 p-2 rounded-lg bg-neutral-900/90 text-white text-center">
      <div class="text-[9px] uppercase tracking-widest text-amber-300">Pattern Detected</div>
      <div class="text-xs font-black" id="pattern-tag">LEARNING</div>
    </div>
  </section>

  <!-- PATTERN MATCHER -->
  <section class="bg-champagne-100 rounded-2xl p-4 border border-champagne-300 shadow-lg text-neutral-900">
    <div class="flex items-center space-x-2 pb-2 border-b border-champagne-300 mb-3">
      <svg class="w-4 h-4 text-neutral-800" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" stroke-linecap="round" stroke-width="2"/></svg>
      <span class="text-xs font-black uppercase tracking-wider">Pattern Matcher</span>
    </div>
    <div class="space-y-2" id="pattern-list">
      <div class="text-[11px] text-neutral-500 italic">Analyzing...</div>
    </div>
  </section>

  <!-- TIME SEQUENCE TABLE -->
  <section class="bg-champagne-100 rounded-2xl p-4 border border-champagne-300 shadow-lg text-neutral-900">
    <div class="flex items-center justify-between pb-2 border-b border-champagne-300 mb-3">
      <div class="flex items-center space-x-2">
        <svg class="w-4 h-4 text-neutral-800" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M12 8v4l3 2m6-2a9 9 0 11-18 0 9 9 0 0118 0z" stroke-linecap="round" stroke-width="2"/></svg>
        <span class="text-xs font-black uppercase tracking-wider">Time Sequence</span>
      </div>
      <span class="text-[10px] text-neutral-500 font-mono">last 60</span>
    </div>
    <div class="overflow-x-auto -mx-1">
      <table class="w-full text-[10px] font-mono">
        <thead>
          <tr class="bg-neutral-900 text-white">
            <th class="px-1.5 py-1 text-left">#</th>
            <th class="px-1.5 py-1 text-left">ISSUE</th>
            <th class="px-1.5 py-1 text-center">SUM5</th>
            <th class="px-1.5 py-1 text-center">NUM</th>
            <th class="px-1.5 py-1 text-center">SIZE</th>
            <th class="px-1.5 py-1 text-center">COLOR</th>
          </tr>
        </thead>
        <tbody id="sequence-body">
          <tr><td colspan="6" class="text-center text-neutral-500 py-3 italic">Loading...</td></tr>
        </tbody>
      </table>
    </div>
  </section>

  <!-- ANALYSIS SUMMARY -->
  <section class="bg-champagne-100 rounded-2xl p-4 border border-champagne-300 shadow-lg text-neutral-900">
    <div class="flex items-center space-x-2 mb-3 pb-2 border-b border-champagne-300">
      <svg class="w-4 h-4 text-neutral-800" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" stroke-linecap="round" stroke-width="2"/></svg>
      <span class="text-xs font-black uppercase tracking-wider">Analysis Summary</span>
    </div>
    <div class="space-y-2">
      <div class="flex items-center justify-between p-2 rounded-xl bg-rose-50 border border-rose-200">
        <div class="flex items-center space-x-2.5">
          <div class="w-8 h-8 rounded-lg bg-rose-600 text-white flex items-center justify-center text-xs font-bold">N</div>
          <div><div class="text-xs font-bold">Number</div><div class="text-[10px] text-neutral-500" id="summary-num-text">Loading...</div></div>
        </div>
        <span class="text-[11px] font-bold text-rose-700 bg-rose-100 px-2 py-0.5 rounded-full" id="summary-num">--</span>
      </div>
      <div class="flex items-center justify-between p-2 rounded-xl bg-purple-50 border border-purple-200">
        <div class="flex items-center space-x-2.5">
          <div class="w-8 h-8 rounded-lg bg-purple-700 text-white flex items-center justify-center text-xs font-bold">S</div>
          <div><div class="text-xs font-bold">Big / Small</div><div class="text-[10px] text-neutral-500" id="summary-bs-text">--</div></div>
        </div>
        <span class="text-[11px] font-bold text-purple-700 bg-purple-100 px-2 py-0.5 rounded-full" id="summary-bs">--</span>
      </div>
      <div class="flex items-center justify-between p-2 rounded-xl bg-emerald-50 border border-emerald-200">
        <div class="flex items-center space-x-2.5">
          <div class="w-8 h-8 rounded-lg bg-emerald-700 text-white flex items-center justify-center text-xs font-bold">C</div>
          <div><div class="text-xs font-bold">Color</div><div class="text-[10px] text-neutral-500" id="summary-color-text">--</div></div>
        </div>
        <span class="text-[11px] font-bold text-emerald-700 bg-emerald-100 px-2 py-0.5 rounded-full" id="summary-color">--</span>
      </div>
    </div>
  </section>

  <!-- LATEST DATA MATRIX -->
  <section class="bg-champagne-100 rounded-2xl p-4 border border-champagne-300 shadow-lg text-neutral-900">
    <div class="flex items-center justify-between pb-2 border-b border-champagne-300 mb-3">
      <div class="flex items-center space-x-1.5">
        <span class="text-rose-600 font-bold">⌘</span>
        <h2 class="text-xs font-black uppercase tracking-wider">LATEST DATA</h2>
      </div>
      <span class="text-[10px] text-neutral-500 font-mono">wingo.json</span>
    </div>

    <div class="space-y-2 mb-3">
      <div class="flex items-center justify-between text-xs">
        <div class="flex items-center space-x-2"><span class="w-3.5 h-3.5 rounded-full bg-red-600"></span><span class="font-bold">Red</span></div>
        <div class="font-mono text-[11px]"><span class="font-bold" id="red-count">0</span> <span class="text-neutral-500" id="red-pct">(0%)</span></div>
      </div>
      <div class="w-full bg-neutral-200 h-1.5 rounded-full overflow-hidden"><div class="bg-red-600 h-full" id="red-bar" style="width:0%"></div></div>
      <div class="flex items-center justify-between text-xs pt-1">
        <div class="flex items-center space-x-2"><span class="w-3.5 h-3.5 rounded-full bg-emerald-600"></span><span class="font-bold">Green</span></div>
        <div class="font-mono text-[11px]"><span class="font-bold" id="green-count">0</span> <span class="text-neutral-500" id="green-pct">(0%)</span></div>
      </div>
      <div class="w-full bg-neutral-200 h-1.5 rounded-full overflow-hidden"><div class="bg-emerald-600 h-full" id="green-bar" style="width:0%"></div></div>
      <div class="flex items-center justify-between text-xs pt-1">
        <div class="flex items-center space-x-2"><span class="w-3.5 h-3.5 rounded-full bg-purple-600"></span><span class="font-bold">Violet</span></div>
        <div class="font-mono text-[11px]"><span class="font-bold" id="violet-count">0</span> <span class="text-neutral-500" id="violet-pct">(0%)</span></div>
      </div>
      <div class="w-full bg-neutral-200 h-1.5 rounded-full overflow-hidden"><div class="bg-purple-600 h-full" id="violet-bar" style="width:0%"></div></div>
    </div>

    <div class="pt-2 border-t border-champagne-300">
      <div class="text-[10px] font-bold text-neutral-600 uppercase tracking-wider mb-2">Numbers 0-9</div>
      <div class="grid grid-cols-5 gap-1.5" id="numbers-grid"></div>
    </div>
  </section>

  <!-- DOWNLOADS -->
  <section class="bg-champagne-100 rounded-2xl p-4 border border-champagne-300 shadow-lg text-neutral-900">
    <div class="flex items-center space-x-2 pb-2 border-b border-champagne-300 mb-3">
      <svg class="w-4 h-4 text-neutral-800" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" stroke-linecap="round" stroke-width="2"/></svg>
      <span class="text-xs font-black uppercase tracking-wider">Backup / Export</span>
    </div>
    <div class="grid grid-cols-2 gap-2">
      <a href="/api/backup" class="block text-center p-2.5 rounded-lg bg-neutral-900 text-amber-300 text-[11px] font-bold hover:bg-neutral-800">⬇ JSON Backup</a>
      <a href="/api/export-csv" class="block text-center p-2.5 rounded-lg bg-neutral-900 text-emerald-300 text-[11px] font-bold hover:bg-neutral-800">⬇ Training CSV</a>
    </div>
  </section>

  <!-- JSON Explanation -->
  <section class="bg-[#0f0207] border border-crimson-800/60 rounded-2xl p-4 shadow-xl text-neutral-200">
    <div class="flex items-center justify-between pb-2 border-b border-crimson-900 mb-3">
      <div class="flex items-center space-x-2">
        <svg class="w-4 h-4 text-amber-400" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M16 18l6-6-6-6M8 6l-6 6 6 6"/></svg>
        <h2 class="text-xs font-black tracking-wider text-rose-300 uppercase">RNG ANALYSIS (JSON)</h2>
      </div>
      <span class="text-[10px] text-neutral-500 font-mono">v8.2</span>
    </div>
    <div class="bg-black/90 p-3 rounded-xl border border-neutral-800 font-mono text-[10px] leading-relaxed overflow-x-auto text-neutral-300 max-h-72 overflow-y-auto">
      <pre id="json-view" class="text-neutral-300">Loading...</pre>
    </div>
  </section>

  <footer class="pt-4 text-center space-y-3">
    <div class="p-4 rounded-2xl bg-gradient-to-r from-crimson-950 via-rose-950 to-crimson-950 border border-crimson-800/40">
      <div class="text-sm italic text-amber-200/90">"Data Speaks... Xomat AI Understands..."</div>
      <div class="flex items-center justify-center space-x-2 mt-2">
        <span class="text-xs font-black text-neutral-300 tracking-wider">XOMAT AI v8.2</span>
      </div>
    </div>
    <div class="text-[10px] text-neutral-500">© 2026 Xomat AI | WinGo 1M | Upload + Train + Predict</div>
  </footer>
</main>

<script>
const $ = id => document.getElementById(id);
let lastIssue = null;

function tickClock() {
  const now = new Date();
  const hh = String(now.getUTCHours()).padStart(2,'0');
  const mm = String(now.getUTCMinutes()).padStart(2,'0');
  const ss = String(now.getUTCSeconds()).padStart(2,'0');
  $('live-clock').textContent = `${hh}:${mm}:${ss} UTC`;
  const rem = 60 - now.getUTCSeconds();
  $('countdown').textContent = rem + 's';
}
setInterval(tickClock, 1000); tickClock();

function setText(id, val) { const el = $(id); if (el) el.textContent = val; }
function numColorClass(colors) {
  const cs = (colors || []).map(c=>c.toUpperCase());
  if (cs.includes('VIOLET') && cs.includes('RED')) return 'bg-red-600';
  if (cs.includes('VIOLET') && cs.includes('GREEN')) return 'bg-purple-600';
  if (cs.includes('GREEN')) return 'bg-emerald-600';
  if (cs.includes('VIOLET')) return 'bg-purple-600';
  return 'bg-red-600';
}
function colorChipClass(c) {
  c = (c||'').toUpperCase();
  if (c === 'BIG') return 'bg-red-600';
  if (c === 'SMALL') return 'bg-emerald-600';
  if (c === 'GREEN') return 'bg-emerald-600';
  if (c === 'VIOLET') return 'bg-purple-600';
  return 'bg-red-600';
}
function sizeCellCls(s) { return s === 'BIG' ? 'bg-red-600 text-white' : 'bg-emerald-600 text-white'; }
function colorCellCls(c) {
  c = (c||'').toUpperCase();
  if (c.includes('VIOLET')) return 'bg-purple-600 text-white';
  if (c.includes('GREEN')) return 'bg-emerald-600 text-white';
  return 'bg-red-600 text-white';
}

/* ═════════════ UPLOAD ═════════════ */
const fileInput = $('file-input');
const uploadZone = $('upload-zone');
const uploadStatus = $('upload-status');
const uploadResult = $('upload-result');

if (fileInput) {
  fileInput.addEventListener('change', async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    uploadStatus.textContent = `Uploading ${f.name}...`;
    uploadStatus.className = 'text-[10px] text-amber-600 font-mono mt-1';
    const fd = new FormData();
    fd.append('file', f);
    try {
      const r = await fetch('/api/upload', { method: 'POST', body: fd });
      const d = await r.json();
      if (d.ok) {
        uploadStatus.textContent = `✅ ${d.fileName}`;
        uploadStatus.className = 'text-[10px] text-emerald-700 font-mono mt-1';
        uploadResult.classList.remove('hidden');
        uploadResult.innerHTML = `📁 <b>${d.fileName}</b><br>
          Parsed: <b>${d.parsed}</b> · Added: <b>${d.added}</b> new<br>
          Total: <b>${d.total}</b> records`;
        loadUploads(); loadSequence(); loadTraining(); refresh();
      } else {
        uploadStatus.textContent = `❌ ${d.msg}`;
        uploadStatus.className = 'text-[10px] text-red-600 font-mono mt-1';
      }
    } catch (err) {
      uploadStatus.textContent = `❌ Network error`;
      uploadStatus.className = 'text-[10px] text-red-600 font-mono mt-1';
    }
    fileInput.value = '';
  });
  ['dragover','dragenter'].forEach(ev =>
    uploadZone.addEventListener(ev, e => { e.preventDefault(); uploadZone.classList.add('border-rose-500','bg-rose-50/40'); }));
  ['dragleave','drop'].forEach(ev =>
    uploadZone.addEventListener(ev, e => { e.preventDefault(); uploadZone.classList.remove('border-rose-500','bg-rose-50/40'); }));
  uploadZone.addEventListener('drop', e => {
    if (e.dataTransfer.files[0]) { fileInput.files = e.dataTransfer.files; fileInput.dispatchEvent(new Event('change')); }
  });
}

async function loadUploads() {
  try {
    const r = await fetch('/api/uploads');
    const d = await r.json();
    const box = $('uploads-list');
    if (!d.ok || !d.files.length) {
      box.innerHTML = '<div class="text-[10px] text-neutral-500 italic">No uploads yet</div>';
      return;
    }
    box.innerHTML = d.files.map(f => `
      <div class="flex justify-between items-center px-2 py-1 rounded bg-neutral-900 text-white text-[10px]">
        <span class="truncate flex-1">${f.name}</span>
        <span class="text-amber-300 ml-2">${f.count} rec</span>
      </div>`).join('');
  } catch(e) {}
}

/* ═════════════ SEQUENCE ═════════════ */
async function loadSequence() {
  try {
    const r = await fetch('/api/sequence?limit=60');
    const d = await r.json();
    const body = $('sequence-body');
    if (!d.ok || !d.records.length) {
      body.innerHTML = '<tr><td colspan="6" class="text-center text-neutral-500 py-3 italic">No data</td></tr>';
      return;
    }
    body.innerHTML = d.records.map((rec, i) => {
      const si = String(rec.issue).slice(-8);
      return `<tr class="border-b border-neutral-200 hover:bg-rose-50">
        <td class="px-1.5 py-1 text-neutral-500">${i+1}</td>
        <td class="px-1.5 py-1 text-neutral-700 truncate max-w-[70px]">${si}</td>
        <td class="px-1.5 py-1 text-center font-bold text-amber-700">${rec.sum}</td>
        <td class="px-1.5 py-1 text-center font-black text-neutral-900">${rec.number}</td>
        <td class="px-1.5 py-1 text-center"><span class="px-1.5 py-0.5 rounded text-[9px] font-bold ${sizeCellCls(rec.size)}">${rec.size}</span></td>
        <td class="px-1.5 py-1 text-center"><span class="px-1.5 py-0.5 rounded text-[9px] font-bold ${colorCellCls(rec.color)}">${rec.color.split('/')[0]}</span></td>
      </tr>`;
    }).join('');
  } catch(e) {}
}

/* ═════════════ TRAINING PROGRESS ═════════════ */
async function loadTraining() {
  try {
    const r = await fetch('/api/training-status');
    const d = await r.json();
    if (!d.ok) return;
    $('train-pct').textContent = d.progress.toFixed(1) + '%';
    $('train-bar').style.width = d.progress + '%';
    $('train-current').textContent = d.jsonRecords;
    $('train-days').textContent = d.daysOfData;
    if (d.ready) {
      $('train-bar').classList.remove('from-amber-500','to-rose-500');
      $('train-bar').classList.add('from-emerald-500','to-emerald-400');
      $('learning-status').textContent = 'Training Data COMPLETE — Ready!';
    }
  } catch(e) {}
}

function renderPatterns(p) {
  const wrap = $('pattern-list');
  if (!p) { wrap.innerHTML = '<div class="text-[11px] text-neutral-500 italic">No patterns</div>'; return; }
  const rows = [
    {key:'len6', label:'6-Length', data:p.len6},
    {key:'len5', label:'5-Length', data:p.len5},
    {key:'len4', label:'4-Length', data:p.len4},
  ];
  let html = '';
  rows.forEach(r => {
    const d = r.data;
    if (!d || !d.sample) {
      html += `<div class="p-2 rounded-lg bg-neutral-100 border border-neutral-200"><div class="flex justify-between text-[11px]"><span class="font-bold text-neutral-700">${r.label} Pattern</span><span class="text-neutral-500">no match</span></div></div>`;
      return;
    }
    const bsBreak = Object.entries(d.bsBreak||{}).map(([k,v])=>`${k}:${Math.round(v*100)}%`).join(' • ') || '-';
    const colBreak = Object.entries(d.colorBreak||{}).map(([k,v])=>`${k}:${Math.round(v*100)}%`).join(' • ') || '-';
    const bsCls = colorChipClass(d.dominantBS||'');
    const colCls = colorChipClass(d.dominantColor||'');
    html += `
      <div class="p-2 rounded-lg bg-white border border-neutral-200 shadow-sm">
        <div class="flex justify-between items-center text-[11px] mb-1">
          <span class="font-bold text-neutral-800">${r.label} Pattern</span>
          <span class="text-[10px] font-mono text-neutral-500">${d.pattern}</span>
        </div>
        <div class="flex justify-between text-[10px] mb-1">
          <span class="text-neutral-600">Sample: <b>${d.sample}</b></span>
          <span class="flex gap-1">
            <span class="${bsCls} text-white px-1.5 py-0.5 rounded text-[9px] font-bold">${d.dominantBS||'-'}</span>
            <span class="${colCls} text-white px-1.5 py-0.5 rounded text-[9px] font-bold">${d.dominantColor||'-'}</span>
          </span>
        </div>
        <div class="text-[9px] text-neutral-500">BS: ${bsBreak}</div>
        <div class="text-[9px] text-neutral-500">Color: ${colBreak}</div>
      </div>`;
  });
  wrap.innerHTML = html;
}

function renderNumbers(freq) {
  const grid = $('numbers-grid');
  if (!freq || !freq.length) return;
  let html = '';
  for (let i = 0; i < 10; i++) {
    const cnt = freq[i] || 0;
    const col = numColorClass(window._colorByNum ? window._colorByNum[i] : null);
    html += `<div class="bg-neutral-900 text-white rounded-lg p-1.5 text-center shadow-sm">
      <div class="w-5 h-5 mx-auto rounded-full ${col} text-[11px] font-black flex items-center justify-center">${i}</div>
      <div class="text-[9px] font-mono text-neutral-300 mt-1">${cnt}</div>
    </div>`;
  }
  grid.innerHTML = html;
}

async function loadNumbersMeta() {
  try {
    const r = await fetch('/api/numbers');
    const d = await r.json();
    window._colorByNum = d.colorByNum;
  } catch(e) {}
}

async function refresh() {
  try {
    const r = await fetch('/api/analysis');
    const d = await r.json();
    if (!d.ok || !d.analysis) { setText('stat-status', 'Loading'); return; }
    const a = d.analysis;

    setText('stat-total', d.total ?? 0);
    setText('stat-issue-short', (d.lastIssue||'--').slice(-5));
    const now = new Date();
    setText('stat-update', now.toLocaleDateString('en-GB',{day:'2-digit',month:'short',year:'numeric'}));
    setText('stat-update-time', now.toISOString().slice(11,19) + ' UTC');
    setText('stat-status', d.online ? 'Online' : 'Offline');
    setText('stat-conf', (a.confidence||0).toFixed(1)+'%');
    setText('online-tag', d.online ? 'Online' : 'Offline');

    // Training progress
    if (d.trainingProgress !== undefined) {
      $('train-pct').textContent = d.trainingProgress + '%';
      $('train-bar').style.width = d.trainingProgress + '%';
      $('train-current').textContent = d.total;
    }

    setText('primary-signal', a.primarySignal || 'BALANCED');
    setText('p1-number', a.number);
    setText('p1-bs', a.bs);
    setText('p1-color', a.color);
    setText('p1-conf', (a.confidence||0).toFixed(1)+'%');
    setText('p2-number', a.number2);
    setText('p2-bs', a.bs2);
    setText('p2-color', a.color2);
    setText('p2-conf', (a.confidence2||0).toFixed(1)+'%');
    $('p1-bs').className = 'text-[10px] px-2 py-0.5 rounded font-bold text-white ' + colorChipClass(a.bs);
    $('p1-color').className = 'text-[10px] px-2 py-0.5 rounded font-bold text-white ' + colorChipClass(a.color);
    $('p2-bs').className = 'text-[10px] px-2 py-0.5 rounded font-bold text-white ' + colorChipClass(a.bs2);
    $('p2-color').className = 'text-[10px] px-2 py-0.5 rounded font-bold text-white ' + colorChipClass(a.color2);

    setText('pattern-tag', a.patternTag);
    setText('size-streak-info', `${a.sizeStreakType} x${a.sizeStreak}`);
    setText('color-streak-info', `${a.colorStreakType} x${a.colorStreak}`);
    setText('learning-status', a.ready ? 'Model Trained — Ready' : 'Learning patterns...');

    setText('summary-num', `${a.number} & ${a.number2}`);
    setText('summary-num-text', `Top: ${a.top?.join(', ') || '-'}`);
    setText('summary-bs', `${a.bs} / ${a.bs2}`);
    setText('summary-bs-text', `${a.sizeStreakType} streak x${a.sizeStreak}`);
    setText('summary-color', `${a.color} / ${a.color2}`);
    setText('summary-color-text', `${a.colorStreakType} streak x${a.colorStreak}`);

    const totalCol = (a.red + a.green + a.violet) || 1;
    const rPct = (a.red/totalCol*100);
    const gPct = (a.green/totalCol*100);
    const vPct = (a.violet/totalCol*100);
    setText('red-count', a.red); setText('red-pct', `(${rPct.toFixed(2)}%)`);
    setText('green-count', a.green); setText('green-pct', `(${gPct.toFixed(2)}%)`);
    setText('violet-count', a.violet); setText('violet-pct', `(${vPct.toFixed(2)}%)`);
    $('red-bar').style.width = rPct+'%';
    $('green-bar').style.width = gPct+'%';
    $('violet-bar').style.width = vPct+'%';

    renderNumbers(a.freq);
    renderPatterns(a.patterns);

    const jsonObj = {
      system: "XOMAT AI v8.2",
      current_issue: d.lastIssue, next_issue: d.nextIssue,
      training: { records: d.total, target: d.trainingTarget, progress: d.trainingProgress + "%" },
      analysis: {
        primary_signal: a.primarySignal,
        size_streak: `${a.sizeStreakType} x${a.sizeStreak}`,
        color_streak: `${a.colorStreakType} x${a.colorStreak}`,
        pattern_tag: a.patternTag,
        top_numbers: a.top, missing: a.missing, frequency: a.freq,
      },
      prediction: {
        rank1: { number: a.number, bs: a.bs, color: a.color, confidence: a.confidence },
        rank2: { number: a.number2, bs: a.bs2, color: a.color2, confidence: a.confidence2 },
      },
    };
    $('json-view').textContent = JSON.stringify(jsonObj, null, 2);

    if (lastIssue !== d.lastIssue) {
      lastIssue = d.lastIssue;
      document.querySelectorAll('[id^="p"]').forEach(el => {
        if (el.textContent.match(/^\d+$/)) { el.classList.remove('digit-roll'); void el.offsetWidth; el.classList.add('digit-roll'); }
      });
    }
  } catch(e) { console.error('refresh err', e); }
}

loadNumbersMeta();
loadUploads();
loadSequence();
loadTraining();
setInterval(refresh, 3000);
setInterval(loadSequence, 8000);
setInterval(loadTraining, 10000);
refresh();
</script>
</body></html>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("═"*60)
    print("  XOMAT AI v8.2 — Upload + Train + Pattern Matcher")
    print(f"  ➜  http://localhost:{port}")
    print(f"  🎯 Training Target: {TRAINING_TARGET} records (4 days)")
    print("═"*60)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
