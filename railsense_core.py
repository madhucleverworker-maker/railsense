"""RailSense logic: prediction helpers, risk levels, waterfall, gauge, best-hour scan, report.

Nothing in here imports Streamlit, so it can be tested on its own.
"""
import math

import numpy as np
import pandas as pd
from scipy.stats import norm

# ------------------------------------------------------------------ constants
INK = "#0F1B2D"
MUTED = "#5B6B80"
RAIL = "#0E7490"
GO = "#15803D"
CAUTION = "#C2710C"
STOP = "#B91C1C"
SOFT = "#9FB3C4"

# Must match the column order of data/trains.csv (minus delay_min).
FEATURES = ["train_type", "hour", "day_of_week", "month", "distance_km", "weather", "prev_delay_min"]
FEATURE_LABELS = {
    "train_type": "Train type",
    "hour": "Arrival hour",
    "day_of_week": "Day of week",
    "month": "Month",
    "distance_km": "Distance",
    "weather": "Weather",
    "prev_delay_min": "Delay at last station",
}
TRAIN_TYPES = ["Express", "Superfast", "Passenger", "Rajdhani"]
WEATHERS = ["Clear", "Rain", "Fog"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
Z90 = float(norm.ppf(0.90))

# A "typical" trip used as the starting point of the waterfall.
NEUTRAL = {
    "train_type": "Express", "hour": 12, "day_of_week": 2, "month": 6,
    "weather": "Clear", "prev_delay_min": 0,
}


# ------------------------------------------------------------------ prediction
def predict_many(model, rows):
    """Predicted delay (minutes, never below 0) for a list of trip dicts."""
    frame = pd.DataFrame(rows)[FEATURES]
    return np.clip(model.predict(frame), 0, None)


def predict_one(model, trip):
    return float(predict_many(model, [trip])[0])


def miss_chance(gap, delay, sigma):
    """Probability (0 to 1) that the delay is longer than the gap between trains."""
    return 1 - norm.cdf((np.asarray(gap, dtype=float) - delay) / sigma)


def risk_level(p):
    """Return (colour, label, advice) for a miss probability between 0 and 1."""
    if p < 0.10:
        return GO, "Low risk", "Your connection looks safe. Walk, don't run."
    if p < 0.50:
        return CAUTION, "Medium risk", "Move quickly on arrival and know your backup train."
    return STOP, "High risk", "Plan a later connection and check alternatives now."


def level_name(percent):
    return "Low" if percent < 10 else "Medium" if percent < 50 else "High"


# ------------------------------------------------------------------ waterfall
def waterfall_steps(model, trip):
    """Split the predicted delay into the contribution of each input.

    Starts from a typical trip (Express, clear weather, no earlier delay, same distance)
    and switches in your real choices one at a time. The split depends on the order,
    so it is a guide, not an exact accounting.
    """
    cur = dict(NEUTRAL, distance_km=trip["distance_km"])

    def raw(d):
        return float(model.predict(pd.DataFrame([d])[FEATURES])[0])

    prev = raw(cur)
    steps = [{"label": "Baseline", "start": 0.0, "end": prev, "delta": prev, "kind": "Baseline"}]
    order = [
        ("Train type", "train_type"), ("Weather", "weather"), ("Month", "month"),
        ("Time of day", "hour"), ("Day", "day_of_week"), ("Previous delay", "prev_delay_min"),
    ]
    for label, key in order:
        cur[key] = trip[key]
        new = raw(cur)
        delta = new - prev
        kind = "No change" if abs(delta) < 0.5 else "Adds delay" if delta > 0 else "Reduces delay"
        steps.append({"label": label, "start": prev, "end": new, "delta": delta, "kind": kind})
        prev = new
    steps.append(
        {"label": "Predicted delay", "start": 0.0, "end": max(prev, 0.0),
         "delta": max(prev, 0.0), "kind": "Total"}
    )

    out = pd.DataFrame(steps)
    out["top"] = out[["start", "end"]].max(axis=1)

    def text(r):
        if r["kind"] in ("Baseline", "Total"):
            return f"{r['delta']:.0f}"
        d = int(round(r["delta"]))
        return "0" if d == 0 else f"{d:+d}"

    out["text"] = out.apply(text, axis=1)
    return out


# ------------------------------------------------------------------ best time
def hour_scan(model, trip, gap, sigma):
    """Predicted delay and miss chance for every arrival hour, keeping the gap fixed."""
    rows = [dict(trip, hour=h) for h in range(24)]
    delay = predict_many(model, rows)
    risk = miss_chance(gap, delay, sigma) * 100
    return pd.DataFrame(
        {
            "Hour": range(24),
            "Time": [f"{h:02d}:00" for h in range(24)],
            "Delay": delay,
            "Risk": risk,
            "Level": [level_name(r) for r in risk],
        }
    )


# ------------------------------------------------------------------ gauge
def gauge_svg(p):
    """Half-circle risk gauge as an inline SVG string. p is 0 to 1."""
    cx, cy = 100, 100

    def pt(frac, r):
        a = math.pi * (1 - frac)
        return cx + r * math.cos(a), cy - r * math.sin(a)

    parts = []
    for lo, hi, colour in [(0.0, 0.10, GO), (0.10, 0.50, CAUTION), (0.50, 1.0, STOP)]:
        x0, y0 = pt(lo, 78)
        x1, y1 = pt(hi, 78)
        parts.append(
            f'<path d="M {x0:.2f} {y0:.2f} A 78 78 0 0 1 {x1:.2f} {y1:.2f}" '
            f'fill="none" stroke="{colour}" stroke-width="16"/>'
        )
    nx, ny = pt(min(max(p, 0.0), 1.0), 60)
    colour, _, _ = risk_level(p)
    parts.append(f'<line x1="{cx}" y1="{cy}" x2="{nx:.2f}" y2="{ny:.2f}" stroke="{INK}" stroke-width="4" stroke-linecap="round"/>')
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="7" fill="{INK}"/>')
    parts.append(
        f'<text x="{cx}" y="128" text-anchor="middle" font-size="26" font-weight="700" '
        f'fill="{colour}" font-family="IBM Plex Sans Condensed, sans-serif">{p * 100:.0f}%</text>'
    )
    return f'<svg viewBox="0 0 200 138" width="100%" role="img" aria-label="Risk gauge">{"".join(parts)}</svg>'


# ------------------------------------------------------------------ report
def report_html(info):
    """Self-contained HTML report. Open it in a browser and print to save as PDF."""
    colour, label, advice = risk_level(info["p"])
    rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in info["inputs"].items()
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>RailSense report</title>
<style>
body{{font-family:Arial,sans-serif;color:{INK};max-width:680px;margin:2rem auto;padding:0 1rem}}
h1{{margin-bottom:0}} .sub{{color:{MUTED};margin-top:.2rem}}
.card{{border:1px solid #D5DDE6;border-left:10px solid {colour};border-radius:10px;padding:1rem 1.4rem;margin:1.2rem 0}}
.pct{{font-size:3rem;font-weight:700;color:{colour}}}
table{{border-collapse:collapse;width:100%}} td{{border-bottom:1px solid #E3E9F0;padding:.45rem .2rem}}
td:first-child{{color:{MUTED};width:55%}} small{{color:{MUTED}}}
</style></head><body>
<h1>RailSense report</h1><p class="sub">Created {info['created']}</p>
<div class="card"><div class="pct">{info['p'] * 100:.0f}%</div>
<b>{label}: chance of missing your connection</b><p>{advice}</p></div>
<table>
<tr><td>Predicted delay</td><td>{info['mu']:.0f} min (typical error about {info['mae']:.0f} min)</td></tr>
<tr><td>Expected arrival</td><td>{info['expected']} (scheduled {info['scheduled']})</td></tr>
<tr><td>Next train leaves</td><td>{info['departs']}</td></tr>
<tr><td>Time needed for 90% safety</td><td>{info['need']:.0f} min</td></tr>
</table>
<h3>Trip details</h3><table>{rows}</table>
<p><small>The model is trained on simulated trips, so treat this as a demonstration.</small></p>
</body></html>"""