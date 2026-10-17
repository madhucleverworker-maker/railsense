import datetime as dt
import inspect

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from railsense_core import (
    CAUTION, DAYS, FEATURE_LABELS, GO, INK, MONTHS, MUTED, RAIL, SOFT, STOP,
    TRAIN_TYPES, WEATHERS, Z90,
    gauge_svg, hour_scan, miss_chance, predict_many, predict_one, report_html,
    risk_level, waterfall_steps,
)

st.set_page_config(page_title="RailSense", page_icon="🚆", layout="wide")

PAPER, CARD = "#EEF2F6", "#FFFFFF"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@500;600;700&display=swap');

.stApp, .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stMarkdown {{
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
}}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 4rem; max-width: 1100px; }}

[data-testid="stSidebar"] {{ background: {CARD}; border-right: 1px solid #D5DDE6; }}
[data-testid="stSidebar"] h3 {{ font-size: 1rem; margin: .4rem 0 .2rem; color: {INK}; }}

.brand {{ display:flex; align-items:baseline; gap:.75rem; margin-bottom:.5rem; }}
.brand .name {{
    font-family: 'IBM Plex Sans Condensed', sans-serif; font-weight: 700;
    font-size: 2.6rem; letter-spacing: -0.01em; color: {INK}; line-height: 1.1;
}}
.brand span {{ color: {MUTED}; font-size: 1rem; }}

.signal {{
    background: {CARD}; border-radius: 14px; padding: 1.4rem 1.6rem;
    border: 1px solid #D5DDE6; border-left: 10px solid var(--c);
    display:flex; align-items:center; gap:1.6rem; flex-wrap:wrap;
}}
.signal .pct {{
    font-family: 'IBM Plex Sans Condensed', sans-serif; font-weight: 700;
    font-size: 4.6rem; line-height: 1; color: var(--c); min-width: 9rem;
}}
.signal .pct small {{ font-size: 1.6rem; font-weight: 600; }}
.signal .what {{ flex:1; min-width: 15rem; }}
.signal .what b {{ font-size: 1.25rem; color: {INK}; }}
.signal .what p {{ margin: .25rem 0 0; color: {MUTED}; font-size: 1rem; }}
.meter {{ height: 8px; background:#E3E9F0; border-radius: 99px; margin-top:.8rem; overflow:hidden; }}
.meter i {{ display:block; height:100%; background: var(--c); border-radius: 99px; }}

.tiles {{ display:grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 1rem 0 1.4rem; }}
@media (max-width: 800px) {{ .tiles {{ grid-template-columns: 1fr; }} }}
.tile {{ background:{CARD}; border:1px solid #D5DDE6; border-radius: 10px; padding: 1rem 1.2rem; }}
.tile .k {{ color:{MUTED}; font-size:.9rem; }}
.tile .v {{
    font-family:'IBM Plex Sans Condensed', sans-serif; font-weight:600;
    font-size:2rem; color:{INK}; line-height:1.15; margin-top:.15rem;
}}
.tile .n {{ color:{MUTED}; font-size:.85rem; margin-top:.15rem; }}

.mini {{ background:{CARD}; border:1px solid #D5DDE6; border-radius:10px; padding:.9rem 1.1rem; }}
.mini.on {{ border-color:{RAIL}; box-shadow: inset 0 0 0 1px {RAIL}; }}
.mini .k {{ color:{MUTED}; font-size:.9rem; }}
.mini .v {{ font-family:'IBM Plex Sans Condensed', sans-serif; font-weight:600; font-size:1.6rem; color:{INK}; }}
.mini .n {{ color:{MUTED}; font-size:.85rem; }}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ helpers
@st.cache_resource
def load():
    return joblib.load("models/railsense.joblib")


def stretch(fn, obj, **kw):
    """Full-width rendering that works on old and new Streamlit versions."""
    if "width" in inspect.signature(fn).parameters:
        return fn(obj, width="stretch", **kw)
    return fn(obj, use_container_width=True, **kw)


def show(chart, height=320):
    chart = (
        chart.properties(height=height)
        .configure_view(strokeWidth=0)
        .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor="#E3E9F0")
    )
    stretch(st.altair_chart, chart)


def mini(title, value, note, on=False, colour=None):
    style = f' style="border-left:8px solid {colour}"' if colour else ""
    return (
        f'<div class="mini{" on" if on else ""}"{style}><div class="k">{title}</div>'
        f'<div class="v">{value}</div><div class="n">{note}</div></div>'
    )


art = load()
model, sigma, mae = art["model"], max(float(art["sigma"]), 1e-6), art["mae"]
model_name = art.get("model_name", "Gradient Boosting")

ROUTES = {
    "Chennai → Bengaluru": 350,
    "Chennai → Coimbatore": 500,
    "Chennai → Madurai": 460,
    "Chennai → Delhi": 2180,
    "Chennai → Mumbai": 1280,
    "Chennai → Hyderabad": 710,
    "Custom (enter km)": None,
}

# ------------------------------------------------------------------ sidebar inputs
with st.sidebar:
    st.markdown("### Your train")
    train_type = st.selectbox("Train type", TRAIN_TYPES)
    route = st.selectbox("Route", list(ROUTES.keys()))
    if ROUTES[route] is None:
        distance = st.number_input("Distance (km)", 50, 2200, 800)
    else:
        distance = ROUTES[route]
        st.caption(f"{distance:,} km")
    arrival = st.time_input("Scheduled arrival", dt.time(18, 0), step=300)
    day = st.selectbox("Day", range(7), format_func=lambda i: DAYS[i])
    month = st.slider("Month", 1, 12, dt.date.today().month)
    weather = st.radio("Weather", WEATHERS, horizontal=True)
    prev_delay = st.slider("Delay at last station (min)", 0, 90, 30)

    st.markdown("### Your connection")
    buffer_min = st.number_input(
        "Next train leaves this many minutes after scheduled arrival", 0, 300, 45
    )

hour = arrival.hour
trip = {
    "train_type": train_type, "hour": hour, "day_of_week": day, "month": month,
    "distance_km": distance, "weather": weather, "prev_delay_min": prev_delay,
}

# ------------------------------------------------------------------ prediction
mu = predict_one(model, trip)
p_miss = float(miss_chance(buffer_min, mu, sigma))
pct = p_miss * 100
need = mu + Z90 * sigma
short = max(need - buffer_min, 0)
colour, label, advice = risk_level(p_miss)

sched = dt.datetime.combine(dt.date.today(), arrival)
expected = sched + dt.timedelta(minutes=mu)
departs = sched + dt.timedelta(minutes=buffer_min)

# ------------------------------------------------------------------ header + result
st.markdown(
    '<div class="brand"><div class="name">RailSense</div>'
    "<span>Will you make your connection?</span></div>",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="signal" style="--c:{colour}">
<div class="pct">{pct:.0f}<small>%</small></div>
<div class="what">
<b>{label}: chance of missing your connection</b>
<p>{advice}</p>
<div class="meter"><i style="width:{min(max(pct, 1), 100):.0f}%"></i></div>
</div>
</div>
""",
    unsafe_allow_html=True,
)

gap_note = f"{short:.0f} min more than you have" if short > 0 else "You have enough time"
st.markdown(
    f"""
<div class="tiles">
<div class="tile"><div class="k">Predicted delay</div>
<div class="v">{mu:.0f} min</div>
<div class="n">Typically off by about {mae:.0f} min</div></div>
<div class="tile"><div class="k">Expected arrival</div>
<div class="v">{expected:%H:%M}</div>
<div class="n">Scheduled {sched:%H:%M}. Next train leaves {departs:%H:%M}</div></div>
<div class="tile"><div class="k">Time needed for 90% safety</div>
<div class="v">{need:.0f} min</div>
<div class="n">{gap_note}</div></div>
</div>
""",
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state["history"] = []

tabs = st.tabs(
    ["Risk curve", "What if…", "Best time", "Breakdown", "Compare", "Model", "Save & export"]
)

# ------------------------------------------------------------------ 1. risk curve + gauge
with tabs[0]:
    st.caption("How the chance of missing your train falls as the gap between trains grows.")
    left, right = st.columns([1, 2])
    with left:
        st.markdown(gauge_svg(p_miss), unsafe_allow_html=True)
        st.caption("Green up to 10%, amber up to 50%, red above.")
    with right:
        buffers = np.arange(0, 181, 2)
        curve = pd.DataFrame({"gap": buffers, "risk": miss_chance(buffers, mu, sigma) * 100})
        base = alt.Chart(curve).encode(
            x=alt.X("gap:Q", title="Minutes between scheduled arrival and next train"),
            y=alt.Y("risk:Q", title="Chance of missing (%)", scale=alt.Scale(domain=[0, 100])),
        )
        guides = (
            alt.Chart(pd.DataFrame({"y": [10, 50]}))
            .mark_rule(strokeDash=[4, 4], color=SOFT)
            .encode(y="y:Q")
        )
        here = pd.DataFrame({"gap": [buffer_min], "risk": [pct]})
        show(
            base.mark_area(opacity=0.10, color=RAIL)
            + base.mark_line(color=RAIL, strokeWidth=3)
            + guides
            + alt.Chart(here).mark_rule(color=colour, strokeWidth=2).encode(x="gap:Q")
            + alt.Chart(here)
            .mark_point(filled=True, size=170, color=colour, stroke="white", strokeWidth=2)
            .encode(x="gap:Q", y="risk:Q"),
            height=300,
        )

# ------------------------------------------------------------------ 2. what if
with tabs[1]:
    st.caption("Change one thing and see what happens. Everything else stays as in the sidebar.")

    st.markdown("**Weather**")
    cols = st.columns(3)
    for col, w in zip(cols, WEATHERS):
        d = predict_one(model, dict(trip, weather=w))
        r = float(miss_chance(buffer_min, d, sigma)) * 100
        col.markdown(
            mini(w, f"{d:.0f} min delay", f"{r:.0f}% chance of missing", on=(w == weather)),
            unsafe_allow_html=True,
        )

    st.markdown("&nbsp;")
    st.markdown("**Train type**")
    types_df = pd.DataFrame(
        {
            "Train": TRAIN_TYPES,
            "Delay": predict_many(model, [dict(trip, train_type=t) for t in TRAIN_TYPES]),
            "Current": [t == train_type for t in TRAIN_TYPES],
        }
    )
    show(
        alt.Chart(types_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Train:N", sort=TRAIN_TYPES, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Delay:Q", title="Predicted delay (min)"),
            color=alt.condition(alt.datum.Current, alt.value(RAIL), alt.value(SOFT)),
            tooltip=["Train:N", alt.Tooltip("Delay:Q", format=".0f")],
        ),
        height=260,
    )

    st.markdown("**Month of travel**")
    month_df = pd.DataFrame(
        {
            "Month": MONTHS,
            "Delay": predict_many(model, [dict(trip, month=m) for m in range(1, 13)]),
        }
    )
    line = (
        alt.Chart(month_df)
        .mark_line(color=RAIL, strokeWidth=3, point=True)
        .encode(
            x=alt.X("Month:N", sort=MONTHS, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Delay:Q", title="Predicted delay (min)"),
            tooltip=["Month:N", alt.Tooltip("Delay:Q", format=".0f")],
        )
    )
    now_pt = (
        alt.Chart(month_df[month_df["Month"] == MONTHS[month - 1]])
        .mark_point(filled=True, size=170, color=colour, stroke="white", strokeWidth=2)
        .encode(x=alt.X("Month:N", sort=MONTHS), y="Delay:Q")
    )
    show(line + now_pt, height=260)

    st.markdown("**Delay at the last station**")
    sweep = np.arange(0, 91, 5)
    sw = pd.DataFrame(
        {
            "now": sweep,
            "final": predict_many(model, [dict(trip, prev_delay_min=int(s)) for s in sweep]),
        }
    )
    sline = alt.Chart(sw).mark_line(color=RAIL, strokeWidth=3).encode(
        x=alt.X("now:Q", title="Delay at last station (min)"),
        y=alt.Y("final:Q", title="Predicted delay at arrival (min)"),
    )
    spt = (
        alt.Chart(pd.DataFrame({"now": [prev_delay], "final": [mu]}))
        .mark_point(filled=True, size=150, color=colour, stroke="white", strokeWidth=2)
        .encode(x="now:Q", y="final:Q")
    )
    show(sline + spt, height=260)

# ------------------------------------------------------------------ 3. best time
with tabs[2]:
    st.caption(
        "Predicted delay for every arrival hour, with your other inputs and your gap "
        f"of {buffer_min} min unchanged."
    )
    scan = hour_scan(model, trip, buffer_min, sigma)
    best3 = scan.sort_values(["Risk", "Delay"]).head(3)
    mine = scan.loc[scan["Hour"] == hour].iloc[0]
    st.markdown(
        f"**Safest arrival hours:** {', '.join(best3['Time'])}. "
        f"Your hour ({mine['Time']}) has a {mine['Risk']:.0f}% chance of missing "
        f"and the safest has {best3.iloc[0]['Risk']:.0f}%."
    )

    # Heatmap: hour x weather
    heat_rows = [dict(trip, weather=w, hour=h) for w in WEATHERS for h in range(24)]
    heat = pd.DataFrame(
        {
            "Hour": [r["hour"] for r in heat_rows],
            "Weather": [r["weather"] for r in heat_rows],
            "Delay": predict_many(model, heat_rows),
        }
    )
    lo, hi = float(heat["Delay"].min()), float(heat["Delay"].max())
    if hi - lo < 1:
        hi = lo + 1
    mid = (lo + hi) / 2
    hb = alt.Chart(heat).encode(
        x=alt.X("Hour:O", title="Scheduled arrival hour", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Weather:O", sort=WEATHERS, title=None),
    )
    rect = hb.mark_rect(cornerRadius=3).encode(
        color=alt.Color(
            "Delay:Q",
            scale=alt.Scale(domain=[lo, mid, hi], range=["#E4F1F4", "#F2C879", STOP]),
            legend=alt.Legend(title="Delay (min)"),
        )
    )
    txt = hb.mark_text(fontSize=10).encode(
        text=alt.Text("Delay:Q", format=".0f"),
        color=alt.condition(alt.datum.Delay > mid + (hi - mid) * 0.4, alt.value("white"), alt.value(INK)),
    )
    st.markdown("**Delay by hour and weather**")
    show(rect + txt, height=200)

    st.markdown("**Miss chance by arrival hour**")
    show(
        alt.Chart(scan)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("Time:N", sort=list(scan["Time"]), title=None, axis=alt.Axis(labelAngle=-90)),
            y=alt.Y("Risk:Q", title="Chance of missing (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color(
                "Level:N",
                scale=alt.Scale(domain=["Low", "Medium", "High"], range=[GO, CAUTION, STOP]),
                legend=alt.Legend(title="Risk"),
            ),
            tooltip=["Time:N", alt.Tooltip("Delay:Q", format=".0f"), alt.Tooltip("Risk:Q", format=".0f")],
        ),
        height=260,
    )

# ------------------------------------------------------------------ 4. breakdown (waterfall)
with tabs[3]:
    st.caption(
        "Where your predicted delay comes from. The baseline is a typical Express in clear "
        "weather with no earlier delay on your route. Each bar shows how one of your choices "
        "moves the delay up or down. The split depends on the order, so read it as a guide."
    )
    wf = waterfall_steps(model, trip)
    order = list(wf["label"])
    kinds = ["Baseline", "Adds delay", "Reduces delay", "No change", "Total"]
    kcolours = [SOFT, CAUTION, GO, "#CBD5E1", RAIL]
    wb = alt.Chart(wf).encode(x=alt.X("label:N", sort=order, title=None, axis=alt.Axis(labelAngle=-20)))
    bars = wb.mark_bar(cornerRadius=3, size=38).encode(
        y=alt.Y("start:Q", title="Delay (min)"),
        y2="end:Q",
        color=alt.Color("kind:N", scale=alt.Scale(domain=kinds, range=kcolours), legend=alt.Legend(title=None)),
        tooltip=["label:N", alt.Tooltip("delta:Q", format="+.1f", title="Change (min)")],
    )
    numbers = wb.mark_text(dy=-8, fontWeight="bold", color=INK).encode(y="top:Q", text="text:N")
    show(bars + numbers, height=340)

# ------------------------------------------------------------------ 5. compare
with tabs[4]:
    st.caption(
        "Compare your train with another one that connects to the same next train. "
        "Your connecting train leaves at "
        f"{departs:%H:%M}, so a train that arrives earlier gives you more time."
    )
    c1, c2, c3 = st.columns(3)
    alt_type = c1.selectbox("Other train type", TRAIN_TYPES, index=3, key="alt_type")
    alt_arr = c2.time_input(
        "Other train's scheduled arrival",
        dt.time((hour - 2) % 24, arrival.minute), step=300, key="alt_arr",
    )
    alt_prev = c3.slider("Its delay at last station (min)", 0, 90, prev_delay, key="alt_prev")

    alt_trip = dict(trip, train_type=alt_type, hour=alt_arr.hour, prev_delay_min=alt_prev)
    alt_sched = dt.datetime.combine(dt.date.today(), alt_arr)
    alt_gap = (departs - alt_sched).total_seconds() / 60
    alt_mu = predict_one(model, alt_trip)
    alt_p = float(miss_chance(alt_gap, alt_mu, sigma))
    alt_colour, alt_label, _ = risk_level(alt_p)

    a, b = st.columns(2)
    a.markdown(
        mini("Your train", f"{pct:.0f}% risk",
             f"{train_type}. Predicted delay {mu:.0f} min, gap {buffer_min} min.", colour=colour),
        unsafe_allow_html=True,
    )
    b.markdown(
        mini("Other train", f"{alt_p * 100:.0f}% risk",
             f"{alt_type}. Predicted delay {alt_mu:.0f} min, gap {alt_gap:.0f} min.", colour=alt_colour),
        unsafe_allow_html=True,
    )
    st.markdown("&nbsp;")
    if alt_gap < 0:
        st.error("The other train is scheduled to arrive after your connecting train leaves.")
    elif abs(alt_p - p_miss) < 0.02:
        st.info("Both options carry about the same risk.")
    elif alt_p < p_miss:
        st.success(f"The other train is safer by {(p_miss - alt_p) * 100:.0f} percentage points.")
    else:
        st.warning(f"Your train is safer by {(alt_p - p_miss) * 100:.0f} percentage points.")

# ------------------------------------------------------------------ 6. model
with tabs[5]:
    st.markdown(
        f"""
**How it works.** A **{model_name}** model predicts the delay at arrival from train type,
time, day, month, distance, weather and the delay at the last station.
The model's typical error then turns that single number into a probability: if the
predicted delay is longer than your gap between trains, you miss the connection.

**How good is it.** On held-out trips the average error is **{mae:.1f} min**
and the spread of errors is **{sigma:.1f} min**.

**Important.** The model is trained on 8,000 *simulated* trips, not real Indian Railways
records. Treat results as a demonstration of the method.
"""
    )

    if "comparison" in art:
        st.markdown("**Model comparison**")
        comp = art["comparison"]
        stretch(st.dataframe, comp, hide_index=True)
        st.caption("Lower MAE is better. The best model is the one used in this app.")

    if "importance" in art:
        st.markdown("**What matters most**")
        imp = pd.DataFrame(
            {
                "Input": [FEATURE_LABELS.get(k, k) for k in art["importance"]],
                "Importance": list(art["importance"].values()),
            }
        ).sort_values("Importance", ascending=False)
        show(
            alt.Chart(imp)
            .mark_bar(color=RAIL, cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
            .encode(
                x=alt.X("Importance:Q", title="Extra error (min) if this input is scrambled"),
                y=alt.Y("Input:N", sort="-x", title=None),
                tooltip=["Input:N", alt.Tooltip("Importance:Q", format=".2f")],
            ),
            height=260,
        )

    if "test_actual" in art:
        st.markdown("**Predicted vs actual delay**")
        sc = pd.DataFrame({"Actual": art["test_actual"], "Predicted": art["test_pred"]})
        top = float(max(sc["Actual"].max(), sc["Predicted"].max()))
        diag = pd.DataFrame({"Actual": [0, top], "Predicted": [0, top]})
        show(
            alt.Chart(sc)
            .mark_circle(color=RAIL, opacity=0.35, size=28)
            .encode(
                x=alt.X("Actual:Q", title="Actual delay (min)"),
                y=alt.Y("Predicted:Q", title="Predicted delay (min)"),
            )
            + alt.Chart(diag).mark_line(color=INK, strokeDash=[5, 4]).encode(x="Actual:Q", y="Predicted:Q"),
            height=320,
        )
        st.caption("Dots on the dashed line are perfect predictions. 600 held-out trips are shown.")

    if not all(k in art for k in ("comparison", "importance", "test_actual")):
        st.info("Run `python train.py` again to add the comparison table and the extra charts.")

# ------------------------------------------------------------------ 7. save & export
with tabs[6]:
    now = dt.datetime.now()
    row = {
        "Saved at": f"{now:%H:%M:%S}",
        "Route": route,
        "Train": train_type,
        "Weather": weather,
        "Arrival": f"{arrival:%H:%M}",
        "Gap (min)": int(buffer_min),
        "Delay (min)": round(mu),
        "Miss chance (%)": round(pct),
    }

    st.markdown("**This prediction**")
    d1, d2, d3 = st.columns(3)
    if d1.button("Save to history", type="primary"):
        st.session_state["history"].append(row)

    info = {
        "p": p_miss, "mu": mu, "mae": mae, "need": need, "created": f"{now:%d %b %Y, %H:%M}",
        "expected": f"{expected:%H:%M}", "scheduled": f"{sched:%H:%M}", "departs": f"{departs:%H:%M}",
        "inputs": {
            "Route": route, "Distance": f"{distance:,} km", "Train type": train_type,
            "Scheduled arrival": f"{arrival:%H:%M}", "Day": DAYS[day], "Month": MONTHS[month - 1],
            "Weather": weather, "Delay at last station": f"{prev_delay} min",
            "Gap to next train": f"{buffer_min} min",
        },
    }
    d2.download_button(
        "Download report (HTML)", report_html(info).encode("utf-8"),
        file_name="railsense_report.html", mime="text/html",
    )
    d3.download_button(
        "Download this prediction (CSV)", pd.DataFrame([row]).to_csv(index=False).encode("utf-8"),
        file_name="railsense_prediction.csv", mime="text/csv",
    )
    st.caption("Open the HTML report in your browser and choose Print, then Save as PDF.")

    st.markdown("**History**")
    hist = st.session_state["history"]
    if hist:
        hist_df = pd.DataFrame(hist)
        stretch(st.dataframe, hist_df, hide_index=True)
        h1, h2 = st.columns(2)
        h1.download_button(
            "Download history (CSV)", hist_df.to_csv(index=False).encode("utf-8"),
            file_name="railsense_history.csv", mime="text/csv",
        )
        if h2.button("Clear history"):
            st.session_state["history"] = []
            st.rerun()
    else:
        st.info("Nothing saved yet. Choose Save to history to keep a prediction for comparison.")