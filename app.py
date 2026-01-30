import time
from pathlib import Path
from collections import deque
from datetime import datetime
import base64

import numpy as np
import pandas as pd
import requests
import joblib
import streamlit as st


# =======================
# BRANDING
# =======================
PROJECT_TITLE = "Smart Agri-Tech Mobility System for Real-time Environmental Tracking"
TEAM_LINE = "Team: Group 8 – Section 1"
ORG_LINE = "College of Engineering and Computer Science • VinUniversity"
ADDRESS_LINE = "Vinhomes Ocean Park, Gia Lam District, Hanoi, Vietnam"
COURSE_LINE = "Course: Introduction to Engineering and Computer Science"

AUTHORS = [
    ("Pham Gia Hung", "V202502278", "Hardware Specialist"),
    ("Le Tri Duc", "V202502315", "Software & Algorithm Developer"),
    ("Dang Duong Tam", "V202502431", "Backend & Integration Developer"),
    ("Ngo Quy Khang", "V202502421", "Project Manager & Systems Integrator"),
    ("Le Cong Duy", "V202502713", "Control Systems Engineer / AI Research"),
]

ASSETS = Path("assets")
LOGO_PATH = ASSETS / "vinuni_logo.png"
HERO_CANDIDATES = [ASSETS / "hero.jpg", ASSETS / "hero.jpeg", ASSETS / "hero.png", ASSETS / "hero.webp"]
REPORT_PATH = ASSETS / "report.pdf"

EVIDENCE_DIR = ASSETS / "media"      # rover/team/testing photos
DIAGRAM_DIR  = ASSETS / "diagrams"   # flow/architecture/wiring images


# =======================
# LIVE SYSTEM
# =======================
BLYNK_TOKEN = st.secrets.get("BLYNK_TOKEN", "")
if not BLYNK_TOKEN:
    st.error("Missing BLYNK_TOKEN in Streamlit Secrets. Go to Manage app → Settings → Secrets and set BLYNK_TOKEN.")
    st.stop()

BASE_URL = "https://blynk.cloud/external/api/get"
MODEL_PATH = Path("models/model_30s.pkl")

# ✅ FIXED ORDER (do NOT change)
FEATURES = ["temp", "humidity", "soil", "ph"]
PINS = {"temp": "V0", "soil": "V1", "ph": "V2", "humidity": "V3"}

LABELS = {"temp": "Temperature", "humidity": "Humidity", "soil": "Soil Moisture", "ph": "pH"}
ICONS  = {"temp": "🌡️", "humidity": "💧", "soil": "🌱", "ph": "🧪"}
UNITS  = {"temp": "°C", "humidity": "%", "soil": "%", "ph": ""}

RANGE = {
    "temp": (0.0, 60.0),
    "humidity": (0.0, 100.0),
    "soil": (0.0, 100.0),
    "ph": (0.0, 14.0),
}

# "No clear change" thresholds
NOISE = {
    "temp": {"range": 0.6, "delta": 0.4},
    "humidity": {"range": 3.5, "delta": 2.0},
    "soil": {"range": 2.5, "delta": 1.5},
    "ph": {"range": 0.12, "delta": 0.08},
}


# =======================
# PAGE CONFIG (MOBILE FRIENDLY)
# =======================
st.set_page_config(
    page_title="Smart Agri-Tech Mobility System (VinUniversity)",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Force viewport for mobile
st.markdown(
    '<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">',
    unsafe_allow_html=True
)


# =======================
# UTIL: files & images
# =======================
def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def guess_mime(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(ext, "image/jpeg")

def file_to_b64(path: Path):
    if not path or not path.exists():
        return None, None
    return base64.b64encode(path.read_bytes()).decode("utf-8"), guess_mime(path)

def list_images(folder: Path):
    if not folder.exists():
        return []
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])

HERO_PATH = first_existing(HERO_CANDIDATES)
hero_b64, hero_mime = file_to_b64(HERO_PATH) if HERO_PATH else (None, None)
logo_b64, logo_mime = file_to_b64(LOGO_PATH)


# =======================
# VIBRANT CSS + MOBILE
# =======================
CSS = """
<style>
:root {
  --accent1: #7C3AED;
  --accent2: #06B6D4;
  --accent3: #22C55E;
  --accent4: #F97316;
  --ink: rgba(10, 10, 10, 0.92);
  --card: rgba(255,255,255,0.78);
  --border: rgba(0,0,0,0.08);
}
html, body { -webkit-text-size-adjust: 100%; }

.delta-up   { color: #EF4444; font-weight: 950; }
.delta-down { color: #22C55E; font-weight: 950; }
.delta-flat { color: rgba(0,0,0,0.55); font-weight: 900; }

.trend-up   { background: rgba(239,68,68,0.10); border: 1px solid rgba(239,68,68,0.22); }
.trend-down { background: rgba(34,197,94,0.10); border: 1px solid rgba(34,197,94,0.22); }
.trend-flat { background: rgba(0,0,0,0.05); border: 1px solid rgba(0,0,0,0.10); }

.kpi-line { padding: 10px 12px; border-radius: 14px; margin-top: 10px; }

.stApp {
  background:
    radial-gradient(1200px 800px at 12% 0%, rgba(124,58,237,0.20), transparent 55%),
    radial-gradient(1200px 800px at 88% 0%, rgba(6,182,212,0.18), transparent 55%),
    radial-gradient(1200px 800px at 50% 100%, rgba(34,197,94,0.12), transparent 55%);
}
header[data-testid="stHeader"] { background: rgba(0,0,0,0); }

.hero {
  border-radius: 28px;
  overflow: hidden;
  border: 1px solid var(--border);
  box-shadow: 0 22px 60px rgba(0,0,0,0.12);
  position: relative;
}
.hero-img {
  position:absolute; inset:0;
  width:100%; height:100%;
  object-fit: cover;
  transform: scale(1.05);
  filter: saturate(1.05) contrast(1.02);
}
.hero-fallback {
  position:absolute; inset:0;
  background: linear-gradient(135deg, rgba(124,58,237,0.95), rgba(6,182,212,0.85), rgba(34,197,94,0.70));
}
.hero-overlay {
  position:absolute; inset:0;
  /* DARKER overlay for readability */
  background: linear-gradient(120deg, rgba(0,0,0,0.78), rgba(0,0,0,0.28));
}
.hero-inner { position: relative; padding: 26px 26px 22px 26px; }
.hero-content { display:flex; gap:18px; justify-content: space-between; align-items:flex-start; }
.brand { display:flex; gap:14px; align-items:center; }
.brand-logo {
  width:68px; height:68px;
  border-radius:18px;
  background: rgba(255,255,255,0.92);
  border: 1px solid rgba(255,255,255,0.22);
  padding: 10px;
  object-fit: contain;
}
.brand-title {
  color: rgba(255,255,255,0.96);
  font-size: 30px;
  font-weight: 950;
  line-height: 1.12;
  margin:0;
}
.brand-sub{
  color: rgba(255,255,255,0.92) !important;
  text-shadow: 0 4px 16px rgba(0,0,0,0.55);
  font-size: 13px;
  line-height: 1.50;
  margin-top: 10px;
}
.brand-panel{
  background: rgba(0,0,0,0.22);
  border: 1px solid rgba(255,255,255,0.16);
  padding: 12px 14px;
  border-radius: 18px;
  backdrop-filter: blur(10px);
}
.badges { display:flex; gap:10px; flex-wrap:wrap; justify-content:flex-end; }
.badge {
  display:inline-flex; align-items:center; gap:8px;
  padding: 8px 12px; border-radius: 999px;
  font-size: 12px; font-weight: 950;
  border: 1px solid rgba(255,255,255,0.22);
  background: rgba(255,255,255,0.14);
  color: rgba(255,255,255,0.94);
}

.pill {
  display:inline-block; padding: 8px 12px; border-radius: 999px;
  font-weight: 950; font-size: 12px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.76);
}
.pill.ok { color:#15803D; }
.pill.bad { color:#B91C1C; }

.section {
  padding: 16px 18px;
  border-radius: 18px;
  border: 1px solid var(--border);
  background: var(--card);
  backdrop-filter: blur(12px);
  box-shadow: 0 12px 34px rgba(0,0,0,0.07);
}

.grid4 { display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }
@media (max-width: 1100px) { .grid4 { grid-template-columns: repeat(2, minmax(0, 1fr)); } }

.card {
  padding: 14px 16px;
  border-radius: 18px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.82);
  backdrop-filter: blur(12px);
  box-shadow: 0 14px 40px rgba(0,0,0,0.08);
  transition: transform .18s ease, box-shadow .18s ease;
}
.card:hover { transform: translateY(-3px); box-shadow: 0 22px 60px rgba(0,0,0,0.12); }
.card-top { display:flex; justify-content:space-between; align-items:center; gap: 10px; }
.card-name { font-weight: 950; opacity: 0.94; }
.card-value { font-size: 32px; font-weight: 950; margin-top: 6px; }
.card-delta { font-weight: 900; margin-top: 6px; font-size: 13px; }
.card-hint { margin-top: 6px; font-size: 12px; color: rgba(0,0,0,0.56); line-height: 1.35; }

.tag {
  display:inline-flex; align-items:center; gap:6px;
  padding: 6px 10px; border-radius: 999px;
  font-size: 12px; font-weight: 950;
  border: 1px solid var(--border);
}
.level-ok   { background: rgba(34,197,94,0.16);  color:#15803D; }
.level-warm { background: rgba(249,115,22,0.16); color:#9A3412; }
.level-hot  { background: rgba(239,68,68,0.16);  color:#B91C1C; }
.level-cold { background: rgba(6,182,212,0.16);  color:#0E7490; }

.media-card { border-radius: 18px; overflow:hidden; border: 1px solid var(--border); background: rgba(255,255,255,0.86); box-shadow: 0 12px 34px rgba(0,0,0,0.08); }
.media-card img { width:100%; max-height: 420px; object-fit: cover; display:block; }
.media-cap { padding: 10px 12px; color: rgba(0,0,0,0.70); font-weight: 800; font-size: 12px; }

.diagram-img { width:100%; max-height: 520px; object-fit: contain; display:block; background: rgba(0,0,0,0.03); }

.footer { margin-top: 18px; padding: 10px 14px; border-radius: 14px; border: 1px solid var(--border); background: rgba(255,255,255,0.74); color: rgba(0,0,0,0.62); font-size: 12px; }
hr { border:none; border-top:1px solid var(--border); margin:16px 0; }

/* MOBILE */
@media (max-width: 768px) {
  .hero-inner { padding: 18px 16px; }
  .hero-content { flex-direction: column; gap: 12px; }
  .badges { justify-content: flex-start; }
  .brand-logo { width:54px; height:54px; border-radius:16px; }
  .brand-title { font-size: 20px; line-height: 1.15; }
  .brand-sub { font-size: 12px; }
  .card-value { font-size: 26px; }
  .card { padding: 12px 12px; }
  .section { padding: 14px 14px; }
  .grid4 { grid-template-columns: 1fr !important; }
  .diagram-img { max-height: 320px; }
  .media-card img { max-height: 260px; }
}
@media (max-width: 420px) {
  .brand-title { font-size: 18px; }
  .badge { font-size: 11px; padding: 7px 10px; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# =======================
# CORE HELPERS
# =======================
def clamp(name: str, v: float) -> float:
    lo, hi = RANGE[name]
    return float(max(lo, min(hi, v)))

def fmt(name: str, v: float) -> str:
    if name == "ph":
        return f"{v:.2f}"
    return f"{v:.2f} {UNITS[name]}".strip()

def delta_fmt(name: str, d: float) -> str:
    sign = "+" if d >= 0 else ""
    if name == "ph":
        return f"{sign}{d:.2f}"
    return f"{sign}{d:.2f} {UNITS[name]}".strip()

def get_value(pin: str) -> float:
    r = requests.get(BASE_URL, params={"token": BLYNK_TOKEN, pin: ""}, timeout=10)
    r.raise_for_status()
    return float(r.text.strip())

def read_sensors() -> dict:
    # Read strictly in FEATURES order to avoid swapping
    vals = {}
    for name in FEATURES:
        vals[name] = get_value(PINS[name])
    return vals

@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Ensure models/model_30s.pkl is in the repo.")
    bundle = joblib.load(MODEL_PATH)
    if bundle.get("features") != FEATURES:
        raise ValueError(f"FEATURES mismatch. Model={bundle.get('features')} App={FEATURES}")
    return bundle

def level_temp(t: float, ideal_min: float, ideal_max: float, margin: float):
    if t < ideal_min - margin: return "cold", "Too Cold"
    if t < ideal_min:          return "cold", "Slightly Low"
    if t <= ideal_max:         return "ok",   "Optimal"
    if t <= ideal_max + margin:return "warm", "Slightly High"
    return "hot", "Too Hot"

def level_humidity(h: float, low: float, high: float, margin: float):
    if h < low - margin: return "cold", "Too Dry"
    if h < low:          return "cold", "Slightly Dry"
    if h <= high:        return "ok",   "Comfort"
    if h <= high + margin:return "warm","Humid"
    return "hot","Too Humid"

def level_soil(s: float, low: float, high: float, margin: float):
    if s < low - margin: return "hot",  "Very Dry"
    if s < low:          return "warm", "Dry"
    if s <= high:        return "ok",   "Good"
    if s <= high + margin:return "cold","Wet"
    return "cold","Very Wet"

def level_ph(p: float, low: float, high: float, margin: float):
    if p < low - margin: return "warm", "Strongly Acidic"
    if p < low:          return "warm", "Acidic"
    if p <= high:        return "ok",   "Neutral Range"
    if p <= high + margin:return "cold","Alkaline"
    return "cold","Strongly Alkaline"

def card_html(icon, title, value, level, label, delta=None, hint=None):
    delta_html = f"<div class='card-delta'>{delta}</div>" if delta else ""
    hint_html = f"<div class='card-hint'>{hint}</div>" if hint else ""
    return f"""
<div class="card">
  <div class="card-top">
    <div class="card-name">{icon} {title}</div>
    <div class="tag level-{level}">● {label}</div>
  </div>
  <div class="card-value">{value}</div>
  {delta_html}
  {hint_html}
</div>
"""

def insight_block(title, level, summary, bullets):
    lis = "".join([f"<li>{b}</li>" for b in bullets])
    return f"""
<div class="section">
  <div class="tag level-{level}">● Insight</div>
  <h3 style="margin-top:10px;">{title}</h3>
  <p>{summary}</p>
  <ul>{lis}</ul>
</div>
"""

def render_diagram(path: Path, caption: str):
    b64, mime = file_to_b64(path)
    if not b64:
        st.info(f"Missing: {path.name} (put it in {path.parent.as_posix()}/)")
        return
    st.markdown(
        f"""
<div class="section">
  <div class="tag level-ok">● Diagram</div>
  <h3 style="margin-top:10px;">{caption}</h3>
  <img class="diagram-img" src="data:{mime};base64,{b64}" />
</div>
""",
        unsafe_allow_html=True
    )

def pdf_embed(path: Path, height=900):
    if not path.exists():
        st.warning("Report PDF not found. Put it at assets/report.pdf")
        return
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    st.markdown(
        f"""<iframe src="data:application/pdf;base64,{b64}" width="100%" height="{height}"
        style="border:none;border-radius:16px;"></iframe>""",
        unsafe_allow_html=True
    )

# Highlight helpers
def delta_class(delta: float, thr: float) -> str:
    if abs(delta) <= thr:
        return "delta-flat"
    return "delta-up" if delta > 0 else "delta-down"

def trend_box_class(delta: float, thr: float) -> str:
    if abs(delta) <= thr:
        return "trend-flat"
    return "trend-up" if delta > 0 else "trend-down"

def arrow(delta: float, thr: float) -> str:
    if abs(delta) <= thr:
        return "→"
    return "↑" if delta > 0 else "↓"

def colored_delta_html(name: str, d: float, context: str) -> str:
    thr = NOISE[name]["delta"]
    cls = delta_class(d, thr)
    sym = arrow(d, thr)
    return f"{context} <span class='{cls}'>Δ {delta_fmt(name, d)} {sym}</span>"

def describe_last_window(
    name: str,
    w: pd.Series,
    current_val: float,
    pred_val: float | None,
    target_low: float,
    target_high: float,
    margin: float,
    unit: str
) -> tuple[str, str]:
    w = w.dropna()
    if len(w) < 2:
        return "cold", f"<p>Not enough samples in the last 30 seconds to analyze {LABELS[name].lower()}.</p>"

    start = float(w.iloc[0])
    end = float(w.iloc[-1])
    d = end - start
    r = float(w.max() - w.min())

    delta_thr = NOISE[name]["delta"]
    range_thr = NOISE[name]["range"]

    if r <= range_thr and abs(d) <= delta_thr:
        trend_word = "stable (no clear change)"
    elif abs(d) <= delta_thr and r > range_thr:
        trend_word = "fluctuating (no clear direction)"
    elif d > delta_thr:
        trend_word = "increasing"
    else:
        trend_word = "decreasing"

    # level
    if name == "temp":
        lvl, tag = level_temp(current_val, target_low, target_high, margin)
    elif name == "humidity":
        lvl, tag = level_humidity(current_val, target_low, target_high, margin)
    elif name == "soil":
        lvl, tag = level_soil(current_val, target_low, target_high, margin)
    else:
        lvl, tag = level_ph(current_val, target_low, target_high, margin)

    in_band = (target_low <= current_val <= target_high)
    band_word = "within the target range" if in_band else "outside the target range"

    box_cls = trend_box_class(d, delta_thr)
    d_cls = delta_class(d, delta_thr)
    sym = arrow(d, delta_thr)

    forecast_clause = ""
    if pred_val is not None and np.isfinite(pred_val):
        forecast_clause = f" <span style='color:rgba(0,0,0,0.55);font-weight:800;'>Forecast (~30s):</span> <b>{pred_val:.2f}{unit}</b>."

    html = f"""
<div class="kpi-line {box_cls}">
  <div style="font-weight:950;color:rgba(0,0,0,0.86);">
    Over the last 30 seconds, <b>{LABELS[name]}</b> was <b>{trend_word}</b> {sym}.
  </div>
  <div style="margin-top:6px;color:rgba(0,0,0,0.70);line-height:1.45;">
    Δ=<span class="{d_cls}">{d:+.2f}{unit} {sym}</span>,
    range=<b>{r:.2f}{unit}</b>.
    Current: <b>{current_val:.2f}{unit}</b> ({tag}), {band_word}
    (<b>{target_low:.2f}–{target_high:.2f}{unit}</b>).
    {forecast_clause}
  </div>
</div>
"""
    return lvl, html


# =======================
# SIDEBAR (controls only)
# =======================
st.sidebar.title("⚙️ Controls")
auto_refresh = st.sidebar.toggle("Auto refresh (Live)", value=True)
refresh_sec  = st.sidebar.slider("Refresh seconds", 2, 15, 5, 1)
history_len  = st.sidebar.slider("Trend history points", 30, 600, 200, 10)

st.sidebar.markdown("---")
st.sidebar.subheader("Targets / Thresholds")

ideal_min_t, ideal_max_t = st.sidebar.slider("Temperature ideal (°C)", 10.0, 40.0, (22.0, 30.0), 0.5)
margin_t = st.sidebar.slider("Temperature margin (°C)", 0.5, 5.0, 2.0, 0.5)

low_h, high_h = st.sidebar.slider("Humidity comfort (%)", 20.0, 95.0, (40.0, 75.0), 1.0)
margin_h = st.sidebar.slider("Humidity margin (%)", 2.0, 20.0, 8.0, 1.0)

low_s, high_s = st.sidebar.slider("Soil good range (%)", 0.0, 100.0, (30.0, 70.0), 1.0)
margin_s = st.sidebar.slider("Soil margin (%)", 2.0, 30.0, 10.0, 1.0)

low_p, high_p = st.sidebar.slider("pH neutral-ish range", 3.0, 11.0, (6.0, 7.5), 0.1)
margin_p = st.sidebar.slider("pH margin", 0.1, 2.0, 0.5, 0.1)

if not HERO_PATH:
    st.sidebar.warning("Hero image missing. Add one of: assets/hero.jpg / hero.png / hero.webp")
if not LOGO_PATH.exists():
    st.sidebar.warning("Logo missing: assets/vinuni_logo.png")
if not MODEL_PATH.exists():
    st.sidebar.warning("Model missing: models/model_30s.pkl")


# =======================
# STATE + MODEL
# =======================
bundle = load_bundle()
model = bundle["model"]
WINDOW_STEPS = int(bundle["window_steps"])

if "pred_buffer" not in st.session_state:
    st.session_state.pred_buffer = deque(maxlen=WINDOW_STEPS)

if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=history_len)
if st.session_state.history.maxlen != history_len:
    st.session_state.history = deque(list(st.session_state.history)[-history_len:], maxlen=history_len)


# =======================
# HERO
# =======================
now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
logo_html = (
    f"<img class='brand-logo' src='data:{logo_mime};base64,{logo_b64}'/>"
    if logo_b64 else
    "<div class='brand-logo' style='display:flex;align-items:center;justify-content:center;font-weight:950;'>VU</div>"
)
hero_bg = f"<img class='hero-img' src='data:{hero_mime};base64,{hero_b64}' />" if hero_b64 else "<div class='hero-fallback'></div>"

st.markdown(
    f"""
<div class="hero">
  {hero_bg}
  <div class="hero-overlay"></div>
  <div class="hero-inner">
    <div class="hero-content">
      <div>
        <div class="brand">
          {logo_html}
          <div class="brand-panel">
              <h1 class="brand-title">{PROJECT_TITLE}</h1>
              <div class="brand-sub">
        
              {TEAM_LINE}<br/>
              {ORG_LINE}<br/>
              {ADDRESS_LINE}<br/>
              {COURSE_LINE}<br/>
              Last updated: <b>{now_str}</b>
            </div>
          </div>
        </div>
      </div>
      <div class="badges">
        <div class="badge">🚗 <b>4WD Rover</b></div>
        <div class="badge">📡 <b>IoT Telemetry</b></div>
        <div class="badge">🧠 <b>AI Forecast</b> (~30s)</div>
        <div class="badge">🧪 <b>Soil + Air</b> sensing</div>
      </div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True
)
st.write("")


# =======================
# TOP NAV (best for mobile)
# =======================
NAV_ITEMS = ["Home", "Live", "Deep Analysis", "Architecture", "Evidence", "Report"]
page = st.radio("Navigation", NAV_ITEMS, horizontal=True, label_visibility="collapsed")
st.write("")


# =======================
# LIVE READ ONCE
# =======================
def read_live_once():
    try:
        vals = read_sensors()
        return True, None, vals
    except Exception as e:
        return False, str(e), {n: np.nan for n in FEATURES}

status_ok, err_msg, current = read_live_once()

if status_ok:
    st.session_state.pred_buffer.append([current[n] for n in FEATURES])
    st.session_state.history.append({"ts": datetime.now(), **{n: float(current[n]) for n in FEATURES}})

pill = '<span class="pill ok">● ONLINE</span>' if status_ok else '<span class="pill bad">● OFFLINE</span>'
st.markdown(f"{pill} <span style='color:rgba(0,0,0,0.62);font-weight:900;'>Blynk connection status</span>", unsafe_allow_html=True)
if err_msg:
    st.error(f"Blynk read error: {err_msg}")

pred_map = None
if len(st.session_state.pred_buffer) == WINDOW_STEPS and status_ok:
    X = np.array(st.session_state.pred_buffer).reshape(1, -1)
    pred = model.predict(X)[0]
    pred_map = {FEATURES[i]: clamp(FEATURES[i], float(pred[i])) for i in range(len(FEATURES))}


# =======================
# PAGES
# =======================
def page_home():
    c1, c2 = st.columns([1.25, 1])
    with c1:
        st.markdown(
            """
<div class="section">
  <div class="tag level-ok">● Overview</div>
  <h3 style="margin-top:10px;">Project Snapshot</h3>
  <p>
    Our autonomous mobile rover addresses the spatial “blind spot” problem of stationary environmental sensors by
    collecting <b>soil + atmospheric</b> measurements across a physical area, streaming telemetry to a cloud dashboard,
    and enabling AI-ready datasets for proactive monitoring.
  </p>
  <ul>
    <li><b>Mobility</b>: 4WD chassis + obstacle avoidance</li>
    <li><b>Sensing</b>: Temperature, Humidity, Soil Moisture, pH</li>
    <li><b>IoT</b>: Real-time cloud telemetry</li>
    <li><b>AI</b>: ~30s forecast for short-term planning</li>
  </ul>
</div>
""",
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            """
<div class="section">
  <div class="tag level-warm">● Quick Guide</div>
  <h3 style="margin-top:10px;">What to explore</h3>
  <ul>
    <li><b>Live</b>: real-time values + 30s prediction</li>
    <li><b>Deep Analysis</b>: 30s narrative + insights</li>
    <li><b>Architecture</b>: pipeline + diagrams</li>
    <li><b>Report</b>: embedded PDF for full documentation</li>
  </ul>
</div>
""",
            unsafe_allow_html=True
        )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.subheader("Team")
    st.dataframe(pd.DataFrame(AUTHORS, columns=["Name", "Student ID", "Role"]), use_container_width=True)

    photos = list_images(EVIDENCE_DIR)
    if photos:
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.subheader("Project Highlights")
        cols = st.columns(3)
        for i, p in enumerate(photos[:3]):
            b64, mime = file_to_b64(p)
            if not b64:
                continue
            with cols[i % 3]:
                st.markdown(
                    f"""
<div class="media-card">
  <img src="data:{mime};base64,{b64}" />
  <div class="media-cap">{p.stem.replace("_"," ").title()}</div>
</div>
""",
                    unsafe_allow_html=True
                )

def page_live():
    st.subheader("Live Dashboard: Current + AI Forecast (~30s)")

    prev = st.session_state.history[-2] if len(st.session_state.history) >= 2 else None

    st.markdown("#### Live (Current)")
    st.markdown("<div class='grid4'>", unsafe_allow_html=True)

    for name in FEATURES:
        v = current[name]
        if not np.isfinite(v):
            st.markdown(card_html(ICONS[name], LABELS[name], "—", "cold", "No data"), unsafe_allow_html=True)
            continue

        if name == "temp":
            lvl, tag = level_temp(float(v), ideal_min_t, ideal_max_t, margin_t)
            hint = f"Target: {ideal_min_t:.1f}–{ideal_max_t:.1f}°C"
        elif name == "humidity":
            lvl, tag = level_humidity(float(v), low_h, high_h, margin_h)
            hint = f"Comfort: {low_h:.0f}–{high_h:.0f}%"
        elif name == "soil":
            lvl, tag = level_soil(float(v), low_s, high_s, margin_s)
            hint = f"Good: {low_s:.0f}–{high_s:.0f}%"
        else:
            lvl, tag = level_ph(float(v), low_p, high_p, margin_p)
            hint = f"Neutral-ish: {low_p:.1f}–{high_p:.1f}"

        dtext = None
        if prev is not None and np.isfinite(prev.get(name, np.nan)):
            d = float(v) - float(prev[name])
            dtext = colored_delta_html(name, d, "vs last sample:")

        st.markdown(card_html(ICONS[name], LABELS[name], fmt(name, float(v)), lvl, tag, dtext, hint), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<hr/>", unsafe_allow_html=True)

    st.markdown("#### AI Prediction (~30s later)")
    if pred_map is None:
        st.info(f"Collecting window… ({len(st.session_state.pred_buffer)}/{WINDOW_STEPS})")
    else:
        st.markdown("<div class='grid4'>", unsafe_allow_html=True)
        for name in FEATURES:
            pv = pred_map[name]
            cv = float(current[name]) if np.isfinite(current[name]) else None

            if name == "temp":
                lvl, tag = level_temp(pv, ideal_min_t, ideal_max_t, margin_t)
            elif name == "humidity":
                lvl, tag = level_humidity(pv, low_h, high_h, margin_h)
            elif name == "soil":
                lvl, tag = level_soil(pv, low_s, high_s, margin_s)
            else:
                lvl, tag = level_ph(pv, low_p, high_p, margin_p)

            dtext = colored_delta_html(name, (pv - cv), "(pred - now):") if cv is not None else None
            st.markdown(card_html(ICONS[name], f"{LABELS[name]} →", fmt(name, pv), lvl, tag, dtext, "Forecast horizon: ~30s"),
                        unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.subheader("Trends")
    if len(st.session_state.history) >= 5:
        df = pd.DataFrame(list(st.session_state.history)).set_index("ts")
        a, b = st.columns(2)
        with a:
            st.caption("Temperature & Humidity")
            st.line_chart(df[["temp", "humidity"]].tail(120))
        with b:
            st.caption("Soil Moisture & pH")
            st.line_chart(df[["soil", "ph"]].tail(120))
    else:
        st.info("Not enough data points yet. Keep the dashboard running.")

def page_deep():
    st.subheader("Deep Analysis (Detailed Interpretation)")

    if len(st.session_state.history) < 5:
        st.info("Need at least a few samples to compute trend/analysis. Keep it running for ~1 minute.")
        return

    df = pd.DataFrame(list(st.session_state.history)).set_index("ts")
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last
    window = df.tail(min(20, len(df)))

    end_ts = df.index[-1]
    last30 = df[df.index >= (end_ts - pd.Timedelta(seconds=30))]

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.subheader("Narrative Summary (Last 30 seconds)")

    if len(last30) < 2:
        st.info("Not enough samples in the last 30 seconds yet. Keep the app running a bit longer.")
    else:
        targets = {
            "temp": (ideal_min_t, ideal_max_t, margin_t, "°C"),
            "humidity": (low_h, high_h, margin_h, "%"),
            "soil": (low_s, high_s, margin_s, "%"),
            "ph": (low_p, high_p, margin_p, ""),
        }
        severities = {"ok": 0, "warm": 1, "cold": 1, "hot": 2}
        worst_lvl = "ok"
        worst_name = None

        for name in FEATURES:
            low, high, m, unit = targets[name]
            curv = float(last[name])
            pv = float(pred_map[name]) if pred_map and name in pred_map else None

            lvl, html = describe_last_window(
                name=name,
                w=last30[name],
                current_val=curv,
                pred_val=pv,
                target_low=low,
                target_high=high,
                margin=m,
                unit=unit
            )

            if severities[lvl] > severities[worst_lvl]:
                worst_lvl = lvl
                worst_name = name

            st.markdown(
                f"""
<div class="section">
  <div class="tag level-{lvl}">● {LABELS[name]}</div>
  {html}
</div>
""",
                unsafe_allow_html=True
            )

        overall = (
            "Overall, conditions were stable over the last 30 seconds with no major deviations."
            if worst_name is None else
            f"Overall, the most notable issue is <b>{LABELS[worst_name]}</b>, currently flagged as <b>{worst_lvl.upper()}</b>."
        )

        st.markdown(
            f"""
<div class="section">
  <div class="tag level-{worst_lvl}">● Overall Conclusion</div>
  <h3 style="margin-top:10px;">Conclusion</h3>
  <p>{overall}</p>
</div>
""",
            unsafe_allow_html=True
        )

    def slope(series):
        if len(series) < 3:
            return 0.0
        x = np.arange(len(series))
        y = series.values.astype(float)
        m = np.polyfit(x, y, 1)[0]
        return float(m)

    insights = []

    # Temperature
    t = float(last["temp"])
    t_delta = t - float(prev["temp"])
    t_slope = slope(window["temp"])
    t_lvl, t_tag = level_temp(t, ideal_min_t, ideal_max_t, margin_t)
    t_summary = f"Current temperature is {t:.2f}°C ({t_tag}). Δ={t_delta:+.2f}°C vs previous. Trend slope≈{t_slope:+.3f} per sample."
    t_actions = ["Maintain current settings", "Monitor trend for sudden changes"]
    if t_lvl in ["hot", "warm"]:
        t_actions = ["Increase ventilation/fan", "Add shading if available", "Check soil moisture; mist lightly if dry"]
    elif t_lvl == "cold":
        t_actions = ["Reduce drafts", "Water during warmer periods", "Check if crop requires heating"]
    if pred_map:
        t_actions += [f"30s forecast: {float(pred_map['temp']):.2f}°C → prepare adjustment if crossing thresholds."]
    insights.append(("Temperature", t_lvl, t_summary, t_actions))

    # Humidity
    h = float(last["humidity"])
    h_delta = h - float(prev["humidity"])
    h_slope = slope(window["humidity"])
    h_lvl, h_tag = level_humidity(h, low_h, high_h, margin_h)
    h_summary = f"Current humidity is {h:.2f}% ({h_tag}). Δ={h_delta:+.2f}% vs previous. Trend slope≈{h_slope:+.3f} per sample."
    h_actions = ["Keep stable environment", "Watch for rapid drops (door open / wind)"]
    if h_lvl in ["hot", "warm"]:
        h_actions = ["Increase airflow to reduce condensation risk", "Inspect for wet surfaces / mold risk", "Avoid over-watering"]
    elif h_lvl == "cold":
        h_actions = ["Consider misting (crop-dependent)", "Reduce excessive ventilation if drying too fast"]
    if pred_map:
        h_actions += [f"30s forecast: {float(pred_map['humidity']):.2f}%."]
    insights.append(("Humidity", h_lvl, h_summary, h_actions))

    # Soil
    s = float(last["soil"])
    s_delta = s - float(prev["soil"])
    s_slope = slope(window["soil"])
    s_lvl, s_tag = level_soil(s, low_s, high_s, margin_s)
    s_summary = f"Soil moisture is {s:.2f}% ({s_tag}). Δ={s_delta:+.2f}% vs previous. Trend slope≈{s_slope:+.3f} per sample."
    s_actions = ["Reduce watering", "Improve drainage/airflow", "Watch pH drift due to excess water"]
    if s_lvl in ["hot", "warm"]:
        s_actions = ["Water/irrigate gradually", "Re-check after 1–2 minutes", "Avoid sudden flooding to prevent root shock"]
    if pred_map:
        s_actions += [f"30s forecast: {float(pred_map['soil']):.2f}%."]
    insights.append(("Soil Moisture", s_lvl, s_summary, s_actions))

    # pH
    p = float(last["ph"])
    p_delta = p - float(prev["ph"])
    p_slope = slope(window["ph"])
    p_lvl, p_tag = level_ph(p, low_p, high_p, margin_p)
    p_summary = f"pH is {p:.2f} ({p_tag}). Δ={p_delta:+.2f} vs previous. Trend slope≈{p_slope:+.3f} per sample."
    p_actions = ["Within target band; keep monitoring", "Ensure probe is stable (warm-up ~30s)"]
    if p_lvl == "warm":
        p_actions = ["If persistent: consider liming (context-dependent)", "Recalibrate probe / ensure warm-up", "Compare with reference buffer if available"]
    elif p_lvl == "cold":
        p_actions = ["If persistent: consider acidifying amendments (context-dependent)", "Re-check after probe stabilization", "Avoid over-correction"]
    if pred_map:
        p_actions += [f"30s forecast: {float(pred_map['ph']):.2f}."]
    insights.append(("pH", p_lvl, p_summary, p_actions))

    st.markdown("<hr/>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Temperature & Humidity")
        st.line_chart(df[["temp", "humidity"]].tail(120))
    with c2:
        st.caption("Soil Moisture & pH")
        st.line_chart(df[["soil", "ph"]].tail(120))

    st.markdown("<hr/>", unsafe_allow_html=True)
    for name, lvl, summary, actions in insights:
        st.markdown(insight_block(name, lvl, summary, actions), unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.subheader("Data Log (for grading)")
    st.dataframe(df.tail(30).reset_index(), use_container_width=True)

def page_architecture():
    st.subheader("Architecture")
    st.markdown(
        """
<div class="section">
  <div class="tag level-ok">● System</div>
  <h3 style="margin-top:10px;">Sense–Think–Act Pipeline</h3>
  <ul>
    <li><b>Sense</b>: Temperature, Humidity, Soil Moisture, pH + navigation sensors</li>
    <li><b>Think</b>: control logic + filtering + sampling schedule</li>
    <li><b>Act</b>: 4WD motor control (PWM), obstacle avoidance, line tracking</li>
  </ul>
</div>
""",
        unsafe_allow_html=True
    )
    st.write("")
    render_diagram(DIAGRAM_DIR / "flow.png", "Operational Flow / Control Flow")
    st.write("")
    render_diagram(DIAGRAM_DIR / "architecture.png", "System Architecture")
    st.write("")
    render_diagram(DIAGRAM_DIR / "wiring.png", "Wiring / Hardware Integration")

def page_evidence():
    st.subheader("Evidence (Project Photos)")
    photos = list_images(EVIDENCE_DIR)
    if not photos:
        st.info("No images found. Add your photos to assets/media/ (jpg/png/webp).")
        return
    cols = st.columns(3)
    for i, p in enumerate(photos[:15]):
        b64, mime = file_to_b64(p)
        if not b64:
            continue
        with cols[i % 3]:
            st.markdown(
                f"""
<div class="media-card">
  <img src="data:{mime};base64,{b64}" />
  <div class="media-cap">{p.stem.replace("_"," ").title()}</div>
</div>
""",
                unsafe_allow_html=True
            )

def page_report():
    st.subheader("Report (PDF)")
    if REPORT_PATH.exists():
        st.download_button("Download Report (PDF)", REPORT_PATH.read_bytes(), file_name=REPORT_PATH.name, mime="application/pdf")
        st.write("")
        pdf_embed(REPORT_PATH, height=920)
    else:
        st.warning("Report PDF not found. Put it at assets/report.pdf")

if page == "Home":
    page_home()
elif page == "Live":
    page_live()
elif page == "Deep Analysis":
    page_deep()
elif page == "Architecture":
    page_architecture()
elif page == "Evidence":
    page_evidence()
elif page == "Report":
    page_report()

st.markdown(
    f"""
<div class="footer">
  <b>{ORG_LINE}</b> • {ADDRESS_LINE} • {TEAM_LINE}<br/>
  Project: {PROJECT_TITLE}
</div>
""",
    unsafe_allow_html=True
)

# Auto refresh only where it matters
if auto_refresh and page in ["Live", "Deep Analysis"]:
    time.sleep(refresh_sec)
    st.rerun()

