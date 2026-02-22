import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.sparse import hstack, csr_matrix
import re

# ═══════════════════════════════════════════════════════════
#  PAGE CONFIG & CUSTOM CSS
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PharmaRisk AI — Pharmacogenomic Risk Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700&family=Source+Serif+4:wght@600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg-page: #f4f6f9;
    --bg-white: #ffffff;
    --border-light: #e0e4eb;
    --border-subtle: #eaedf2;
    --text-heading: #1a2332;
    --text-body: #3d4f65;
    --text-muted: #8293a7;
    --primary: #2563eb;
    --primary-hover: #1d4ed8;
    --teal: #0d9488;
    --coral: #e8573d;
    --amber: #d97706;
    --violet: #7c3aed;
    --radius: 12px;
    --shadow-xs: 0 1px 2px rgba(0,0,0,0.04);
    --shadow-sm: 0 1px 4px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.03);
    --shadow-md: 0 4px 20px rgba(0,0,0,0.06), 0 1px 4px rgba(0,0,0,0.04);
    --shadow-card-hover: 0 8px 30px rgba(0,0,0,0.08);
}

* { font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important; }

.stApp {
    background: var(--bg-page) !important;
    color: var(--text-body) !important;
}

header[data-testid="stHeader"] {
    background: rgba(244,246,249,0.88) !important;
    backdrop-filter: blur(16px) !important;
    border-bottom: 1px solid var(--border-subtle) !important;
}

/* ══════════════════════════════════════════
   SIDEBAR — deep rich background
   ══════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #0f1b33 0%, #162544 45%, #1a2f52 100%) !important;
    border-right: none !important;
    box-shadow: 4px 0 24px rgba(0,0,0,0.1) !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] * {
    color: #a8b8cc !important;
}
section[data-testid="stSidebar"] b,
section[data-testid="stSidebar"] strong {
    color: #dce4ee !important;
}

/* ── Cards ── */
.pr-card {
    background: var(--bg-white);
    border: 1px solid var(--border-light);
    border-radius: var(--radius);
    padding: 22px 24px;
    box-shadow: var(--shadow-sm);
    margin-bottom: 14px;
    transition: box-shadow 0.25s ease;
}
.pr-card:hover {
    box-shadow: var(--shadow-md);
}

/* ── Metric Cards ── */
.metric-card {
    background: var(--bg-white);
    border: 1px solid var(--border-light);
    border-radius: var(--radius);
    padding: 20px 16px 18px;
    text-align: center;
    box-shadow: var(--shadow-sm);
    position: relative;
    overflow: hidden;
    transition: box-shadow 0.25s ease, transform 0.2s ease;
}
.metric-card:hover {
    box-shadow: var(--shadow-card-hover);
    transform: translateY(-1px);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: var(--radius) var(--radius) 0 0;
}
.metric-card.mc-blue::before  { background: var(--primary); }
.metric-card.mc-teal::before  { background: var(--teal); }
.metric-card.mc-amber::before { background: var(--amber); }
.metric-card.mc-coral::before { background: var(--coral); }
.metric-card.mc-violet::before { background: var(--violet); }

.metric-value {
    font-family: 'Source Serif 4', Georgia, serif !important;
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-heading);
    line-height: 1.15;
}
.metric-label {
    font-size: 0.74rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-top: 6px;
    font-weight: 500;
}

/* ── Risk Badges ── */
.risk-badge {
    display: inline-block;
    padding: 9px 24px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.3px;
}
.risk-critical { background: #fde8e8; color: #b91c1c; border: 1px solid #f5c6c6; }
.risk-high     { background: #fef3e2; color: #b45309; border: 1px solid #fcd69f; }
.risk-moderate { background: #fefce8; color: #92400e; border: 1px solid #fde68a; }
.risk-low      { background: #ecfdf5; color: #065f46; border: 1px solid #a7f3d0; }
.risk-minimal  { background: #eff4ff; color: #1e40af; border: 1px solid #bfdbfe; }

/* ── Page Title ── */
.page-title {
    font-family: 'Source Serif 4', Georgia, serif !important;
    font-size: 2.1rem;
    font-weight: 700;
    color: var(--text-heading);
    margin-bottom: 2px;
    line-height: 1.15;
}
.page-subtitle {
    font-size: 0.95rem;
    color: var(--text-muted);
    font-weight: 400;
    margin-top: 0;
    line-height: 1.5;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-white) !important;
    border-radius: var(--radius);
    padding: 5px;
    gap: 3px;
    border: 1px solid var(--border-light);
    box-shadow: var(--shadow-xs);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    padding: 9px 16px !important;
    font-size: 0.88rem !important;
    transition: all 0.15s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: #f0f3f8 !important;
    color: var(--text-body) !important;
}
.stTabs [aria-selected="true"] {
    background: var(--primary) !important;
    color: #ffffff !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.2) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 22px !important;
}

/* ── Input Fields ── */
.stTextArea textarea, .stTextInput input {
    background: var(--bg-white) !important;
    border: 1.5px solid var(--border-light) !important;
    color: var(--text-body) !important;
    border-radius: 10px !important;
    padding: 12px 14px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
}
.stTextArea label, .stTextInput label, .stSelectbox label {
    font-weight: 500 !important;
    color: var(--text-body) !important;
    font-size: 0.88rem !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--primary) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 11px 28px !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 6px rgba(37,99,235,0.15) !important;
}
.stButton > button:hover {
    background: var(--primary-hover) !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.25) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

.stDownloadButton > button {
    background: var(--teal) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 6px rgba(13,148,136,0.15) !important;
}

/* ── Divider ── */
.section-line {
    height: 1px;
    background: var(--border-subtle);
    margin: 24px 0;
    border: none;
}

/* ── Section Labels ── */
.section-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.3px;
    font-weight: 600;
    margin-bottom: 10px;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: var(--text-muted);
    font-size: 0.78rem;
    padding: 32px 0 12px;
}

/* ══════════════════════════════════════════
   SIDEBAR COMPONENTS
   ══════════════════════════════════════════ */

/* Brand area */
.sb-brand {
    text-align: center;
    padding: 28px 16px 20px;
}
.sb-brand-icon {
    width: 52px; height: 52px;
    margin: 0 auto 10px;
    background: linear-gradient(135deg, rgba(37,99,235,0.2), rgba(124,58,237,0.2));
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.7rem;
    border: 1px solid rgba(255,255,255,0.06);
}
.sb-brand-name {
    font-family: 'Source Serif 4', Georgia, serif !important;
    font-size: 1.3rem;
    font-weight: 700;
    color: #ffffff !important;
    letter-spacing: -0.3px;
}
.sb-brand-tag {
    font-size: 0.72rem;
    color: #6b7f99 !important;
    margin-top: 3px;
    letter-spacing: 0.5px;
}

/* Separator */
.sb-sep {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
    margin: 6px 16px;
    border: none;
}

/* Status indicator */
.sb-status {
    margin: 14px 16px;
    padding: 10px 14px;
    background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.15);
    border-radius: 10px;
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.82rem;
    font-weight: 500;
    color: #6ee7b7 !important;
}
.sb-status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #34d399;
    flex-shrink: 0;
    animation: sb-pulse 2.5s ease-in-out infinite;
}
@keyframes sb-pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(52,211,153,0.4); }
    50%      { opacity: 0.7; box-shadow: 0 0 0 4px rgba(52,211,153,0); }
}

/* Stat cards */
.sb-stats {
    margin: 14px 16px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
}
.sb-stat-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 12px 10px;
    text-align: center;
}
.sb-stat-num {
    font-family: 'Source Serif 4', Georgia, serif !important;
    font-size: 1.2rem;
    font-weight: 700;
    color: #e8ecf1 !important;
    line-height: 1.2;
}
.sb-stat-label {
    font-size: 0.68rem;
    color: #6b7f99 !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 3px;
}

/* Tier legend */
.sb-tiers {
    margin: 14px 16px;
}
.sb-tier-title {
    font-size: 0.68rem;
    color: #6b7f99 !important;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 600;
    margin-bottom: 10px;
}
.sb-tier {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 5px 0;
}
.sb-tier-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 5px;
    font-size: 0.72rem;
    font-weight: 600;
    min-width: 68px;
    text-align: center;
}
.sb-tier-desc {
    font-size: 0.78rem;
    color: #7a8da3 !important;
}

/* Footer */
.sb-footer {
    text-align: center;
    padding: 20px 16px 12px;
    font-size: 0.7rem;
    color: #4a5f78 !important;
    line-height: 1.6;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  LOAD ARTIFACTS
# ═══════════════════════════════════════════════════════════
@st.cache_resource
def load_artifacts():
    with open("model_artifacts.pkl", "rb") as f:
        arts = pickle.load(f)
    with open("dashboard_data.pkl", "rb") as f:
        dash = pickle.load(f)
    return arts, dash

try:
    artifacts, dashboard = load_artifacts()
    model = artifacts["model"]
    vectorizer = artifacts["vectorizer"]
    drug_encoder = artifacts["drug_encoder"]
    condition_encoder = artifacts["condition_encoder"]
    top_drugs = artifacts["top_drugs"]
    top_conditions = artifacts["top_conditions"]
    RISK_LABELS = artifacts["risk_labels"]
    NEG_WORDS = artifacts["negative_words"]
    POS_WORDS = artifacts["positive_words"]
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.error(f"Model artifacts not found. Run `python main.py` first to train the model.\n\nError: {e}")

# ═══════════════════════════════════════════════════════════
#  CONSTANTS & HELPERS
# ═══════════════════════════════════════════════════════════
RISK_COLORS = {
    0: "#2563eb",  # Minimal - blue
    1: "#0d9488",  # Low - teal
    2: "#d97706",  # Moderate - amber
    3: "#ea580c",  # High - orange
    4: "#dc2626",  # Critical - red
}

RISK_BG_COLORS = {
    0: "#eff4ff",
    1: "#ecfdf5",
    2: "#fefce8",
    3: "#fef3e2",
    4: "#fde8e8",
}

RISK_CSS_CLASSES = {
    0: "risk-minimal",
    1: "risk-low",
    2: "risk-moderate",
    3: "risk-high",
    4: "risk-critical",
}

RISK_ICONS = {0: "✓", 1: "✓", 2: "●", 3: "▲", 4: "✕"}

def clean_review(text):
    text = str(text)
    text = re.sub(r"&#\d+;", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

def count_keywords(text, keywords):
    words = text.lower().split()
    return sum(1 for w in words if w in keywords)

def predict_single(review_text, drug_name="", condition=""):
    cleaned = clean_review(review_text)
    text_features = vectorizer.transform([cleaned])

    review_length = len(cleaned)
    word_count = len(cleaned.split())
    neg_count = count_keywords(cleaned, NEG_WORDS)
    pos_count = count_keywords(cleaned, POS_WORDS)
    useful_log = 0.0
    numeric = np.array([[review_length, word_count, neg_count, pos_count, useful_log]])

    drug_enc = drug_encoder.transform([drug_name if drug_name in top_drugs else "_OTHER_"])[0]
    cond_enc = condition_encoder.transform([condition if condition in top_conditions else "_OTHER_"])[0]
    categorical = np.array([[drug_enc, cond_enc]])

    X = hstack([text_features, csr_matrix(numeric), csr_matrix(categorical)])

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    feature_names = vectorizer.get_feature_names_out()
    tfidf_array = text_features.toarray()[0]
    top_indices = np.argsort(tfidf_array)[-10:][::-1]
    top_keywords = [(feature_names[i], tfidf_array[i]) for i in top_indices if tfidf_array[i] > 0]

    return {
        "risk_tier": prediction,
        "risk_label": RISK_LABELS[prediction],
        "probabilities": {RISK_LABELS[i]: round(p * 100, 2) for i, p in enumerate(probabilities)},
        "confidence": round(max(probabilities) * 100, 2),
        "top_keywords": top_keywords,
        "review_stats": {
            "word_count": word_count,
            "neg_keywords": neg_count,
            "pos_keywords": pos_count,
        }
    }

# ── Chart helpers (light theme) ──

def create_gauge_chart(risk_tier, confidence):
    color = RISK_COLORS.get(risk_tier, "#2563eb")
    fig = go.Figure(go.Pie(
        values=[confidence, 100 - confidence],
        hole=0.78,
        marker=dict(colors=[color, "#f0f2f5"]),
        textinfo="none", hoverinfo="skip",
        rotation=270, direction="clockwise",
    ))
    fig.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=8, b=8, l=8, r=8), height=200,
        annotations=[
            dict(text=f"<b>{confidence}%</b>", x=0.5, y=0.55,
                 font=dict(size=26, color=color, family="DM Sans"), showarrow=False),
            dict(text="Confidence", x=0.5, y=0.38,
                 font=dict(size=11, color="#8293a7", family="DM Sans"), showarrow=False),
        ],
    )
    return fig

def create_probability_bar(probabilities):
    labels = list(probabilities.keys())
    values = list(probabilities.values())

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(
            color=[RISK_COLORS[i] for i in range(len(labels))],
            cornerradius=5,
        ),
        text=[f"{v}%" for v in values], textposition="outside",
        textfont=dict(color="#3d4f65", family="DM Sans", size=12),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=8, b=8, l=8, r=35), height=200,
        xaxis=dict(showgrid=False, showticklabels=False,
                   range=[0, max(values) * 1.3 if values else 100], zeroline=False),
        yaxis=dict(showgrid=False, tickfont=dict(color="#5a6b80", family="DM Sans", size=11),
                   autorange="reversed"),
        bargap=0.35,
    )
    return fig

def create_keyword_chart(keywords):
    if not keywords:
        return None
    words, scores = zip(*keywords[:8])
    fig = go.Figure(go.Bar(
        x=list(scores), y=list(words), orientation="h",
        marker=dict(
            color=list(scores),
            colorscale=[[0, "#93c5fd"], [0.5, "#2563eb"], [1, "#7c3aed"]],
            cornerradius=4,
        ),
        text=[f"{s:.3f}" for s in scores], textposition="outside",
        textfont=dict(color="#5a6b80", family="IBM Plex Mono", size=11),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=8, b=8, l=8, r=45), height=260,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, tickfont=dict(color="#3d4f65", family="IBM Plex Mono", size=12),
                   autorange="reversed"),
        bargap=0.3,
    )
    return fig


# ═══════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    # ── Brand ──
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-brand-icon">🧬</div>
        <div class="sb-brand-name">PharmaRisk AI</div>
        <div class="sb-brand-tag">Pharmacogenomic Risk Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-sep"></div>', unsafe_allow_html=True)

    if MODEL_LOADED:
        # ── Status Indicator ──
        st.markdown("""
        <div class="sb-status">
            <span class="sb-status-dot"></span>
            <span>System Online</span>
        </div>
        """, unsafe_allow_html=True)

        # ── Stat Grid ──
        st.markdown(f"""
        <div class="sb-stats">
            <div class="sb-stat-card">
                <div class="sb-stat-num">{dashboard['total_reviews']:,}</div>
                <div class="sb-stat-label">Reviews</div>
            </div>
            <div class="sb-stat-card">
                <div class="sb-stat-num">{dashboard['total_drugs']:,}</div>
                <div class="sb-stat-label">Drugs</div>
            </div>
            <div class="sb-stat-card">
                <div class="sb-stat-num">{dashboard['total_conditions']:,}</div>
                <div class="sb-stat-label">Conditions</div>
            </div>
            <div class="sb-stat-card">
                <div class="sb-stat-num">5</div>
                <div class="sb-stat-label">Risk Tiers</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sb-sep"></div>', unsafe_allow_html=True)

    # ── Risk Tier Legend ──
    st.markdown("""
    <div class="sb-tiers">
        <div class="sb-tier-title">Risk Classification</div>
        <div class="sb-tier">
            <span class="sb-tier-badge" style="background:rgba(220,38,38,0.15);color:#fca5a5;">Critical</span>
            <span class="sb-tier-desc">Severe reactions</span>
        </div>
        <div class="sb-tier">
            <span class="sb-tier-badge" style="background:rgba(234,88,12,0.15);color:#fdba74;">High</span>
            <span class="sb-tier-desc">Significant effects</span>
        </div>
        <div class="sb-tier">
            <span class="sb-tier-badge" style="background:rgba(217,119,6,0.15);color:#fcd34d;">Moderate</span>
            <span class="sb-tier-desc">Notable concerns</span>
        </div>
        <div class="sb-tier">
            <span class="sb-tier-badge" style="background:rgba(13,148,136,0.15);color:#5eead4;">Low</span>
            <span class="sb-tier-desc">Minor issues</span>
        </div>
        <div class="sb-tier">
            <span class="sb-tier-badge" style="background:rgba(37,99,235,0.15);color:#93c5fd;">Minimal</span>
            <span class="sb-tier-desc">Well tolerated</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-sep"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-footer">Built with Streamlit & Scikit-Learn<br>© 2026 PharmaRisk AI</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  MAIN HEADER
# ═══════════════════════════════════════════════════════════
st.markdown("""
<div style="padding: 8px 0 0;">
    <div class="page-title">PharmaRisk AI</div>
    <div class="page-subtitle">Pharmacogenomic risk intelligence — ML-driven analysis of drug reviews across 5 risk tiers</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════
if MODEL_LOADED:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Single Analysis",
        "Batch Analysis",
        "Analytics",
        "Drug Explorer",
        "About"
    ])

    # ─────────────────────────────────────────────────────
    #  TAB 1: SINGLE ANALYSIS
    # ─────────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-label">Analyze a Drug Review</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#5a6b80;font-size:0.92rem;margin-top:-4px;">Paste a drug review below. Providing the drug name and condition improves accuracy.</p>', unsafe_allow_html=True)

        col_input_left, col_input_right = st.columns([2, 1])

        with col_input_left:
            review_input = st.text_area(
                "Drug Review Text",
                placeholder="e.g. 'This medication caused severe headaches and nausea for the first week, but symptoms improved after adjusting the dose...'",
                height=150,
                key="single_review"
            )

        with col_input_right:
            drug_input = st.selectbox(
                "Drug Name (optional)",
                options=["— Not specified —"] + sorted(top_drugs[:50]),
                key="single_drug"
            )
            condition_input = st.selectbox(
                "Condition (optional)",
                options=["— Not specified —"] + sorted(top_conditions[:50]),
                key="single_condition"
            )

        analyze_btn = st.button("Analyze Risk", use_container_width=True, key="analyze_single")

        if analyze_btn and review_input.strip():
            drug_val = drug_input if drug_input != "— Not specified —" else ""
            cond_val = condition_input if condition_input != "— Not specified —" else ""

            with st.spinner("Analyzing review..."):
                result = predict_single(review_input, drug_val, cond_val)

            st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)

            # ── Risk Result Header ──
            tier = result["risk_tier"]
            css_class = RISK_CSS_CLASSES[tier]
            icon = RISK_ICONS[tier]
            color = RISK_COLORS[tier]

            st.markdown(f"""
            <div class="pr-card" style="text-align:center; border-top: 3px solid {color};">
                <div class="section-label">Risk Assessment Result</div>
                <div class="risk-badge {css_class}" style="font-size:1.15rem; padding:10px 28px; margin-top:6px;">
                    {icon}&ensp;{result['risk_label']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Gauge + Probabilities + Stats ──
            col_g, col_p, col_s = st.columns([1, 1.5, 1])

            with col_g:
                st.markdown('<div class="pr-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-label" style="text-align:center;">Risk Confidence</div>', unsafe_allow_html=True)
                fig_gauge = create_gauge_chart(tier, result["confidence"])
                st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})
                st.markdown('</div>', unsafe_allow_html=True)

            with col_p:
                st.markdown('<div class="pr-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-label">Probability Distribution</div>', unsafe_allow_html=True)
                fig_prob = create_probability_bar(result["probabilities"])
                st.plotly_chart(fig_prob, use_container_width=True, config={"displayModeBar": False})
                st.markdown('</div>', unsafe_allow_html=True)

            with col_s:
                wc = result['review_stats']['word_count']
                nk = result['review_stats']['neg_keywords']
                pk = result['review_stats']['pos_keywords']
                st.markdown(f"""
                <div class="pr-card">
                    <div class="section-label">Review Stats</div>
                    <div style="margin-bottom: 14px;">
                        <div style="color:#8293a7; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.8px;">Word Count</div>
                        <div style="font-family:'Source Serif 4',Georgia,serif !important;
                             font-size:1.6rem; font-weight:700; color:#2563eb;">{wc}</div>
                    </div>
                    <div style="margin-bottom: 14px;">
                        <div style="color:#8293a7; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.8px;">Negative Keywords</div>
                        <div style="font-family:'Source Serif 4',Georgia,serif !important;
                             font-size:1.6rem; font-weight:700; color:#dc2626;">{nk}</div>
                    </div>
                    <div>
                        <div style="color:#8293a7; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.8px;">Positive Keywords</div>
                        <div style="font-family:'Source Serif 4',Georgia,serif !important;
                             font-size:1.6rem; font-weight:700; color:#0d9488;">{pk}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Top Keywords ──
            if result["top_keywords"]:
                st.markdown('<div class="section-label" style="margin-top:8px;">Top Contributing Keywords</div>', unsafe_allow_html=True)
                fig_kw = create_keyword_chart(result["top_keywords"])
                if fig_kw:
                    st.plotly_chart(fig_kw, use_container_width=True, config={"displayModeBar": False})

        elif analyze_btn:
            st.warning("Please enter a drug review to analyze.")

    # ─────────────────────────────────────────────────────
    #  TAB 2: BATCH ANALYSIS
    # ─────────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-label">Batch Review Analysis</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#5a6b80;font-size:0.92rem;margin-top:-4px;">Upload a CSV file with a <code>review</code> column. You can also include <code>drugName</code> and <code>condition</code> columns for better accuracy.</p>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="batch_upload")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.markdown(f'<div class="pr-card">Loaded <b>{len(df):,}</b> rows &nbsp;·&nbsp; Columns: <code>{", ".join(df.columns)}</code></div>', unsafe_allow_html=True)

                if "review" not in df.columns:
                    st.error("CSV must have a `review` column.")
                else:
                    if st.button("Analyze All Reviews", use_container_width=True, key="batch_analyze"):
                        progress = st.progress(0)
                        results_list = []

                        for idx, row in df.iterrows():
                            drug = row.get("drugName", "")
                            cond = row.get("condition", "")
                            res = predict_single(str(row["review"]), str(drug), str(cond))
                            results_list.append({
                                "Review": str(row["review"])[:100] + "...",
                                "Drug": drug,
                                "Condition": cond,
                                "Risk Level": res["risk_label"],
                                "Confidence (%)": res["confidence"],
                            })
                            progress.progress((idx + 1) / len(df))

                        results_df = pd.DataFrame(results_list)
                        progress.empty()

                        st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)

                        # Summary
                        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                        risk_counts = results_df["Risk Level"].value_counts()

                        with mcol1:
                            st.markdown(f'<div class="metric-card mc-blue"><div class="metric-value">{len(results_df)}</div><div class="metric-label">Total Analyzed</div></div>', unsafe_allow_html=True)
                        with mcol2:
                            crit_count = risk_counts.get("Critical Risk", 0) + risk_counts.get("High Risk", 0)
                            st.markdown(f'<div class="metric-card mc-coral"><div class="metric-value">{crit_count}</div><div class="metric-label">High / Critical</div></div>', unsafe_allow_html=True)
                        with mcol3:
                            avg_conf = results_df["Confidence (%)"].mean()
                            st.markdown(f'<div class="metric-card mc-violet"><div class="metric-value">{avg_conf:.1f}%</div><div class="metric-label">Avg Confidence</div></div>', unsafe_allow_html=True)
                        with mcol4:
                            safe_count = risk_counts.get("Minimal Risk", 0) + risk_counts.get("Low Risk", 0)
                            st.markdown(f'<div class="metric-card mc-teal"><div class="metric-value">{safe_count}</div><div class="metric-label">Low / Minimal</div></div>', unsafe_allow_html=True)

                        st.markdown('<div class="section-label" style="margin-top:20px;">Risk Distribution</div>', unsafe_allow_html=True)
                        fig_dist = px.pie(
                            values=risk_counts.values, names=risk_counts.index,
                            color=risk_counts.index,
                            color_discrete_map={
                                "Critical Risk": "#dc2626", "High Risk": "#ea580c",
                                "Moderate Risk": "#d97706", "Low Risk": "#0d9488",
                                "Minimal Risk": "#2563eb"
                            },
                            hole=0.55,
                        )
                        fig_dist.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#3d4f65", family="DM Sans"),
                            legend=dict(font=dict(size=12)),
                            height=340, margin=dict(t=16, b=16),
                        )
                        st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})

                        st.markdown('<div class="section-label">Detailed Results</div>', unsafe_allow_html=True)
                        st.dataframe(results_df, use_container_width=True, height=400)

                        csv_data = results_df.to_csv(index=False)
                        st.download_button(
                            "Download Results CSV",
                            csv_data, "pharmarisk_batch_results.csv", "text/csv",
                            use_container_width=True
                        )

            except Exception as e:
                st.error(f"Error processing file: {e}")

    # ─────────────────────────────────────────────────────
    #  TAB 3: ANALYTICS DASHBOARD
    # ─────────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-label">Analytics Overview</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#5a6b80;font-size:0.92rem;margin-top:-4px;">Key metrics and visualizations from the full training dataset.</p>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-card mc-blue"><div class="metric-value">{dashboard["total_reviews"]:,}</div><div class="metric-label">Total Reviews</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card mc-violet"><div class="metric-value">{dashboard["total_drugs"]:,}</div><div class="metric-label">Unique Drugs</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card mc-teal"><div class="metric-value">{dashboard["total_conditions"]:,}</div><div class="metric-label">Conditions</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card mc-amber"><div class="metric-value">5</div><div class="metric-label">Risk Tiers</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)

        col_dist, col_top = st.columns([1, 1])

        with col_dist:
            st.markdown('<div class="section-label">Overall Risk Distribution</div>', unsafe_allow_html=True)
            risk_dist = dashboard["overall_risk_distribution"]
            labels = [RISK_LABELS[k] for k in sorted(risk_dist.keys())]
            values = [risk_dist[k] for k in sorted(risk_dist.keys())]
            colors = [RISK_COLORS[k] for k in sorted(risk_dist.keys())]

            fig_overall = go.Figure(go.Bar(
                x=labels, y=values,
                marker=dict(color=colors, cornerradius=6),
                text=[f"{v:,}" for v in values], textposition="outside",
                textfont=dict(color="#3d4f65", family="DM Sans", size=12),
            ))
            fig_overall.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=16, b=40, l=40, r=16), height=350,
                xaxis=dict(showgrid=False, tickfont=dict(color="#5a6b80", family="DM Sans", size=11)),
                yaxis=dict(showgrid=True, gridcolor="#eceef2",
                           tickfont=dict(color="#8293a7", family="DM Sans", size=10)),
                bargap=0.3,
            )
            st.plotly_chart(fig_overall, use_container_width=True, config={"displayModeBar": False})

        with col_top:
            st.markdown('<div class="section-label">Top 10 Most Reviewed Drugs</div>', unsafe_allow_html=True)
            drug_stats = dashboard["drug_risk_stats"].nlargest(10, "review_count")

            fig_drugs = go.Figure(go.Bar(
                x=drug_stats["review_count"].values,
                y=drug_stats["drugName"].values,
                orientation="h",
                marker=dict(
                    color=drug_stats["avg_rating"].values,
                    colorscale=[[0, "#dc2626"], [0.5, "#d97706"], [1, "#2563eb"]],
                    cornerradius=5,
                    colorbar=dict(title=dict(text="Avg Rating", font=dict(color="#5a6b80", family="DM Sans", size=11)),
                                  tickfont=dict(color="#8293a7", family="DM Sans")),
                ),
                text=[f"{v:,}" for v in drug_stats["review_count"].values],
                textposition="outside",
                textfont=dict(color="#5a6b80", family="DM Sans", size=11),
            ))
            fig_drugs.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=16, b=16, l=8, r=50), height=350,
                xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                yaxis=dict(showgrid=False,
                           tickfont=dict(color="#3d4f65", family="DM Sans", size=11),
                           autorange="reversed"),
                bargap=0.3,
            )
            st.plotly_chart(fig_drugs, use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-label">Top 15 Conditions by Review Count</div>', unsafe_allow_html=True)
        cond_stats = dashboard["condition_risk_stats"].nlargest(15, "review_count")

        fig_cond = go.Figure(go.Bar(
            x=cond_stats["condition"].values if "condition" in cond_stats.columns else cond_stats.iloc[:, 0].values,
            y=cond_stats["review_count"].values,
            marker=dict(
                color=cond_stats["avg_rating"].values,
                colorscale=[[0, "#dc2626"], [0.5, "#7c3aed"], [1, "#2563eb"]],
                cornerradius=6,
                colorbar=dict(title=dict(text="Avg Rating", font=dict(color="#5a6b80", family="DM Sans", size=11)),
                              tickfont=dict(color="#8293a7", family="DM Sans")),
            ),
            text=[f"{v:,}" for v in cond_stats["review_count"].values],
            textposition="outside",
            textfont=dict(color="#5a6b80", family="DM Sans", size=11),
        ))
        fig_cond.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=16, b=80, l=40, r=16), height=380,
            xaxis=dict(showgrid=False,
                       tickfont=dict(color="#5a6b80", family="DM Sans", size=10),
                       tickangle=-40),
            yaxis=dict(showgrid=True, gridcolor="#eceef2",
                       tickfont=dict(color="#8293a7", family="DM Sans", size=10)),
            bargap=0.3,
        )
        st.plotly_chart(fig_cond, use_container_width=True, config={"displayModeBar": False})


    # ─────────────────────────────────────────────────────
    #  TAB 4: DRUG EXPLORER
    # ─────────────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-label">Drug Explorer</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#5a6b80;font-size:0.92rem;margin-top:-4px;">Select a drug to view its risk profile and patient review analytics.</p>', unsafe_allow_html=True)

        drug_list = dashboard["drug_risk_stats"].sort_values("review_count", ascending=False)
        drug_names = drug_list["drugName"].tolist()

        selected_drug = st.selectbox("Select a Drug", drug_names[:200], key="drug_explorer")

        if selected_drug:
            drug_row = drug_list[drug_list["drugName"] == selected_drug].iloc[0]

            st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)

            avg_rating = drug_row["avg_rating"]
            if avg_rating <= 3:
                overall_label, overall_color = "Critical", "#dc2626"
            elif avg_rating <= 5:
                overall_label, overall_color = "High", "#ea580c"
            elif avg_rating <= 7:
                overall_label, overall_color = "Moderate", "#d97706"
            elif avg_rating <= 8.5:
                overall_label, overall_color = "Low", "#0d9488"
            else:
                overall_label, overall_color = "Minimal", "#2563eb"

            dc1, dc2, dc3 = st.columns(3)
            with dc1:
                st.markdown(f"""
                <div class="metric-card mc-blue">
                    <div class="metric-value" style="color:{overall_color};">{avg_rating:.1f}</div>
                    <div class="metric-label">Average Rating</div>
                </div>
                """, unsafe_allow_html=True)
            with dc2:
                st.markdown(f"""
                <div class="metric-card mc-violet">
                    <div class="metric-value">{drug_row['review_count']:,}</div>
                    <div class="metric-label">Total Reviews</div>
                </div>
                """, unsafe_allow_html=True)
            with dc3:
                st.markdown(f"""
                <div class="metric-card mc-teal">
                    <div class="metric-value" style="font-size:1.4rem; color:{overall_color};">{overall_label} Risk</div>
                    <div class="metric-label">Overall Assessment</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="section-label" style="margin-top:20px;">Risk Tier Breakdown</div>', unsafe_allow_html=True)
            risk_breakdown = drug_row["risk_distribution"]
            if isinstance(risk_breakdown, dict) and risk_breakdown:
                br_labels = [RISK_LABELS.get(k, str(k)) for k in sorted(risk_breakdown.keys())]
                br_values = [risk_breakdown[k] for k in sorted(risk_breakdown.keys())]
                br_colors = [RISK_COLORS.get(k, "#5a6b80") for k in sorted(risk_breakdown.keys())]

                fig_drug_risk = go.Figure(go.Pie(
                    labels=br_labels, values=br_values,
                    marker=dict(colors=br_colors),
                    hole=0.55,
                    textinfo="label+percent",
                    textfont=dict(color="#3d4f65", family="DM Sans", size=12),
                ))
                fig_drug_risk.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#3d4f65", family="DM Sans"),
                    legend=dict(font=dict(size=12, color="#5a6b80")),
                    height=380, margin=dict(t=16, b=16),
                )
                st.plotly_chart(fig_drug_risk, use_container_width=True, config={"displayModeBar": False})


    # ─────────────────────────────────────────────────────
    #  TAB 5: ABOUT
    # ─────────────────────────────────────────────────────
    with tab5:
        st.markdown('<div class="section-label">About PharmaRisk AI</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="pr-card" style="border-left: 3px solid #2563eb;">
            <h4 style="color:#1a2332; margin-top:0; font-family:'Source Serif 4',Georgia,serif !important;">Mission</h4>
            <p style="color:#5a6b80;">PharmaRisk AI uses natural language processing and machine learning to analyze
            drug reviews and predict pharmacogenomic adverse reaction risks. The 5-tier classification system
            provides granular risk insights for healthcare professionals and researchers.</p>
        </div>
        """, unsafe_allow_html=True)

        ab1, ab2 = st.columns(2)

        with ab1:
            st.markdown("""
            <div class="pr-card" style="border-left: 3px solid #7c3aed;">
                <h4 style="color:#1a2332; margin-top:0; font-family:'Source Serif 4',Georgia,serif !important;">Model Architecture</h4>
                <div style="color:#5a6b80; font-size:0.9rem; line-height:1.9;">
                    <b>Algorithm:</b> Random Forest Classifier (200 trees)<br>
                    <b>Text Features:</b> TF-IDF with 10K features and bigrams<br>
                    <b>Numeric Features:</b> Review length, word count, sentiment keyword counts, usefulness score<br>
                    <b>Categorical:</b> Drug name and condition encoding<br>
                    <b>Classification:</b> 5-tier risk system
                </div>
            </div>
            """, unsafe_allow_html=True)

        with ab2:
            st.markdown("""
            <div class="pr-card" style="border-left: 3px solid #0d9488;">
                <h4 style="color:#1a2332; margin-top:0; font-family:'Source Serif 4',Georgia,serif !important;">Risk Classification</h4>
                <div style="color:#5a6b80; font-size:0.9rem; line-height:1.9;">
                    <span style="color:#dc2626;">●</span> <b>Critical Risk (Rating 1–2):</b> Severe adverse reactions<br>
                    <span style="color:#ea580c;">●</span> <b>High Risk (Rating 3–4):</b> Significant side effects<br>
                    <span style="color:#d97706;">●</span> <b>Moderate Risk (Rating 5–6):</b> Notable concerns present<br>
                    <span style="color:#0d9488;">●</span> <b>Low Risk (Rating 7–8):</b> Minor issues only<br>
                    <span style="color:#2563eb;">●</span> <b>Minimal Risk (Rating 9–10):</b> Well tolerated
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="pr-card" style="border-left: 3px solid #d97706;">
            <h4 style="color:#1a2332; margin-top:0; font-family:'Source Serif 4',Georgia,serif !important;">Disclaimer</h4>
            <p style="color:#5a6b80;">This tool is for research and educational purposes only. It should not be used
            as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a
            qualified healthcare provider regarding any medical decisions.</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
st.markdown('<div class="footer">PharmaRisk AI v2.0 — Pharmacogenomic Risk Intelligence Platform</div>', unsafe_allow_html=True)
