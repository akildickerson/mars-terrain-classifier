import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from src.planner.visualize import colorize_segmentation, draw_path
from src.planner.astar import AStarPlanner
from src.model.unet import MarsUNet
import time
import html

st.set_page_config(page_title="Mars Terrain Classifier", page_icon="🔴", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600&display=swap');

* { color: #aaaaaa !important; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0b0e14;
}

.block-container {
    padding-top: 2rem;
    max-width: 1500px;
}

/* HUD panel base */
.hud-panel {
    background: #0d0d0d;
    border: 1px solid #2a2a2a;
    border-top: 1px solid #e8623a;
    position: relative;
    padding: 0;
    font-family: 'Share Tech Mono', monospace;
}

.hud-panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, #e8623a 0%, #e8623a44 60%, transparent 100%);
}

.hud-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 10px;
    border-bottom: 1px solid #1e1e1e;
    background: #111111;
}

.hud-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #e8623a !important;
}

.hud-status {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem;
    color: #4caf50 !important;
    letter-spacing: 1px;
}

.hud-status::before {
    content: '● ';
    color: #4caf50 !important;
}

.hud-meta {
    display: flex;
    justify-content: space-between;
    padding: 5px 10px;
    border-top: 1px solid #1e1e1e;
    background: #0a0a0a;
}

.hud-coord {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.58rem;
    color: #555555 !important;
    letter-spacing: 1px;
}

.hud-coord span {
    color: #888888 !important;
}

/* image wrapper — scanlines effect */
.hud-image-wrap {
    position: relative;
    margin: 0;
    line-height: 0;
}

.hud-image-wrap::after {
    content: '';
    position: absolute;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,0,0,0.08) 2px,
        rgba(0,0,0,0.08) 4px
    );
    pointer-events: none;
}

/* corner brackets */
.hud-image-wrap::before {
    content: '';
    position: absolute;
    inset: 6px;
    border: 1px solid transparent;
    background:
        linear-gradient(#e8623a, #e8623a) top left / 14px 1px no-repeat,
        linear-gradient(#e8623a, #e8623a) top left / 1px 14px no-repeat,
        linear-gradient(#e8623a, #e8623a) top right / 14px 1px no-repeat,
        linear-gradient(#e8623a, #e8623a) top right / 1px 14px no-repeat,
        linear-gradient(#e8623a, #e8623a) bottom left / 14px 1px no-repeat,
        linear-gradient(#e8623a, #e8623a) bottom left / 1px 14px no-repeat,
        linear-gradient(#e8623a, #e8623a) bottom right / 14px 1px no-repeat,
        linear-gradient(#e8623a, #e8623a) bottom right / 1px 14px no-repeat;
    pointer-events: none;
    z-index: 10;
}

/* stats panel */
.stats-panel {
    background: #0d0d0d;
    border: 1px solid #2a2a2a;
    border-top: 1px solid #e8623a;
    font-family: 'Share Tech Mono', monospace;
    height: 100%;
    box-sizing: border-box;
}

.stats-header {
    padding: 6px 10px;
    border-bottom: 1px solid #1e1e1e;
    background: #111111;
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #e8623a !important;
}

.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 10px;
    border-bottom: 1px solid #141414;
}

.stat-label {
    font-size: 0.6rem;
    letter-spacing: 2px;
    color: #555555 !important;
    text-transform: uppercase;
}

.stat-value {
    font-size: 0.72rem;
    color: #cccccc !important;
}

.stat-bar-wrap {
    padding: 6px 10px 10px;
    border-bottom: 1px solid #141414;
}

.stat-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.58rem;
    color: #555 !important;
    margin-bottom: 4px;
    letter-spacing: 1px;
    gap: 8px;
}

.stat-bar-track {
    height: 3px;
    background: #1e1e1e;
    border-radius: 1px;
    overflow: hidden;
}

.stat-bar-fill {
    height: 100%;
    border-radius: 1px;
}

/* Path found badge */
.path-found {
    display: inline-block;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 2px;
    color: #4caf50 !important;
    border: 1px solid #4caf50;
    padding: 2px 6px;
}

.path-notfound {
    display: inline-block;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 2px;
    color: #e8623a !important;
    border: 1px solid #e8623a;
    padding: 2px 6px;
}

/* section divider */
.hud-divider {
    border: none;
    border-top: 1px solid #1e1e1e;
    margin: 1.2rem 0;
}

/* override Streamlit image default margin */
div[data-testid="stImage"] {
    margin: 0 !important;
}

div[data-testid="stImage"] img {
    display: block;
}

/* number input styling */
div[data-testid="stNumberInput"] input {
    background: #1a1a1a;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    color: #aaaaaa !important;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
}

.mars-container {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 320px;
}

@keyframes pulse {
    0%   { filter: drop-shadow(0 0 20px rgba(232,98,58,0.4)); }
    50%  { filter: drop-shadow(0 0 50px rgba(232,98,58,0.7)); }
    100% { filter: drop-shadow(0 0 20px rgba(232,98,58,0.4)); }
}

.mars-svg {
    animation: pulse 4s ease-in-out infinite;
}

.mars-title {
    text-align: center;
    color: #e8623a !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    font-size: 1.1rem !important;
}

.section-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #e8623a !important;
    margin-bottom: 8px;
}

.mission-id {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.58rem;
    color: #333 !important;
    text-align: center;
    letter-spacing: 2px;
    margin-top: 6px;
}

/* Make the HUD row sit directly under planner costs */
.planner-shell {
    padding-top: 4.5rem;
}

.hud-shell {
    margin-top: 1.5rem;
}

/* tighter Streamlit gaps */
[data-testid="stHorizontalBlock"] {
    gap: 1rem;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_image():
    path = next(Path("data/ai4mars/ai4mars-dataset-merged-0.6/msl/ncam/images/edr").glob("*.JPG"))
    return Image.open(path).resize((512, 512)), path.stem


@st.cache_resource
def load_model():
    model = MarsUNet()
    state = torch.load("checkpoints/unet_v2_epoch_2.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


# ── Load data & run model ─────────────────────────────────────────────────────
image, frame_id = load_image()
model = load_model()

img_array = np.array(image.convert("RGB"))
img_tensor = torch.tensor(img_array.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

with torch.no_grad():
    seg = model.predict(img_tensor).squeeze(0).numpy()


# ── Main Layout ───────────────────────────────────────────────────────────────
left, right = st.columns([1, 2.6], gap="large")

with left:
    st.markdown("""
    <div class="mars-container">
    <svg class="mars-svg" width="240" height="240" viewBox="0 0 280 280" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <radialGradient id="marsGrad" cx="38%" cy="35%" r="60%">
          <stop offset="0%" stop-color="#e8724a"/>
          <stop offset="40%" stop-color="#c0392b"/>
          <stop offset="100%" stop-color="#5a1008"/>
        </radialGradient>
        <radialGradient id="crater1" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="#7b1a0a" stop-opacity="0.8"/>
          <stop offset="100%" stop-color="#c0392b" stop-opacity="0"/>
        </radialGradient>
        <radialGradient id="crater2" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="#5a1008" stop-opacity="0.9"/>
          <stop offset="100%" stop-color="#c0392b" stop-opacity="0"/>
        </radialGradient>
        <clipPath id="marsClip">
          <circle cx="140" cy="140" r="130"/>
        </clipPath>
        <filter id="blur">
          <feGaussianBlur stdDeviation="2"/>
        </filter>
      </defs>
      <circle cx="140" cy="140" r="130" fill="url(#marsGrad)"/>
      <g clip-path="url(#marsClip)">
        <ellipse cx="140" cy="170" rx="130" ry="30" fill="#8b2500" opacity="0.4"/>
        <ellipse cx="100" cy="120" rx="80" ry="20" fill="#7b1a0a" opacity="0.3"/>
        <ellipse cx="180" cy="90" rx="60" ry="15" fill="#9b2a00" opacity="0.25"/>
        <ellipse cx="160" cy="140" rx="50" ry="35" fill="#e8825a" opacity="0.3"/>
        <ellipse cx="90" cy="160" rx="40" ry="25" fill="#d4603a" opacity="0.2"/>
        <circle cx="110" cy="130" r="18" fill="url(#crater1)"/>
        <circle cx="110" cy="130" r="18" stroke="#6b1508" stroke-width="1.5" fill="none" opacity="0.6"/>
        <circle cx="170" cy="170" r="12" fill="url(#crater2)"/>
        <circle cx="170" cy="170" r="12" stroke="#6b1508" stroke-width="1" fill="none" opacity="0.5"/>
        <circle cx="80" cy="90" r="8" fill="url(#crater1)"/>
        <circle cx="80" cy="90" r="8" stroke="#6b1508" stroke-width="1" fill="none" opacity="0.4"/>
        <circle cx="200" cy="110" r="10" fill="url(#crater2)"/>
        <circle cx="195" cy="155" r="6" fill="url(#crater1)"/>
        <circle cx="130" cy="195" r="9" fill="url(#crater2)"/>
        <path d="M 80 145 Q 140 138 210 150" stroke="#6b1508" stroke-width="4" fill="none" opacity="0.5" filter="url(#blur)"/>
        <path d="M 80 148 Q 140 141 210 153" stroke="#8b2500" stroke-width="2" fill="none" opacity="0.3"/>
        <ellipse cx="140" cy="18" rx="45" ry="18" fill="white" opacity="0.7"/>
        <ellipse cx="140" cy="15" rx="35" ry="12" fill="white" opacity="0.5"/>
      </g>
      <ellipse cx="100" cy="95" rx="55" ry="45" fill="white" opacity="0.07"/>
      <path d="M 140 10 A 130 130 0 0 1 270 140 A 130 130 0 0 1 140 270 A 130 130 0 0 0 140 10" fill="black" opacity="0.35"/>
    </svg>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p class='mars-title'>Mars Terrain Classifier</p>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;font-size:0.75rem;letter-spacing:1px;'>"
        "Autonomous terrain perception<br>and path planning from NASA rover imagery</p>",
        unsafe_allow_html=True
    )
    st.markdown("<p class='mission-id'>MSL · CURIOSITY ROVER · NAVCAM</p>", unsafe_allow_html=True)


with right:
    st.markdown("<div class='planner-shell'>", unsafe_allow_html=True)
    st.markdown("<p class='section-label'>Path Planner Costs</p>", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        soil_cost = st.number_input("Soil", min_value=0.1, max_value=20.0, value=1.0, step=0.1)

    with c2:
        bedrock_cost = st.number_input("Bedrock", min_value=0.1, max_value=20.0, value=2.0, step=0.1)

    with c3:
        sand_cost = st.number_input("Sand", min_value=0.1, max_value=20.0, value=3.0, step=0.1)

    with c4:
        large_rock_cost = st.number_input("Large Rock", min_value=0.1, max_value=20.0, value=10.0, step=0.1)

    with c5:
        rover_track_cost = st.number_input("Rover Track", min_value=0.1, max_value=20.0, value=0.5, step=0.1)

    st.markdown("<hr class='hud-divider'>", unsafe_allow_html=True)


    # ── Planner data ──────────────────────────────────────────────────────────
    costs = {
        "rover_track": rover_track_cost,
        "soil": soil_cost,
        "bedrock": bedrock_cost,
        "sand": sand_cost,
        "large_rock": large_rock_cost,
    }

    TERRAIN_LABELS = ["soil", "bedrock", "sand", "large_rock", "rover_track"]
    TERRAIN_COLORS = ["#b87333", "#8a7560", "#c8b07a", "#e8623a", "#4caf50"]

    planner = AStarPlanner(costs)
    path = planner.plan(seg)
    colored = colorize_segmentation(seg)

    flat = seg.flatten()
    total = flat.size

    terrain_counts = {
        label: int(np.sum(flat == i))
        for i, label in enumerate(TERRAIN_LABELS)
    }

    terrain_pcts = {
        label: count / total * 100
        for label, count in terrain_counts.items()
    }

    path_len = len(path) if path is not None else 0

    frame_display = frame_id[:16].upper() if frame_id else "UNKNOWN"
    ts = time.strftime("%H:%M:%S")

    st.markdown("<div class='hud-shell'>", unsafe_allow_html=True)

    col_img1, col_img2, col_stats = st.columns([5, 5, 3])

    with col_img1:
        st.markdown(f"""
        <div class="hud-panel">
          <div class="hud-header">
            <span class="hud-label">RAW FEED · NAVCAM</span>
            <span class="hud-status">LIVE</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="hud-image-wrap">', unsafe_allow_html=True)
        st.image(image.convert("L"), use_container_width=True, clamp=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="hud-panel">
          <div class="hud-meta">
            <span class="hud-coord">FRAME <span>{frame_display}</span></span>
            <span class="hud-coord">RES <span>512×512</span></span>
            <span class="hud-coord">T+ <span>{ts}</span></span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_img2:
        path_badge = (
            '<span class="path-found">PATH OK</span>'
            if path is not None
            else '<span class="path-notfound">NO PATH</span>'
        )

        st.markdown(f"""
        <div class="hud-panel">
          <div class="hud-header">
            <span class="hud-label">SEG · PLANNED ROUTE</span>
            {path_badge}
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="hud-image-wrap">', unsafe_allow_html=True)

        if path is not None:
            path_img = draw_path(colored, path)
            st.image(path_img, use_container_width=True, clamp=True)
        else:
            st.image(colored, use_container_width=True, clamp=True)

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="hud-panel">
          <div class="hud-meta">
            <span class="hud-coord">NODES <span>{path_len}</span></span>
            <span class="hud-coord">MODEL <span>UNET-V2</span></span>
            <span class="hud-coord">ALGO <span>A*</span></span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_stats:
        dominant = max(terrain_pcts, key=terrain_pcts.get)

        bars_html = ""

        for i, label in enumerate(TERRAIN_LABELS):
            safe_label = html.escape(label.replace("_", " ").upper())
            pct = terrain_pcts[label]
            cost_val = costs[label]
            color = TERRAIN_COLORS[i]

            bars_html += f"""
            <div class="stat-bar-wrap">
              <div class="stat-bar-label">
                <span>{safe_label}</span>
                <span>{pct:.1f}% · cost {cost_val:.1f}</span>
              </div>
              <div class="stat-bar-track">
                <div class="stat-bar-fill" style="width:{min(pct, 100):.1f}%; background:{color};"></div>
              </div>
            </div>
            """

        stats_html = f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

        body {{
            margin: 0;
            background: transparent;
            color: #aaaaaa;
            font-family: 'Share Tech Mono', monospace;
        }}

        .stats-panel {{
            background: #0d0d0d;
            border: 1px solid #2a2a2a;
            border-top: 1px solid #e8623a;
            font-family: 'Share Tech Mono', monospace;
            min-height: 100%;
            box-sizing: border-box;
        }}

        .stats-header {{
            padding: 6px 10px;
            border-bottom: 1px solid #1e1e1e;
            background: #111111;
            font-size: 0.65rem;
            letter-spacing: 3px;
            text-transform: uppercase;
            color: #e8623a;
        }}

        .stat-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 7px 10px;
            border-bottom: 1px solid #141414;
        }}

        .stat-label {{
            font-size: 0.6rem;
            letter-spacing: 2px;
            color: #555555;
            text-transform: uppercase;
        }}

        .stat-value {{
            font-size: 0.72rem;
            color: #cccccc;
        }}

        .stat-bar-wrap {{
            padding: 6px 10px 10px;
            border-bottom: 1px solid #141414;
        }}

        .stat-bar-label {{
            display: flex;
            justify-content: space-between;
            font-size: 0.58rem;
            color: #777777;
            margin-bottom: 4px;
            letter-spacing: 1px;
            gap: 8px;
        }}

        .stat-bar-track {{
            height: 3px;
            background: #1e1e1e;
            border-radius: 1px;
            overflow: hidden;
        }}

        .stat-bar-fill {{
            height: 100%;
            border-radius: 1px;
        }}
        </style>

        <div class="stats-panel">
          <div class="stats-header">Terrain Analysis</div>

          <div class="stat-row">
            <span class="stat-label">Dominant</span>
            <span class="stat-value">{html.escape(dominant.replace("_", " ").upper())}</span>
          </div>

          <div class="stat-row">
            <span class="stat-label">Path nodes</span>
            <span class="stat-value">{path_len}</span>
          </div>

          <div class="stat-row">
            <span class="stat-label">Coverage</span>
            <span class="stat-value">512×512 px</span>
          </div>

          {bars_html}
        </div>
        """

        components.html(stats_html, height=340, scrolling=False)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)