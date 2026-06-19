"""
Extra Innings Win Probability & Market Pricing Tool
Run distributions derived from 2.1M Statcast plate appearances (2021-2023).
"""

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

# ─────────────────────────────────────────────────────────────────────────────
# EMPIRICAL RUN DISTRIBUTIONS
# Source: 2021-2023 Statcast, innings 1-9 only (excludes ghost-runner distortion)
# Format: {(bases_tuple, outs): [P(0 runs), P(1 run), ..., P(10 runs)]}
# bases_tuple: (on_1b, on_2b, on_3b) — 1 = occupied
# ─────────────────────────────────────────────────────────────────────────────
RUN_DIST = {
    ((0, 0, 0), 0): [0.728034, 0.144646, 0.069615, 0.032879, 0.014256, 0.006218, 0.002465, 0.001056, 0.000472, 0.000202, 0.000075],
    ((0, 0, 0), 1): [0.840053, 0.095685, 0.038637, 0.016274, 0.005985, 0.002033, 0.000846, 0.000341, 0.000052, 0.000041, 0.000031],
    ((0, 0, 0), 2): [0.930429, 0.047807, 0.015071, 0.004643, 0.001479, 0.000324, 0.000143, 0.000078, 0.000000, 0.000013, 0.000000],
    ((0, 0, 1), 0): [0.150831, 0.574822, 0.140143, 0.079572, 0.033254, 0.014252, 0.002375, 0.004751, 0.000000, 0.000000, 0.000000],
    ((0, 0, 1), 1): [0.329067, 0.486348, 0.104096, 0.050057, 0.018771, 0.006542, 0.002275, 0.001706, 0.000853, 0.000284, 0.000000],
    ((0, 0, 1), 2): [0.734181, 0.185484, 0.054125, 0.019386, 0.004342, 0.000931, 0.001086, 0.000310, 0.000155, 0.000000, 0.000000],
    ((0, 1, 0), 0): [0.396900, 0.316911, 0.149322, 0.074730, 0.035428, 0.014254, 0.006643, 0.003045, 0.001937, 0.000554, 0.000138],
    ((0, 1, 0), 1): [0.600727, 0.225985, 0.101016, 0.045511, 0.016767, 0.006360, 0.002395, 0.000991, 0.000165, 0.000083, 0.000000],
    ((0, 1, 0), 2): [0.784637, 0.143092, 0.050180, 0.014644, 0.005050, 0.001515, 0.000442, 0.000316, 0.000126, 0.000000, 0.000000],
    ((0, 1, 1), 0): [0.159978, 0.278200, 0.284165, 0.130694, 0.080260, 0.036334, 0.013015, 0.010846, 0.004338, 0.000542, 0.001627],
    ((0, 1, 1), 1): [0.326464, 0.298680, 0.202825, 0.095161, 0.046075, 0.020607, 0.006946, 0.001852, 0.001158, 0.000232, 0.000000],
    ((0, 1, 1), 2): [0.735751, 0.072539, 0.125648, 0.041667, 0.016623, 0.005397, 0.001727, 0.000648, 0.000000, 0.000000, 0.000000],
    ((1, 0, 0), 0): [0.592924, 0.153349, 0.131772, 0.065434, 0.031025, 0.014809, 0.005796, 0.002714, 0.001340, 0.000536, 0.000134],
    ((1, 0, 0), 1): [0.740801, 0.105786, 0.088621, 0.039676, 0.015739, 0.005868, 0.002166, 0.000795, 0.000302, 0.000110, 0.000110],
    ((1, 0, 0), 2): [0.880697, 0.050953, 0.046376, 0.014861, 0.004825, 0.001599, 0.000358, 0.000248, 0.000028, 0.000028, 0.000000],
    ((1, 0, 1), 0): [0.150099, 0.416238, 0.165941, 0.146535, 0.066931, 0.029703, 0.015446, 0.005149, 0.002772, 0.001188, 0.000000],
    ((1, 0, 1), 1): [0.367757, 0.373241, 0.111863, 0.085908, 0.037105, 0.014257, 0.005849, 0.002376, 0.001097, 0.000548, 0.000000],
    ((1, 0, 1), 2): [0.736509, 0.140006, 0.050936, 0.049559, 0.015694, 0.003992, 0.001927, 0.000826, 0.000275, 0.000275, 0.000000],
    ((1, 1, 0), 0): [0.392324, 0.236000, 0.144973, 0.117297, 0.057514, 0.030162, 0.012216, 0.004973, 0.002703, 0.001405, 0.000108],
    ((1, 1, 0), 1): [0.594909, 0.162728, 0.100718, 0.085836, 0.033943, 0.012728, 0.005940, 0.002350, 0.000457, 0.000131, 0.000261],
    ((1, 1, 0), 2): [0.777622, 0.109763, 0.050189, 0.042619, 0.013947, 0.003992, 0.000933, 0.000570, 0.000104, 0.000104, 0.000104],
    ((1, 1, 1), 0): [0.148119, 0.249216, 0.219436, 0.145768, 0.114028, 0.065439, 0.031740, 0.016850, 0.006270, 0.002743, 0.000000],
    ((1, 1, 1), 1): [0.327037, 0.263554, 0.169465, 0.097587, 0.081322, 0.037426, 0.015740, 0.006296, 0.001049, 0.000350, 0.000175],
    ((1, 1, 1), 2): [0.671510, 0.110145, 0.107810, 0.048651, 0.044110, 0.012195, 0.003892, 0.001038, 0.000259, 0.000389, 0.000000],
}

# Expected runs per state (mean of empirical distribution) — for reference table only
RE24 = {
    k: sum(r * p for r, p in enumerate(v))
    for k, v in RUN_DIST.items()
}

BASE_STATE_LABELS = {
    (0,0,0): "Bases Empty",
    (1,0,0): "Man on 1st",
    (0,1,0): "Man on 2nd",
    (0,0,1): "Man on 3rd",
    (1,1,0): "1st & 2nd",
    (1,0,1): "1st & 3rd",
    (0,1,1): "2nd & 3rd",
    (1,1,1): "Bases Loaded",
}
BASE_STATES_LIST = list(BASE_STATE_LABELS.keys())
BASE_STATE_NAMES = list(BASE_STATE_LABELS.values())

EXTRA_INNING_START = ((0, 1, 0), 0)  # Runner on 2nd, 0 outs (ghost runner rule)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def ordinal(n: int) -> str:
    return {10:"10th",11:"11th",12:"12th",13:"13th",14:"14th",
            15:"15th",16:"16th",17:"17th"}.get(n, f"{n}th")


def inning_label(inn: int, h: str) -> str:
    return f"{'Top' if h == 'top' else 'Bot'} {ordinal(inn)}"


def american_odds(p: float) -> str:
    if p <= 0.001:
        return "N/A"
    if p >= 0.999:
        return "N/A"
    if p >= 0.5:
        return f"{-round((p / (1 - p)) * 100):+d}"
    return f"{round(((1 - p) / p) * 100):+d}"


def get_run_distribution(bases: tuple, outs: int) -> np.ndarray:
    """Return empirical run distribution for this base-out state as a numpy array."""
    dist = RUN_DIST.get((bases, outs))
    if dist is None:
        dist = RUN_DIST[((0, 0, 0), min(outs, 2))]
    arr = np.array(dist, dtype=float)
    return arr / arr.sum()


def tilt_dist(dist: np.ndarray, theta: float) -> np.ndarray:
    """
    Exponential tilt of a run distribution.
    theta > 0 shifts weight toward more runs (stronger offense).
    theta < 0 shifts weight toward fewer runs (weaker offense).
    Calibrated so theta_max=0.10 moves a 50/50 line to ~40/60 (+150).
    """
    runs = np.arange(len(dist))
    weights = np.exp(theta * runs)
    tilted = dist * weights
    return tilted / tilted.sum()


def extra_inning_run_dist() -> np.ndarray:
    """Run distribution for a fresh extra inning half (runner on 2nd, 0 outs)."""
    return get_run_distribution(*EXTRA_INNING_START)


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_win_probabilities(
    inning: int,
    half: str,
    home_lead: int,
    bases: tuple,
    outs: int,
    strength: float = 0.0,
    manfred_runner: bool = True,
    max_innings: int = 17,
) -> dict:
    """
    Simulate forward from current game state using empirical run distributions.
    strength > 0: home team is stronger. strength < 0: away team is stronger.
    At |strength|=1, the weaker team's win probability drops to ~40% from a tied game.
    Returns home/away win probabilities and P(game reaches each inning).
    """
    THETA_MAX = 0.10  # calibrated: s=±1 → ~40/60 win split from tied game
    theta = strength * THETA_MAX

    MAX_DELTA = 15
    SCORES = np.arange(-MAX_DELTA, MAX_DELTA + 1)

    fresh_start = EXTRA_INNING_START if manfred_runner else ((0, 0, 0), 0)
    base_fresh = get_run_distribution(*fresh_start)
    # Home gets positive tilt when strength>0; away gets the mirror
    home_fresh = tilt_dist(base_fresh, +theta)
    away_fresh = tilt_dist(base_fresh, -theta)

    # Current half-inning: apply same tilt to the batting team's distribution
    base_current = get_run_distribution(bases, outs)
    if half == "top":
        run_d_current = tilt_dist(base_current, -theta)   # away is batting
    else:
        run_d_current = tilt_dist(base_current, +theta)   # home is batting

    score_dist = np.zeros(len(SCORES))
    clamped = int(np.clip(home_lead, -MAX_DELTA, MAX_DELTA))
    score_dist[clamped + MAX_DELTA] = 1.0

    home_win_prob = 0.0
    away_win_prob = 0.0
    inning_reach_probs = {}

    def apply_runs(dist: np.ndarray, batting: str, run_probs: np.ndarray) -> np.ndarray:
        new_dist = np.zeros(len(SCORES))
        sign = -1 if batting == "away" else +1
        for runs, p_runs in enumerate(run_probs):
            if p_runs < 1e-12:
                continue
            shift = sign * runs
            shifted = np.roll(dist, shift)
            if shift > 0:
                shifted[:shift] = 0.0
            elif shift < 0:
                shifted[shift:] = 0.0
            new_dist += p_runs * shifted
        return new_dist

    for sim_inning in range(inning, max_innings + 1):
        p_still_playing = score_dist.sum()
        if p_still_playing < 1e-8:
            break

        inning_reach_probs[sim_inning] = float(p_still_playing)

        # Top half (away bats) — skip if we're already in bottom of this inning
        if not (sim_inning == inning and half == "bottom"):
            away_run_d = run_d_current if (sim_inning == inning and half == "top") else away_fresh
            score_dist = apply_runs(score_dist, "away", away_run_d)

        # Bottom half (home bats)
        home_run_d = run_d_current if (sim_inning == inning and half == "bottom") else home_fresh
        score_dist = apply_runs(score_dist, "home", home_run_d)

        # End-of-inning resolution
        new_dist = np.zeros(len(SCORES))
        for i, score_after in enumerate(SCORES):
            p = score_dist[i]
            if p < 1e-12:
                continue
            if score_after > 0:
                home_win_prob += p   # home leads at end of bottom half → home wins
            elif score_after < 0:
                away_win_prob += p   # away leads at end of bottom half → away wins
            else:
                new_dist[i] += p    # tied → next inning
        score_dist = new_dist

    # Remaining probability (past max innings) — split by current distribution
    remaining = score_dist.sum()
    if remaining > 1e-6:
        for i, score in enumerate(SCORES):
            p = score_dist[i]
            if p < 1e-12:
                continue
            if score > 0:
                home_win_prob += p
            elif score < 0:
                away_win_prob += p
            else:
                home_win_prob += p * 0.5
                away_win_prob += p * 0.5

    total = home_win_prob + away_win_prob
    if total > 0:
        home_win_prob /= total
        away_win_prob /= total

    return {
        "home_win": home_win_prob,
        "away_win": away_win_prob,
        "inning_reach_probs": inning_reach_probs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT APP
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Extra Innings Pricer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 0rem; }
    .win-box {
        border-radius: 8px;
        padding: 16px 12px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 8px;
    }
    /* Diamond base buttons: override Streamlit's default button padding */
    div[data-testid="stButton"] > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# ── SESSION STATE DEFAULTS ────────────────────────────────────────────────────
# These fields are mutated by the OUT button; all other widgets read from them.
def _ss_init():
    defaults = {
        "inning": 10,
        "half": "top",
        "outs": 0,
        "on_1b": False,
        "on_2b": True,   # ghost runner default
        "on_3b": False,
        "manfred_runner": True,
        "away_score": 0,
        "home_score": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_ss_init()


# ── TEAM COLORS ───────────────────────────────────────────────────────────────
TEAM_COLORS = {
    "Angels":       "#BA0021",  # red
    "Astros":       "#EB6E1F",  # orange
    "Athletics":    "#003831",  # dark green
    "Blue Jays":    "#134A8E",  # royal blue
    "Braves":       "#CE1141",  # red
    "Brewers":      "#12284B",  # navy
    "Cardinals":    "#C41E3A",  # cardinal red
    "Cubs":         "#0E3386",  # cubs blue
    "Diamondbacks": "#A71930",  # sedona red
    "Dodgers":      "#005A9C",  # dodger blue
    "Giants":       "#FD5A1E",  # orange
    "Guardians":    "#00385D",  # navy
    "Mariners":     "#005C5C",  # teal
    "Marlins":      "#00A3E0",  # miami blue
    "Mets":         "#002D72",  # mets blue
    "Nationals":    "#AB0003",  # red
    "Orioles":      "#DF4601",  # orioles orange
    "Padres":       "#2F241D",  # brown
    "Phillies":     "#E81828",  # phillies red
    "Pirates":      "#FDB827",  # gold/yellow
    "Rangers":      "#003278",  # blue
    "Rays":         "#8FBCE6",  # columbia blue
    "Red Sox":      "#BD3039",  # red
    "Reds":         "#C6011F",  # red
    "Rockies":      "#33006F",  # purple
    "Royals":       "#004687",  # royal blue
    "Tigers":       "#0C2340",  # navy
    "Twins":        "#002B5C",  # navy
    "White Sox":    "#27251F",  # black
    "Yankees":      "#1C2841",  # navy
}
NEUTRAL_AWAY = "#7B241C"   # fallback
NEUTRAL_HOME = "#1A5276"   # fallback
NEUTRAL_NEXT = "#6B6B6B"   # gray for "goes to next inning" slice


def team_color(name: str) -> str:
    return TEAM_COLORS.get(name, NEUTRAL_HOME)


def text_color(bg_hex: str) -> str:
    """Return black or white depending on background luminance."""
    h = bg_hex.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#000000" if luminance > 0.55 else "#ffffff"


def _run_scored():
    if st.session_state["half"] == "top":
        st.session_state["away_score"] += 1
    else:
        st.session_state["home_score"] += 1


def _advance_out():
    """Record an out. On the 3rd out, flip the half-inning and reset base state."""
    if st.session_state["outs"] < 2:
        st.session_state["outs"] += 1
    else:
        # Third out — end of half-inning
        if st.session_state["half"] == "top":
            st.session_state["half"] = "bottom"
        else:
            # End of bottom half → next inning
            st.session_state["half"] = "top"
            st.session_state["inning"] = min(st.session_state["inning"] + 1, 17)
        # Reset for new half — place ghost runner only if Manfred rule is on
        st.session_state["outs"] = 0
        st.session_state["on_1b"] = False
        st.session_state["on_2b"] = bool(st.session_state.get("manfred_runner", True))
        st.session_state["on_3b"] = False


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚾ Controls")
    st.markdown("---")

    MLB_TEAMS = [
        "Angels", "Astros", "Athletics", "Blue Jays", "Braves",
        "Brewers", "Cardinals", "Cubs", "Diamondbacks", "Dodgers",
        "Giants", "Guardians", "Mariners", "Marlins", "Mets",
        "Nationals", "Orioles", "Padres", "Phillies", "Pirates",
        "Rangers", "Rays", "Red Sox", "Reds", "Rockies",
        "Royals", "Tigers", "Twins", "White Sox", "Yankees",
    ]

    st.markdown("**Teams**")
    tc1, tc2 = st.columns(2)
    with tc1:
        away_team = st.selectbox("Away", options=MLB_TEAMS, index=0, key="away_team_sel")
    with tc2:
        home_team = st.selectbox("Home", options=MLB_TEAMS, index=1, key="home_team_sel")

    st.markdown("**Team Strength**")
    strength = st.slider(
        f"{away_team} ◀ weaker / stronger ▶ {home_team}",
        min_value=-1.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="0 = league average. +1 = home ~60% win from tie (~+150 away). -1 = away ~60% from tie (~+150 home).",
    )
    if strength > 0.01:
        st.caption(f"{home_team} favored — offense boosted, {away_team} suppressed")
    elif strength < -0.01:
        st.caption(f"{away_team} favored — offense boosted, {home_team} suppressed")
    else:
        st.caption("League average — no strength adjustment")

    st.markdown("---")
    st.markdown("**Score**")
    score_cols = st.columns(2)
    with score_cols[0]:
        away_score = st.number_input(f"{away_team}", min_value=0, max_value=30, value=0, step=1, key="away_score")
    with score_cols[1]:
        home_score = st.number_input(f"{home_team}", min_value=0, max_value=30, value=0, step=1, key="home_score")
    home_lead = int(home_score - away_score)

    st.markdown("---")
    st.markdown("**Inning**")
    inn_cols = st.columns([2, 1])
    with inn_cols[0]:
        st.selectbox(
            "Inning", options=list(range(10, 18)),
            key="inning",
        )
    with inn_cols[1]:
        st.radio(
            "Half", options=["top", "bottom"],
            format_func=lambda x: "Top" if x == "top" else "Bot",
            key="half",
        )

    st.markdown("---")
    # ── BASE-OUT STATE ────────────────────────────────────────────────────────
    st.markdown("**Base — Out State**")

    # OUT button + outs display side by side
    # Use on_click callback so state is updated BEFORE the next render pass,
    # which means outs icons and base labels always reflect current state.
    out_col, outs_display_col = st.columns([1, 1])
    with out_col:
        st.button("OUT", use_container_width=True, type="primary", on_click=_advance_out)
    with outs_display_col:
        outs_icons = "⬤ " * st.session_state["outs"] + "○ " * (2 - st.session_state["outs"])
        st.markdown(
            f"<div style='font-size:1.4rem;padding-top:6px;text-align:center'>{outs_icons}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── DIAMOND ──────────────────────────────────────────────────────────────
    # Use on_click callbacks so the toggled state is committed before labels render.
    ON  = "🟥"   # occupied
    OFF = "⬜"   # empty

    # Row 1: 2nd base (center top)
    _, mid_col, _ = st.columns([1, 2, 1])
    with mid_col:
        b2_label = f"{ON} 2nd" if st.session_state["on_2b"] else f"{OFF} 2nd"
        st.button(b2_label, use_container_width=True, key="btn_2b",
                  on_click=lambda: st.session_state.update({"on_2b": not st.session_state["on_2b"]}))

    # Row 2: 3rd base (left) and 1st base (right)
    left_col, _, right_col = st.columns([2, 1, 2])
    with left_col:
        b3_label = f"{ON} 3rd" if st.session_state["on_3b"] else f"{OFF} 3rd"
        st.button(b3_label, use_container_width=True, key="btn_3b",
                  on_click=lambda: st.session_state.update({"on_3b": not st.session_state["on_3b"]}))
    with right_col:
        b1_label = f"{ON} 1st" if st.session_state["on_1b"] else f"{OFF} 1st"
        st.button(b1_label, use_container_width=True, key="btn_1b",
                  on_click=lambda: st.session_state.update({"on_1b": not st.session_state["on_1b"]}))

    # Row 3: home plate
    _, hp_col, _ = st.columns([1, 2, 1])
    with hp_col:
        st.markdown("<div style='text-align:center;font-size:1.1rem;color:#aaa'>⬡ Home</div>",
                    unsafe_allow_html=True)

    _, run_btn_col, _ = st.columns([1, 2, 1])
    with run_btn_col:
        st.button("RUN SCORED", use_container_width=True, on_click=_run_scored)

    # State summary in plain English
    _cur_bases = (
        int(st.session_state["on_1b"]),
        int(st.session_state["on_2b"]),
        int(st.session_state["on_3b"]),
    )
    _out_word = {0: "0 outs", 1: "1 out", 2: "2 outs"}[st.session_state["outs"]]
    _base_desc = BASE_STATE_LABELS.get(_cur_bases, "Unknown")
    st.markdown(
        f"<div style='text-align:center;font-size:0.95rem;color:#ccc;margin-top:6px'>"
        f"{_base_desc}, {_out_word}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.session_state["manfred_runner"] = st.toggle(
        "Manfred Runner (ghost runner on 2nd)",
        value=st.session_state["manfred_runner"],
        help="When on, each new extra half-inning starts with a runner on 2nd. Turning off models pre-2020 extra innings rules.",
    )


# ── READ FINAL STATE FROM SESSION STATE ──────────────────────────────────────
inning = st.session_state["inning"]
half   = st.session_state["half"]
outs   = st.session_state["outs"]
bases  = (
    int(st.session_state["on_1b"]),
    int(st.session_state["on_2b"]),
    int(st.session_state["on_3b"]),
)


# ── COMPUTE ───────────────────────────────────────────────────────────────────
result = compute_win_probabilities(
    inning=inning,
    half=half,
    home_lead=home_lead,
    bases=bases,
    outs=outs,
    strength=strength,
    manfred_runner=st.session_state["manfred_runner"],
)

home_win = result["home_win"]
away_win = result["away_win"]
inning_reach = result["inning_reach_probs"]
base_state_name = BASE_STATE_LABELS.get(bases, "Unknown")
# Run distribution shown reflects the batting team's tilted distribution
_theta = strength * 0.10
_batting_team = away_team if half == "top" else home_team
_batting_theta = -_theta if half == "top" else +_theta
current_run_dist = tilt_dist(get_run_distribution(bases, outs), _batting_theta)


# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("Extra Innings Pricer")

if home_lead == 0:
    lead_str = "Tied"
elif home_lead > 0:
    lead_str = f"{home_team} +{home_lead}"
else:
    lead_str = f"{away_team} +{abs(home_lead)}"

st.markdown(
    f"### {inning_label(inning, half)}  ·  {away_team} {away_score} – {home_score} {home_team}"
    f"  ·  {lead_str}  ·  {outs} out(s)  ·  {base_state_name}"
)
st.markdown("---")


# ── WIN PROBABILITIES ─────────────────────────────────────────────────────────
col_away, col_home = st.columns(2)

_away_bg = team_color(away_team)
_home_bg = team_color(home_team)
_away_fg = text_color(_away_bg)
_home_fg = text_color(_home_bg)

with col_away:
    st.markdown(
        f'<div class="win-box" style="background-color:{_away_bg};color:{_away_fg}">'
        f'{away_team}<br>{away_win*100:.1f}%<br>'
        f'<span style="font-size:0.85rem;font-weight:normal">Win Probability</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='text-align:center'>Fair ML: <b>{american_odds(away_win)}</b></p>",
        unsafe_allow_html=True,
    )

with col_home:
    st.markdown(
        f'<div class="win-box" style="background-color:{_home_bg};color:{_home_fg}">'
        f'{home_team}<br>{home_win*100:.1f}%<br>'
        f'<span style="font-size:0.85rem;font-weight:normal">Win Probability</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='text-align:center'>Fair ML: <b>{american_odds(home_win)}</b></p>",
        unsafe_allow_html=True,
    )

st.markdown("---")


# ── RUN DISTRIBUTION  +  INNING REACH ────────────────────────────────────────
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Run Distribution — This Half-Inning")
    re24_base = RE24.get((bases, outs), 0.0)
    re24_tilted = sum(r * p for r, p in enumerate(current_run_dist))
    strength_note = f"  |  Strength adj: {_batting_theta:+.2f}θ → xR {re24_tilted:.3f}" if strength != 0.0 else ""
    st.caption(
        f"Batting: **{_batting_team}**  |  {base_state_name}, {outs} out(s)  |  "
        f"Base xR: {re24_base:.3f}{strength_note}"
    )

    run_labels = [str(k) for k in range(10)]
    run_chart_df = pd.DataFrame({
        "Runs": run_labels,
        "Probability": current_run_dist[:10],
    })
    chart = (
        alt.Chart(run_chart_df)
        .mark_bar()
        .encode(
            x=alt.X("Runs:O", sort=run_labels, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format="%")),
            tooltip=[
                alt.Tooltip("Runs:O", title="Runs"),
                alt.Tooltip("Probability:Q", title="Probability", format=".1%"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)

    run_table = pd.DataFrame({
        "Runs": run_labels,
        "P(exactly)": [f"{p*100:.2f}%" for p in current_run_dist[:10]],
        "P(≥ this many)": [
            f"{(1 - current_run_dist[:k].sum())*100:.2f}%"
            for k in range(10)
        ],
    }).set_index("Runs")
    st.dataframe(run_table, use_container_width=True, height=220)

with col_right:
    st.subheader("Game Length Markets")
    st.caption("P(game still active / reaches each inning)")

    reach_rows = []
    for inn in range(10, 18):
        if inn < inning:
            p = 1.0   # already reached — game is here or past it
        else:
            p = inning_reach.get(inn, 0.0)
        reach_rows.append({
            "Inning": ordinal(inn),
            "P(Reached)": p,
            "P(Reached) %": f"{p*100:.1f}%",
            "Fair ML": american_odds(p) if 0.005 < p < 0.995 else "N/A",
        })

    def style_prob(val):
        try:
            num = float(str(val).replace("%", ""))
        except Exception:
            return ""
        if num >= 60:   return "background-color:#1e8449;color:white"
        elif num >= 40: return "background-color:#27ae60;color:white"
        elif num >= 20: return "background-color:#f39c12;color:black"
        elif num >= 5:  return "background-color:#e74c3c;color:white"
        else:           return "background-color:#7b241c;color:white"

    display_df = pd.DataFrame(reach_rows)[["Inning", "P(Reached) %", "Fair ML"]].set_index("Inning")
    st.dataframe(
        display_df.style.applymap(style_prob, subset=["P(Reached) %"]),
        use_container_width=True,
        height=320,
    )

    st.markdown("---")
    st.subheader(f"Outcomes — {ordinal(inning)}")

    p_goes_next = inning_reach.get(inning + 1, 0.0)
    p_ends_this = inning_reach.get(inning, 1.0) - p_goes_next
    p_away_wins_this = p_ends_this * away_win
    p_home_wins_this = p_ends_this * home_win

    _home_lbl = f"{home_team} wins {ordinal(inning)}"
    _next_lbl = f"Goes to {ordinal(inning + 1)}"
    _away_lbl = f"{away_team} wins {ordinal(inning)}"

    pie_df = pd.DataFrame({
        "Outcome": [_home_lbl, _next_lbl, _away_lbl],
        "Probability": [p_home_wins_this, p_goes_next, p_away_wins_this],
        "sort_order": [0, 1, 2],
    })

    pie_chart = (
        alt.Chart(pie_df)
        .mark_arc(innerRadius=40)
        .encode(
            theta=alt.Theta("Probability:Q"),
            order=alt.Order("sort_order:Q"),
            color=alt.Color(
                "Outcome:N",
                scale=alt.Scale(
                    domain=[_home_lbl, _next_lbl, _away_lbl],
                    range=[_home_bg, NEUTRAL_NEXT, _away_bg],
                ),
                legend=alt.Legend(orient="bottom", labelLimit=200),
            ),
            tooltip=[
                alt.Tooltip("Outcome:N"),
                alt.Tooltip("Probability:Q", format=".1%"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(pie_chart, use_container_width=True)


st.markdown("---")


# ── RE24 REFERENCE ────────────────────────────────────────────────────────────
with st.expander("RE24 Reference Table (empirical expected runs)", expanded=False):
    re24_rows = []
    for bases_key, label in BASE_STATE_LABELS.items():
        row = {"Base State": label}
        for o in [0, 1, 2]:
            row[f"{o} Outs"] = f"{RE24.get((bases_key, o), 0.0):.3f}"
        re24_rows.append(row)
    st.dataframe(
        pd.DataFrame(re24_rows).set_index("Base State"),
        use_container_width=True,
    )

with st.expander("Full Distribution Table (all 24 states)", expanded=False):
    dist_rows = []
    for bases_key, label in BASE_STATE_LABELS.items():
        for o in [0, 1, 2]:
            dist = RUN_DIST.get((bases_key, o), [])
            row = {"State": f"{label}, {o} outs"}
            for k in range(11):
                row[f"{k}R"] = f"{dist[k]*100:.1f}%" if k < len(dist) else "—"
            dist_rows.append(row)
    st.dataframe(
        pd.DataFrame(dist_rows).set_index("State"),
        use_container_width=True,
    )

with st.expander("Methodology & Notes", expanded=False):
    st.markdown("""
**How This Works**

1. **Empirical Run Distributions** — Each of the 24 base-out states has a full probability
   distribution P(0 runs), P(1 run), ..., P(10+ runs) derived directly from 2.1 million
   Statcast plate appearances across the 2021, 2022, and 2023 MLB regular seasons
   (innings 1–9 only, to exclude ghost-runner distortion).

2. **Extra Innings Ghost Runner** — Each new extra inning starts with runner on 2nd, 0 outs.
   The (2nd base, 0 outs) empirical distribution is used: ~39.7% chance of scoring 0,
   ~31.7% chance of exactly 1 run, etc. Adjust the base state mid-inning as needed.

3. **Win Probability** — Score distributions are propagated inning-by-inning by convolving
   the run distribution with the current score probability mass. At the end of each bottom
   half, home leads resolve as home wins and away leads resolve as away wins; ties continue.

4. **Game Length Markets** — P(reaches inning N) is the total probability mass still
   in the tied/unresolved state at the start of inning N. Shown as fair moneyline (no margin).

**Limitations**
- Team strength adjustments not included — uses league-average offense from 2021-2023.
- Manager decisions (IBB, PH, pitching changes) are not modeled.
- Run distributions are right-truncated at 10 runs; extreme multi-run innings are grouped.
    """)
