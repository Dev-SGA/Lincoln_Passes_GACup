import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from streamlit_image_coordinates import streamlit_image_coordinates

# ==========================
# Page Configuration
# ==========================
st.set_page_config(layout="wide", page_title="Match Analysis Dashboard")
st.title("Match Analysis Dashboard")
st.caption("Click on the maps to inspect events.")

# ==========================
# Configuration
# ==========================
FINAL_THIRD_LINE_X = 80
GOAL_X = 120
GOAL_Y = 40

PROG_OWN_HALF_THRESHOLD = 24
PROG_CROSS_HALF_THRESHOLD = 12
PROG_OPP_HALF_THRESHOLD = 8

COLOR_PASS_SUCCESS = "#8E8E8E"
COLOR_PASS_FAIL = "#F2A3A3"
COLOR_PROGRESSIVE = "#2F80ED"

FIG_W, FIG_H = 8, 5.6
FIG_DPI = 110

# ==========================
# Pass Data
# ==========================
df_passes = pd.DataFrame([
    ("PASS WON",  65.15, 26.54, 99.40, 22.38, None),
    ("PASS LOST", 62.99,  7.09, 71.30,  2.93, None),
    ("PASS LOST", 46.70, 45.82, 42.88, 67.10, None),
    ("PASS LOST", 93.91, 49.81, 100.23, 58.12, None),
], columns=["type", "x_start", "y_start", "x_end", "y_end", "video"])
df_passes["number"] = np.arange(1, len(df_passes) + 1)

# ==========================
# Duel Data
# ==========================
df_duels = pd.DataFrame([
    ("FOUL COMMITTED",        88.92,  4.09, None),
    ("FOUL COMMITTED",        30.08, 22.05, None),
    ("OFFENSIVE DUEL LOST",  106.71, 25.04, None),
    ("OFFENSIVE DUEL LOST",   27.09, 22.88, None),
    ("DEFENSIVE DUEL WON",    26.09, 29.03, None),
    ("DEFENSIVE DUEL WON",    41.38, 22.21, None),
], columns=["type", "x", "y", "video"])
df_duels["number"] = np.arange(1, len(df_duels) + 1)

# ==========================
# Helpers
# ==========================
def has_video_value(v) -> bool:
    return pd.notna(v) and str(v).strip() != ""


def distance_to_goal(x, y):
    return np.sqrt((GOAL_X - x) ** 2 + (GOAL_Y - y) ** 2)


def is_progressive_pass(x_start, y_start, x_end, y_end) -> bool:
    start_dist = distance_to_goal(x_start, y_start)
    end_dist = distance_to_goal(x_end, y_end)
    gain = start_dist - end_dist

    start_own = x_start < 60
    end_own = x_end < 60
    end_opp = x_end >= 60
    start_opp = x_start >= 60

    if start_own and end_own:
        return gain >= PROG_OWN_HALF_THRESHOLD
    elif start_own and end_opp:
        return gain >= PROG_CROSS_HALF_THRESHOLD
    elif start_opp and end_opp:
        return gain >= PROG_OPP_HALF_THRESHOLD
    return False


df_passes["progressive"] = df_passes.apply(
    lambda r: is_progressive_pass(r["x_start"], r["y_start"], r["x_end"], r["y_end"]),
    axis=1,
)

# ==========================
# Pass Stats
# ==========================
def compute_pass_stats(df: pd.DataFrame) -> dict:
    total = len(df)
    won = int(df["type"].str.contains("WON", case=False).sum())
    lost = total - won
    acc = (won / total * 100) if total else 0

    prog_total = int(df["progressive"].sum())
    prog_won = int((df["progressive"] & df["type"].str.contains("WON", case=False)).sum())
    prog_acc = (prog_won / prog_total * 100) if prog_total else 0

    ft = df["x_end"] >= FINAL_THIRD_LINE_X
    ft_total = int(ft.sum())
    ft_won = int((ft & df["type"].str.contains("WON", case=False)).sum())
    ft_lost = ft_total - ft_won
    ft_acc = (ft_won / ft_total * 100) if ft_total else 0

    return dict(
        total=total, won=won, lost=lost, acc=acc,
        prog_total=prog_total, prog_won=prog_won, prog_acc=prog_acc,
        ft_total=ft_total, ft_won=ft_won, ft_lost=ft_lost, ft_acc=ft_acc,
    )

# ==========================
# Duel Stats
# ==========================
def compute_duel_stats(df: pd.DataFrame) -> dict:
    is_duel = df["type"].str.contains("DUEL|AERIAL", case=False, na=False)
    is_won = df["type"].str.contains("WON", case=False, na=False)
    is_foul = df["type"].str.contains("FOUL", case=False, na=False)

    duels = df[is_duel]
    d_total = len(duels)
    d_won = int((duels["type"].str.contains("WON", case=False)).sum()) if d_total else 0
    d_lost = d_total - d_won
    d_rate = (d_won / d_total * 100) if d_total else 0

    ft_mask = df["x"] > 80
    ft_duels = df[ft_mask & is_duel]
    ft_total = len(ft_duels)
    ft_won = int((ft_duels["type"].str.contains("WON", case=False)).sum())
    ft_lost = ft_total - ft_won
    ft_rate = (ft_won / ft_total * 100) if ft_total else 0

    fouls = int(is_foul.sum())

    return dict(
        d_total=d_total, d_won=d_won, d_lost=d_lost, d_rate=d_rate,
        ft_total=ft_total, ft_won=ft_won, ft_lost=ft_lost, ft_rate=ft_rate,
        fouls=fouls,
    )

# ==========================
# Draw Pass Map
# ==========================
def draw_pass_map(df: pd.DataFrame):
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#f5f5f5", line_color="#4a4a4a")
    fig, ax = pitch.draw(figsize=(FIG_W, FIG_H))
    fig.set_dpi(FIG_DPI)

    ax.axvline(x=FINAL_THIRD_LINE_X, color="#FFD54F", linewidth=1.2, alpha=0.25)

    for _, row in df.iterrows():
        is_lost = "LOST" in row["type"].upper()
        is_prog = bool(row["progressive"]) and not is_lost
        has_vid = has_video_value(row["video"])

        if is_lost:
            color, alpha = COLOR_PASS_FAIL, 0.45
        elif is_prog:
            color, alpha = COLOR_PROGRESSIVE, 0.82
        else:
            color, alpha = COLOR_PASS_SUCCESS, 0.75

        pitch.arrows(
            row["x_start"], row["y_start"], row["x_end"], row["y_end"],
            color=color, width=1.55, headwidth=2.25, headlength=2.25,
            ax=ax, zorder=3, alpha=alpha,
        )
        if has_vid:
            pitch.scatter(
                row["x_start"], row["y_start"], s=95, marker="o",
                facecolors="none", edgecolors="#FFD54F", linewidths=2.0,
                ax=ax, zorder=4,
            )
        pitch.scatter(
            row["x_start"], row["y_start"], s=45, marker="o", color=color,
            edgecolors="white", linewidths=0.8, ax=ax, zorder=5, alpha=alpha,
        )

    ax.set_title("Pass Map", fontsize=12)

    legend_elements = [
        Line2D([0], [0], color=COLOR_PASS_SUCCESS, lw=2.5, label="Successful Pass"),
        Line2D([0], [0], color=COLOR_PASS_FAIL, lw=2.5, label="Unsuccessful Pass"),
        Line2D([0], [0], color=COLOR_PROGRESSIVE, lw=2.5, label="Progressive Pass (Opta)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="#FFD54F", markeredgewidth=2, markersize=7, label="Has video"),
    ]
    legend = ax.legend(
        handles=legend_elements, loc="upper left", bbox_to_anchor=(0.01, 0.99),
        frameon=True, facecolor="white", edgecolor="#cccccc", shadow=False,
        fontsize="x-small", labelspacing=0.5, borderpad=0.5,
    )
    legend.get_frame().set_alpha(1.0)

    arrow = FancyArrowPatch(
        (0.45, 0.05), (0.55, 0.05), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=15, linewidth=2, color="#333333",
    )
    fig.patches.append(arrow)
    fig.text(0.5, 0.02, "Attack Direction", ha="center", va="center",
             fontsize=9, color="#333333")

    fig.tight_layout()
    fig.canvas.draw()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI)
    buf.seek(0)
    img = Image.open(buf)
    return img, ax, fig

# ==========================
# Draw Duel Map
# ==========================
def get_duel_style(event_type):
    t = event_type.upper()
    if "FOUL" in t:
        # Falta cometida → vermelho/laranja
        return "P", (0.90, 0.30, 0.10, 1.00), 130, 0.8
    if "OFFENSIVE" in t:
        if "WON" in t:
            return "o", (0.10, 0.85, 0.10, 0.95), 110, 0.8
        return "x", (0.95, 0.15, 0.15, 0.95), 120, 3.0
    if "DEFENSIVE" in t:
        if "WON" in t:
            return "s", (0.00, 0.60, 0.00, 0.90), 110, 0.8
        return "D", (0.70, 0.00, 0.00, 0.90), 110, 2.5
    if "AERIAL" in t:
        if "WON" in t:
            return "^", (0.20, 0.50, 0.95, 0.90), 120, 0.8
        return "v", (0.55, 0.20, 0.85, 0.85), 120, 0.8
    return "o", (0.5, 0.5, 0.5, 0.8), 90, 0.5


def draw_duel_map(df: pd.DataFrame):
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#f8f8f8", line_color="#4a4a4a")
    fig, ax = pitch.draw(figsize=(FIG_W, FIG_H))
    fig.set_dpi(FIG_DPI)

    for _, row in df.iterrows():
        has_vid = has_video_value(row["video"])
        marker, color, size, lw = get_duel_style(row["type"])
        edge = "black" if has_vid else "none"

        pitch.scatter(
            row["x"], row["y"], marker=marker, s=size, color=color,
            edgecolors=edge, linewidths=lw, ax=ax, zorder=3,
        )

    ax.set_title("Duel Map", fontsize=12)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Offensive Won",
               markerfacecolor=(0.10, 0.85, 0.10, 0.95), markersize=10, linestyle="None"),
        Line2D([0], [0], marker="x", color=(0.95, 0.15, 0.15, 0.95),
               label="Offensive Lost", markersize=10, markeredgewidth=2.5, linestyle="None"),
        Line2D([0], [0], marker="s", color="w", label="Defensive Won",
               markerfacecolor=(0.00, 0.60, 0.00, 0.90), markersize=9, linestyle="None"),
        Line2D([0], [0], marker="D", color="w", label="Defensive Lost",
               markerfacecolor=(0.70, 0.00, 0.00, 0.90), markersize=9, linestyle="None"),
        Line2D([0], [0], marker="P", color="w", label="Foul Committed",
               markerfacecolor=(0.90, 0.30, 0.10, 1.00), markersize=10, linestyle="None"),
    ]
    legend = ax.legend(
        handles=legend_elements, loc="upper left", bbox_to_anchor=(0.01, 0.99),
        frameon=True, facecolor="white", edgecolor="#333333", fontsize="x-small",
        title="Events", title_fontsize="small", labelspacing=0.8, borderpad=0.8,
        framealpha=0.95,
    )
    legend.get_title().set_fontweight("bold")

    arrow = FancyArrowPatch(
        (0.45, 0.05), (0.55, 0.05), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=15, linewidth=2, color="#333333",
    )
    fig.patches.append(arrow)
    fig.text(0.5, 0.02, "Attack Direction", ha="center", va="center",
             fontsize=9, color="#333333")

    fig.tight_layout()
    fig.canvas.draw()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI)
    buf.seek(0)
    img = Image.open(buf)
    return img, ax, fig

# ==========================
# Layout — Two maps side by side
# ==========================
col_pass, col_duel = st.columns(2)

# ---------- Pass Map ----------
with col_pass:
    st.subheader("Passes")

    pass_img, pass_ax, pass_fig = draw_pass_map(df_passes)
    pass_click = streamlit_image_coordinates(pass_img, key="pass_map", width=700)

    selected_pass = None
    if pass_click is not None:
        rw, rh = pass_img.size
        dw, dh = pass_click["width"], pass_click["height"]
        px = pass_click["x"] * (rw / dw)
        py = pass_click["y"] * (rh / dh)
        mpl_py = rh - py
        fx, fy = pass_ax.transData.inverted().transform((px, mpl_py))

        tmp = df_passes.copy()
        tmp["dist"] = np.sqrt((tmp["x_start"] - fx) ** 2 + (tmp["y_start"] - fy) ** 2)
        cands = tmp[tmp["dist"] < 5.0]
        if not cands.empty:
            selected_pass = cands.loc[cands["dist"].idxmin()]

    plt.close(pass_fig)

    st.divider()
    if selected_pass is None:
        st.info("Click a start dot to inspect the pass.")
    else:
        st.success(f"Pass #{int(selected_pass['number'])} — {selected_pass['type']}")
        st.write(
            f"Start: ({selected_pass['x_start']:.2f}, {selected_pass['y_start']:.2f})  \n"
            f"End: ({selected_pass['x_end']:.2f}, {selected_pass['y_end']:.2f})"
        )
        st.write(f"Progressive: {'Yes' if selected_pass['progressive'] else 'No'}")
        if has_video_value(selected_pass["video"]):
            try:
                st.video(selected_pass["video"])
            except Exception:
                st.error(f"Video not found: {selected_pass['video']}")

# ---------- Duel Map ----------
with col_duel:
    st.subheader("Duels & Fouls")

    duel_img, duel_ax, duel_fig = draw_duel_map(df_duels)
    duel_click = streamlit_image_coordinates(duel_img, key="duel_map", width=700)

    selected_duel = None
    if duel_click is not None:
        rw, rh = duel_img.size
        dw, dh = duel_click["width"], duel_click["height"]
        px = duel_click["x"] * (rw / dw)
        py = duel_click["y"] * (rh / dh)
        mpl_py = rh - py
        fx, fy = duel_ax.transData.inverted().transform((px, mpl_py))

        tmp = df_duels.copy()
        tmp["dist"] = np.sqrt((tmp["x"] - fx) ** 2 + (tmp["y"] - fy) ** 2)
        cands = tmp[tmp["dist"] < 5.0]
        if not cands.empty:
            selected_duel = cands.loc[cands["dist"].idxmin()]

    plt.close(duel_fig)

    st.divider()
    if selected_duel is None:
        st.info("Click a marker to inspect the duel.")
    else:
        st.success(f"Event #{int(selected_duel['number'])} — {selected_duel['type']}")
        st.write(f"Position: ({selected_duel['x']:.2f}, {selected_duel['y']:.2f})")
        if has_video_value(selected_duel["video"]):
            try:
                st.video(selected_duel["video"])
            except Exception:
                st.error(f"Video not found: {selected_duel['video']}")

# ==========================
# Statistics Row
# ==========================
st.divider()

col_pass_stats, col_duel_stats = st.columns(2)

pass_stats = compute_pass_stats(df_passes)
duel_stats = compute_duel_stats(df_duels)

with col_pass_stats:
    st.subheader("Pass Statistics")

    st.markdown("**Overall**")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Total", pass_stats["total"])
    a2.metric("Successful", pass_stats["won"])
    a3.metric("Unsuccessful", pass_stats["lost"])
    a4.metric("Accuracy", f"{pass_stats['acc']:.1f}%")

    st.divider()

    st.markdown("**Progressive Passes**")
    b1, b2, b3 = st.columns(3)
    b1.metric("Total", pass_stats["prog_total"])
    b2.metric("Successful", pass_stats["prog_won"])
    b3.metric("Accuracy", f"{pass_stats['prog_acc']:.1f}%")

    st.divider()

    st.markdown("**Final Third**")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Total", pass_stats["ft_total"])
    e2.metric("Successful", pass_stats["ft_won"])
    e3.metric("Unsuccessful", pass_stats["ft_lost"])
    e4.metric("Accuracy", f"{pass_stats['ft_acc']:.1f}%")

with col_duel_stats:
    st.subheader("Duel Statistics")

    st.markdown("**Overall Duels**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", duel_stats["d_total"])
    c2.metric("Won", duel_stats["d_won"])
    c3.metric("Lost", duel_stats["d_lost"])
    c4.metric("Success %", f"{duel_stats['d_rate']:.1f}%")

    st.divider()

    st.markdown("**Final Third Duels**")
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Total", duel_stats["ft_total"])
    f2.metric("Won", duel_stats["ft_won"])
    f3.metric("Lost", duel_stats["ft_lost"])
    f4.metric("Success %", f"{duel_stats['ft_rate']:.1f}%")

    st.divider()

    st.markdown("**Fouls Committed**")
    st.metric("Total", duel_stats["fouls"])
