"""
dashboard/app.py
================
Football Analytics — Streamlit Dashboard
Phase 5 Final Version: All features integrated.
Run: streamlit run dashboard/app.py
"""

import io
import sys
import tempfile
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Football Analytics Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.metric-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
    border: 1px solid #30363d; border-radius: 12px;
    padding: 18px 22px; text-align: center; transition: border-color 0.2s;
}
.metric-card:hover { border-color: #1E90FF; }
.metric-label { color: #8b949e; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; }
.metric-value { color: #e6edf3; font-size: 1.8rem; font-weight: 700; line-height: 1; }
.metric-unit  { color: #1E90FF; font-size: 0.75rem; margin-top: 4px; }

.team-badge { display: inline-block; padding: 4px 14px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; }
.team1 { background: rgba(60,80,255,0.2); color: #4d7cff; border: 1px solid #4d7cff; }
.team2 { background: rgba(255,120,40,0.2); color: #ff7828; border: 1px solid #ff7828; }

.section-header {
    font-size: 1.1rem; font-weight: 600; color: #e6edf3;
    border-left: 3px solid #1E90FF; padding-left: 10px; margin: 20px 0 12px 0;
}
.disclaimer {
    background: rgba(255,165,0,0.1); border: 1px solid rgba(255,165,0,0.4);
    border-radius: 8px; padding: 10px 16px; color: #ffa500; font-size: 0.82rem; margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def metric_card(col, label, value, unit=""):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-unit">{unit}</div>
    </div>""", unsafe_allow_html=True)


def _encode_video_to_bytes(frames, fps: float) -> bytes:
    """Encode a list of BGR frames to MP4 bytes in memory."""
    if not frames:
        return b""
    h, w = frames[0].shape[:2]
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    writer = cv2.VideoWriter(
        tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    for f in frames:
        writer.write(f)
    writer.release()
    with open(tmp.name, "rb") as fh:
        return fh.read()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# ⚽")
    st.title("Football Analytics")
    st.caption("End-to-end player tracking & performance insights")
    st.divider()

    st.subheader("📤 Input")
    source = st.radio("Video source", ["Upload file", "Local path"], key="vid_source")

    video_path_str = None
    if source == "Upload file":
        uploaded = st.file_uploader("Choose a video", type=["mp4", "mkv", "avi"])
        if uploaded:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(uploaded.read())
            tmp.flush()
            video_path_str = tmp.name
    else:
        video_path_str = st.text_input(
            "Local video path",
            placeholder="e.g. data/soccernet/tracking/train/SNMOT-060/img1/%06d.jpg",
            key="local_path",
        )

    st.divider()
    st.subheader("⚙️ Settings")
    all_frames = st.checkbox("Process ALL frames", value=False, key="all_frames",
                             help="Tick to process the entire video. For a 20-min clip this can be 30 000+ frames — expect long runtimes.")
    max_frames = None if all_frames else st.number_input(
        "Max frames to process", min_value=30, max_value=36_000,
        value=300, step=150, key="max_frames",
        help="30 000 ≈ 20 min @ 25 fps. Reduce for faster previews.",
    )
    conf       = st.slider("Detection confidence",  0.10, 0.90, 0.35, step=0.05, key="conf")
    run_btn    = st.button("▶ Run Analysis", type="primary", width='stretch', key="run")
    st.divider()
    st.caption("Football Tracker v2.0  •  YOLOv8 + ByteTrack + mplsoccer")


# ── Hero Header ───────────────────────────────────────────────────────────────
c_logo, c_title = st.columns([1, 9])
with c_logo:
    st.markdown("# ⚽")
with c_title:
    st.markdown("## Football Player Analytics Dashboard")
    st.caption("Upload a match video → detection → tracking → full tactical analysis.")
st.divider()


# ── Session state ─────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None
if "video_bytes" not in st.session_state:
    st.session_state.video_bytes = None


# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_btn:
    if not video_path_str:
        st.error("Please provide a video source first.")
    else:
        from dashboard.pipeline_runner import run_pipeline

        prog = st.progress(0.0, text="Initialising pipeline…")

        def _cb(pct):
            prog.progress(min(pct, 1.0), text=f"Processing… {int(pct*100)}%")

        with st.spinner("Running analytics pipeline…"):
            results = run_pipeline(
                video_path=video_path_str,
                max_frames=max_frames,
                conf_threshold=conf,
                progress_callback=_cb,
            )

        st.session_state.results = results
        st.session_state.video_bytes = None   # reset cached video
        prog.progress(1.0, text="✅ Done!")
        st.success(f"Processed {results['total_frames']} frames successfully!")


# ── Guard ─────────────────────────────────────────────────────────────────────
results = st.session_state.results
if results is None:
    st.markdown("""
    <div style='text-align:center; padding:60px 20px; color:#8b949e;'>
        <div style='font-size:4rem;'>⚽</div>
        <h3 style='color:#8b949e;'>Ready to analyse</h3>
        <p>Upload a video in the sidebar and click <strong>Run Analysis</strong>.</p>
    </div>""", unsafe_allow_html=True)
    st.stop()

metrics        = results["metrics"]
formations     = results["formations"]
pressing       = results["pressing"]
possession_pct = results["possession_pct"]


# ════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════
tab_overview, tab_video, tab_tactical, tab_players, tab_export = st.tabs([
    "📊 Overview", "🎬 Annotated Video",
    "⚙️ Tactical", "👤 Player Stats", "📥 Export",
])


# ── TAB 1: OVERVIEW ──────────────────────────────────────────────────────────
with tab_overview:
    st.markdown('<div class="section-header">Match Summary</div>', unsafe_allow_html=True)

    n_players    = len(metrics)
    total_dist   = sum(m["Distance (km)"] for m in metrics)
    total_sprint = sum(m["Sprints"] for m in metrics)
    max_spd      = max((m["Max Speed (km/h)"] for m in metrics), default=0)
    ev_sum       = results.get("event_summary", {})

    c1, c2, c3, c4, c5 = st.columns(5)
    metric_card(c1, "Frames Processed", results["total_frames"], "frames")
    metric_card(c2, "Players Tracked",  n_players,               "players")
    metric_card(c3, "Total Distance",   f"{total_dist:.1f}",     "km")
    metric_card(c4, "Total Sprints",    total_sprint,            "events")
    metric_card(c5, "Peak Speed",       f"{max_spd:.1f}",        "km/h")

    st.markdown('<div class="section-header">Team Overview</div>', unsafe_allow_html=True)
    t1col, t2col, posc = st.columns(3)

    with t1col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Team 1 Formation</div>
            <div class="metric-value">{formations.get(1,'—')}</div>
            <div class="metric-unit team-badge team1">Team 1</div>
        </div>""", unsafe_allow_html=True)

    with t2col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Team 2 Formation</div>
            <div class="metric-value">{formations.get(2,'—')}</div>
            <div class="metric-unit team-badge team2">Team 2</div>
        </div>""", unsafe_allow_html=True)

    with posc:
        p1, p2 = possession_pct.get(1, 50), possession_pct.get(2, 50)
        fig_pie = go.Figure(go.Pie(
            labels=["Team 1", "Team 2"], values=[p1, p2], hole=0.65,
            marker_colors=["#3c50ff", "#ff7828"], textinfo="percent",
        ))
        fig_pie.update_layout(
            showlegend=True, height=200,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", font_color="#e6edf3",
            legend=dict(orientation="h", y=-0.15),
            annotations=[dict(text="Ball<br>Possession", x=0.5, y=0.5,
                               font_size=12, showarrow=False, font_color="#8b949e")],
        )
        st.plotly_chart(fig_pie, width='stretch', key="pie_possession")

    # Pressing bars
    st.markdown('<div class="section-header">Pressing Intensity</div>', unsafe_allow_html=True)
    press_fig = go.Figure()
    press_fig.add_bar(name="Team 1", x=["Team 1"], y=[pressing.get(1, 0)], marker_color="#3c50ff")
    press_fig.add_bar(name="Team 2", x=["Team 2"], y=[pressing.get(2, 0)], marker_color="#ff7828")
    press_fig.update_layout(
        yaxis_title="Pressing Intensity", height=240,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e6edf3", yaxis=dict(range=[0, 100], gridcolor="#21262d"),
        xaxis=dict(gridcolor="#21262d"), showlegend=False, bargap=0.5,
    )
    st.plotly_chart(press_fig, width='stretch', key="press_bar")

    # Event summary row
    if ev_sum:
        st.markdown('<div class="section-header">Possession Events</div>', unsafe_allow_html=True)
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Total Events",   ev_sum.get("total_events", 0))
        e2.metric("Team 1 Passes",  ev_sum.get("team1_passes", 0))
        e3.metric("Team 2 Passes",  ev_sum.get("team2_passes", 0))
        e4.metric("Interceptions",  ev_sum.get("interceptions", 0))


# ── TAB 2: ANNOTATED VIDEO ────────────────────────────────────────────────────
with tab_video:
    st.markdown('<div class="section-header">Annotated Frames Preview</div>', unsafe_allow_html=True)
    frames = results.get("frames", [])

    if frames:
        st.info(f"Showing {len(frames)} processed frames.")
        idx = st.slider("Frame", 0, len(frames) - 1, 0, key="frame_slider")
        frame_rgb = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption=f"Frame {idx+1}/{len(frames)}", width='stretch')

        st.markdown('<div class="section-header">Download Annotated Video (Task 5.2)</div>', unsafe_allow_html=True)

        # Encode once and cache in session state
        if st.session_state.video_bytes is None:
            with st.spinner("Encoding MP4…"):
                st.session_state.video_bytes = _encode_video_to_bytes(
                    frames, float(results.get("fps", 25))
                )

        if st.session_state.video_bytes:
            st.download_button(
                label="📥 Download Annotated MP4",
                data=st.session_state.video_bytes,
                file_name="football_analysis_annotated.mp4",
                mime="video/mp4",
                width='stretch',
                key="dl_video",
            )
            col_info = st.columns(3)
            col_info[0].metric("Frames",  len(frames))
            col_info[1].metric("FPS",     int(results.get("fps", 25)))
            col_info[2].metric("Size",    f"{len(st.session_state.video_bytes)//1024} KB")
    else:
        st.warning("No frames were processed.")





# ── TAB 4: TACTICAL ──────────────────────────────────────────────────────────
with tab_tactical:
    st.markdown('<div class="section-header">Tactical Analysis</div>', unsafe_allow_html=True)

    tc1, tc2 = st.columns(2)
    with tc1:
        st.caption("🟡 Pressing Timeline")
        if results.get("pressing_chart_path") and Path(results["pressing_chart_path"]).exists():
            st.image(results["pressing_chart_path"], width='stretch')
    with tc2:
        st.caption("⚽ Possession by Zone")
        if results.get("possession_chart_path") and Path(results["possession_chart_path"]).exists():
            st.image(results["possession_chart_path"], width='stretch')

    # Pass Map
    st.markdown('<div class="section-header">🗺️ Pass Map & Possession Events</div>', unsafe_allow_html=True)
    ev_sum = results.get("event_summary", {})
    if ev_sum:
        pm1, pm2, pm3, pm4 = st.columns(4)
        pm1.metric("Total Events",  ev_sum.get("total_events", 0))
        pm2.metric("Team 1 Passes", ev_sum.get("team1_passes", 0))
        pm3.metric("Team 2 Passes", ev_sum.get("team2_passes", 0))
        pm4.metric("Interceptions", ev_sum.get("interceptions", 0))

    if results.get("pass_map_path") and Path(results["pass_map_path"]).exists():
        st.image(results["pass_map_path"], width='stretch')
    else:
        st.info("Pass map requires ball detection in at least 2 frames.")

    # Voronoi
    st.markdown('<div class="section-header">🗺️ Voronoi Pitch Control</div>', unsafe_allow_html=True)
    pitch_ctrl = results.get("pitch_control", {})
    if pitch_ctrl:
        v1, v2 = st.columns(2)
        v1.metric("Team 1 Pitch Control", f"{pitch_ctrl.get('team1_pct', 0):.1f}%")
        v2.metric("Team 2 Pitch Control", f"{pitch_ctrl.get('team2_pct', 0):.1f}%")

    if results.get("voronoi_path") and Path(results["voronoi_path"]).exists():
        st.image(results["voronoi_path"], width='stretch')
    else:
        st.info("Voronoi requires at least 3 players with valid world coordinates.")

    # Offside
    st.markdown('<div class="section-header">🚩 Indicative Offside Lines</div>', unsafe_allow_html=True)
    st.markdown('<div class="disclaimer">⚠️ INDICATIVE ONLY — Not for official use.</div>', unsafe_allow_html=True)
    if results.get("offside_img_path") and Path(results["offside_img_path"]).exists():
        st.image(results["offside_img_path"], width='stretch')

    alerts = results.get("offside_alerts", {})
    if alerts:
        st.markdown('<div class="section-header">Offside Alert Counts</div>', unsafe_allow_html=True)
        alert_df = pd.DataFrame(
            [{"Player ID": k, "Alert Frames": v}
             for k, v in sorted(alerts.items(), key=lambda x: -x[1])]
        )
        st.dataframe(alert_df, width='stretch', hide_index=True)


# ── TAB 5: PLAYER STATS ──────────────────────────────────────────────────────
with tab_players:
    st.markdown('<div class="section-header">Player Performance Metrics</div>', unsafe_allow_html=True)

    if metrics:
        df = pd.DataFrame(metrics).sort_values("Distance (km)", ascending=False)
        # Coerce mixed-type columns so Arrow serialisation doesn't crash
        for _col in ["Fatigue Index", "Workload Index"]:
            if _col in df.columns:
                df[_col] = pd.to_numeric(df[_col], errors="coerce")
        team_filter = st.multiselect("Filter by Team", [1, 2], default=[1, 2], key="team_filter")
        df_f = df[df["Team"].isin(team_filter)] if team_filter else df

        # Safely apply gradients only for columns that exist
        styler = df_f.style
        for col, cmap in [
            ("Distance (km)", "Blues"),
            ("Max Speed (km/h)", "Oranges"),
            ("Sprints", "Purples"),
        ]:
            if col in df_f.columns:
                styler = styler.background_gradient(subset=[col], cmap=cmap)
        if "Max Sprint Spd (km/h)" in df_f.columns:
            styler = styler.background_gradient(subset=["Max Sprint Spd (km/h)"], cmap="Reds")

        st.dataframe(styler, width='stretch', hide_index=True)

        # Distance bar chart
        st.markdown('<div class="section-header">Distance Covered by Player</div>', unsafe_allow_html=True)
        fig_dist = px.bar(
            df_f.head(15), x="Player ID", y="Distance (km)", color="Team",
            color_discrete_map={1: "#3c50ff", 2: "#ff7828"}, template="plotly_dark",
        )
        fig_dist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_dist, width='stretch', key="dist_bar")

        # Speed vs Workload scatter
        st.markdown('<div class="section-header">Speed vs Workload Index</div>', unsafe_allow_html=True)
        fig_sc = px.scatter(
            df_f, x="Max Speed (km/h)", y="Workload Index",
            color="Team", size="Sprints",
            hover_data=["Player ID", "Distance (km)"],
            color_discrete_map={1: "#3c50ff", 2: "#ff7828"}, template="plotly_dark",
        )
        fig_sc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_sc, width='stretch', key="speed_scatter")

        # Fatigue Index chart (if available)
        if "Fatigue Index" in df_f.columns:
            df_fatigue = df_f[df_f["Fatigue Index"] != "N/A"].copy()
            if not df_fatigue.empty:
                st.markdown('<div class="section-header">Fatigue Index by Player</div>', unsafe_allow_html=True)
                df_fatigue["Fatigue Index"] = pd.to_numeric(df_fatigue["Fatigue Index"], errors="coerce")
                fig_fat = px.bar(
                    df_fatigue.dropna(subset=["Fatigue Index"]).head(15),
                    x="Player ID", y="Fatigue Index", color="Team",
                    color_discrete_map={1: "#3c50ff", 2: "#ff7828"}, template="plotly_dark",
                )
                fig_fat.add_hline(y=1.0, line_dash="dash", line_color="#ff4d4d",
                                  annotation_text="Baseline (no fatigue)")
                fig_fat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_fat, width='stretch', key="fatigue_bar")
    else:
        st.info("No player metrics available.")


# ── TAB 6: EXPORT ────────────────────────────────────────────────────────────
with tab_export:
    st.markdown('<div class="section-header">Export Results</div>', unsafe_allow_html=True)

    # CSV download
    if metrics:
        df_export = pd.DataFrame(metrics)
        st.download_button(
            "📥 Download Player Metrics CSV",
            data=df_export.to_csv(index=False).encode("utf-8"),
            file_name="player_metrics.csv",
            mime="text/csv",
            width='stretch',
            key="dl_csv",
        )

    # Video download (reuse cached bytes from Tab 2)
    if st.session_state.get("video_bytes"):
        st.download_button(
            "📥 Download Annotated Video (MP4)",
            data=st.session_state.video_bytes,
            file_name="football_analysis_annotated.mp4",
            mime="video/mp4",
            width='stretch',
            key="dl_video_export",
        )
    else:
        if st.button("🎬 Encode Annotated Video", width='stretch', key="enc_btn"):
            frames = results.get("frames", [])
            if frames:
                with st.spinner("Encoding MP4…"):
                    st.session_state.video_bytes = _encode_video_to_bytes(
                        frames, float(results.get("fps", 25))
                    )
                st.rerun()

    st.divider()

    # PDF Report — Task 5.3
    st.markdown('<div class="section-header">📄 PDF Match Report (Task 5.3)</div>', unsafe_allow_html=True)
    if st.button("Generate PDF Match Report", width='stretch', key="gen_pdf"):
        from src.reporting.report_generator import ReportGenerator
        from src.data_models import PlayerMetrics as PM

        pm_objects = {}
        for m in metrics:
            tid = m["Player ID"]
            pm_objects[tid] = PM(
                track_id=tid,
                team_id=m.get("Team"),
                total_distance_km=m.get("Distance (km)", 0.0),
                avg_speed_kmh=m.get("Avg Speed (km/h)", 0.0),
                max_speed_kmh=m.get("Max Speed (km/h)", 0.0),
                sprint_count=m.get("Sprints", 0),
                max_sprint_speed_kmh=m.get("Max Sprint Spd (km/h)", 0.0),
                total_sprint_distance_km=m.get("Sprint Dist (km)", 0.0),
                time_in_zones={},
                workload_index=m.get("Workload Index", 0.0),
            )

        image_map = {
            k: results[k] for k in [
                "heatmap_path", "team_heatmap_path", "sprint_hm_path",
                "pass_map_path", "voronoi_path", "offside_img_path",
            ] if results.get(k) and Path(results[k]).exists()
        }

        with st.spinner("Generating PDF…"):
            gen = ReportGenerator(output_dir="outputs/reports")
            pdf_path = gen.generate(
                match_name="Match_Analysis",
                metrics=pm_objects,
                images=image_map,
                formations=results.get("formations"),
                possession_pct=results.get("possession_pct"),
                pressing=results.get("pressing"),
                event_summary=results.get("event_summary"),
                pitch_control=results.get("pitch_control"),
            )

        with open(pdf_path, "rb") as fh:
            pdf_bytes = fh.read()

        st.download_button(
            "📥 Download PDF Report",
            data=pdf_bytes,
            file_name="match_report.pdf",
            mime="application/pdf",
            width='stretch',
            key="dl_pdf",
        )
        st.success(f"Report saved: {pdf_path}")

    st.divider()
    st.markdown("#### Session JSON Summary")
    st.json({
        "total_frames":      results["total_frames"],
        "players_tracked":   len(metrics),
        "fps":               results.get("fps"),
        "team_1_formation":  formations.get(1),
        "team_2_formation":  formations.get(2),
        "team_1_pressing":   pressing.get(1),
        "team_2_pressing":   pressing.get(2),
        "possession_pct":    possession_pct,
        "pitch_control":     results.get("pitch_control", {}),
        "event_summary":     results.get("event_summary", {}),
    })
