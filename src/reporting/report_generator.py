"""
src/reporting/report_generator.py
=================================
Generates the final PDF match report summarizing player stats and tactical maps.

Task 5.3 Update:
  - Includes heatmap, team heatmap, pass map, Voronoi and offside diagrams
  - Formation, possession %, pressing intensity, and pass event stats
  - Multi-page layout with automatic pagination

PRD Reference: Goal-10, Task 2.3
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

from src.data_models import PlayerMetrics

DARK_BG   = HexColor("#0d1117")
HEADER_BG = HexColor("#161b22")
ACCENT    = HexColor("#1E90FF")
TEXT      = HexColor("#e6edf3")
MUTED     = HexColor("#8b949e")
T1_COLOR  = HexColor("#3c50ff")
T2_COLOR  = HexColor("#ff7828")


def _draw_header(c: canvas.Canvas, width: float, height: float, title: str) -> None:
    """Draw the dark header bar on the current page."""
    c.setFillColor(HEADER_BG)
    c.rect(0, height - 0.85 * inch, width, 0.85 * inch, fill=True, stroke=False)
    c.setFillColor(ACCENT)
    c.rect(0, height - 0.85 * inch, 4, 0.85 * inch, fill=True, stroke=False)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(0.4 * inch, height - 0.52 * inch, title)
    c.setFont("Helvetica", 8)
    c.setFillColor(MUTED)
    c.drawRightString(width - 0.4 * inch, height - 0.52 * inch,
                      "Football Analytics Engine  •  YOLOv8 + ByteTrack")


def _draw_footer(c: canvas.Canvas, width: float, page_num: int) -> None:
    c.setFont("Helvetica-Oblique", 7)
    c.setFillColor(MUTED)
    c.drawString(0.4 * inch, 0.35 * inch, "INDICATIVE ONLY — Not for official match use")
    c.drawRightString(width - 0.4 * inch, 0.35 * inch, f"Page {page_num}")


def _section_title(c: canvas.Canvas, text: str, x: float, y: float) -> None:
    c.setFillColor(ACCENT)
    c.rect(x - 0.05 * inch, y - 0.02 * inch, 0.04 * inch, 0.18 * inch, fill=True, stroke=False)
    c.setFillColor(TEXT)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, text)


def _stat_row(c: canvas.Canvas, label: str, value: str, x: float, y: float,
              label_color=None, value_color=None) -> None:
    c.setFont("Helvetica", 10)
    c.setFillColor(label_color or MUTED)
    c.drawString(x, y, label)
    c.setFillColor(value_color or TEXT)
    c.drawRightString(x + 2.8 * inch, y, value)


def _embed_image(c: canvas.Canvas, img_path: str,
                 x: float, y: float, w: float, h: float) -> bool:
    """Embed an image; return True on success."""
    if not img_path or not Path(img_path).exists():
        return False
    try:
        c.drawImage(img_path, x, y, width=w, height=h,
                    preserveAspectRatio=True, anchor="nw")
        return True
    except Exception as e:
        logging.warning(f"Could not embed image {img_path}: {e}")
        return False


class ReportGenerator:
    """
    Compiles PlayerMetrics and all tactical visualizations into a structured PDF.
    """

    def __init__(self, output_dir: str = "outputs/reports") -> None:
        self.out_dir = Path(output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        match_name: str,
        metrics: Dict[int, PlayerMetrics],
        images: Dict[str, str],
        formations: Optional[Dict[int, str]] = None,
        possession_pct: Optional[Dict[int, float]] = None,
        pressing: Optional[Dict[int, float]] = None,
        event_summary: Optional[Dict[str, Any]] = None,
        pitch_control: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Generate a multi-page PDF report.

        Parameters
        ----------
        match_name     : Name of the clip / match
        metrics        : Dict[track_id → PlayerMetrics]
        images         : Dict[title → file_path] for all visuals
        formations     : Dict[team_id → formation_str]
        possession_pct : Dict[team_id → float percent]
        pressing       : Dict[team_id → float]
        event_summary  : Dict with keys total_events, passes, interceptions, etc.
        pitch_control  : Dict with keys team1_pct, team2_pct

        Returns
        -------
        str : Absolute path to the saved PDF.
        """
        pdf_path = self.out_dir / f"{match_name}_Tactical_Report.pdf"
        c = canvas.Canvas(str(pdf_path), pagesize=A4)
        W, H = A4
        page = 1

        # ════════════════════════════════════════════════════════════════
        # PAGE 1 — Match Summary & Player Stats
        # ════════════════════════════════════════════════════════════════
        c.setFillColor(DARK_BG)
        c.rect(0, 0, W, H, fill=True, stroke=False)
        _draw_header(c, W, H, f"Football Analytics Report: {match_name}")
        _draw_footer(c, W, page)

        y = H - 1.1 * inch

        # ── Match summary stats ──────────────────────────────────────────
        _section_title(c, "Match Summary", 0.4 * inch, y)
        y -= 0.32 * inch

        if metrics:
            total_dist    = sum(m.total_distance_km for m in metrics.values())
            total_sprints = sum(m.sprint_count for m in metrics.values())
            max_spd       = max((m.max_speed_kmh for m in metrics.values()), default=0.0)
            n_players     = len(metrics)
        else:
            total_dist = total_sprints = max_spd = n_players = 0

        col1_x, col2_x = 0.4 * inch, 3.5 * inch
        rows_left = [
            ("Players Tracked",   str(n_players)),
            ("Total Distance",    f"{total_dist:.2f} km"),
            ("Total Sprints",     str(total_sprints)),
            ("Peak Speed",        f"{max_spd:.1f} km/h"),
        ]
        rows_right: List = []

        if formations:
            rows_right.append(("Team 1 Formation", formations.get(1, "—")))
            rows_right.append(("Team 2 Formation", formations.get(2, "—")))
        if possession_pct:
            rows_right.append(("Team 1 Possession", f"{possession_pct.get(1, 0):.1f}%"))
            rows_right.append(("Team 2 Possession", f"{possession_pct.get(2, 0):.1f}%"))
        if pressing:
            rows_right.append(("Team 1 Pressing",  f"{pressing.get(1, 0):.1f}"))
            rows_right.append(("Team 2 Pressing",  f"{pressing.get(2, 0):.1f}"))

        for i, (lbl, val) in enumerate(rows_left):
            _stat_row(c, lbl, val, col1_x, y - i * 0.22 * inch)
        for i, (lbl, val) in enumerate(rows_right):
            _stat_row(c, lbl, val, col2_x, y - i * 0.22 * inch)

        y -= (max(len(rows_left), len(rows_right)) + 1) * 0.22 * inch

        # ── Event summary ────────────────────────────────────────────────
        if event_summary:
            _section_title(c, "Possession Events", 0.4 * inch, y)
            y -= 0.28 * inch
            ev_rows = [
                ("Total Events",   str(event_summary.get("total_events", 0))),
                ("Passes",         str(event_summary.get("passes", 0))),
                ("Interceptions",  str(event_summary.get("interceptions", 0))),
                ("Team 1 Passes",  str(event_summary.get("team1_passes", 0))),
                ("Team 2 Passes",  str(event_summary.get("team2_passes", 0))),
            ]
            for i, (lbl, val) in enumerate(ev_rows):
                _stat_row(c, lbl, val, col1_x, y - i * 0.21 * inch)
            if pitch_control:
                pc_rows = [
                    ("Team 1 Pitch Control", f"{pitch_control.get('team1_pct', 0):.1f}%"),
                    ("Team 2 Pitch Control", f"{pitch_control.get('team2_pct', 0):.1f}%"),
                ]
                for i, (lbl, val) in enumerate(pc_rows):
                    _stat_row(c, lbl, val, col2_x, y - i * 0.21 * inch)
            y -= (len(ev_rows) + 1) * 0.21 * inch

        # ── Top performers table ─────────────────────────────────────────
        if metrics:
            _section_title(c, "Top Performers by Distance", 0.4 * inch, y)
            y -= 0.28 * inch

            headers = ["ID", "Team", "Distance", "Avg Spd", "Max Spd", "Sprints", "Workload"]
            col_xs  = [0.4, 1.1, 1.9, 2.9, 3.8, 4.7, 5.4]
            col_xs  = [x * inch for x in col_xs]

            c.setFont("Helvetica-Bold", 8.5)
            c.setFillColor(ACCENT)
            for lbl, cx in zip(headers, col_xs):
                c.drawString(cx, y, lbl)
            y -= 0.08 * inch
            c.setStrokeColor(ACCENT)
            c.line(0.4 * inch, y, W - 0.4 * inch, y)
            y -= 0.15 * inch

            top = sorted(metrics.values(), key=lambda m: m.total_distance_km, reverse=True)[:8]
            c.setFont("Helvetica", 8.5)
            for pm in top:
                c.setFillColor(T1_COLOR if pm.team_id == 1 else T2_COLOR)
                vals = [
                    str(pm.track_id),
                    str(pm.team_id or "?"),
                    f"{pm.total_distance_km:.3f} km",
                    f"{pm.avg_speed_kmh:.1f}",
                    f"{pm.max_speed_kmh:.1f}",
                    str(pm.sprint_count),
                    f"{pm.workload_index:.2f}",
                ]
                for val, cx in zip(vals, col_xs):
                    c.drawString(cx, y, val)
                y -= 0.2 * inch
                if y < 0.7 * inch:
                    break

        c.showPage()
        page += 1

        # ════════════════════════════════════════════════════════════════
        # SUBSEQUENT PAGES — Tactical Visualizations
        # ════════════════════════════════════════════════════════════════
        visual_order = [
            ("Heatmap — All Players",       "heatmap_path"),
            ("Team Heatmap Comparison",     "team_heatmap_path"),
            ("Sprint Locations",            "sprint_hm_path"),
            ("Pass Map & Possession Events","pass_map_path"),
            ("Voronoi Pitch Control",       "voronoi_path"),
            ("Offside Lines (Indicative)",  "offside_img_path"),
        ]

        images_to_draw = [
            (title, images[key])
            for title, key in visual_order
            if key in images and images[key] and Path(images[key]).exists()
        ]

        # Two images per page
        for i in range(0, len(images_to_draw), 2):
            c.setFillColor(DARK_BG)
            c.rect(0, 0, W, H, fill=True, stroke=False)
            _draw_header(c, W, H, "Tactical Visualizations")
            _draw_footer(c, W, page)

            y_top = H - 1.1 * inch
            img_h = (y_top - 0.65 * inch) / 2 - 0.3 * inch

            for slot, (title, img_path) in enumerate(images_to_draw[i:i+2]):
                y_slot = y_top - slot * (img_h + 0.35 * inch)
                c.setFont("Helvetica-Bold", 10)
                c.setFillColor(ACCENT)
                c.drawString(0.4 * inch, y_slot, title)
                _embed_image(c, img_path, 0.4 * inch,
                             y_slot - img_h - 0.05 * inch,
                             W - 0.8 * inch, img_h)

            c.showPage()
            page += 1

        c.save()
        logging.info(f"PDF report saved → {pdf_path}")
        return str(pdf_path)
