import os
from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

from src.data_models import PlayerMetrics
from src.reporting.report_generator import ReportGenerator


def test_report_generation(tmp_path):
    out_dir = tmp_path / "outputs"
    generator = ReportGenerator(output_dir=str(out_dir))
    
    # Create fake metrics
    metrics = {
        1: PlayerMetrics(
            track_id=1, team_id=1, total_distance_km=10.5, 
            avg_speed_kmh=8.0, max_speed_kmh=29.5, sprint_count=15, 
            max_sprint_speed_kmh=29.5, total_sprint_distance_km=1.2, 
            time_in_zones={"defensive_third": 100}, workload_index=0.8
        ),
        2: PlayerMetrics(
            track_id=2, team_id=2, total_distance_km=8.2, 
            avg_speed_kmh=6.0, max_speed_kmh=25.0, sprint_count=5, 
            max_sprint_speed_kmh=25.0, total_sprint_distance_km=0.3, 
            time_in_zones={"attacking_third": 200}, workload_index=0.5
        )
    }
    
    # Create a dummy image
    img_path = tmp_path / "dummy.png"
    # Actually, reportlab requires a valid image or we just skip it in the test.
    # Let's pass an invalid path to ensure it doesn't crash (should log warning and continue).
    images = {"Test Map": str(img_path)}
    
    pdf_path = generator.generate("Test_Match", metrics, images)
    
    # Verify file was created
    assert Path(pdf_path).exists()
    assert Path(pdf_path).stat().st_size > 0
    assert "Test_Match_Tactical_Report.pdf" in pdf_path
