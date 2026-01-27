from datetime import datetime, timezone

from infrawatch.ingestion import timeseries
from infrawatch.ingestion.models import SceneSummary


def test_timeseries_dry_run_plan(monkeypatch, capsys, tmp_path):
    scenes = [
        SceneSummary(
            product_id="S2A_CLOUDY",
            title="Cloudy scene",
            acquisition_datetime=datetime(2025, 6, 10, 9, 15, tzinfo=timezone.utc),
            cloud_cover=45.0,
            bbox=(14.0, 52.0, 15.0, 53.0),
            footprint=None,
            assets={
                "B04": "https://example.com/B04.jp2",
                "B08": "https://example.com/B08.jp2",
                "SCL": "https://example.com/SCL.jp2",
            },
        ),
        SceneSummary(
            product_id="S2A_CLEAR",
            title="Clear scene",
            acquisition_datetime=datetime(2025, 6, 10, 8, 5, tzinfo=timezone.utc),
            cloud_cover=5.0,
            bbox=(14.0, 52.0, 15.0, 53.0),
            footprint=None,
            assets={
                "B04": "https://example.com/B04.jp2",
                "B08": "https://example.com/B08.jp2",
                "SCL": "https://example.com/SCL.jp2",
            },
        ),
    ]

    def fake_search(_client, _request, _assets, search_limit):
        assert search_limit >= 1
        return scenes, {"query": "stub"}

    monkeypatch.setattr(timeseries, "search_scenes", fake_search)

    exit_code = timeseries.main(
        [
            "--bbox",
            "14.0",
            "52.0",
            "15.0",
            "53.0",
            "--date-from",
            "2025-06-01",
            "--date-to",
            "2025-06-30",
            "--max-days",
            "1",
            "--data-dir",
            str(tmp_path),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Planned Sentinel-2 time-series downloads" in output
    assert "S2A_CLEAR" in output
    assert "S2A_CLOUDY" not in output
    assert "Dry-run completed" in output
