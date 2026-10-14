"""
CLI smoke tests using the FakeProvider.
"""
from __future__ import annotations

import csv
import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.cli import build_parser, main


# ──────────────────────────────────────────────────────────────────────────────
# Minimal config fixture
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_config(tmp_path) -> Path:
    """
    Creates a minimal config dir + constituent CSV + returns config_dir path.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # data.yaml
    (config_dir / "data.yaml").write_text(
        f"start_date: '2020-01-01'\n"
        f"end_date: '2020-03-31'\n"
        f"cache_dir: '{tmp_path / 'data'}'\n"
        f"provider: 'fake'\n"
        f"overlap_sessions: 5\n"
        f"no_data_retry_days: 30\n"
        f"adj_tolerance: 0.02\n"
        f"large_return_threshold: 0.40\n"
        f"stale_price_run: 5\n"
        f"with_data_threshold: 0.90\n"
        f"throttle_sleep: 0.0\n"
        f"max_retries: 1\n"
        f"retry_backoff: 1.0\n"
    )

    # universe.yaml
    csv_path = tmp_path / "sp500_history.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "tickers"])
        w.writerow(["2020-01-01", "AAPL,SPY"])
    constituent_src = str(csv_path).replace("\\", "/")

    (config_dir / "universe.yaml").write_text(
        f"constituent_source: '{constituent_src}'\n"
        f"constituent_sha256: ''\n"
        f"constituent_commit: ''\n"
        f"aliases: {{}}\n"
        f"etfs:\n"
        f"  - SPY\n"
    )

    return config_dir


# ──────────────────────────────────────────────────────────────────────────────
# Parser
# ──────────────────────────────────────────────────────────────────────────────

def test_parser_builds_without_error():
    parser = build_parser()
    assert parser is not None


def test_data_update_subcommand_parsed():
    parser = build_parser()
    args = parser.parse_args(["data", "update", "--start", "2022-01-01"])
    assert args.start == "2022-01-01"
    assert hasattr(args, "func")


def test_universe_members_subcommand_parsed():
    parser = build_parser()
    args = parser.parse_args(["universe", "members", "--date", "2022-06-01"])
    assert args.date == "2022-06-01"


# ──────────────────────────────────────────────────────────────────────────────
# Smoke: universe members
# ──────────────────────────────────────────────────────────────────────────────

def test_universe_members_smoke(minimal_config, capsys):
    """CLI universe members runs without error against the fixture CSV."""
    # Patch the config loader to use our minimal config
    with patch("src.cli._load_everything") as mock_load:
        from src.config import DataConfig, UniverseConfig, Config
        from src.data.cache import CacheManager
        from src.universe.etfs import ETFUniverse

        cfg = Config(
            data=DataConfig(
                start_date=date(2020, 1, 1),
                end_date=date(2020, 3, 31),
                cache_dir=minimal_config.parent / "data",
                provider="fake",
            ),
            universe=UniverseConfig(
                constituent_source=str(minimal_config.parent / "sp500_history.csv"),
                constituent_sha256="",
                constituent_commit="",
                aliases={},
                etfs=["SPY"],
            ),
        )
        cache = CacheManager(cfg.data.cache_dir)
        etfs = ETFUniverse.from_config(["SPY"])
        mock_load.return_value = (cfg, MagicMock(), MagicMock(), cache, etfs)

        rc = main(["--config-dir", str(minimal_config), "universe", "members", "--date", "2020-02-01"])
        assert rc == 0

    captured = capsys.readouterr()
    assert "AAPL" in captured.out


# ──────────────────────────────────────────────────────────────────────────────
# Smoke: data report
# ──────────────────────────────────────────────────────────────────────────────

def test_data_report_smoke(minimal_config, tmp_path, capsys):
    """CLI data report runs without error and writes a file."""
    from src.data.cache import CacheManager
    from src.universe.etfs import ETFUniverse
    from src.config import DataConfig, UniverseConfig, Config
    from tests.fakes.fake_provider import FakeProvider

    csv_path = minimal_config.parent / "sp500_history.csv"
    cache_dir = tmp_path / "data2"

    cfg = Config(
        data=DataConfig(
            start_date=date(2020, 1, 1),
            end_date=date(2020, 3, 31),
            cache_dir=cache_dir,
            provider="fake",
        ),
        universe=UniverseConfig(
            constituent_source=str(csv_path),
            constituent_sha256="",
            constituent_commit="",
            aliases={},
            etfs=["SPY"],
        ),
    )

    cache = CacheManager(cache_dir)
    provider = FakeProvider(["AAPL", "SPY"])
    cache.update_symbol(provider, "AAPL")
    cache.update_symbol(provider, "SPY")
    etfs = ETFUniverse.from_config(["SPY"])

    with patch("src.cli._load_everything") as mock_load, \
         patch("src.cli._build_calendar") as mock_cal:
        mock_load.return_value = (cfg, MagicMock(), MagicMock(), cache, etfs)
        mock_cal.return_value = None

        args = build_parser().parse_args([
            "--config-dir", str(minimal_config),
            "data", "report",
        ])
        # Override the output path by monkeypatching generate_data_report
        with patch("src.cli.Path") as mock_path_cls:
            from pathlib import Path as RealPath
            mock_path_cls.side_effect = lambda *a: RealPath(*a)

            import src.reports.data_report as dr_mod
            original = dr_mod.generate_data_report

            written = []
            def _capture(*args, **kwargs):
                kwargs["output_path"] = tmp_path / "report.md"
                result = original(*args, **kwargs)
                written.append(result)
                return result

            with patch.object(dr_mod, "generate_data_report", _capture):
                rc = main([
                    "--config-dir", str(minimal_config),
                    "data", "report",
                ])

    # Report was written and contains the bias statement
    assert len(written) == 1
    assert "KNOWN BIAS" in written[0]
