from pathlib import Path

import pytest

from src.data.ingest import DEFAULT_TICKERS_CONFIG
from src.data.universe import load_universe


def test_project_universe_has_explicit_target_and_context_roles():
    universe = load_universe(DEFAULT_TICKERS_CONFIG)

    assert universe.schema_version == 1
    assert len(universe.target_symbols) == 34
    assert len(universe.context_symbols) == 1
    assert len(universe.all_symbols) == 35
    assert "AAPL" in universe.target_symbols
    assert "^VIX" not in universe.target_symbols
    assert universe.context_symbol("implied_volatility") == "^VIX"


def test_universe_rejects_symbol_reused_as_target_and_context(tmp_path: Path):
    config = tmp_path / "tickers.yaml"
    config.write_text(
        """\
schema_version: 1
targets:
  - symbol: AAPL
    group: technology
context:
  - symbol: AAPL
    role: implied_volatility
"""
    )

    with pytest.raises(ValueError, match="must be unique"):
        load_universe(config)


def test_universe_requires_one_symbol_per_context_role(tmp_path: Path):
    config = tmp_path / "tickers.yaml"
    config.write_text(
        """\
schema_version: 1
targets:
  - symbol: AAPL
    group: technology
context:
  - symbol: ^VIX
    role: implied_volatility
  - symbol: OTHER
    role: implied_volatility
"""
    )

    with pytest.raises(ValueError, match="roles must be unique"):
        load_universe(config)


def test_universe_requires_implied_volatility_context(tmp_path: Path):
    config = tmp_path / "tickers.yaml"
    config.write_text(
        """\
schema_version: 1
targets:
  - symbol: AAPL
    group: technology
context:
  - symbol: OTHER
    role: unrelated_context
"""
    )

    with pytest.raises(ValueError, match="implied_volatility"):
        load_universe(config)


def test_universe_rejects_non_mapping_entries(tmp_path: Path):
    config = tmp_path / "tickers.yaml"
    config.write_text(
        """\
schema_version: 1
targets:
  - AAPL
context:
  - symbol: ^VIX
    role: implied_volatility
"""
    )

    with pytest.raises(ValueError, match="target entry must be a mapping"):
        load_universe(config)
