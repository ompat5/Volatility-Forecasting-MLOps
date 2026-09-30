"""Typed, validated definition of the forecast universe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TargetAsset:
    """One asset for which the global model must produce forecasts."""

    symbol: str
    group: str


@dataclass(frozen=True)
class ContextSeries:
    """One market series consumed as input but never used as a target."""

    symbol: str
    role: str


@dataclass(frozen=True)
class Universe:
    """Complete target and context contract shared across the system."""

    schema_version: int
    targets: tuple[TargetAsset, ...]
    context: tuple[ContextSeries, ...]

    @property
    def target_symbols(self) -> tuple[str, ...]:
        return tuple(asset.symbol for asset in self.targets)

    @property
    def context_symbols(self) -> tuple[str, ...]:
        return tuple(series.symbol for series in self.context)

    @property
    def all_symbols(self) -> tuple[str, ...]:
        return self.target_symbols + self.context_symbols

    def context_symbol(self, role: str) -> str:
        matches = [series.symbol for series in self.context if series.role == role]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one context series with role {role!r}, "
                f"found {len(matches)}"
            )
        return matches[0]


def _require_non_empty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _require_mapping(value: object, field: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return value


def load_universe(path: Path) -> Universe:
    """Load and validate the single source of truth for supported symbols."""
    with path.open() as config_file:
        raw = yaml.safe_load(config_file)

    if not isinstance(raw, dict):
        raise ValueError("Universe config must contain a mapping")
    if raw.get("schema_version") != 1:
        raise ValueError("Universe config schema_version must be 1")

    raw_targets = raw.get("targets")
    raw_context = raw.get("context")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("Universe must define at least one target")
    if not isinstance(raw_context, list) or not raw_context:
        raise ValueError("Universe must define at least one context series")

    target_items = [
        _require_mapping(item, "target entry") for item in raw_targets
    ]
    context_items = [
        _require_mapping(item, "context entry") for item in raw_context
    ]
    targets = tuple(
        TargetAsset(
            symbol=_require_non_empty_string(item.get("symbol"), "target symbol"),
            group=_require_non_empty_string(item.get("group"), "target group"),
        )
        for item in target_items
    )
    context = tuple(
        ContextSeries(
            symbol=_require_non_empty_string(item.get("symbol"), "context symbol"),
            role=_require_non_empty_string(item.get("role"), "context role"),
        )
        for item in context_items
    )

    universe = Universe(
        schema_version=raw["schema_version"],
        targets=targets,
        context=context,
    )
    if len(set(universe.all_symbols)) != len(universe.all_symbols):
        raise ValueError("Universe symbols must be unique across targets and context")
    if len(set(series.role for series in context)) != len(context):
        raise ValueError("Context roles must be unique")
    universe.context_symbol("implied_volatility")
    return universe
