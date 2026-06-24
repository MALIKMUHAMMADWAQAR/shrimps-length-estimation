"""Minimal YAML configuration loader with attribute access and overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


class Config(dict):
    """A ``dict`` subclass that also supports attribute access.

    Nested dictionaries are recursively wrapped so that ``cfg.model.lr`` works
    in addition to ``cfg["model"]["lr"]``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for key, value in list(self.items()):
            self[key] = self._wrap(value)

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, dict) and not isinstance(value, Config):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(v) for v in value]
        return value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = self._wrap(value)

    def to_dict(self) -> Dict[str, Any]:
        """Recursively convert back to plain Python dictionaries."""
        out: Dict[str, Any] = {}
        for key, value in self.items():
            if isinstance(value, Config):
                out[key] = value.to_dict()
            elif isinstance(value, list):
                out[key] = [v.to_dict() if isinstance(v, Config) else v for v in value]
            else:
                out[key] = value
        return out


def load_config(path: str | Path) -> Config:
    """Load a YAML file into a :class:`Config`."""
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return Config(data)


def merge_overrides(cfg: Config, overrides: Iterable[str]) -> Config:
    """Apply ``key.subkey=value`` command-line overrides to ``cfg``.

    Values are parsed as YAML scalars so numbers and booleans are typed correctly.
    """
    merged = Config(copy.deepcopy(cfg.to_dict()))
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override (expected key=value): {override!r}")
        key, raw_value = override.split("=", 1)
        value = yaml.safe_load(raw_value)
        node: Dict[str, Any] = merged
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, Config())
        node[parts[-1]] = Config._wrap(value)
    return merged
