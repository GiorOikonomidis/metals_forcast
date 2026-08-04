"""Local (on-disk) artifact-saving configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LocalConfig:
    use: bool = True
    dir: str = ""   # empty -> consts.OUTPUT_ROOT is used as the base output directory
