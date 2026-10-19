from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from mapdoctor.model import MapModel


@dataclass(frozen=True)
class AdapterInspection:
    backend: str
    resolved_path: Path
    model_format: str
    notes: tuple[str, ...] = ()


class MapAdapter(ABC):
    """Stable producer-specific interface into MapDoctor's unified MapModel."""

    backend: str
    display_name: str

    @abstractmethod
    def inspect(self, path: str | Path) -> AdapterInspection:
        """Validate input and describe how the reconstruction will be loaded."""

    @abstractmethod
    def load(self, path: str | Path) -> MapModel:
        """Load a producer's output into MapDoctor's unified representation."""
