from __future__ import annotations

from pathlib import Path

from mapdoctor.adapters import ColmapAdapter, GlomapAdapter, GluemapAdapter
from mapdoctor.model import MapModel


def load_colmap(path: str | Path) -> MapModel:
    return ColmapAdapter().load(path)


def load_glomap(path: str | Path) -> MapModel:
    return GlomapAdapter().load(path)


def load_gluemap(path: str | Path) -> MapModel:
    return GluemapAdapter().load(path)
