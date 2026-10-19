from __future__ import annotations

from importlib import import_module

from mapdoctor.adapters.base import MapAdapter
from mapdoctor.adapters.colmap import ColmapAdapter
from mapdoctor.adapters.glomap import GlomapAdapter
from mapdoctor.adapters.gluemap import GluemapAdapter

_ADAPTER_TYPES: dict[str, type[MapAdapter]] = {
    "colmap": ColmapAdapter,
    "glomap": GlomapAdapter,
    "gluemap": GluemapAdapter,
}


def register_adapter(
    name: str,
    adapter_type: type[MapAdapter],
    *,
    replace: bool = False,
) -> None:
    """Register a map adapter without changing MapDoctor core code."""
    key = name.strip().lower()
    if not key:
        raise ValueError("adapter name must not be empty")
    if not isinstance(adapter_type, type) or not issubclass(adapter_type, MapAdapter):
        raise TypeError("adapter_type must be a MapAdapter subclass")
    if key in _ADAPTER_TYPES and not replace:
        raise ValueError(f"Map adapter '{key}' is already registered")
    _ADAPTER_TYPES[key] = adapter_type


def _load_external_adapter(spec: str) -> MapAdapter:
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(
            "External map adapters must use 'package.module:AdapterClass' syntax"
        )
    candidate = getattr(import_module(module_name), attribute)
    adapter = candidate() if isinstance(candidate, type) else candidate
    if not isinstance(adapter, MapAdapter):
        raise TypeError(f"External map adapter '{spec}' must implement MapAdapter")
    return adapter


def get_adapter(backend: str) -> MapAdapter:
    token = backend.strip()
    if ":" in token:
        return _load_external_adapter(token)
    try:
        return _ADAPTER_TYPES[token.lower()]()
    except KeyError as exc:
        supported = ", ".join(sorted(_ADAPTER_TYPES))
        raise ValueError(
            f"Unknown map adapter '{backend}'. Built-ins: {supported}. "
            "External adapters use package.module:AdapterClass."
        ) from exc


def list_adapters() -> tuple[str, ...]:
    return tuple(_ADAPTER_TYPES)
