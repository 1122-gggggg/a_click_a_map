"""Map-first orchestrator over MapDoctor and sfm-diagnosis."""

from .bridge import (
    map_model_to_map_data,
    mapdoctor_rows_to_history_rows,
    nearest_mapping_rotation,
    rotation_from_viewing_direction,
)
from .pipeline import analyze, attribute_query, check, check_localize, check_map

__version__ = "0.1.0"

__all__ = [
    "analyze",
    "attribute_query",
    "check",
    "check_localize",
    "check_map",
    "map_model_to_map_data",
    "mapdoctor_rows_to_history_rows",
    "nearest_mapping_rotation",
    "rotation_from_viewing_direction",
]

try:
    from .session_select.run import select_sessions
except ImportError:
    pass
else:
    __all__ = [*__all__, "select_sessions"]
