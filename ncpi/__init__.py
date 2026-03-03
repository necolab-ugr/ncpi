"""
Initialization file for the ncpi package.
"""

from .Inference import *
from .Features import *
from .Simulation import *
from .Analysis import *
from . import tools

_FIELD_POTENTIAL_IMPORT_ERROR = None
try:
    from .FieldPotential import *
except ModuleNotFoundError as exc:
    optional_modules = {"neuron", "LFPy", "lfpykernels", "lfpykit", "h5py"}
    if getattr(exc, "name", None) not in optional_modules:
        raise
    # Allow the core package (and WebUI) to import even when optional
    # NEURON/field-potential dependencies are not installed.
    _FIELD_POTENTIAL_IMPORT_ERROR = exc

    class FieldPotential:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            missing = getattr(_FIELD_POTENTIAL_IMPORT_ERROR, "name", None) or "optional dependency"
            raise ModuleNotFoundError(
                "ncpi.FieldPotential requires optional dependency "
                f"'{missing}', which is not installed in this environment. "
                "Install NEURON-related dependencies to use field-potential features."
            ) from _FIELD_POTENTIAL_IMPORT_ERROR
