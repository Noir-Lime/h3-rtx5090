"""h3-rtx5090: MiniMax H3 optimizations for a single 32 GB GPU.

Each module registers independently; a failure in one (for example a missing
optional dependency) does not prevent the others from loading.
"""
import importlib
import os
import sys

_ROOT = os.path.dirname(__file__)
if _ROOT not in sys.path:
    # vdn_h3_fast uses absolute imports of its own package.
    sys.path.insert(0, _ROOT)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _register(name, loader):
    try:
        module = loader()
    except Exception as exc:  # pragma: no cover
        print(f"[h3-rtx5090] {name} not loaded: {exc}")
        return
    NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)


def _vdn():
    from vdn_h3_fast.compiler_guard import install_layout_guard
    from vdn_h3_fast import nodes
    install_layout_guard()
    return nodes


_register("h3_memory", lambda: importlib.import_module(".h3_memory", __name__))
_register("h3_helpers", lambda: importlib.import_module(".h3_helpers", __name__))
_register("vdn_h3_fast", _vdn)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
