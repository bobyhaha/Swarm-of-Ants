"""Vendored EoM. Lazy exports keep optional runtimes out of the colony."""
__version__ = "0.1.0"


def __getattr__(name):
    from importlib import import_module
    base = import_module("hayekmas.base")
    return getattr(base, name)
