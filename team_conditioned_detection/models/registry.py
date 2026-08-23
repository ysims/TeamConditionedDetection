"""Architecture registry so train.py can stay model-agnostic.

Add a new architecture by subclassing Detector (see base.py) and
registering it:

    @register_model("my_arch")
    class MyDetector(Detector):
        ...

then point a config's model.architecture at "my_arch".
"""
from __future__ import annotations

from typing import Callable

_REGISTRY: dict[str, Callable] = {}


def register_model(name: str):
    def decorator(cls):
        if name in _REGISTRY:
            raise ValueError(f"Model '{name}' is already registered to {_REGISTRY[name]}")
        _REGISTRY[name] = cls
        return cls

    return decorator


def build_model(name: str, **kwargs):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown architecture '{name}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
