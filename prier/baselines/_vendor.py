"""Import the authors' baseline code from ``vendor/`` without its drivers.

The vendored files import packages that only their command-line drivers need
(gin-config, faiss, OpenAI clients, ...). We register small stand-ins for
those modules while loading, so the algorithm code itself runs unchanged.
"""
from __future__ import annotations

import importlib.util
import random
import sys
import types
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np

VENDOR = Path(__file__).resolve().parents[2] / "vendor"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def _stub_modules(stubs):
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        yield
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _gin_stub():
    gin = types.ModuleType("gin")

    def configurable(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f

    gin.configurable = configurable
    gin.REQUIRED = None
    return gin


@lru_cache(maxsize=1)
def kwikbucks():
    """Return (model_utils, data_utils) of the KwikBucks reference code."""
    package = types.ModuleType("kwikbucks")
    package.__path__ = [str(VENDOR / "kwikbucks")]
    with _stub_modules({"gin": _gin_stub(), "kwikbucks": package}):
        package.cluster_utils = _load(VENDOR / "kwikbucks/cluster_utils.py", "kwikbucks.cluster_utils")
        package.data_utils = _load(VENDOR / "kwikbucks/data_utils.py", "kwikbucks.data_utils")
        model_utils = _load(VENDOR / "kwikbucks/model_utils.py", "kwikbucks.model_utils")
    return model_utils, package.data_utils


@lru_cache(maxsize=1)
def alper():
    stubs = {}
    for name, cls in (("data_loader", "DataLoader"), ("encoder_topk", "ANNTopKSearch"), ("llm_oracle", "LLMOracle")):
        module = types.ModuleType(name)
        setattr(module, cls, type(cls, (), {}))
        stubs[name] = module
    with _stub_modules(stubs):
        return _load(VENDOR / "alper/alper_framework.py", "alper_framework")


@lru_cache(maxsize=1)
def acc():
    queries = _load(VENDOR / "acc/query_strategies.py", "acc_query_strategies")
    decoder = _load(VENDOR / "acc/correlation_clustering.py", "acc_correlation_clustering")
    return queries, decoder


@contextmanager
def seeded(seed, python=True):
    """The vendored code draws from the global RNGs; seed them temporarily."""
    py_state, np_state = random.getstate(), np.random.get_state()
    if python:
        random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
