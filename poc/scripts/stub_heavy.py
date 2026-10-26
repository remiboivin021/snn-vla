"""Stub tensorflow-stack modules that BitVLA imports for training data only (never used at inference)."""
import sys, importlib.abc, importlib.machinery
from unittest.mock import MagicMock
PREFIXES = ('dlimp', 'tensorflow', 'tensorflow_datasets', 'tensorflow_graphics')
class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name.split('.')[0] in PREFIXES:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec):
        m = MagicMock(name=spec.name); m.__path__ = []; m.__spec__ = spec; m.__name__ = spec.name; return m
    def exec_module(self, module): pass
sys.meta_path.insert(0, _Finder())
