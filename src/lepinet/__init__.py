"""lepinet — hierarchical Lepidoptera classification (species / genus / family).

The reusable library version of the pipeline developed in dev/ (dev/030 is the reference). See
src/lepinet/README.md for the architecture, the migration status of each module, and the lessons
each one encodes. Migration in progress: the modules exported below are DONE; data/heads/train/
test are still PORTING from dev/ and should be imported from there for now.
"""
from __future__ import annotations

__version__ = "0.1.0.dev0"

# DONE modules -- safe to import and use.
from . import gpu_decode, memory, schedules  # noqa: E402,F401

__all__ = ["gpu_decode", "memory", "schedules", "__version__"]
