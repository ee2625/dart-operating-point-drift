"""Development library for the DART cross-market project.

Imported from the Colab notebook as::

    import sys
    sys.path.insert(0, "/content/drive/MyDrive/ICAIF26/dart_dev")
    from src import labels, features, evaluation

Everything under ``DART_Spike_v2_Clean/`` is read-only; see ``paths.py``.
"""

from __future__ import annotations

from . import cache, evaluation, features, labels, paths

__all__ = ["cache", "evaluation", "features", "labels", "paths"]
