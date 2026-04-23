# -*- coding: utf-8 -*-
"""
face-match: Motor de búsqueda de rostros basado en OpenCV y FAISS.
"""

from .search import find_matches, run_search

__version__ = "0.2.0"
__all__ = ["find_matches", "run_search"]
