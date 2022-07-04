#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By: Lucas Ribeiro Borges
# ---------------------------------------------------------------------------
""" KeepLargestHeap testing module """

from structures.KeepLargestHeap import KeepLargestHeap
import pytest

def test_inserts():
    """
    Tests KeepLargestHeap behavior when adding a number of elements
    larger than capacity.
    """
    h = KeepLargestHeap(4)
    h.add(-1)
    h.add(-5)
    h.add(-2)
    h.add(-4)
    h.add(-3)
    h.add(-6)

    assert h.getData() == [-1,-2,-3,-4]
