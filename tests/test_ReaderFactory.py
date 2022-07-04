#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By: Lucas Ribeiro Borges
# ---------------------------------------------------------------------------
""" Reader Factory testing module """

from readers.Reader import Reader
from readers.ReaderFactory import ReaderFactory
from readers.BvecsReader import BvecsReader
from readers.FvecsReader import FvecsReader
from readers.IvecsReader import IvecsReader
import pytest

def test_getFormat():
    """
    Tests getFormat behavior on different filenames.
    """
    assert ReaderFactory.getFormat("filename.ivecs") == "ivecs"
    assert ReaderFactory.getFormat("filename.fvecs") == "fvecs"
    assert ReaderFactory.getFormat("filename.bvecs") == "bvecs"
    assert ReaderFactory.getFormat("filename.asdf.ivecs") == "ivecs"
    assert ReaderFactory.getFormat("filename") == ""

def test_getReader():
    assert isinstance(ReaderFactory.getReader("filename.ivecs"), IvecsReader)
    assert isinstance(ReaderFactory.getReader("filename.fvecs"), FvecsReader)
    assert isinstance(ReaderFactory.getReader("filename.bvecs"), BvecsReader)
    assert isinstance(ReaderFactory.getReader("filename.asdf.ivecs"), IvecsReader)

    with pytest.raises(ValueError):
        ReaderFactory.getReader("filename.ivecsa")

    with pytest.raises(ValueError):
        ReaderFactory.getReader("filename")

    with pytest.raises(ValueError):
        ReaderFactory.getReader("filename.asdf")

def test_registerReader():
    ReaderFactory.registerReader("dummy", DummyReader)

    assert isinstance(ReaderFactory.getReader("filename.dummy"), DummyReader)


class DummyReader(Reader):
    def read(self):
        return "dummy"
