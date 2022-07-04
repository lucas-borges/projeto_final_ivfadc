#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By: Lucas Ribeiro Borges
# ---------------------------------------------------------------------------
""" Reader factory class module for nearest neighbor datasets"""

from typing import ClassVar
from .BvecsReader import BvecsReader
from .FvecsReader import FvecsReader
from .IvecsReader import IvecsReader
from .Reader import Reader


class ReaderFactory:
    """
    Factory class for Reader class instances.

    Instantiates a concrete Reader instance depending on file extension or
    requested format.
    """
    readerImplementations: ClassVar[dict[str, Reader]] = {
        "fvecs":FvecsReader,
        "bvecs":BvecsReader,
        "ivecs":IvecsReader
    }

    @staticmethod
    def getFormat(datasetPath: str) -> str:
        """
        Static helper function for parsing file extension.

        Parameters:
            datasetPath: str
                Path of file to parse extension.

        Returns:
            String containing file extension.
        """
        dotPosition = datasetPath.rfind(".")
        if dotPosition == -1:
            return ""
        return datasetPath[dotPosition+1:]

    @staticmethod
    def getReader(datasetPath: str, format: str=None) -> Reader:
        """
        Static factory function to instantiate concrete Reader.

        Automatically parses file extension to decide Reader to instatiate if
        format is None. Supports fvecs, bvecs and ivecs formats by default with other
        formats being able to be registered.

        Parameters:
            datasetPath: str
                Path of file to get a Reader for.
            format: str, default=None
                Requested reader format, if None, file extension is
                parsed to decide which reader to instantiate.

        Returns:
            Instantiated Reader for specified file.
        """
        if format == None:
            format = ReaderFactory.getFormat(datasetPath)

        if format in ReaderFactory.readerImplementations:
            return ReaderFactory.readerImplementations[format](datasetPath)
        else:
            raise ValueError(format)

    @staticmethod
    def registerReader(format: str, reader: Reader) -> None:
        """
        Registers a Reader as associated with the specified format, allowing the
        factory to instantiate this new implementation.

        Parameters:
            format: str
                File extension to be associated with reader.
            reader: Reader
                Reader implementation to be registered and associated with format.
        """
        ReaderFactory.readerImplementations[format] = reader
