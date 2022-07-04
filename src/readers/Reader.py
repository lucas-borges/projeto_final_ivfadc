#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By: Lucas Ribeiro Borges
# ---------------------------------------------------------------------------
""" Abstract file reader class module"""

import abc
import logging
import os
import sys
import numpy as np


class Reader(metaclass=abc.ABCMeta):
    """
    This abstract class declares functions to read data from a file.

    Attributes:
        filepath: str
            Path of file to be read.
    """

    def __init__(self, filePath: str) -> None:
        """
        The constructor for abstract Reader class.

        Parameters:
            filepath: str
                Path of file to be read.
        """
        self.filepath = filePath

    @abc.abstractmethod
    def read(self) -> np.ndarray:
        """
        Reads binary data on file pointed to by filepath according to
        dtypeStr binary data representation.

        Returns:
            Array of shape (nSamples, dataDimension)
        """
        pass
