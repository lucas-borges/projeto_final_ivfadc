#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By: Lucas Ribeiro Borges
# ---------------------------------------------------------------------------
""" Abstract binary file reader class module for nearest neighbor datasets"""

import abc
import logging
import os
import struct
import sys
import numpy as np

from .Reader import Reader


class BinaryFileReader(Reader):
    """
    This abstract class implements functions to read data from
    nearest neighbor datasets in the format specified on
    http://corpus-texmex.irisa.fr/ into a numpy array.

    Classes that inherit from it must implement the dtypeStr property
    to describe the binary data representation on the file.

    http://corpus-texmex.irisa.fr/
    The vectors are stored in raw little endian.
    Each vector takes 4+d*4 bytes .ivecs formats, where d is the dimensionality of the vector, as shown below.

    field       field type                          description
    d           int                                 the vector dimension
    components  (unsigned char|float | int)*d       the vector components

    Properties:
        dtypeStr: str
            String with numpy data type representation.
            See: https://numpy.org/doc/stable/reference/arrays.dtypes.html

    Attributes:
        filepath: str
            Path of file to be read.
    """

    def __init__(self, filePath: str) -> None:
        """
        The constructor for abstract BinaryFileReader class.

        Parameters:
            filepath: str
                Path of file to be read.
        """
        self.filepath = filePath

    @property
    @abc.abstractmethod
    def dtypeStr(self):
        """
        Abstract property describing binary data on the file.
        See: https://numpy.org/doc/stable/reference/arrays.dtypes.html
        """
        pass

    def read(self) -> np.ndarray:
        """
        Reads binary data on file pointed to by filepath according to
        dtypeStr binary data representation.

        Returns:
            Array of shape (nSamples, dataDimension)
        """
        if not os.path.exists(self.filepath):
            logging.error(f"File {self.filepath} does not exist, exiting.")
            sys.exit(2)

        dim: int = self.__getVectorDimension()
        dt = np.dtype(self.dtypeStr.format(dim=dim))
        dataset = np.fromfile(self.filepath, dt)
        return dataset['f1']

    def __getVectorDimension(self) -> int:
        """
        Helper function to read data dimension.

        Returns:
            Integer representing data dimension
        """
        with open(self.filepath, "rb") as fh:
            dimension: int = struct.unpack("<i", fh.read(4))[0]
        return dimension
