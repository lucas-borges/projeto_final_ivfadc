#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By: Lucas Ribeiro Borges
# ---------------------------------------------------------------------------
""" Module containing assorted utility functions """

import logging
import sys


def isPowerOfTwo(n: int) -> bool:
    """
    Function to check if a number is a power of two.

    Parameters:
        n: int
            The number to be checked.

    Returns:
        bool: A boolean indicating if n is a power of two.
    """
    return (n != 0) and (n & (n-1) == 0)

def configureLogging(logLevel: str) -> None:
    """
    Function to set up logging settings.

    Sets the log level, redirects output to stdout and
    formats log messages.

    Parameters:
        logLevel: str
            Log level to be set for logging.
            One of the following in order of most restrictive to most verbose.
                'CRITICAL'
                'FATAL'
                'ERROR'
                'WARN'/'WARNING'
                'INFO'
                'DEBUG'
    """
    root = logging.getLogger()
    root.setLevel(logging.getLevelName(logLevel))

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.getLevelName(logLevel))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)
