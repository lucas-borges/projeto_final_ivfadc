#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By: Lucas Ribeiro Borges
# ---------------------------------------------------------------------------
""" Fixed capacity max heap module"""

from heapq import heapify, heappush, heappushpop, nlargest
from typing import Any, List, Tuple

class KeepLargestHeap():
    """
    This class implements a fixed capacity max heap as a wrapper around
    Python heapq.

    Upon reaching max capacity, when a new element is added it is inserted into
    the heap and then the smallest one after the addition is removed.

    Attributes:
        mem: List
            Memory area heapified.
        capacity: int
           Maximum capacity of elements of the heap.
    """

    def __init__(self, capacity: int) -> None:
        """
        The constructor for KeepLargestHeap class.

        capacity: int
           Maximum capacity of elements of the heap.
       """
        self.mem: List = []
        self.capacity = capacity
        heapify(self.mem)

    def add(self, element: Tuple[int, Any]) -> None:
        """
        Adds an element to the heap.

        If the heap is at max capacity, the element is first added and then
        the smallest element is removed from the heap.

        Parameters:
            element: Tuple(int, Any)
                Element to be added to the heap.
                First field of the tuple is the priority of the element.
                Second field is the data if needed.
        """
        if len(self.mem) < self.capacity:
            heappush(self.mem, element)
        else:
            heappushpop(self.mem, element)

    def getData(self) -> List[Tuple[int,Any]]:
        """
        Gets all the data currently on the heap, sorted in decresing order
        of priority.

        Returns:
            List of tuples of elements on the heap.
        """
        return nlargest(self.capacity, self.mem)
