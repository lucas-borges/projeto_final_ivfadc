#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By: Lucas Ribeiro Borges
# ---------------------------------------------------------------------------
""" IVFADC testing module """

import numpy as np
from IVFADC import IVFADC
import pytest

from quantizers.kmeans import KMeans
from quantizers.productQuantizer import ProductQuantizer

def test_insert():
    """
    Tests IVFADC insert behavior to insert element into correct inverted index.

    Coarse quantizer centroids with seed of 0:
    [[ 20.  20.   0.   0.]
    [-20. -20.   0.   0.]
    [-20.  20.   0.   0.]
    [ 20. -20.   0.   0.]]
    """
    coarseQuantizer = KMeans(4, 50, 0)
    coarseQuantizer.fit(np.array([[20,20,0,0], [20,-20,0,0], [-20,-20,0,0], [-20,20,0,0]]))

    productQuantizer = ProductQuantizer(2, 4, 50, 0)
    productQuantizer.centroids = np.array([
                                    [[1,1], [1,-1], [-1,-1], [-1,1]],
                                    [[1,-1], [-1,-1], [-1,1], [1,1]],
                                    ])
    productQuantizer._ProductQuantizer__fillDataDimensionInformation(4)
    productQuantizer.trained = True

    ivfadc = IVFADC(nearestCoarseNeighborsSearched=2)
    ivfadc.coarseQuantizer=coarseQuantizer
    ivfadc.productQuantizer=productQuantizer
    ivfadc.dataDimension=4
    ivfadc.trained=True

    ivfadc.insert(5, np.array([21,21,-1,-1]))
    assert len(ivfadc.IVF[0])==1
    assert np.array_equal(ivfadc.IVF[0][0].getCode(), np.array([0,1]))
    assert ivfadc.IVF[0][0].id == 5

    ivfadc.insert(7, np.array([18,21,-2,-2]))
    assert len(ivfadc.IVF[0])==2
    assert np.array_equal(ivfadc.IVF[0][1].getCode(), np.array([3,1]))
    assert ivfadc.IVF[0][1].id == 7

    ivfadc.insert(11, np.array([18,-21,-2,-2]))
    assert len(ivfadc.IVF[3])==1
    assert np.array_equal(ivfadc.IVF[3][0].getCode(), np.array([2,1]))
    assert ivfadc.IVF[3][0].id == 11

def test_search():
    """
    Tests IVFADC search behavior to find correct elements.

    Coarse quantizer centroids with seed of 0:
    [[ 20.  20.   0.   0.]
    [-20. -20.   0.   0.]
    [-20.  20.   0.   0.]
    [ 20. -20.   0.   0.]]
    """
    coarseQuantizer = KMeans(4, 50, 0)
    coarseQuantizer.fit(np.array([[20,20,0,0], [20,-20,0,0], [-20,-20,0,0], [-20,20,0,0]]))

    productQuantizer = ProductQuantizer(2, 4, 50, 0)
    productQuantizer.centroids = np.array([
                                    [[1,1], [1,-1], [-1,-1], [-1,1]],
                                    [[1,-1], [-1,-1], [-1,1], [1,1]],
                                    ])
    productQuantizer._ProductQuantizer__fillDataDimensionInformation(4)
    productQuantizer.trained = True

    ivfadc = IVFADC(nearestCoarseNeighborsSearched=2)
    ivfadc.coarseQuantizer=coarseQuantizer
    ivfadc.productQuantizer=productQuantizer
    ivfadc.dataDimension=4
    ivfadc.trained=True

    ivfadc.insert(5, np.array([21,21,-1,-1]))
    ivfadc.insert(7, np.array([18,21,-2,-2]))
    ivfadc.insert(11, np.array([18,-21,-2,-2]))

    results = ivfadc.search(np.array([21,21,1,1]), 2)
    assert len(results) == 2
    assert results[0] == 5
    assert results[1] == 7

    results = ivfadc.search(np.array([10,-15,1,3]), 1)
    assert len(results) == 1
    assert results[0] == 11

    results = ivfadc.search(np.array([25,-15,1,3]), 2)
    assert len(results) == 2
    assert results[0] == 11
    assert results[1] == 5 or results[1] == 7

def test_train_incorrectTrainData():
    """
    Train with non 2D array should raise ValueError.
    """
    ivfadc = IVFADC()

    with pytest.raises(ValueError):
        ivfadc.train(np.arange(27).reshape(3,3,3))

def test_insert_notTrained():
    """
    insert on untrained model should raise Exception.
    """
    ivfadc = IVFADC()

    with pytest.raises(Exception):
        ivfadc.insert(np.array([1,1,1,1]))

def test_search_notTrained():
    """
    search on untrained model should raise Exception.
    """
    ivfadc = IVFADC()

    with pytest.raises(Exception):
        ivfadc.search(np.array([1,1,1,1]))
