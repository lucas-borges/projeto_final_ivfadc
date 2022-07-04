#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By: Lucas Ribeiro Borges
# ---------------------------------------------------------------------------
""" Product Quantizer testing module """

import pytest
from quantizers.productQuantizer import ProductQuantizer
import numpy as np

def test_predict_singleVector3m6D():
    """
    Tests predict behavior on simple 3 subquantizers 6 dimensional model.
    """
    quantizer = ProductQuantizer(3, 4, 50, 0)
    quantizer.centroids = np.array([
                                    [[1,1], [1,-1], [-1,-1], [-1,1]],
                                    [[1,-1], [-1,-1], [-1,1], [1,1]],
                                    [[-1,-1], [-1,1], [1,1], [1,-1]]
                                    ])
    quantizer._ProductQuantizer__fillDataDimensionInformation(6)
    quantizer.trained = True

    # distancesToCentroids = quantizer.getDistancesToCentroids(np.array([2,2,2,2,2,2]))
    # print(distancesToCentroids)
    assert np.array_equal(quantizer.predict(np.array([2,2,2,2,2,2])), np.array([0,3,2]))
    assert np.array_equal(quantizer.predict(np.array([2,-2,2,-2,2,-2])), np.array([1,0,3]))
    assert np.array_equal(quantizer.predict(np.array([-2,-2,-2,-2,-2,-2])), np.array([2,1,0]))
    assert np.array_equal(quantizer.predict(np.array([-2,2,-2,2,-2,2])), np.array([3,2,1]))


def test_predict_singleVector3m9D():
    """
    Tests predict behavior on simple 3 subquantizers 9 dimensional model.
    """
    quantizer = ProductQuantizer(3, 8, 50, 0)
    quantizer._ProductQuantizer__fillDataDimensionInformation(9)
    quantizer.centroids = np.array([
                                    [[1,1,1], [1,1,-1], [1,-1,-1], [1,-1,1], [-1,1,1], [-1,1,-1], [-1,-1,-1], [-1,-1,1]],
                                    [[1,-1,-1], [1,-1,1], [-1,1,1], [-1,1,-1], [-1,-1,-1], [-1,-1,1], [1,1,1], [1,1,-1]], # 6rot
                                    [[-1,1,-1], [-1,-1,-1], [-1,-1,1], [1,1,1], [1,1,-1], [1,-1,-1], [1,-1,1], [-1,1,1]] # 3rot
                                    ])
    quantizer.trained = True

    # print(quantizer.predict(np.array([2,2,2, 2,2,2, 2,2,2])))
    assert np.array_equal(quantizer.predict(np.array([2,2,2, 2,2,2, 2,2,2])), np.array([0,6,3]))
    assert np.array_equal(quantizer.predict(np.array([-2,2,2 ,-2,2,2, -2,2,2])), np.array([4,2,7]))
    assert np.array_equal(quantizer.predict(np.array([2,-2,2, 2,-2,2, 2,-2,2])), np.array([3,1,6]))
    assert np.array_equal(quantizer.predict(np.array([2,2,-2, 2,2,-2, 2,2,-2])), np.array([1,7,4]))



def test_getCentroids_untrained():
    """
    GetCentroids on untrained model should raise Exception.
    """
    quantizer = ProductQuantizer(3, 4, 50, 0)
    quantizer.centroids = np.array([
                                    [[1,1], [1,-1], [-1,-1], [-1,1]],
                                    [[1,-1], [-1,-1], [-1,1], [1,1]],
                                    [[-1,-1], [-1,1], [1,1], [1,-1]]
                                    ])

    with pytest.raises(Exception):
        quantizer.getCentroids()

def test_predict_untrained():
    """
    Predict on untrained model should raise Exception.
    """
    quantizer = ProductQuantizer(3, 4, 50, 0)
    quantizer.centroids = np.array([
                                    [[1,1], [1,-1], [-1,-1], [-1,1]],
                                    [[1,-1], [-1,-1], [-1,1], [1,1]],
                                    [[-1,-1], [-1,1], [1,1], [1,-1]]
                                    ])

    with pytest.raises(Exception):
        quantizer.predict(np.array([2,2,2,2,2,2]))

def test_fit_simple3m6D_equalSubvectors():
    """
    Tests fit behavior on simple 3 subquantizers 6 dimensional model.
    """
    rng = np.random.default_rng(1)
    centers = [[2,2], [2,-2], [-2,-2], [-2,2]]
    data = rng.normal(centers[0], [0.2,0.2], [1000,2])
    for center in centers[1:]:
        data = np.concatenate([data,rng.normal(center, [0.2,0.2], [1000,2])])
    data = np.concatenate([data]*3, axis=1)

    quantizer = ProductQuantizer(3, 4, 10, 1)
    quantizer.fit(data)

    # print(quantizer.getCentroids())

    assert quantizer.getCentroids().shape == (3,4,2)
    assert not np.isnan(quantizer.getCentroids()).any()
    assert np.allclose(quantizer.getCentroids()[0], quantizer.getCentroids()[1])
    assert np.allclose(quantizer.getCentroids()[1], quantizer.getCentroids()[2])

def test_fit_simple3m6D():
    """
    Tests predict behavior on simple 3 subquantizers 6 dimensional model.
    """
    rng = np.random.default_rng(1)
    centers = [[2,2], [2,-2], [-2,-2], [-2,2]]
    data = rng.normal(centers[0], [0.2,0.2], [1000,2])
    for center in centers[1:]:
        data = np.concatenate([data,rng.normal(center, [0.2,0.2], [1000,2])])
    data = np.concatenate([data,rng.permutation(data), rng.permutation(data)], axis=1)

    quantizer = ProductQuantizer(3, 4, 10, 1)
    quantizer.fit(data)

    # print(quantizer.getCentroids())

    assert quantizer.getCentroids().shape == (3,4,2)
    assert not np.isnan(quantizer.getCentroids()).any()
    assert not np.array_equal(quantizer.getCentroids()[0], quantizer.getCentroids()[1])
    assert not np.array_equal(quantizer.getCentroids()[1], quantizer.getCentroids()[2])
