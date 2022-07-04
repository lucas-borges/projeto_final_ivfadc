#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By: Lucas Ribeiro Borges
# ---------------------------------------------------------------------------
""" kmeans wrapper testing module """

from quantizers.kmeans import KMeans
import numpy as np
import pytest

def test_predict_singleVector2D():
    """
    Tests predict behavior on simple 4 centroids model.
    Seed of 0 generates following centroids:
    [[ 1.  1.]
    [-1. -1.]
    [-1.  1.]
    [ 1. -1.]]
    """
    quantizer = KMeans(4, 50, 0)
    quantizer.fit(np.array([[1,1], [1,-1], [-1,-1], [-1,1]]))

    assert quantizer.predict(np.array([2,2])) == 0
    assert quantizer.predict(np.array([2,-2])) == 3
    assert quantizer.predict(np.array([-2,-2])) == 1
    assert quantizer.predict(np.array([-2,2])) == 2
    assert quantizer.predict(np.array([1,1])) == 0

    assert np.array_equal(quantizer.predict(np.array([[2,2], [2,-2], [-2,2]])), np.array([0,3,2]))

def test_predict_singleVector3D():
    """
    Tests predict behavior on simple 8 centroids model.
    Seed of 0 generates following centroids:
    [[-1.  1.  1.]
    [ 1. -1.  1.]
    [-1.  1. -1.]
    [ 1.  1. -1.]
    [ 1. -1. -1.]
    [-1. -1. -1.]
    [ 1.  1.  1.]
    [-1. -1.  1.]]
    """
    quantizer = KMeans(8, 50, 0)
    quantizer.fit(np.array([[1,1,1], [1,1,-1], [1,-1,-1], [1,-1,1],
                                    [-1,1,1], [-1,1,-1], [-1,-1,-1], [-1,-1,1]]))

    assert quantizer.predict(np.array([2,2,2])) == 6
    assert quantizer.predict(np.array([2,2,-2])) == 3
    assert quantizer.predict(np.array([2,-2,-2])) == 4
    assert quantizer.predict(np.array([2,-2,2])) == 1
    assert quantizer.predict(np.array([-2,2,2])) == 0
    assert quantizer.predict(np.array([-2,2,-2])) == 2
    assert quantizer.predict(np.array([-2,-2,-2])) == 5
    assert quantizer.predict(np.array([-2,-2,2])) == 7

    predictions = quantizer.predict(np.array([[2,2,2],[2,2,-2],[2,-2,-2],[2,-2,2],[-2,2,2],[-2,2,-2],[-2,-2,-2],[-2,-2,2]]))
    assert np.array_equal(predictions, np.array([6,3,4,1,0,2,5,7]))

def test_predictNClosestCentroids_2Dvectors():
    """
    Test predictNClosestCentroids on simple 4 centroids model.
    Seed of 0 generates following centroids:
    [[ 1.  1.]
    [-1. -1.]
    [-1.  1.]
    [ 1. -1.]]
    """
    quantizer = KMeans(4, 50, 0)
    quantizer.fit(np.array([[1,1], [1,-1], [-1,-1], [-1,1]]))

    knearest = quantizer.predictNClosestCentroids(np.array([2,0]),2)
    print(knearest)
    c1 = (knearest == 0).any()
    c2 =  (knearest == 3).any()
    assert c1 and c2

    knearest = quantizer.predictNClosestCentroids(np.array([0,-2]),2)
    c1 = (knearest == 1).any()
    c2 = (knearest == 3).any()
    assert c1 and c2

    knearest = quantizer.predictNClosestCentroids(np.array([-2,0]),2)
    c1 = (knearest == 1).any()
    c2 = (knearest == 2).any()
    assert c1 and c2

    knearest = quantizer.predictNClosestCentroids(np.array([0,2]),2)
    c1 = (knearest == 0).any()
    c2 = (knearest == 2).any()
    assert c1 and c2

    knearest = quantizer.predictNClosestCentroids(np.array([[0,2], [-2,0]]),2)
    print(knearest)
    c1 = (knearest[0] == 0).any()
    c2 = (knearest[0] == 2).any()
    c3 = (knearest[1] == 1).any()
    c4 = (knearest[1] == 2).any()
    assert c1 and c2 and c3 and c4

    knearest = quantizer.predictNClosestCentroids(np.array([2,2]),3)
    print(knearest)
    c1 = (knearest == 0).any()
    c2 = (knearest == 2).any()
    c3 = (knearest == 3).any()
    assert c1 and c2 and c3

def test_getCentroids_untrained():
    """
    getCentroids should raise Exception when model is not trained.
    """
    quantizer = KMeans(8, 50, 0)

    with pytest.raises(Exception):
        quantizer.getCentroids()

def test_predict_untrained():
    """
    predict should raise Exception when model is not trained.
    """
    quantizer = KMeans(8, 50, 0)

    with pytest.raises(Exception):
        quantizer.predict(np.array([2,2,2]))

def test_predictNClosestCentroids_untrained():
    """
    predictNClosestCentroids should raise Exception when model is not trained.
    """
    quantizer = KMeans(8, 50, 0)

    with pytest.raises(Exception):
        quantizer.predictNClosestCentroids(np.array([2,2,0]),2)

def test_fit_simple2D():
    """
    fit should produce expected number of centroids.
    """
    rng = np.random.default_rng(1)
    centers = [[2,2], [2,-2], [-2,-2], [-2,2]]
    data = rng.normal(centers[0], [0.2,0.2], [1000,2])
    for center in centers[1:]:
        data = np.concatenate([data,rng.normal(center, [0.2,0.2], [1000,2])])

    quantizer = KMeans(4, 10, 1)
    quantizer.fit(data)

    print(quantizer.getCentroids())

    assert quantizer.getCentroids().shape == (4,2)
    assert not np.isnan(quantizer.getCentroids()).any()

def test_fit_simple3D():
    """
    fit should produce expected number of centroids.
    """
    rng = np.random.default_rng(1)
    centers = [[2,2,2], [2,2,-2], [2,-2,-2], [2,-2,2], [-2,2,2], [-2,2,-2], [-2,-2,-2], [-2,-2,2]]
    data = rng.normal(centers[0], [0.2,0.2,0.2], [1000,3])
    for center in centers[1:]:
        data = np.concatenate([data,rng.normal(center, [0.2,0.2,0.2], [1000,3])])

    quantizer = KMeans(8, 10, 0)
    quantizer.fit(data)

    assert quantizer.getCentroids().shape == (8,3)
    assert not np.isnan(quantizer.getCentroids()).any()
