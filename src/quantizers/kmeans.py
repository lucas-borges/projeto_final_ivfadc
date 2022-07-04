#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By: Lucas Ribeiro Borges
# ---------------------------------------------------------------------------
""" K-Means algorithm wrapper module """

import numpy as np
import sklearn.cluster


class KMeans(sklearn.cluster.KMeans):
    """
    This class is a wrapper around the k-means clustering algorithm on sci-kit learn
    that adds a function to get the t nearest centroids to a given vector.

    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

    KMeans is initialized once through k-means++ and uses Lloyd's algorithm.

    Attributes:
        nClusters: int
            Number of clusters to form and consequently number of centroids generated.
        trained: bool
           Boolean representing wether model has been trained or not.
    """

    def __init__(self, nClusters: int, maxIter: int, seed: int=0) -> None:
        """
        The constructor for KMeans wrapper class.

        Parameters:
            nClusters: int
                Number of clusters to form and consequently number of centroids generated.
            maxIter: int
                Maximum number of iterations.
            seed: int, default=0
                Initial seed value.
        """
        if nClusters <= 0:
            raise ValueError("Number of clusters must be greater than 0")
        if maxIter <= 0:
            raise ValueError("Number of iterations must be greater than 0")

        super().__init__(n_clusters=nClusters,
                        init='k-means++',
                        n_init=1,
                        max_iter=maxIter,
                        tol=0.0001,
                        random_state=seed,
                        algorithm='lloyd')
        self.nClusters = nClusters
        self.trained = False

    def fit(self, data: np.ndarray) -> None:
        """
        Trains the clustering model.

        Parameters:
            data: np.ndarray of shape (n_samples, data_dimension)
                Train data for the model.
        """
        super().fit(data)
        self.trained = True

    def getCentroids(self) -> np.ndarray:
        """
        Returns array with cluster centers.

        Returns:
            Array of shape (nClusters, dataDimension)
        """
        if not self.trained:
            raise Exception("k-means not trained, no centroids.")
        return self.cluster_centers_

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Quantizes x: returns the index of the nearest centroid for each
        element of x

        Parameters:
            x: np.ndarray of shape (nSamples, dataDimension) or (dataDimension, )
                Elements to find nearest cluster center of.

        Returns:
            Array of shape (nSamples, )
        """
        if not self.trained:
            raise Exception("k-means not trained, can't predict.")
        elif x.ndim == 1:
            return super().predict(x.reshape(1,-1))
        elif x.ndim == 2:
            return super().predict(x)
        else:
            raise Exception("Invalid input array dimensions.")

    def predictNClosestCentroids(self, x: np.ndarray, N: int) -> np.ndarray:
        """
        Quantizes x: returns the indexes of the N nearest centroids for each
        element of x

        Parameters:
            x: np.ndarray of shape (nSamples, dataDimension) or (dataDimension, )
                Elements to find nearest cluster center of.
            N: int
                Number of nearest centroids to quantize.

        Returns:
            Array of shape (nSamples, N)
        """
        if not self.trained:
            raise Exception("k-means not trained, can't predict.")
        elif x.ndim == 1:
            return np.argpartition(super().transform(x.reshape(1,-1)),N)[:,:N]
        elif x.ndim == 2:
            return np.argpartition(super().transform(x),N)[:,:N]
        else:
            raise Exception("Invalid input array dimensions.")

