#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By: Lucas Ribeiro Borges
# ---------------------------------------------------------------------------
""" Product Quantizer algorithm module """

from typing import List, Tuple
import numpy as np

from quantizers.kmeans import KMeans
from utils import isPowerOfTwo


class ProductQuantizer:
    """
    This class implements the product quantizer algorithm.

    Each subquantizer is an instance of a Lloyd's k-means algorithm which is used
    to create a codebook for a section of the vector. The encoding of a vector is
    the concatenation of each subquantizer encoding.

    Attributes:
        nSubquantizers: int
            Number of subquantizers to create. The dimension of the data to be trained
            must be divisible by this.
        nClusters: int
            Number of clusters to form for each subquantizer and consequently number of
            centroids generated for each subquantizer.
        trained: boolean
            Boolean representing wether model has been trained or not.
        keepSubquantizers: boolean
            Boolean representing wether to keep subquantizer data after training.
            Default value of False frees up memory of subquantizers and keeps only
            the data representing the centroids
        subquantizers: List[KMeans]
            List of subquantizers. If keepSubquantzers is False, this will be set
            to an empty list after training.
        dataDimension: int
            Dimension of data used to train the model.
        subquantizerDimension: int
            Dimension of data dealt by each subquantizer.
            subquantizerDimension = dataDimension//nSubquantizers
    """
    def __init__(self, nSubquantizers: int, nClusters: int, maxIter: int, seed: int=0, keepSubquantizers: bool=False) -> None:
        """
        The constructor for ProductQuantizer class.

        Parameters:
            nSubquantizers: int
                Number of subquantizers to create. The dimension of the data to be trained
                must be divisible by this.
            nClusters: int
                Number of clusters to form for each subquantizer and consequently number of
                centroids generated for each subquantizer.
            maxIter: int
                Maximum number of iterations for each subquantizer.
            seed: int, default=0
                Initial seed value for each subquantizer.
            keepSubquantizers: boolean, default=False
                Boolean representing wether to keep subquantizer data after training.
                Default value of False frees up memory of subquantizers and keeps only
                the data representing the centroids
        """
        if nClusters <= 0:
            raise ValueError("Number of clusters must be greater than 0")
        if maxIter <= 0:
            raise ValueError("Number of iterations must be greater than 0")
        if not isPowerOfTwo(nClusters):
            raise ValueError(f"Number of clusters per subquantizer must be a power of 2, was {nClusters}")

        self.nSubquantizers = nSubquantizers
        self.nClusters = nClusters
        self.trained = False
        self.keepSubquantizers = keepSubquantizers

        self.dataDimension = -1
        self.subquantizerDimension = -1

        self.subquantizers = [KMeans(self.nClusters, maxIter, seed) for i in range(self.nSubquantizers)]

    def fit(self, data: np.ndarray) -> None:
        """
        Trains the clustering model.

        Parameters:
            data: np.ndarray of shape (n_samples, data_dimension)
                Train data for the model.
                The dimension of the training data must be divisible by nSubquantizers.
        """
        self.__fillDataDimensionInformation(data.shape[1])

        for i, slice in enumerate(self.__getSubvectorSlices()):
            self.subquantizers[i].fit(data[:,slice[0]:slice[1]])

        self.centroids = np.stack([self.subquantizers[i].getCentroids() for i in range(self.nSubquantizers)])
        self.trained = True
        if not self.keepSubquantizers:
            self.subquantizers = []

    def getCentroids(self) -> np.ndarray:
        """
        Returns array with subquantizer cluster centers.

        Returns:
            Array of shape (nSubquantizers, nClusters, subquantizerDimension)
        """
        if not self.trained:
            raise Exception("Product quantizer not trained, no centroids.")
        return self.centroids

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Quantizes x: returns the index of the nearest centroid for each
        subvector of x from each subquantizer.

        Parameters:
            x: np.ndarray of shape (data_dimension, )
                Element to find nearest cluster center of for each subquantizer.

        Returns:
            Array of shape (nSubquantizers, )
        """
        if not self.trained:
            raise Exception("Product Quantizer not trained, can't predict.")
        if x.ndim == 1:
            distances = self.getDistancesToCentroids(x)
            # (m, k*, 1) operation with (m, 1) result
            return np.argmin(distances, axis=1)
        else:
            raise Exception("Invalid input array dimensions.")

    def getDistancesToCentroids(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the distance to each centroid for each subquantizer.

        Parameters:
            x: np.ndarray of shape (data_dimension, )
                Element to to calculate distances for each subquantizer.

        Returns:
            Array of shape (nSubquantizers, nClusters)
        """
        if not self.trained:
            raise Exception("Product Quantizer not trained, can't get distances to centroids.")
        # m=self.nSubquantizers, (D* = dataDimension/m), k*=self.nClusters
        # x is a (1, D) array
        # self.centroids is (m, k*, D*) array
        # change x to (m, 1, D*)
        x_reshaped = x.view().reshape(self.nSubquantizers, 1, self.subquantizerDimension)
        # (m, k*, D*) on (m, 1, D*) operation with (m, k*, 1) result
        return np.linalg.norm(self.centroids-x_reshaped, axis=2)

    def __getSubvectorSlices(self) -> List[Tuple[int,int]]:
        """
        Computes list of tuples determining subvectors starts and end points.

        Example:
            For 6 dimensional data with 3 subquantizers, returns:
            [(0,2), (2,4), (4,6)]

        Returns:
            List of tuples (int, int)
        """
        return [(i*self.subquantizerDimension, (i+1)*self.subquantizerDimension) for i in range(self.nSubquantizers)]

    def __fillDataDimensionInformation(self, dataDimension: int) -> None:
        """
        Helper function for dealing with fields regarding data dimension.

        Parameters:
            dataDimension: int
                Dimension of data that will be used with this quantizer.
        """
        self.dataDimension = dataDimension
        if self.dataDimension%self.nSubquantizers != 0:
            raise ValueError(f"Dataset dimension ({self.dataDimension}) is not a mutliple of number of subquantizers ({self.nSubquantizers}).")
        self.subquantizerDimension = self.dataDimension//self.nSubquantizers

    def getCodeBitLength(self) -> int:
        """
        Calculates how many bits a code for the product quantizer needs.

        Returns:
            Integer
        """
        # m [log2 k*]
        return self.nSubquantizers * (self.nClusters-1).bit_length()

