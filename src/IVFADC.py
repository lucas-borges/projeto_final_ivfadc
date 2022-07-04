#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By: Lucas Ribeiro Borges
# ---------------------------------------------------------------------------
""" Inverted File System with Asymmetric Distance Computation (IVFADC) algorithm module """

import logging
from multiprocessing.sharedctypes import Value
import os
import pickle
from typing import List
import numpy as np

from quantizers.kmeans import KMeans
from quantizers.productQuantizer import ProductQuantizer
from structures.KeepLargestHeap import KeepLargestHeap


class IVFADC:
    """
    This class implements the Inverted File System with Asymmetric Distance Computation (IVFADC)
    structure described in 'Product quantization for nearest neighbor search' by Jegou et al.

    It makes use of a coarse quantizer and a product quantizer to index
    data samples and perform approximate nearest neighbor search on the dataset.

    Attributes:
        coarseQuantizer: KMeans
            Coarse quantizer for first level indexing.
        productQuantizer: ProductQuantizer
            Product quantizer for secondary indexing.
        nearestCoarseNeighborsSearched: int
            Number of nearest coarse centroids added to search space.
        IVF: List[List[IVFentry]]
            Indexing structure.
        trained: boolean
           Boolean representing wether model has been trained or not.
        dataDimension: int
            Dimension of data used to train the model.
    """
    def __init__(self,
                nearestCoarseNeighborsSearched: int=8,
                coarseQuantizerCentroids: int=1000,
                coarseQuantizerMaxIter: int=50,
                coarseQuantizerSeed: int=0,
                productQuantizerNSubquantizers: int=8,
                productQuantizerCentroids: int=256,
                productQuantizerMaxIter: int=50,
                productQuantizerSeed: int=0) -> None:
        """
        The constructor for IVFADC class.

        Parameters:
            nearestCoarseNeighborsSearched: int, default=8
                Number of nearest coarse centroids added to search space.
            coarseQuantizerCentroids: int, default=1000
                Number of clusters to form for the coarse quantizer.
            coarseQuantizerMaxIter: int, default=50
                Maximum number of iterations for the coarse quantizer.
            coarseQuantizerSeed: int, default=0
                Initial seed value for the coarse quantizer.
            productQuantizerNSubquantizers: int, default=8
                Number of subquantizers to create for the product quantizer.
            productQuantizerCentroids: int, default=256
                Number of clusters to form for each subquantizer of the product quantizer.
            productQuantizerMaxIter: int, default=50
                Maximum number of iterations for each subquantizer of the product quantizer.
            productQuantizerSeed: int, default=0
                Initial seed value for each subquantizer of the product quantizer.
        """
        self.coarseQuantizer = KMeans(coarseQuantizerCentroids, coarseQuantizerMaxIter, coarseQuantizerSeed)
        self.productQuantizer = ProductQuantizer(productQuantizerNSubquantizers, productQuantizerCentroids, productQuantizerMaxIter, productQuantizerSeed)
        self.nearestCoarseNeighborsSearched = nearestCoarseNeighborsSearched
        self.IVF: List[List[IVFentry]] = [[] for i in range(self.coarseQuantizer.nClusters)]
        self.trained = False

    def train(self, trainData: np.ndarray) -> None:
        """
        Trains the clustering models.

        First the coarse quantizer is trained on the original data, then the
        product quantizer is trained on the residuals.

        Parameters:
            trainData: np.ndarray of shape (n_samples, data_dimension)
                Train data for the models.
        """
        if not trainData.ndim == 2:
            raise ValueError(f"trainData is not a 2 dimensional array.")
        self.dataDimension = trainData.shape[1]

        logging.debug(f"Training coarse quantizer with data of shape {trainData.shape}.")
        self.coarseQuantizer.fit(trainData)
        logging.debug(f"Coarse quantizer trained in {self.coarseQuantizer.n_iter_} iterations.")

        # calculate residuals
        trainData = trainData - self.coarseQuantizer.getCentroids()[self.coarseQuantizer.labels_]
        logging.debug(f"Train data adjusted to residuals")

        logging.debug(f"Training product quantizer with data of shape {trainData.shape}.")
        self.productQuantizer.fit(trainData)
        logging.debug(f"Product quantizer trained")

        self.trained = True

    def insert(self, id: int, x: np.ndarray) -> None:
        """
        Inserts data into the inverted file system.

        First the data is mapped to the nearest coarse centroid then its residual
        is mapped by the product quantizer into an entry.

        The entry is inserted into the IVF list on the index of the coarse centroid.

        Parameters:
            id: int
                Id of data sample.
            x: np.ndarray of shape (data_dimension,)
                Sample to be inserted.
        """
        if not self.trained:
            raise Exception("Quantizers not trained, can't insert data.")
        if x.shape[0] != self.dataDimension:
            raise ValueError(f"IVFADC was trained on data with {self.dataDimension} dimensions but trying to insert {x.shape} vector.")

        centroidIndex = self.coarseQuantizer.predict(x)[0]
        residual = x - self.coarseQuantizer.getCentroids()[centroidIndex]

        code = self.productQuantizer.predict(residual)
        self.IVF[centroidIndex].append(IVFentry(id,code))

    def search(self, x: np.ndarray, k: int) -> np.ndarray:
        """
        Searches inverted file system for k approximate nearest neighbors of x.

        First nearestCoarseNeighborsSearched coarse indexes are added to the search space
        then the asymetric distance to every entry under those indices is computed.

        Parameters:
            x: np.ndarray of shape (data_dimension,)
                Reference array to find nearest neighbors of.
            k: int
                Number of approximate nearest neighbors to find.

        Returns:
            Array of shape (approximateNearestNeighborsFound,) with ids of samples.
            Number of found nearest neighbors can be lower than k.
        """
        if not self.trained:
            raise Exception("Quantizers not trained, can't search data.")
        if x.shape[0] != self.dataDimension:
            raise ValueError(f"IVFADC was trained on data with {self.dataDimension} dimensions but trying to search {x.shape} vector.")

        centroidIndexes = self.coarseQuantizer.predictNClosestCentroids(x, self.nearestCoarseNeighborsSearched)[0]
        residuals = x - self.coarseQuantizer.getCentroids()[centroidIndexes]

        kNearestNeighbors = KeepLargestHeap(k)
        for residualIndex, residual in enumerate(residuals):
            # nsubquantizers x nclusters x 1
            distancesToCentroids = self.productQuantizer.getDistancesToCentroids(residual)
            for entry in self.IVF[centroidIndexes[residualIndex]]:
                distance = distancesToCentroids[np.arange(entry.code.size),entry.code].sum()
                kNearestNeighbors.add((-distance, entry.id))

        listOfTuples = kNearestNeighbors.getData() # (d, id)
        return np.array(listOfTuples)[:,1].astype(int)

    def saveToFile(self, filepath: str) -> None:
        """
        Saves the current instance of the IVFADC to a file.

        Parameters:
            filepath: str
                Path of file to save to.
        """
        file = open(filepath, "wb")
        pickle.dump(self, file)

    @staticmethod
    def loadFromFile(filepath: str) -> 'IVFADC':
        """
        Loads a previously saved instance of the IVFADC from a file.

        Parameters:
            filepath: str
                Path of file to load from.

        Returns:
            Instance of previously saved IVFADC.
        """
        if not os.path.exists(filepath):
            raise ValueError(f"File {filepath} does not exist, failed to load IVFADC from file.")

        file = open(filepath, "rb")
        return pickle.load(file)


class IVFentry:
    """
    This helper class implements an entry of the IVF.

    Attributes:
        id: int
            Id of the sample represented by the entry.
        code: np.ndarray of shape (productQuantizerNSubquantizers)
            Code of sample residual mapped by the product quantizer.
    """
    def __init__(self, id: int, code) -> None:
        """
        The constructor for IVFentry class.

        Parameters:
            id: int
                Id of the sample represented by the entry.
            code: np.ndarray of shape (productQuantizerNSubquantizers,)
                Code of sample residual mapped by the product quantizer.
        """
        self.id = id
        self.code = code

    def __str__(self) -> str:
        """
        Function describing the string representation of IVFentry.

        Returns:
            String representation of IVFentry.
        """
        return f"{{id({self.id}), code({self.getCode()})}}"

    def getCode(self) -> np.ndarray:
        """
        Getter function for IVFentry code.

        Returns:
            Array of shape (productQuantizerNSubquantizers,) with the entry code.
        """
        return self.code
