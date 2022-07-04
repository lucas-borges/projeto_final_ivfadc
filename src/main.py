import json
import logging
import sys
from typing import List
import numpy as np
from configparser import ConfigParser, ExtendedInterpolation

from IVFADC import IVFADC
from readers.ReaderFactory import ReaderFactory
from utils import configureLogging

def main(config: ConfigParser) -> None:
    # Train
    trainData = ReaderFactory.getReader(config["datasets"]["trainSet"]).read()
    ivfadc = IVFADC(nearestCoarseNeighborsSearched=config.getint("ivfadc", "coarseNeighborsLookup"),
                    coarseQuantizerCentroids=config.getint("coarseQuantizer","numberCentroids"),
                    coarseQuantizerMaxIter=config.getint("coarseQuantizer","maxIterations"),
                    coarseQuantizerSeed=config.getint("coarseQuantizer","seed"),
                    productQuantizerNSubquantizers=config.getint("productQuantizer","numberSubquantizers"),
                    productQuantizerCentroids=config.getint("productQuantizer","numberCentroids"),
                    productQuantizerMaxIter=config.getint("productQuantizer","maxIterations"),
                    productQuantizerSeed=config.getint("productQuantizer","seed"))
    logging.info(f"Beginning IVFADC training with {trainData.shape} training data.")
    ivfadc.train(trainData)
    trainData = None
    logging.info(f"Finished training IVFADC.")

    # Fill IVFADC with base dataset
    data = ReaderFactory.getReader(config["datasets"]["baseSet"]).read()
    logging.info(f"Populating IVFADC with {data.shape} data.")
    for i, vector in enumerate(data):
        ivfadc.insert(i, vector)
    data = None
    logging.info(f"IVFADC populated.")

    # Search queries
    queryData = ReaderFactory.getReader(config["datasets"]["querySet"]).read()
    nearestNeighbors = config.getint("ivfadc", "nearestNeighbors")
    logging.info(f"Querying {nearestNeighbors} nearest neighbors for {queryData.shape} query.")
    queryResults = searchQueries(ivfadc, queryData, nearestNeighbors)
    queryData = None
    logging.info(f"Results computed for {nearestNeighbors} nearest neighbors")

    # Compute Recalls
    groundTruth = ReaderFactory.getReader(config["datasets"]["groundTruth"]).read()
    recallRs = json.loads(config.get("misc","recallRs"))
    recallValues = evaluateResults(groundTruth, queryResults, recallRs)

    # Report results
    reportConfig(config)
    reportRecalls(recallRs, recallValues)

def reportConfig(config: ConfigParser):
    print("[Datasets]")
    print(f'datasetName={config.get("datasets","datasetName")}')
    print(f"[Coarse Quantizer]")
    print(f'numberCentroids={config.get("coarseQuantizer","numberCentroids")}, maxIterations={config.get("coarseQuantizer","maxIterations")}, seed={config.get("coarseQuantizer","seed")}')
    print(f"[Product Quantizer]")
    print(f'numberSubquantizers={config.get("productQuantizer","numberSubquantizers")}, numberCentroids={config.get("productQuantizer","numberCentroids")}, maxIterations={config.get("productQuantizer", "maxIterations")}, seed={config.get("productQuantizer", "seed")}')
    print("[IVFADC]")
    print(f'coarseNeighborsLookup={config.get("ivfadc","coarseNeighborsLookup")}, nearestNeighbors={config.get("ivfadc","nearestNeighbors")}')

def reportRecalls(recallRs, recallValues):
    for i in range(len(recallRs)):
        print(f"recall@{recallRs[i]}: {recallValues[i]}")

def evaluateResults(groundTruth: np.ndarray, queryResults: List, recallRs: List[int]) -> List[float]:
    recalls = []
    for r in recallRs:
        recall = calculateRecallR(groundTruth, queryResults, r)
        recalls.append(recall)
    return recalls

def calculateRecallR(groundTruth: np.ndarray, results: List, R: int) -> float:
    n = 0
    for i, result in enumerate(results):
        if groundTruth[i][0] in result[:R]:
            n+=1

    return n/len(results)

def searchQueries(ivfadc: IVFADC, queries: np.ndarray, nearestNeighbors: int) -> List:
    results = []
    for i, query in enumerate(queries):
        results.append(ivfadc.search(query, nearestNeighbors))

    return results

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} CONFIG_PATH")
        print("")
        print(f"    CONFIG_PATH: Path to configuration file.")
        sys.exit(2)
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(sys.argv[1])
    configureLogging(config["misc"]["logLevel"])
    main(config)
