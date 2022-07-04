# IVFADC configuration file

The parameters for IVFADC training and search can be configure with an `.ini`-like file. An example configuration is provided on `siftsmall.ini`.

## Coarse Quantizer parameters

The coarse quantizer is a k-means clustering algorithm.
```
[coarseQuantizer]
numberCentroids = 1000
maxIterations = 50
seed = 0
```
- `numberCentroids`: number of centroids of the k-means clustering.
- `maxIterations`: maximum number of iterations for k-means.
- `seed`: seed for the coarse quantizer random number generator.

## Product Quantizer parameters

The product quantizer divides the space into subvectors and trains a k-means clustering algorithm on each subvector space separately.
```
[productQuantizer]
numberSubquantizers = 8
numberCentroids = 256
maxIterations = 50
seed = 0
```
- `numberSubquantizers`: number of subquantizers to train, the data dimension must be divisible by the number of subquantizers.
- `numberCentroids`: number of centroids for each subquantizer.
- `maxIterations`: maximum number of iterations for each subquantizer.
- `seed`: seed for the random number generator for each subquantizer.

## IVFADC parameters
```
[ivfadc]
coarseNeighborsLookup = 8
nearestNeighbors = 100
```
- `coarseNeighborsLookup`: Number of coarse quantizer indices that will be added to the search space for approximate nearest neighbors.
- `nearestNeighbors`: Number of approximate nearest neighbors the algorithm will try to find for each query vector.

## Datasets parameters
```
[datasets]
datasetName = siftsmall
basePath = datasets/${datasetName}/${datasetName}_
trainSet = ${basePath}learn.fvecs
baseSet = ${basePath}base.fvecs
querySet = ${basePath}query.fvecs
groundTruth = ${basePath}groundtruth.ivecs
```
- `datasetName`: dataset name for configuration reporting.
- `trainSet`: training data filename.
- `baseSet`: base data filename.
- `querySet`: query data filename.
- `groundTruth`: ground truth filename.

## Misc parameters
```
[misc]
logLevel = DEBUG
recallRs = [1,5,10,20,50,100]
```
- `logLevel`: Minimum log level to show on stdout. One of the following in order of most restrictive to most verbose:
  - `CRITICAL`, `FATAL`, `ERROR`, `WARN`/`WARNING`, `INFO`, `DEBUG`.
- `recallRs`: list of values to compute recall for.
  - The performance measure is recall@R: average rate of queries in which the nearest neighbor is ranked within the top R positions.
