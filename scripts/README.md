# Download script for datasets for approximate nearest neighbor search

 Script to download and extract approximate nearest neighbor datasets from
 http://corpus-texmex.irisa.fr/

### USAGE:
```
./download_dataset DATASET_NAME [TARGET_DIR]
```


DATASET_NAME: Name of dataset to download and extract.
- SIFT10K - downloads siftsmall.tar.gz
- SIFT1M  - downloads sift.tar.gz
- GIST1M  - downloads gist.tar.gz

TARGET_DIR:
Optional path to download and extract dataset on.
- Default: ./datasets/
