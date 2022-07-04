#!/bin/bash
# ---------------------------------------------------------------------------
# Created By: Lucas Ribeiro Borges
# ---------------------------------------------------------------------------
# Script to download and extract approximate nearest neighbor datasets from
# http://corpus-texmex.irisa.fr/
#
# USAGE:
#       ./download_dataset DATASET_NAME [TARGET_DIR]
# ARGS:
#   DATASET_NAME:
#       Name of dataset to download and extract.
#           - SIFT10K - downloads siftsmall.tar.gz
#           - SIFT1M  - downloads sift.tar.gz
#           - GIST1M  - downloads gist.tar.gz
#   TARGET_DIR:
#       Optional path to download and extract dataset on.
#           Default: ./datasets/
# ---------------------------------------------------------------------------

URL_BASE="ftp://ftp.irisa.fr/local/texmex/corpus/"
SCRIPT_NAME="$0"

function error_message {
  echo >&2 "$1"
  echo >&2 "Usage: ${SCRIPT_NAME} DATASET_NAME [TARGET_DIR]

  DATASET_NAME    Name of dataset to download and extract. (SIFT10K, SIFT1M, GIST1M)
  TARGET_DIR      Optional path to download and extract dataset on. (Default: ./datasets/)
"
}

DATASET_NAME="$1"
TARGET_DIR=${2:-"./datasets/"}

if [[ $# -lt 1 ]]; then
  error_message "Wrong number of arguments"
  exit 2
fi

if [[ ${DATASET_NAME} == "SIFT10K" ]]; then
  FILENAME="siftsmall.tar.gz"
elif [[ ${DATASET_NAME} == "SIFT1M" ]]; then
  FILENAME="sift.tar.gz"
elif [[ ${DATASET_NAME} == "GIST1M" ]]; then
  FILENAME="gist.tar.gz"
else
  error_message "Invalid dataset name"
  exit 2
fi

mkdir -p ${TARGET_DIR}
wget ${URL_BASE}/${FILENAME} -O ${TARGET_DIR}/${FILENAME}
cd ${TARGET_DIR}
tar -xvf ${FILENAME}
rm ${FILENAME}
cd -
