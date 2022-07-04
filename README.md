# INF2102 - PROJETO FINAL DE PROGRAMACAO - 2022.1 - 3WA

Autor: Lucas Ribeiro Borges

Neste repositório se encontra o Projeto Final de Programação. Ele implementa o algoritmo e estrutura de dados descrito por [Product quantization for nearest neighbor search (Jegou, H. et. al.)](https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf).

Essa estrutura se trata de um índice invertido que utiliza dois quantizadores para indexar um dataset e permitir busca por vizinho mais próximo aproximado.

Você poderá encontrar o seguinte nos diferentes diretórios:
- [configs/](configs/): Exemplo de arquivos de configuração.
- [docs/](docs/): Documentação e especificação.
- [scripts/](scripts/): Script de download dos datasets.
- [src/](src/): Código fonte.
- [tests/](tests/): Testes automatizados.

# Inverted File System with Asymmetric Distance Computation (IVFADC)

This repository implements the algorithm and data structure described by [Product quantization for nearest neighbor search (Jegou, H. et. al.)](https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf).

This structure consist of an inverted file index and utilizes two different quantizers to index a dataset and provide efficient approximate nearest neighbor search.

This repository can be used as both a configurable script to evaluate the effect of different parameters and datasets on recall@R performance as well as a library package providing an IVFADC implementation.

You will find the following in the different directories:
- [configs/](configs/): Example configuration files.
- [docs/](docs/): Extra documentation and specification.
- [scripts/](scripts/): Dataset download script.
- [src/](src/): Source code.
- [tests/](tests/): Automated tests.

# Quickstart

You will need `python3` and `pipenv` to install this package. Checkout the [pipenv](https://pipenv.pypa.io/en/latest/install/) page for instructions on installing `pipenv`.

## Recommended installation steps

Assuming you already have `python3` installed, the following steps are recommended:

```
pip install --user pipenv
export PIPENV_VENV_IN_PROJECT=1
pipenv sync
```

If you are using WSL2 and pipenv hangs, check the [Troubleshooting](#troubleshooting) section.

## Running the script on datasets for approximate nearest neighbor search

You can quickly train and populate an IVFADC using provided scripts.

### Download dataset

A dataset download script is provided on `scripts/download_dataset.sh`, checkout the its [README](scripts/README.md)

```
./scripts/download_dataset.sh SIFT10K
```

### Configure parameters

The algorithm parameters can be declared on a `.ini`-like file. An example and documentation can be found on [configs/](configs/README.md)

### Run script

You can run the base script with the following command:

```
pipenv run python src/main.py configs/siftsmall.ini
```

The script will report the configurations used for that run and the recall@R for the specified R values.

recall@R is the performance measure: average rate of queries in which the nearest neighbor is ranked within the top R positions.

## Installing as a package for personal use

You can install this package and import it for personal use with:

```
from ivf_adc.IVFADC import IVFADC
```

Check the docstrings and `docs/` for full documentation.


# Running tests

To run the test suite located in `/tests` you will need to install the development dependencies with:

```
pipenv sync --dev
```

Once the development dependencies are installed, you can run the entire test suite with:

```
pipenv run pytest
```

## Running specific test file

To run a specific test file you can run:

```
pipenv run pytest tests/<filename>
```

## Running specific test method

To run a specific test file you can run:

```
pipenv run pytest tests/<filename>::<method_name>
```

## Show stdout for passing tests

To show `stdout` output like `print` on passing tests, use the `-rP` option:

```
pipenv run pytest -rP
```

# Troubleshooting

- If running the project on WSL2, you might need to unset your `DISPLAY` environment variable to properly run `pipenv`. You can do so with:
```
DISPLAY=
```
