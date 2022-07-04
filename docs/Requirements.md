# Requirements

## Functional requirements
- System must implement the algorithm and data structure described by [Product quantization for nearest neighbor search (Jegou, H. et. al.)](https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf).
- System must permit the training of a new instance of IVFADC.
- System must permit the insertion of data samples on a trained instance of IVFADC.
- System must permit the search for a user specified number of approximate nearest neighbors on a populated IVFADC instance.
- System must permit to save a trained and populated IVFADC instance to a file.
- System must permit to load a trained and populated IVFADC instance from a file.
- System must permit loading of datasets from files in the formats specified on [Datasets for approximate nearest neighbor search](http://corpus-texmex.irisa.fr/).
- System must permit user expansion of known dataset file formats.

## Non-functional requirements
- System shall be usable as both a configurable script and a library package.
  - Configuration of script shall be simple with persistent configuration files.
- System shall make use of available cores.
- System shall be implemented in Python.
- System shall make use of numpy.
- System shall be carefully documented according to coursework requirements.
- System shall have unit tests.
- System shall make use of mypy type checking


