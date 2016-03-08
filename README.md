# Expresso/PyCAS

## Expresso
A c++11 symbolic expression manipulation library with wrappers for python 2.7. It it based on pattern matching and term rewriting.

### Dependencies

- A c++11 compatible compiler (such as gcc-4.9)
- cmake (optional)

## PyCAS
PyCAS is a minimalistic computer algebra system based on expresso (it also serves as an example for writing your own modules with expresso). Since it is developed in parallel it is distributed inside the expresso module as expresso.pycas. In the examples directory you can find an introductory tutorial for PyCAS.

### Installation
Installing the latest release of Expresso can be done by cloning directly from the repository:

        git clone https://github.com/TheLartians/Expresso.git
        pip2 install .[pycas]

### Dependencies

- python 2.7
- pip >= 8.0
- boost.python >= 1.55
- numpy
- mpmath

## Citing Expresso/PyCAS
To cite Expresso/PyCAS use the Zenodo DOI:

[![DOI](https://zenodo.org/badge/20604/TheLartians/Expresso.svg)](https://zenodo.org/badge/latestdoi/20604/TheLartians/Expresso)

## Outlook
There are still many possible improvements to expresso/pyCAS. Feel free to fork the project and add your own contributions. Planned features include:

### Expresso
- Match an arbitrary number of patterns at once in even faster time (near constant instead of logarithmic) using a lookup table
- Add better support for commutative patterns (at the moment only the outermost function is matched commutatively)
- Add documentation

### PyCAS
- Implement algorithms for factoring polynomials
- Add support for integrals using a heuristic risch algorithm 
