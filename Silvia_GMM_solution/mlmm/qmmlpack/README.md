# qmmlpack - a quantum mechanics / machine learning package

Package for machine learning models that interpolate between first-principles calculations of atomistic systems.
It provides efficient C++ code with bindings to Mathematica and Python.

If you are using this software, please cite
> Matthias Rupp: Machine Learning for Quantum Mechanics in a Nutshell, International Journal of Quantum Chemistry, 115(16): 1058–1073, 2015. [DOI](http://dx.doi.org/10.1002/qua.24954)

## Warning

**This package is currently a pre-release and under active development, including possibility of breaking changes without notice.**

## Requirements

Designed for scientific computing, this package requires recent, but not bleeding-edge versions of the following software:

* C++/14 compatible compiler, such as LLVM/Clang&nbsp;3.4 or later; GCC&nbsp;5 or later; Intel icc&nbsp;17.0.4 or later
* Mathematica&nbsp;10.1 or later
* Python&nbsp;3.5 or later, including py.test, NumPy&nbsp;1.11, and SciPy&nbsp;0.17 or later

The packages Python functionality does not require Mathematica.

When compiling with Intel icc, corresponding GCC C++ standard libraries must be version 5 or later.
This means that both Intel and GNU compilers should be available and meet version requirements.

## Installation

Create a local copy of the repository, compile the C++ core routines, then the desired bindings (Python, Mathematica). Optionally, install a system-wide version. Adjust paths accordingly.

    git clone git@gitlab.com:qmml/qmmlpack.git
    ./make -v cpp
    ./make -v python          # independent of Mathematica bindings
    ./make -v mathematica     # independent of Python bindings
    sudo ./make -v install    # optional

Typical paths to adjust include the shell's `PYTHONPATH` environment variable, Python's `sys.path` variable and Mathematica's `$Path` variable. For example:

    export PYTHONPATH="/usr/local/qmmlpack/python:$PYTHONPATH"
    import sys; sys.path.append('/usr/local/qmmlpack/python')
    AppendTo[$Path, "/usr/local/qmmlpack/mathematica"];

Troubleshooting:
* Run the make script with `-v` option and carefully check the output, in particular whether the used compiler, library and Python versions and paths are correct. If not use optional arguments to specify paths (see `./make -h`)
* Check whether all unit tests ran successfully
* If the qmmlpack library can not be found, check whether the `LD_LIBRARY_PATH` environment variable is set correctly

## About

Created and maintained by [Matthias Rupp](http://mrupp.info/) (Fritz Haber Institute of the Max Planck Society, Berlin, Germany).

With contributions by
* Marcel Langer (Fritz Haber Institute of the Max Planck Society, Berlin, Germany)
* Lucas Deecke (Free University of Berlin, Berlin, Germany)
* Haoyan Huo (Peking University, Beijing, China)
