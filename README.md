## UPDATE 5/25/2020
New version of backend and more extensive examples to match paper's results coming soon

# Mapped Convolutions
**Official PyTorch implementation of the mapped convolution operation**

This repository contains the "Mapped Convolutions" library. It is written to be a Python extension to PyTorch and can run on either GPU (needs CUDA) or CPU.


## Set Up

I highly recommend using some kind of virtual environment, like [Conda](https://www.anaconda.com/), [virtualenv](https://virtualenv.pypa.io/en/latest/), or [Docker](https://www.docker.com/).


### Dependencies

To install the Python dependencies, you can either use the provided Conda YML file (for a Conda environment) or use the `requirements.txt` file for a `pip` installation.

For Conda, first [install Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) and then call the command:

`conda env create -f mapped-conv.yml`

For `pip`, navigate to the top-level of this project and install the dependencies using:

`pip install -r requirements.txt`

This code has been tested with Python 3.7, PyTorch 1.0, and CUDA 10.0. *I cannot help you with driver incompatibility issues.*

Note: I know that PyTorch has been updated since I originally wrote this package. I will try to improve compatibility with the more recent PyTorch version as I find time.*


### Installation

The library has been set up as a PyTorch extension. All you need to do to install is navigate to the `package` directory and run:

`python setup install`

This should take care of all the compilation and installation to your Python environment


### Usage

To use mapped convolutions, simply import the desired subpackage. For example:

```
# my_file.py
import mapped_convolutions.nn as mcnn
import mapped_convolutions.util as mcutil
```

### Tests

After you've installed the package, I highly recommend you run the unit test suite to make sure everything has installed correctly. It only take a minute. To do this, navigate to the `package` directory and then use the command:

`python -m pytest`

This will trigger all unit tests to run. All 52 should pass successfully. If you do not have a CUDA-enabled machine, 26 should pass and 26 should be skipped.


## Related papers

Please also read our related papers:

 - [Mapped Convolutions](https://arxiv.org/abs/1906.11096)
 - [Convolutions on Spherical Images](http://openaccess.thecvf.com/content_CVPRW_2019/papers/SUMO/Eder_Convolutions_on_Spherical_Images_CVPRW_2019_paper.pdf)

## Credit
If you find this code useful for your research, please cite:

```
@article{eder2019mapped,
Author = {Marc Eder and True Price and Thanh Vu and Akash Bapat and Jan-Michael Frahm},
Title = {Mapped Convolutions},
Year = {2019},
Eprint = {arXiv:1906.11096}}

@inproceedings{eder2019convolutions,
  title={Convolutions on Spherical Images},
  author={Eder, Marc and Frahm, Jan-Michael},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  pages={1--5},
  year={2019}
}
```

## TODO

Things I intend to do in the next few weeks

- [x] Push package to public GitHub
- [x] Write some example scripts for resampling to sphere
- [x] Update backend to PyTorch 1.2 compatibility
- [ ] Upload network example
- [ ] Layer docs
- [ ] Clean up some code and improve comments
