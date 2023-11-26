
# Welcome to spinSimulations! 

This is a python module that is being developed to provide some functionality needed to simulate nuclear magnetic resonance pulse sequences (NMR). Currently it is under heavy development and features will be added as and where we need them. If you are interested in contributing or would like to add some functionaility then get in touch and we would love to hear from you. 

Part of this project also started as a way to explore fast and effient methods to do spin simulations so any ideas around this are welcome. 

If you would like to see an example how to use this then please see
SpinSimulation/examples/random_spectrum.py

## Install 

This module requires the following modules: 
- numpy 
- scipy 
- cython

Once these are installed you can install the module as you wish. For example with: 

python3 setup.py install

## Currently implimented

Currently in this module there is basic functionality for
- simulations in hilbert space 
- constructing hamiltonians for chemical shifts and scalar couplings 
- propergating free induction decays 

## For Contributors

Going forwards (currently not implimented yet for existing code) it would be good to 
- format code to PEP8 with black 
- check code with flake8 
- write docstrings with the numpy doc style see: https://numpydoc.readthedocs.io/en/latest/format.html
- I know scientists love one letter varriables but lets try to use descriptive names!

# Regarding editors / IDEs
We have configurations for VSC and .editorconfig available. Feel free to add your own!
