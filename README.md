# ALGOPORT (alpha version)
Proof-of-concept implementation of the generalized algorithmic portfolio management framework. The development is in extremely early stage and thus this implementation might be unstable. 

## Installation
Install through pip using the following command:
'''
pip install git+https://github.com/astekas/algoport.git
'''

## Requirements 
Note that current version of this package is very dependency-heavy. 
Besides the packages listed in requirements.txt, which are considered "light" dependencies, 
some pieces of functionality have stronger dependencies, which we list here as optional. If they are missing, the corresponding methods and classes will be unavailable and through errors if attempted to be used.
- Python v. 3.8. Specific Python version is required to use the C-based version of MSG_Omega_ratio, which speeds up the numeric integration approximately 15 times. Notice, that if this requirement is not satisfied, MSG_Omega_ratio will still work, just using pure Python version. But optimization will become extremely slow. 
- R installation is required for DEA_AS asset preselection, which uses "additiveDEA" R package (will be installed automatically if missing) via rpy2 underhood. If R installation is not found - DEA_AS preselector will be unavailable. 
- "arch" and "copulas" Python packages are required for GARCH-EVT-COPULA information extraction model. They are not included in standard requirements as "copulas" require Visual C++ installed on the machine. If it is available - one may install the abovementioned packages via pip. 
