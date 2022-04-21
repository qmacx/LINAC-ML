# LINAC-ML

Repository containing all relevant codes for the misalignment detection in linear particle accelerators.

## mlmodels 
 - contains the final Machine Learning models configuration and training file, i.e finalclf.py and finalregression.py
 - contains the models and functions used to compute uncertainty, i.e uncertainty.py, clfuncertainty.py and regressionuncertainty.py
 - archived_models contains the historic models used as part of the development process but which are now obsolete
 - the bash script get_results.sh runs the relevant models to produce final results which are stored in results.txt

## data
 - contains the raw data files provided by Craig Edwards who produced the data as part of his thesis

## simulations
 - contains functions which can be used to simulate the transport line across a parameter grid to create a dataset, this later became obsolete once the data had been provided

## plots
 - contains the relevant plots for production of the final report
