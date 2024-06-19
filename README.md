# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data. 

The following techniques have been used: 

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## Execution Instructions

### 1. Environment Setup

To set up the environment, create the `mle-dev` environment from `env.yml` using the following command:

```bash
conda env create -f env.yml
```

Activate the mle-dev environment using:
```bash
conda activate mle-dev
```

To execute the Python script for predicting median housing values, use the following command:

```bash
python nonstandardcode.py
```

After running script, output should look like this:

![bandicam 2024-06-19 18-25-18-047](https://github.com/aayushi-ta/mle-training/assets/171973120/fd1dc0cc-1793-48d2-ada7-610d4d1dc9df)












