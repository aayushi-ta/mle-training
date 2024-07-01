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

##Project structure

```
.
├── README.md
├── artifacts
│   └── best_model.pkl
├── data
│   ├── processed
│   │   ├── test.csv
│   │   └── train.csv
│   └── raw
│       ├── housing.csv
│       └── housing.tgz
├── dist
│   ├── housing_data_project-0.2-py3-none-any.whl
│   └── housing_data_project-0.2.tar.gz
├── docs
│   ├── Makefile
│   ├── _build
│   │   ├── doctrees
│   │   └── html
│   │       ├── _sources
│   │       ├── _static
│   │       ├── genindex.html
│   │       ├── index.html
│   │       ├── ingest_data.html
│   │       ├── modules.html
│   │       ├── objects.inv
│   │       ├── score.html
│   │       ├── search.html
│   │       ├── searchindex.js
│   │       └── train.html
│   ├── _static
│   ├── _templates
│   ├── conf.py
│   ├── index.rst
│   ├── ingest_data.rst
│   ├── make.bat
│   ├── modules.rst
│   ├── score.rst
│   └── train.rst
├── env.yml
├── isort.cfg
├── nonstandardcode.py
├── pyproject.toml
├── scores
│   └── scores.txt
├── setup.py
├── src
│   ├── __init__.py
│   ├── housing_data_project.egg-info
│   │   └── top_level.txt
│   ├── ingest_data.py
│   ├── score.py
│   └── train.py
└── tests
    ├── __pycache__
    ├── test_data_ingestion.py
    ├── test_installation.py
    └── test_training.py
```
markdown
## Execution Instructions

### Installation

To install and run this project, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/aayushi-ta/mle-training.git
cd mle-training
```

2.Install dependencies using Conda:

```bash
conda env create -f env.yml
conda activate mle-dev
```
3. Install the package:

```bash
python -m pip install .
```

or

```bash
pip install housing_data_project-0.2-py3-none-any.whl
```

Usage

Data Ingestion

Run ingest_data.py to download and create training and validation datasets:


```bash
python src/ingest_data.py --output-folder data/processed
```

Model Training

Run train.py to train the model:

```bash
python src/train.py --input-folder data/processed --output-folder artifacts
```

Model Scoring
Run score.py to score the model:

```bash
python src/score.py --model-folder artifacts --dataset-folder data/processed --output-folder
```

scores
Logging
All scripts in src support logging configuration. Example usage:

```bash
python src/train.py --log-level DEBUG --log-path logs/training.log
```

Testing
To verify correct installation and functionality, go inside tests directory and run:

```bash
pytest
```





