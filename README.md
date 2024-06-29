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

### Installation

To install and run this project, follow these steps:

1. Clone the repository:

```bash
git clone <repository_url>
cd <project_directory>
```

2.Install dependencies using Conda:

```bash
conda env create -f env.yml
conda activate <environment_name>
```
3. Install the package:

```bash
python -m pip install .
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
All scripts support logging configuration. Example usage:

```bash
python src/train.py --log-level DEBUG --log-path logs/training.log
```

Testing
To verify correct installation and functionality, run:

```bash
pytest
```





