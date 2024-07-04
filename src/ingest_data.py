import argparse
import logging
import os
import tarfile

import pandas as pd
from six.moves import urllib
from sklearn.model_selection import StratifiedShuffleSplit
import mlflow

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("..", "data", "raw")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """
    Fetches the housing data from a specified URL and saves it locally.

    Parameters:
    housing_url (str): URL of the housing data file.
    housing_path (str): Local directory where the data will be saved.
    """
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    """
    Loads the housing data from a CSV file.

    Parameters:
    housing_path (str): Local directory where the data is stored.

    Returns:
    pandas.DataFrame: Loaded housing data as a pandas DataFrame.
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def prepare_data(output_folder):
    """
    Prepares the housing data for training and validation.

    This function fetches the housing data, splits it into training and testing sets
    using stratified sampling based on income categories, and saves the processed
    data as CSV files.

    Parameters:
    output_folder (str): Path to the directory where processed data will be saved.
    """
    fetch_housing_data()
    housing = load_housing_data()

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, float("inf")],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    os.makedirs(os.path.join(output_folder, "processed"), exist_ok=True)
    strat_train_set.to_csv(
        os.path.join(output_folder, "processed", "train.csv"), index=False
    )
    strat_test_set.to_csv(
        os.path.join(output_folder, "processed", "test.csv"), index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest data for training and validation."
    )
    parser.add_argument(
        "--output-folder",
        default=os.path.join("..", "data"),
        help="Path to save the output data",
    )
    parser.add_argument("--log-level", default="INFO", help="Set the logging level")
    parser.add_argument("--log-path", help="Path to save the log file")
    parser.add_argument(
        "--no-console-log", action="store_true", help="Disable console logging"
    )

    args = parser.parse_args()

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)

    if args.log_path:
        logging.basicConfig(filename=args.log_path, level=log_level, format=log_format)
    else:
        logging.basicConfig(level=log_level, format=log_format)

    if args.no_console_log and not args.log_path:
        logging.getLogger().addHandler(logging.NullHandler())

    with mlflow.start_run(run_name="Data Ingestion") as run:
        mlflow.log_params(vars(args))

        logging.info("Starting data ingestion process")
        train_file, test_file = prepare_data(args.output_folder)
        logging.info("Data ingestion completed successfully")

        mlflow.log_artifact(train_file, "processed_data")
        mlflow.log_artifact(test_file, "processed_data")
