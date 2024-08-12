import argparse
import logging
import os
import tarfile
import pandas as pd
from six.moves import urllib  # type: ignore
from sklearn.model_selection import StratifiedShuffleSplit

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    logging.info(f"Fetching data from {housing_url}")
    os.makedirs(housing_path, exist_ok=True)
    logging.info(f"Created directory {housing_path}")
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    logging.info(f"Downloaded housing data to {tgz_path}")
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    logging.info(f"Extracted housing data to {housing_path}")


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    logging.info(f"Loading data from {csv_path}")
    return pd.read_csv(csv_path)


def prepare_data(output_folder):
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

    processed_path = os.path.join(output_folder, "processed")
    os.makedirs(processed_path, exist_ok=True)
    logging.info(f"Created directory {processed_path}")

    train_file = os.path.join(processed_path, "train.csv")
    test_file = os.path.join(processed_path, "test.csv")

    strat_train_set.to_csv(train_file, index=False)
    strat_test_set.to_csv(test_file, index=False)
    logging.info(f"Saved train data to {train_file}")
    logging.info(f"Saved test data to {test_file}")

    return train_file, test_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest data for training and validation."
    )
    parser.add_argument(
        "--output-folder",
        default=os.path.join(os.path.dirname(__file__), "..", "data"),
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

    logging.info(f"Starting data preparation with output folder {args.output_folder}")
    train_file, test_file = prepare_data(args.output_folder)
    logging.info(f"Data preparation completed. Train file: {train_file}, Test file: {test_file}")
