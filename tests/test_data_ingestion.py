import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from ingest_data import fetch_housing_data, prepare_data

TEST_HOUSING_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"


@pytest.fixture(scope="module")
def setup_test_environment():
    os.makedirs(TEST_HOUSING_PATH, exist_ok=True)
    yield


def test_prepare_data(setup_test_environment):
    output_folder = os.path.join(
        os.path.dirname(__file__), "..", "data"
    )  # Adjusted path to data directory
    prepare_data(output_folder)

    assert os.path.exists(os.path.join(output_folder, "processed"))
    assert os.path.exists(os.path.join(output_folder, "processed", "train.csv"))
    assert os.path.exists(os.path.join(output_folder, "processed", "test.csv"))

    train_data = pd.read_csv(os.path.join(output_folder, "processed", "train.csv"))
    test_data = pd.read_csv(os.path.join(output_folder, "processed", "test.csv"))
    expected_columns = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
    ]
    assert all(col in train_data.columns for col in expected_columns)
    assert all(col in test_data.columns for col in expected_columns)


if __name__ == "__main__":
    pytest.main()
