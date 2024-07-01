import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from train import train_model


@pytest.fixture
def sample_train_data_path():
    return os.path.join(os.path.dirname(__file__), "..", "data", "processed/train.csv")


@pytest.fixture
def sample_output_folder():
    return os.path.join(os.path.dirname(__file__), "..", "artifacts")


def test_train_model(sample_train_data_path, sample_output_folder):
    train_model(sample_train_data_path, sample_output_folder)
    model_path = os.path.join(sample_output_folder, "best_model.pkl")
    assert os.path.exists(model_path), f"Expected model file {model_path} not found"
