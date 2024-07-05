import argparse
import logging
import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import mlflow


def load_data(file_path):
    """
    Loads data from a CSV file into a pandas DataFrame.

    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    pandas.DataFrame: Loaded data.
    """
    return pd.read_csv(file_path)


def prepare_features(data):
    """
    Prepares features for machine learning modeling.

    This function drops unnecessary columns, creates new features, and handles missing values.

    Parameters:
    data (pandas.DataFrame): Input data.

    Returns:
    pandas.DataFrame: Processed data with new features.
    """
    try:
        data = data.drop("ocean_proximity", axis=1)
        data["rooms_per_household"] = data["total_rooms"] / data["households"]
        data["bedrooms_per_room"] = data["total_bedrooms"] / data["total_rooms"]
        data["population_per_household"] = data["population"] / data["households"]
        logging.info("Features prepared successfully")
        return data
    except Exception as e:
        logging.error(f"Failed to prepare features: {e}")
        raise


def score_model(model_path, test_data, output_folder):
    """
    Scores a machine learning model on test data and saves the results.

    Parameters:
    model_path (str): Path to the trained model file.
    test_data (str): Path to the test dataset.
    output_folder (str): Path to save the scores.
    """
    test_set = load_data(test_data)
    test_set_labels = test_set["median_house_value"].copy()
    test_set = test_set.drop("median_house_value", axis=1)

    test_set_prepared = prepare_features(test_set)
    columns1 = test_set_prepared.columns
    imputer = SimpleImputer(strategy="median")
    test_set_prepared = imputer.fit_transform(test_set_prepared)
    test_set_prepared = pd.DataFrame(test_set_prepared, columns=columns1)

    model_path = os.path.join(args.model_folder, "best_model.pkl")  # Updated model path
    model = pd.read_pickle(model_path)
    predictions = model.predict(test_set_prepared)
    mse = mean_squared_error(test_set_labels, predictions)
    rmse = np.sqrt(mse)

    os.makedirs(output_folder, exist_ok=True)
    scores_path = os.path.join(output_folder, "scores.txt")
    with open(scores_path, "w") as f:
        f.write(f"RMSE: {rmse}\n")

    logging.info(f"Model scoring completed. Scores saved to {scores_path}")

    with mlflow.start_run(run_name="Model Scoring", nested=True) as run:
        mlflow.log_params(vars(args))
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_artifact(scores_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score the model.")
    parser.add_argument(
        "--model-folder",
        default=os.path.join("..", "artifacts"),
        help="Path to the model",
    )
    parser.add_argument(
        "--dataset-folder",
        default=os.path.join("..", "data/processed/test.csv"),
        help="Path to the dataset",
    )
    parser.add_argument(
        "--output-folder",
        default=os.path.join("..", "scores"),
        help="Path to save the scores",
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

    if args.no_console_log:
        logging.getLogger().addHandler(logging.NullHandler())

    logging.info("Starting model scoring process")
    score_model(args.model_folder, args.dataset_folder, args.output_folder)
    logging.info("Model scoring completed successfully")
