import argparse
import logging
import os

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV


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


def train_model(train_data, output_folder):
    """
    Trains a RandomForestRegressor model using GridSearchCV and saves the best model.

    Parameters:
    train_data (str): Path to the training dataset.
    output_folder (str): Path to save the trained model.
    """
    train_set = load_data(train_data)
    train_set_labels = train_set["median_house_value"].copy()
    train_set = train_set.drop("median_house_value", axis=1)

    train_set_prepared = prepare_features(train_set)
    columns = train_set_prepared.columns
    imputer = SimpleImputer(strategy="median")
    train_set_prepared = imputer.fit_transform(train_set_prepared)
    train_set_prepared = pd.DataFrame(train_set_prepared, columns=columns)

    param_grid = [
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]

    forest_reg = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(train_set_prepared, train_set_labels)

    best_model = grid_search.best_estimator_

    os.makedirs(output_folder, exist_ok=True)
    model_path = os.path.join(output_folder, "best_model.pkl")
    pd.to_pickle(best_model, model_path)

    logging.info(f"Model training completed. Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument(
        "--input-folder",
        default=os.path.join("..", "data", "processed"),
        help="Path to the input data",
    )
    parser.add_argument(
        "--output-folder",
        default=os.path.join("..", "artifacts"),
        help="Path to save the model",
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

    logging.info("Starting model training process")
    train_data = os.path.join(args.input_folder, "train.csv")
    train_model(train_data, args.output_folder)
    logging.info("Model training completed successfully")
