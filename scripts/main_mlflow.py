import argparse
import logging
import mlflow
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from ingest_data import prepare_data
from train import train_model
from score import score_model


def main():
    parser = argparse.ArgumentParser(
        description="Main script to run mlflow with housing data project."
    )
    parser.add_argument(
        "--train-data",
        default="../data/processed/train.csv",
        help="Path to the training data",
    )
    parser.add_argument(
        "--output-folder", default="../artifacts", help="Path to save the trained model"
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

    logging.info("Starting main mlflow run")

    mlflow.set_tracking_uri("http://127.0.0.1:8080")

    with mlflow.start_run(run_name="Main Run") as main_run:
        mlflow.log_param("project", "housing_data_project")

        # Pass args to train_model function
        train_model(args.train_data, args.output_folder, args)

        logging.info("Main mlflow run completed successfully")


if __name__ == "__main__":
    main()
