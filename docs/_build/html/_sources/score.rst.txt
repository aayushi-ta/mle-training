=====================================
Score Model Module (:mod:`score`)
=====================================

Module Overview
---------------

This module provides functionality to score a machine learning model on test data and save the results.

score_model
-----------

.. function:: score_model(model_path, test_data, output_folder)
   :noindex:

   Scores a machine learning model on test data and saves the results.

   :param model_path: Path to the trained model file.
   :param test_data: Path to the test dataset.
   :param output_folder: Path to save the scores.

   Example::

       # Example usage
       python score.py --model-folder ../artifacts --dataset-folder ../data/processed/test.csv --output-folder ../scores --log-level INFO

   This function loads the test dataset, prepares features, loads the model, makes predictions, computes RMSE, and saves scores to a text file.

   .. note::
      - `model_path` should point to a trained model in pickle format (`best_model.pkl`).
      - `test_data` should be a CSV file with features and target (`median_house_value`).

