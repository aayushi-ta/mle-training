======================================
Train Model Module (:mod:`train`)
======================================

Module Overview
---------------

This module provides functionality to train a RandomForestRegressor model using GridSearchCV and save the best model.

train_model
-----------

.. function:: train_model(train_data, output_folder)
   :noindex:

   Trains a RandomForestRegressor model using GridSearchCV and saves the best model.

   :param train_data: Path to the training dataset.
   :param output_folder: Path to save the trained model.

   Example::

       # Example usage
       python train.py --input-folder ../data/processed --output-folder ../artifacts --log-level INFO

   This function loads the training dataset, prepares features, performs grid search to find the best hyperparameters,
   and saves the trained model (`best_model.pkl`) to the specified output folder.

   .. note::
      - `train_data` should be a CSV file containing features and the target variable (`median_house_value`).
