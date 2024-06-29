=======================================
Ingest Data Module (:mod:`ingest_data`)
=======================================

Module Overview
---------------

This module provides functions to fetch, load, and prepare housing data for training and validation.

fetch_housing_data
------------------

.. function:: fetch_housing_data(housing_url, housing_path)
   :noindex:

   Fetches the housing data from a specified URL and saves it locally.

   :param housing_url: URL of the housing data file.
   :param housing_path: Local directory where the data will be saved.

load_housing_data
-----------------

.. function:: load_housing_data(housing_path)
   :noindex:

   Loads the housing data from a CSV file.

   :param housing_path: Local directory where the data is stored.
   :return: pandas.DataFrame containing the loaded housing data.

prepare_data
------------

.. function:: prepare_data(output_folder)
   :noindex:

   Prepares the housing data for training and validation.

   This function fetches the housing data, splits it into training and testing sets
   using stratified sampling based on income categories, and saves the processed
   data as CSV files.

   :param output_folder: Path to the directory where processed data will be saved.
