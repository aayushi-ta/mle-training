import logging
import os
import sys

# Set up logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Adjust the path to src relative to the conf.py file location
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "../src"))
logger.debug(f"Current Directory: {current_dir}")
logger.debug(f"Src Path: {src_path}")

sys.path.insert(0, src_path)

# Log sys.path contents
logger.debug(f"sys.path: {sys.path}")

# Log contents of src directory
if os.path.exists(src_path) and os.path.isdir(src_path):
    logger.debug(f"Contents of {src_path}: {os.listdir(src_path)}")
else:
    logger.error(f"{src_path} does not exist or is not a directory")

# -- Project information -----------------------------------------------------
project = "Housing training"
copyright = "2024, Aayushi Priya"
author = "Aayushi Priya"
release = "0.2"

# -- General configuration ---------------------------------------------------
autodoc_mock_imports = ["ingest_data", "score", "train"]
extensions = ["sphinx.ext.autodoc", "sphinx.ext.viewcode", "sphinx.ext.napoleon"]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
