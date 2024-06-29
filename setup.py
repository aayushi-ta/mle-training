from setuptools import find_packages, setup

setup(
    name="housing_data_project",
    version="v0.2",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "pytest",
        "matplotlib",
        "scipy",
    ],
    entry_points={
        "console_scripts": [
            "train_model = src.train:main",
            "score_model = src.score:main",
            "ingest_data = src.ingest_data:main",
        ],
    },
)
