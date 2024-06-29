import importlib


def test_import_argparse():
    try:
        importlib.import_module("argparse")
        assert True, "argparse imported successfully."
    except ImportError:
        assert False, "argparse could not be imported."


def test_import_logging():
    try:
        importlib.import_module("logging")
        assert True, "logging imported successfully."
    except ImportError:
        assert False, "logging could not be imported."


def test_import_os():
    try:
        importlib.import_module("os")
        assert True, "os imported successfully."
    except ImportError:
        assert False, "os could not be imported."


def test_import_tarfile():
    try:
        importlib.import_module("tarfile")
        assert True, "tarfile imported successfully."
    except ImportError:
        assert False, "tarfile could not be imported."


def test_import_pandas():
    try:
        importlib.import_module("pandas")
        assert True, "pandas imported successfully."
    except ImportError:
        assert False, "pandas could not be imported."


def test_import_urllib():
    try:
        importlib.import_module("six.moves.urllib")
        assert True, "urllib imported successfully."
    except ImportError:
        assert False, "urllib could not be imported."


def test_import_numpy():
    try:
        importlib.import_module("numpy")
        assert True, "numpy imported successfully."
    except ImportError:
        assert False, "numpy could not be imported."
