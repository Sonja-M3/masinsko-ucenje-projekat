import pandas as pd


TRAIN_PATH = "data/processed/train.txt"
VALIDATION_PATH = "data/processed/val.txt"
TEST_PATH = "data/processed/test.txt"


def get_training_data():
    return _read_data(TRAIN_PATH)


def get_validation_data():
    return _read_data(VALIDATION_PATH)


def get_test_data():
    return _read_data(TEST_PATH)


def _read_data(path):
    return pd.read_csv(
        path,
        sep=";",
        header=None,
        names=["text", "emotion"]
    )