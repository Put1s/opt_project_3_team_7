import numpy as np
from pathlib import Path
from urllib.request import urlretrieve
import ssl
import os
from sklearn.datasets import load_svmlight_file

# bypass SSL cert verification for dataset downloads
ssl._create_default_https_context = ssl._create_unverified_context


def _normalize(X):
    X = X.astype(np.float64)
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)


def _to_pm_one(y):
    y = y.astype(np.float64)
    if set(np.unique(y)) != {-1.0, 1.0}:
        y = 2.0 * y - 1.0
    return y


def create_oracle_functions(X):
    """Create matvec_Ax, matvec_ATx, matmat_ATsA closures for a dense matrix X."""
    def matvec_Ax(x):
        return X.dot(x)

    def matvec_ATx(v):
        return X.T.dot(v)

    def matmat_ATsA(s):
        return X.T.dot(np.diag(s).dot(X))

    return matvec_Ax, matvec_ATx, matmat_ATsA


def load_datasets(data_dir="data"):
    DATA_DIR = Path(data_dir)
    DATA_DIR.mkdir(exist_ok=True)

    datasets = [
        ["phishing", "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/phishing"],
        ["space_ga", "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/space_ga"],
    ]

    for name, url in datasets:
        dataset_path = DATA_DIR / name
        if not dataset_path.exists():
            print(f"downloading {name}...")
            urlretrieve(url, dataset_path)


def load_space_ga(data_dir="data"):
    X, y = load_svmlight_file(os.path.join(data_dir, "space_ga"))
    X = X.toarray()
    y = y.astype(np.float64)
    return _normalize(X), y


def load_phishing(data_dir="data"):
    X, y = load_svmlight_file(os.path.join(data_dir, "phishing"))
    X = X.toarray()
    y = _to_pm_one(y)
    return _normalize(X), y
