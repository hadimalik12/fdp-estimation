import numpy as np

DUMMY_CONSTANT = 1000000000

def _ensure_2dim(X0, X1):
    # sklearn wants X to have dimension>=2
    if X0.ndim == 1:
        X0 = X0[:, np.newaxis]
    if X1.ndim == 1:
        X1 = X1[:, np.newaxis]
    return X0, X1

def convert_bytes_to_mb(x):
    assert np.isreal(x)
    return x / 1048576


def convert_mb_to_bytes(x):
    assert np.isreal(x)
    return x * 1048576


def convert_bytes_to_gb(x):
    assert np.isreal(x)
    return x / 1073741824


def convert_gb_to_bytes(x):
    assert np.isreal(x)
    return x * 1073741824