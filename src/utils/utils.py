import numpy as np

DUMMY_CONSTANT = 1000000000

def _ensure_2dim(X0, X1):
    # sklearn wants X to have dimension>=2
    if X0.ndim == 1:
        X0 = X0[:, np.newaxis]
    if X1.ndim == 1:
        X1 = X1[:, np.newaxis]
    return X0, X1


def _ensure_np_array(arr):
    if np.isscalar(arr):  # Check if eta is a scalar
        eta_array = np.array([arr])
    else:
        eta_array = np.array(arr)
    return eta_array

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


def extract_order_of_magnitude(x):
    if x == 0:
        return "e0"  # Special case for zero
    order = np.floor(np.log10(abs(x))).astype(int)  # Get the exponent
    return f"e{order}"