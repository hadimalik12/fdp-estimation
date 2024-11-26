import numpy as np
import logging
from math import floor, ceil, sqrt
import sys
import os

from sklearn import neighbors
from sklearn.neighbors import KDTree

from utils.utils import convert_bytes_to_gb, convert_gb_to_bytes

def get_k(method, num_samples, d=None):
    if method == 'gyorfi':
        k = floor(num_samples ** (2 / (d + 2)))
    elif 'sqrt' in method:
        k = ceil(sqrt(num_samples))

    k = int(k)
    if k % 2 == 0:  # ensure k is odd
        k += 1
    return k

def invert_permutation(p):
    """Return an array s with which np.array_equal(arr[p][s], arr) is True.
    The array_like argument p must be some permutation of 0, 1, ..., len(p)-1.
    """
    p = np.asanyarray(p)  # in case p is a tuple, etc.
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s

class OneDkNNImpl:
    def __init__(self, k):
        self.k = k  # k nearest neighbor value
        self.train_labels = None
        self.index = None
        self.test_label_output = []
        self.nbytes = None

    def fit(self, train_features, train_labels):
        ids = train_features.ravel().argsort()
        self.index = train_features.ravel()[ids]
        self.train_labels = train_labels.ravel()[ids]

        self.nbytes = self.index.nbytes + self.train_labels.nbytes
        return self

    def predict(self, test_features):
        ids = test_features.ravel().argsort()
        testing_points = test_features[ids]

        tail_p = 0
        head_p = self.k
        label_output = []

        for test_p in range(len(testing_points)):
            value = testing_points[test_p]

            #   Compute the nearest neighbors
            while head_p < len(self.index) - 1:
                if np.abs(self.index[tail_p] - value) >= np.abs(self.index[head_p] - value):
                    tail_p += 1
                    head_p += 1
                else:
                    break

            #   Compute its label
            label_output.append(np.rint(self.train_labels[tail_p:head_p].mean()))

        inv_ids = invert_permutation(ids)
        self.test_label_output = np.array(label_output)[inv_ids]
        return self.test_label_output

    def score(self, test_features, test_labels):
        return (self.predict(test_features) == test_labels).mean()

    def correct_num(self, test_features, test_labels):
        return (self.predict(test_features) == test_labels).sum()

class kdTreekNNImpl:
    """optimized version for our case using sklearn kDtree
       we use the parameter p for the minkowski metric to control the metric we use
       the default p=2 is for Euclidean distance; p=1 for manhattan distance; p tends to infinity for infinity metric
    """
    def __init__(self, k, leaf_size=30, p=2.0):
        self.k = k  # k nearest neighbor value
        self.leaf_size = leaf_size
        self.p = p
        self.tree = None
        self.train_labels = None
        self.nbytes = None

    def fit(self, train_features, train_labels):
        self.tree = KDTree(train_features, self.leaf_size, p=self.p)
        self.train_labels = train_labels.ravel()

        self.nbytes = sys.getsizeof(self.tree.get_arrays()) + self.train_labels.nbytes
        return self

    def predict(self, test_features, allowed_memory_gb=0.4):
        assert isinstance(test_features, np.ndarray), "ERR: required np.ndarray type"
        assert test_features.ndim == 2, f"ERR: data base input is in wrong shape, required 2 dimensions"

        pid = os.getpid()
        estimated_memory_usage = convert_bytes_to_gb(test_features.shape[0] * self.k * test_features[0][0].nbytes) * \
                                 2 + convert_bytes_to_gb(self.nbytes)
        num_samples = test_features.shape[0]
        logging.info(f"Estimated memory usage by process {pid} is {estimated_memory_usage} GB ")
        test_label_output = None

        if estimated_memory_usage > allowed_memory_gb:
            logging.info(f"memory usage is larger than allowed memory size, prediction process runs in chunk by chunk "
                         f"mode")
            chunk_size = int((convert_gb_to_bytes(allowed_memory_gb)) / (test_features[0][0].nbytes *
                                                                                      self.k*2))
            assert chunk_size > 1, "ERR: there is not enough memory, pls decrease the number of process; each process "\
                                   "has only 2GB being used "
            num_batch_samples = np.minimum(chunk_size, num_samples)
            neigh_ind = self.tree.query(X=test_features[:num_batch_samples], k=self.k, return_distance=False,
                                        dualtree=True, breadth_first=False, sort_results=False)
            test_label_output = np.rint(self.train_labels[neigh_ind].mean(axis=1))

            pointer = num_batch_samples
            while pointer < num_samples:
                logging.info(f"{pid} process {pointer}/{num_samples}")
                num_batch_samples = np.minimum(chunk_size, num_samples-pointer)
                neigh_ind = self.tree.query(X=test_features[pointer:pointer+num_batch_samples], k=self.k,
                                            return_distance=False,
                                            dualtree=True, breadth_first=False, sort_results=False)

                test_label_output = np.concatenate((test_label_output,
                                                    np.rint(self.train_labels[neigh_ind].mean(axis=1))))

                # update pointer
                pointer += num_batch_samples
        else:
            neigh_ind = self.tree.query(X=test_features, k=self.k, return_distance=False, dualtree=True,
                                        breadth_first=False, sort_results=False)
            test_label_output = np.rint(self.train_labels[neigh_ind].mean(axis=1))

        return test_label_output

    def score(self, test_features, test_labels):
        return (self.predict(test_features) == test_labels).mean()

    def correct_num(self, test_features, test_labels):
        return (self.predict(test_features) == test_labels).sum()

def train_kNN_model(samples, n_features=1, k=None):
    assert (np.isreal(n_features) and n_features>0), "ERR: number of feature must be a positive integer"
    n_features = int(np.ceil(n_features))
    num_samples = samples['X'].shape[0]
    
    X, y = samples['X'], samples['y']

    if k is None:
        k = 10*get_k('sqrt_random_tiebreak', num_samples)

    if n_features == 1:
        logging.info("use the optimized one dimensional KNN algorithm")
        KNN = OneDkNNImpl(k=k)
    else:
        logging.info("use the multidimensional KNN algorithm with l2 distance")
        KNN = kdTreekNNImpl(k=k, p=2)

    return KNN.fit(X, y)