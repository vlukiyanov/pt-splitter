import networkx as nx
import numpy as np
import scipy as sp
from scipy.sparse import spdiags


def basis_vector(size, i, **kwargs):
    v = np.zeros(size, **kwargs)
    v[i] = 1
    return v


def transition_matrix(G):
    M = nx.to_scipy_sparse_matrix(G, dtype=float)  # TODO extra options
    n, m = M.shape
    DI = spdiags(1.0 / sp.array(M.sum(axis=1).flat), [0], n, n)
    return (DI * M).transpose().tocsr()
