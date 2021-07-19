import sys
import time
from inspect import currentframe, getframeinfo
import numpy as np
from typing import Tuple
import math
import argparse


def setup_mpi_comm(distributed: bool):
    if distributed:
        # get communicator: duplicate from comm world
        from mpi4py import MPI

        return MPI.COMM_WORLD.Dup()
    return None


def setup_mpi(comm=None) -> Tuple[int, int]:
    comm_size = 1
    comm_rank = 0
    if comm is not None:
        comm_size = comm.Get_size()
        comm_rank = comm.Get_rank()

    return comm_size, comm_rank


def timer(
    label: str, start: int = 1, frameinfo=None
):  # start = 1 - start, start = -1 - stop, start = 0 - neither
    t = time.localtime()
    gps = time.mktime(t)
    readable = time.asctime(t)
    if frameinfo is None:
        frameinfo = getframeinfo(currentframe().f_back)
    fractions = time.perf_counter()
    print(
        f"TLaBeL|{label}|{start}|{gps}|{readable}|{frameinfo.filename}|{frameinfo.lineno}|{fractions}"
    )
    sys.stdout.flush()


class Timer:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        frameinfo = getframeinfo(currentframe().f_back)
        timer(self.label, 1, frameinfo)
        return self

    def __exit__(self, type, value, traceback):
        frameinfo = getframeinfo(currentframe().f_back)
        timer(self.label, -1, frameinfo)


def bestk(
    a: np.ndarray, k: int, smallest: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Return the best `k` values and correspdonding indices.

    Parameters
    ----------
    a : np.ndarray
        Array of dim (N,)
    k : int
        Specifies which element to partition upon.
    smallest : bool
        True if the best values are small (or most negative).
        False if the best values are most positive.

    Returns
    -------
    np.ndarray
        Of length `k` containing the `k` smallest values in `a`.
    np.ndarray
        Of length `k` containing indices of input array `a`
        coresponding to the `k` smallest values in `a`.
    """
    # If larger values are considered best, make large values the smallest
    # in order for the sort function to pick the best values.
    arr = a if smallest else -1 * a
    # Only sorts 1 element of `arr`, namely the element that is position
    # k in the sorted array. The elements above and below the kth position
    # are partitioned but not sorted. Returns the indices of the elements
    # on the left hand side of the partition i.e. the top k.
    best_inds = np.argpartition(arr, k)[:k]
    # Get the associated values of the k-partition
    best_values = arr[best_inds]
    # Only sorts an array of size k
    sort_inds = np.argsort(best_values)
    return best_values[sort_inds], best_inds[sort_inds]


def t2Dto1D(A):
    n, m = A.shape
    B = np.zeros(int(n * (n - 1) / 2), dtype=np.uint8)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            B[k] = A[i, j]
            k += 1
    return B


def t1Dto2D(B):
    m = B.shape[0]
    n = int((1 + math.sqrt(1 + 8 * m)) / 2)
    A = np.ones((n, n), dtype=np.uint8)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            A[i, j] = B[k]
            A[j, i] = B[k]
            k += 1
    return A


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", help="YAML config file", type=str, required=True
    )
    parser.add_argument("-p", "--project", action="store_true", help="compute tsne")
    args = parser.parse_args()
    return args


def hash2intarray(h):
    b = []
    for i in range(len(h) // 4):
        b.append(int(h[4 * i : 4 * (i + 1)], 16))
    return np.asarray(b, dtype=np.int64)


def intarray2hash(ia):
    c = list(map(lambda x: "{0:#0{1}x}".format(x, 6).replace("0x", ""), ia))
    return "".join(c)
