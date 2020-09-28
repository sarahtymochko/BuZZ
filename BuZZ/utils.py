import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import random

def minmaxsubsample(pts, n, seed=3):
    '''
    Subsample points using greedy permutation algorithm.

    Parameters
    ----------
    pts: np.array
        Points in point cloud.
    n: int
        Number of points to subsample. Default is None, meaning no subsampling.
    seed: int
        Seed for random generation of starting point.

    '''

    # Get pairwise distance matrix of all points
    D = pairwise_distances(pts, metric='euclidean')

    # Initialize array for subsampled points
    newpts = np.zeros((n,2))

    # Pick a random starting point and add it to array
    st = random.randint(0,len(pts)-1)
    newpts[0,:]= pts[st,:]

    # Only look at distances from starting point
    ds = D[st,:]
    for i in range(1,n):

        # Find point furthest away
        idx = np.argmax(ds)

        newpts[i,:] = pts[idx,:]

        ds = np.minimum(ds, D[idx, :])

    return newpts

def time_delay_embedding(ts, d=2, tau=1):
    '''
    Compute delay embedding

    Parameters
    ----------
    ts: list or np.array
        Time series values
    d: int
        Dimension for delay embedding
    tau: int
        Delay for delay embedding

    '''
    numpts = len(ts) - tau * (d-1)

    pts = [[ts[i+(j*tau)] for j in range(d) ] for i in range(numpts)]

    return np.array(pts)
