import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import random

def minmaxsubsample(pts, n, seed=3):
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
    numpts = len(ts) - tau * (d-1)

    pts = [[data[i+(j*tau)] for j in range(dim) ] for i in range(npts)]

    return np.array(pts)
