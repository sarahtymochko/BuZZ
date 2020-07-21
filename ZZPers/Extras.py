import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
import random

def MinMaxSubsample(pts, n, seed=3):
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

# def plot_ZZ_dgm(dgms, dims=[0,1], save=False, savename='Dgm.png'):
#     '''
#     Plot the zigzag persistence diagram
#
#     Parameters
#     ----------
#
#     dims: list of ints
#         List of integers representing dimensions you want plotted (default is [0,1])
#     save: bool
#         Set to true if you want to save the figure (default is False)
#
#     savename: string
#         Path to save figure (default is 'Cplx.png' in current directory)
#
#     '''
#
#     # # Check diagrams have been calculated
#     # if not hasattr(self,'zz_dgms'):
#     #     print('No diagrams calculated yet...')
#     #     print('Use run_Zigzag first then you can use this function...')
#     #     print('Quitting...')
#     #     return
#
#     # Check dimensions selected are calculated
#     if not set(dims).issubset(range(len(dgms))):
#         print('Those dimensions arent available')
#         print('Plotting up to dimenssion ', len(dgms)-1)
#         dims=range(len(dgms))
#
#     # Plot zigzag diagram
#     fig,ax = plt.subplots(figsize=[4,4])
#
#     plot_diagrams(dgms[min(dims):max(dims)+1])
#
#     # Save figure
#     if save:
#         print('Saving fig at ', savename, '...')
#         plt.savefig(savename, dpi=500, bbox_inches='tight')
