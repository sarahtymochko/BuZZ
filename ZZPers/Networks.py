import numpy as np
import dionysus as dio
import pandas as pd
import matplotlib.pyplot as plt
import time
from ripser import ripser

import script.PHN as PHN


def build_network(ts, d, tau, directed=True):
    PS = PHN.Permutation_Sequence(ts,d,tau)

    ## Build adjacency matrix
    perm_keys = np.unique(PS)

    A = np.zeros((len(perm_keys),len(perm_keys))) #prepares A
    for i in range(1,len(PS)): #go through all permutation transitions (This could be faster without for loop)
        ind_1 = np.where(perm_keys == PS[i-1])[0][0]
        ind_2 = np.where(perm_keys == PS[i])[0][0]
        A[ ind_1 ][ ind_2 ] += 1 #for each transition between permutations increment A_ij
        if not directed:
            A[ ind_2 ][ ind_1 ] += 1

    return A, perm_keys


class Networks(object):
    '''
    Class to hold collection of networks.

    Parameters
    ----------
    verbose: bool
        If true, prints updates when running code. Default is False.

    '''

    def __init__(self, networks, vert_labels, verbose=False):
        '''
        Initialize.

        '''

        self.networks = networks
        self.vert_labels = vert_labels

        self.verbose = verbose

    def __str__(self):
        '''
        Nicely prints.

        '''
        attrs = vars(self)
        output = ''
        output += 'Networks\n'
        output += '---\n'
        for key in attrs.keys():
            output += str(key) + ' : '
            output += str(attrs[key])+ '\n'
        output += '---\n'
        return output

    def run_Zigzag(self, k=2, r=None, alpha=None):
        '''
        Runs zigzag persistence on collections of point clouds, taking unions between each adjacent pair of point clouds.
        Adds attributes to class ``zz``, ``zz_dgms``, ``zz_cells`` which are the outputs of dionysus' zigzag persistence function.
        Also adds attributes ``setup_time`` and ``zigzag_time`` with timing data.

        Parameters
        ----------

        k: int, optional
            Maximum dimension simplex desired (default is 2)

        r: float or list of floats,
            Parameter for Rips complex on each point cloud. Can be a single number or a list with one value per point cloud. (Required for using Rips or Landmark complex)

        alpha: float
            Parameter for the Witness complex (Required for using Witness complex)

        '''

        self.k = k


        filtration, times = self.setup_Zigzag_fixed(r = self.r, k = self.k, verbose = self.verbose)

        self.filtration = filtration
        self.times_list = times

        if self.verbose:
            print('Time to build filtration, times: ', str(ft_end-ft_st))

        # Run zigzag presistence using dionysus
        zz, dgms, cells = dio.zigzag_homology_persistence(self.filtration,self.times_list)

        self.zz = zz
        self.zz_dgms = to_PD_Class(dgms)
        self.zz_cells = cells

    def setup_Zigzag(self, k=2, verbose=False):
        '''
        Helper function for ``run_Zigzag`` that sets up inputs needed for Dionysus' zigzag persistence function.
        This only works for a fixed radius r.

        Parameters
        ----------

        k: int, optional
            Max dimension for rips complex (default is 2)

        verbose: bool, optional
            If true, prints updates when running code

        Returns
        -------
        filtration: dio.Filtration
            Dionysis filtration containing all simplices that exist in zigzag sequence

        times: list
            List of times, where times[i] is a list containing [a,b] where simplex i appears at time a, and disappears at time b

        '''
        def rename_edges(edge_list, vert_labels):
            vert_dict = {i:vert_labels[i] for i in range(len(vert_labels))}
            return np.vectorize(vert_dict.__getitem__)(edge_list)

        simps_list = []; times_list = []
        verts = np.unique(np.concatenate(self.vert_labels))

        # Handle vertices...
        for v in verts:
            simps_list.append(dio.Simplex([v],0))

            s_times = []
            simp_in = False
            for i in range(len(self.networks)):
                if v in self.vert_labels[i] and simp_in == False:
                    s_times.append(i)
                    simp_in = True
                if v not in self.vert_labels[i] and simp_in == True:
                    s_times.append(i)
                    simp_in = False
            times_list.append(s_times)

        # list of lists
        # edges_lists[i] contains the edges in network[i] with correct vert labels
        # note edges are sorted in vert label order so edges are no longer directed
        edges_lists = [ np.sort(rename_edges(np.hstack([np.where(self.networks[i]!=0)]).T, self.vert_labels[i]),axis=1).tolist() for i in range(len(networks)) ]

        # list of unique edges across all networks
        unique_edges = np.unique(np.vstack([ np.array(es) for es in all_edges_list]), axis=0).tolist()

        # Handle edges...
        for e in unique_edges:
            simps_list.append(dio.Simplex(e,0))

            s_times = []
            simp_in = False
            for i in range(len(self.networks)):
                if e in edges_lists[i] and simp_in == False:
                    s_times.append(i)
                    simp_in = True
                if e in edges_lists[i] and simp_in == True:
                    s_times.append(i)
                    simp_in = False
            times_list.append(s_times)

        # Handle triangles...
        ### TO DO 

        f_st = time.time()
        filtration = dio.Filtration(simps_list)
        f_end = time.time()

        return filtration, times_list
