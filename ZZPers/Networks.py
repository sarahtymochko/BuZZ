import numpy as np
import dionysus as dio
import pandas as pd
import matplotlib.pyplot as plt
import time
from ripser import ripser
from scipy.spatial.distance import squareform

import script.PHN as PHN


class Networks(object):
    '''
    Class to hold collection of networks.

    Parameters
    ----------
    verbose: bool
        If true, prints updates when running code. Default is False.

    '''

    def __init__(self, networks, vert_labels, cplx_type = 'intersection', verbose=False):
        '''
        Initialize.

        '''

        self.networks = networks
        self.vert_labels = vert_labels
        self.cplx_type = cplx_type

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

    def run_Zigzag(self):
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


        filtration, times = self.setup_Zigzag()

        self.filtration = filtration
        self.times_list = times


        # Run zigzag presistence using dionysus
        zz, dgms, cells = dio.zigzag_homology_persistence(self.filtration,self.times_list)

        dgm_list = [ np.array([ [p.birth,p.death] for p in dgm]) for dgm in dgms ]
        if len(dgm_list) > 2:
            dgm_list = dgm_list[:2]

        self.zz = zz
        self.zz_dgms = dgm_list
        self.zz_cells = cells

    def setup_Zigzag(self, k=2):
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
        # renames edges in the list based on the vert labels
        # so an edge [0,5] becomes [vert_labels[0], vert_labels[5]]
        def rename_edges(edge_list, vert_labels):
            vert_dict = {i:vert_labels[i] for i in range(len(vert_labels))}
            return np.vectorize(vert_dict.__getitem__)(edge_list)

        def make_undirected(A):
            for i in range(len(A)):
                for j in range(i):
                    if A[i,j] > 0 or A[j,i] > 0:
                        A[i,j] = 1
                        A[j,i] = 1
                    else:
                        A[i,j] = 0
                        A[j,i] = 0
            return A

        def fix_vert_nums(simp, labels):
            vlist = []
            for v in simp:
                vlist.append(labels[v])
            return vlist

        def get_tris(A, labels):
            A_u = make_undirected(A)

            A_u[A_u != 0] = 0.1
            A_u[A_u == 0] = 10
            np.fill_diagonal(A_u, 0)

            f = dio.fill_rips(squareform(A_u), k=2, r=1)
            tris = [fix_vert_nums(i, labels) for i in f if i.dimension() == 2]

            return tris

        simps_list = []; times_list = []
        verts = np.unique(np.concatenate(self.vert_labels))
        num_verts = len(verts)

        # Handle vertices...
        for v in verts:
            simps_list.append(dio.Simplex([v],0))

            s_times = []
            simp_in = False


            # Find time simplex enters filtration
            for i in range(len(self.networks)):
                if v in self.vert_labels[i] and simp_in == False:

                    if self.cplx_type == 'union':
                        if i == 0:
                            st = 0
                        else:
                            st = i-0.5
                        s_times.append(st)
                    elif self.cplx_type == 'intersection':
                        s_times.append(i)
                    else:
                        print('cplx_type not recognized...\nQuitting')
                        return [],[]

                    simp_in = True

                # Find time simplex exits filtration
                if v not in self.vert_labels[i] and simp_in == True:
                    if self.cplx_type == 'union':
                        s_times.append(i)
                    elif self.cplx_type == 'intersection':
                        s_times.append(i-0.5)

                    else:
                        print('cplx_type not recognized...\nQuitting')
                        return [],[]
                    simp_in = False
            times_list.append(s_times)

        if self.verbose:
            print(f"Added {num_verts} vertices to filtration.")

        # list of lists
        # edges_lists[i] contains the edges in network[i] with correct vert labels
        # note edges are sorted in vert label order so edges are no longer directed
        edges_lists = [ np.sort(rename_edges(np.hstack([np.where(self.networks[i]!=0)]).T,
                        self.vert_labels[i]),axis=1).tolist() for i in range(len(self.networks)) ]

        # list of unique edges across all networks
        unique_edges = np.unique(np.vstack([ np.array(es) for es in edges_lists]),
                                 axis=0).tolist()
        num_edges = len(unique_edges)

        # Handle edges...
        for e in unique_edges:
            simps_list.append(dio.Simplex(e,0))

            s_times = []
            simp_in = False
            for i in range(len(self.networks)):
                if e in edges_lists[i] and simp_in == False:

                    if self.cplx_type == 'union':
                        if i == 0:
                            st = 0
                        else:
                            st = i-0.5
                        s_times.append(st)
                    elif self.cplx_type == 'intersection':
                        s_times.append(i)
                    else:
                        print('cplx_type not recognized...\nQuitting')
                        return [],[]

                    simp_in = True
                if not e in edges_lists[i] and simp_in == True:

                    if self.cplx_type == 'union':
                        s_times.append(i)
                    elif self.cplx_type == 'intersection':
                        s_times.append(i-0.5)

                    simp_in = False
            times_list.append(s_times)

        if self.verbose:
            print(f"Added {num_edges} edges to filtration.")


        # Handle triangles...
        tri_lists = [ get_tris(self.networks[i], self.vert_labels[i]) for i in range(len(self.networks)) ]

        unique_tris = np.unique(np.vstack([ np.array(ts) for ts in tri_lists if ts != [] ]),
                                 axis=0).tolist()
        num_tris = len(unique_tris)

        for t in unique_tris:
            simps_list.append(dio.Simplex(t,0))

            s_times = []
            simp_in = False
            for i in range(len(self.networks)):
                if t in tri_lists[i] and simp_in == False:

                    if self.cplx_type == 'union':
                        if i == 0:
                            st = 0
                        else:
                            st = i-0.5
                        s_times.append(st)
                    elif self.cplx_type == 'intersection':
                        s_times.append(i)
                    else:
                        print('cplx_type not recognized...\nQuitting')
                        return [],[]

                    simp_in = True
                if t not in tri_lists[i] and simp_in == True:
                    if self.cplx_type == 'union':
                        s_times.append(i)
                    elif self.cplx_type == 'intersection':
                        s_times.append(i-0.5)

                    simp_in = False
            times_list.append(s_times)

        if self.verbose:
            print(f"Added {num_tris} edges to filtration.")

        f_st = time.time()
        filtration = dio.Filtration(simps_list)
        f_end = time.time()

        return filtration, times_list


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
