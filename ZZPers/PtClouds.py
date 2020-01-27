import numpy as np
import dionysus as dio
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from ZZPers.ZZPers.PD import PD
import time
from ripser import ripser



class PtClouds(object):
    '''
    Point Clouds Class
    ==================

    '''
    def __init__(self, ptclouds, verbose=False):
        '''
        Initialize class to hold collection of pointclouds

        :param ptclouds: List of point clouds
        :param verbose: If true, prints updates when running code

        '''

        if isinstance(ptclouds, pd.DataFrame):
            self.ptclouds = ptclouds
        else:
            self.ptclouds = pd.DataFrame(columns=['PtCloud'])
            self.ptclouds['PtCloud'] = ptclouds

        self.verbose = verbose

    def __str__(self):
        '''
        Nicely prints.
        '''
        attrs = vars(self)
        output = ''
        output += 'Point Cloud\n'
        output += '---\n'
        for key in attrs.keys():
            output += str(key) + ' : '
            output += str(attrs[key])+ '\n'
        output += '---\n'
        return output


    def run_Zigzag_from_PC(self, r, k=2, cplx_type = 'Rips'):
        '''
        Runs zigzag persistence on collections of point clouds, taking unions between each adjacent pair of point clouds.
        Adds attributes to class `zz`, `zz_dgms`, `zz_cells` which are the outputs of dionysus' zigzag persistence function.

        :param delta: Parameter for Rips complex on each point cloud. Can be a single number or a list with one delta value per point cloud.
        :param k: Maximum dimension simplex desired
        :param cplx_type: Type of complex you want created for each point cloud and each point cloud union. Only option currently is 'Rips'.

        '''
        # ############################
        # if self.verbose:
        #     print('Starting rips complex computations...')
        #
        # st_getallcplx = time.time()
        # # if cplx_type == 'Flag':
        # self.get_All_Cplx(r,k,cplx_type)
        # # else:
        #     # self.get_All_Cplx_RIPS(delta,k)
        # end_getallcplx = time.time()
        #
        # if self.verbose:
        #     print('\tTime to build all complexes: ', str(end_getallcplx-st_getallcplx))
        #     print('\nStarting filtration and times computation...')
        #
        # st_getfilt = time.time()
        # self.all_Cplx_to_Filtration()
        # end_getfilt = time.time()
        #
        # if self.verbose:
        #     print('\tTime to build filtration: ', str(end_getfilt-st_getfilt))
        #     print('\nStarting zigzag calculation...')
        # ############################

        ###########################
        self.r = r
        self.k = k

        ft_st = time.time()
        filtration, times = setup_Zigzag(list(self.ptclouds['PtCloud']), r = r, k = k, verbose = self.verbose)
        ft_end = time.time()

        self.filtration = filtration
        self.times_list = times

        if self.verbose:
            print('Time to build filtration, times: ', str(ft_end-ft_st))
        ###########################

        zz_st = time.time()
        zz, dgms, cells = dio.zigzag_homology_persistence(self.filtration,self.times_list)
        zz_end = time.time()

        if self.verbose:
            print('Time to compute zigzag: ', str(zz_end-zz_st))
            # print('\nTotal Time: ', str(end_zz - st_getallcplx), '\n')

        self.zz = zz
        self.zz_dgms = to_PD_Class(dgms)
        self.zz_cells = cells

        self.setup_time = ft_end - ft_st
        self.zigzag_time = zz_end - zz_st


    def run_Ripser(self):
        '''
        Function to run Ripser on all point clouds in the list and save them in the 'Dgms' column of the `ptclouds` attribute as a column of the DataFrame.

        '''
        st_ripser = time.time()
        dgms_list = []
        for PC in list(self.ptclouds['PtCloud']):
            diagrams = ripser(PC)['dgms']
            dgms_dict = {i: diagrams[i] for i in range(len(diagrams))}

            dgms_list.append(dgms_dict)

        end_ripser = time.time()

        if self.verbose:
            print('Time to compute persistence: ', str(end_ripser-st_ripser))
        self.ptclouds['Dgms'] = dgms_list

def setup_Zigzag(lst, r, k=2, verbose=False):
    '''
    Sets up inputs needed for Dionysus' zigzag persistence function.

    :param lst: List of point clouds
    :type lst: list

    :param r: Radius for rips complex
    :type r: float

    :param k: Max dimension for rips complex (default is 2)
    :type k: int, optional

    :param verbose: If true, prints updates when running code
    :type verbose: bool, optional

    :return: Inputs for Dionysus' zigzag persistence function, ``filtration`` and ``times``

    '''

    simps_df = pd.DataFrame(columns = ['Simp','B,D'])
    simps_list = []
    times_list = []

    # Vertex counter
    vertind = 0

    init_st = time.time()
    # Initialize A with R(X_0)
    rips = dio.fill_rips(lst[0], k=2, r=r)
    rips.sort()
    rips_set = set(rips)

    # Add all simps to the list with birth,death=[0,1]
    for s in rips_set:
        s.data = 0
        simps_list.append(s)
        times_list.append([0,1])

    # Initialize with vertices for X_0
    verts = set([dio.Simplex([j+vertind],0) for j,pc in enumerate(lst[0])])

    # Initialize A with set of simplices with verts in X_0
    A = rips_set

    init_end = time.time()
    if verbose:
        print(f'Initializing done in {init_end-init_st} seconds...')

    loop_st = time.time()
    # Loop over the rest of the point clouds
    for i in range(1,len(lst)):

        # Calculate rips of X_{i-1} \cup X_i
        rips = dio.fill_rips(np.vstack([lst[i-1],lst[i]]), k=2, r=r)

        # Adjust vertex numbers, sort, and make into a set
        rips = fix_dio_vert_nums(rips, vertind)
        rips.sort()
        rips_set = set(rips)

        # Increment vertex counter
        vertind = vertind+len(verts)

        # Set of vertices in R(X_i)
        verts_next = set([dio.Simplex([j+vertind],0) for j,pc in enumerate(lst[i])])

        # Set of simplices with verts in X_{i}
        B = set(verts_next.intersection(rips_set))

        # Set of simplices with verts in X_{i-1} AND X_{i}
        M = set()

        # Loop over vertices in R(X_{i-1} \cup R_i)
        for simp in rips:

            # Get list of vertices of simp
            bdy = getVerts(simp) #set([s for s in simp.boundary()])

            # If it has no boundary and its in B, its a vertex we haven't seen yet
            # So add it to the list with appropriate birth,death
            if not bdy:
                if simp in B:
                    simps_list.append(simp)
                    times_list.append([i-0.5,i+1])
                continue

            # If all of its verts are in A, it's been handled in the initialization or the previous iteration
            if bdy.intersection(A) == bdy:
                continue

            # If all of its verts are in B, it exists in B and in the following union
            # Add it to the list with appropriate birth,death
            elif bdy.intersection(B) == bdy:
                simp.data = 0
                simps_list.append(simp)
                times_list.append([i-0.5,i+1])
                B.add(simp)

            # If it has some verts in A and some in B, it only exists in the union
            # Add to list and set birth,death appropriately
            else:
                simp.data = 0
                simps_list.append(simp)
                times_list.append([i-0.5,i])
                M.add(simp)

        # Reinitialize for next iteration
        verts = verts_next
        A = B

    loop_end = time.time()
    if verbose:
        print(f'Loop done in {loop_end-loop_st} seconds...')

    f_st = time.time()
    filtration = dio.Filtration(simps_list)
    f_end = time.time()

    if verbose:
        print(f'Calculated filtration in {f_end-f_st} seconds...\n')

    return filtration, times_list

def to_PD_Class(dio_dgms):
    '''
    Helper function to convert Dionysus diagram class into my PD class.

    :param dio_dgms: Persistence diagrams from Dionysus

    '''
    dgms_list = []
    for i,dgm in enumerate(dio_dgms):
        dgms_list.append([[[p.birth,p.death] for p in dgm],i])

    all_dgms = [PD(dgms_list[i][0],dgms_list[i][1]) for i in range(len(dgms_list))]

    return all_dgms


def fix_dio_vert_nums(D, vertind, data=None):
    '''
    Helper function for `get_All_Cplx` to adjust indexing of vertices in Dionysus from `fill_rips` function.

    :param vertind: starting index for vertices

    '''
    C = []
    for s in D:
        # s.data = data
        vlist = []
        for v in s:
            vlist.append(v+vertind)
        s = dio.Simplex(vlist)
        if data:
            s.data = data
        else:
            s.data = 0
        # print(s)
        C.append(s)
    return C

def getVerts(simp):
    if simp.dimension == 2:
        return [dio.Simplex([v],0) for v in t]
    else:
        return set([s for s in simp.boundary()])
