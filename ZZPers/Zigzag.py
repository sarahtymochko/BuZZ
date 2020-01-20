import numpy as np
import dionysus as dio
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from ZZPers.ZZPers.PersDgm import PD
import time
from ripser import ripser

class PtClouds(object):
    def __init__(self, ptclouds,verbose=False): 
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
    

    def run_Zigzag_from_PC(self, delta, k=2, cplx_type = 'Rips'): 
        '''
        Runs zigzag persistence on collections of point clouds, taking unions between each adjacent pair of point clouds.
        Adds attributes to class `zz`, `zz_dgms`, `zz_cells` which are the outputs of dionysus' zigzag persistence function.

        :param delta: Parameter for Rips complex on each point cloud. Can be a single number or a list with one delta value per point cloud.
        :param k: Maximum dimension simplex desired
        :param cplx_type: Type of complex you want created for each point cloud and each point cloud union. Only option currently is 'Rips'.
        
        '''

        st_getallcplx = time.time()
        # if cplx_type == 'Flag':
        self.get_All_Cplx(delta,k,cplx_type)
        # else:
            # self.get_All_Cplx_RIPS(delta,k)
        end_getallcplx = time.time()
        
        if self.verbose:
            print('Time to build all complexes: ', str(end_getallcplx-st_getallcplx))
        
        st_getfilt = time.time()
        self.all_Cplx_to_Filtration()
        end_getfilt = time.time()
        
        if self.verbose:
            print('Time to build filtration: ', str(end_getfilt-st_getfilt))
        
        st_zz = time.time()
        zz, dgms, cells = dio.zigzag_homology_persistence(self.filtration,self.times_list)
        end_zz = time.time()
        
        if self.verbose:
            print('Time to compute zigzag: ', str(end_zz-st_zz))
                  
        if self.verbose:
            print('\nTotal Time: ', str(end_zz - st_getallcplx), '\n')
            
        self.zz = zz
        self.zz_dgms = to_PD_Class(dgms)
        self.zz_cells = cells
    

    def get_All_Cplx(self,delta,k=2,cplx_type='Rips'):
        '''
        Helper function for `run_Zigzag_from_PC` to get create the Rips complex for each point cloud and each union of adjacent point clouds.

        :param delta: Parameter for Rips complex on each point cloud. Can be a single number or a list with one delta value per point cloud.
        :param k: Maximum dimension simplex desired
        :param cplx_type: Type of complex you want created for each point cloud and each point cloud union. Only option currently is 'Rips'.

        '''
        
        AllPtClds = list(self.ptclouds['PtCloud'])
        
        if type(delta) is not list:
            delta = [delta]
        if len(delta) == len(AllPtClds):
            delta = delta
        else:
            delta = [delta[0]] * len(AllPtClds)
        
        self.delta = delta
        self.k = k
            
        # Initialize empty dataframe to store all complexes
        allCplx = pd.DataFrame(columns=['Time','Cplx','Label'])

        if cplx_type == 'Flag':
            # Initialize first complex
            D = distance_matrix(AllPtClds[0],AllPtClds[0])
            C = from_DistMat_to_Cpx(D,delta = delta[0])

        else:
            D = dio.fill_rips(AllPtClds[0], self.k, delta[0])
            C = []
            for s in D:
                s.data = 0
                C.append(s)

        allCplx.loc[0] = [0, C, 'PC']

        dfind = 1
        vertind = 0
        for i in range(0,len(AllPtClds)-1):  
            if cplx_type == 'Flag':  
                D_union = distance_matrix(np.concatenate([AllPtClds[i],AllPtClds[i+1]]), np.concatenate([AllPtClds[i],AllPtClds[i+1]]) )
                C_union = from_DistMat_to_Cpx(D_union, start_ind = vertind, delta = max(delta[i], delta[i+1]))
            else:
                D_union = dio.fill_rips(np.concatenate([AllPtClds[i],AllPtClds[i+1]]), self.k, max(delta[i], delta[i+1]))
                C_union = fix_dio_vert_nums(D_union,vertind)

            allCplx.loc[dfind] = [dfind,C_union,'Union']

            dfind = dfind + 1
            vertind = vertind + len(AllPtClds[i])

            if cplx_type == 'Flag':
                D_next = distance_matrix(AllPtClds[i+1],AllPtClds[i+1])
                C_next = from_DistMat_to_Cpx(D_next, start_ind = vertind, delta = delta)
            else:
                D_next = dio.fill_rips(AllPtClds[i+1], self.k, delta[i+1])
                C_next = fix_dio_vert_nums(D_next,vertind)

            allCplx.loc[dfind] = [dfind,C_next,'PC']

            dfind = dfind + 1

            D = D_next
            C = C_next

        self.all_Cplx = allCplx
    
    
    def all_Cplx_to_Filtration(self):
        '''
        Helper function for `run_Zigzag_from_PC` to get create the filtration and times to pass into dionysus zigzag persistence function.

        '''

        all_Cplx = self.all_Cplx
    
        # Get filtration consisting of all simplices from all complexes
        f = dio.Filtration(dio.closure(all_Cplx['Cplx'].sum(),2))

        # Initialize empty dataframe to store all complexes
        times_df = pd.DataFrame(columns=['Simp','B,D'])

        timesind = 0
        for simp in f:
            times_df.loc[timesind,'Simp'] = simp

            ins = []
            outs = []
            found_birth = False
            found_death = False
            d = np.inf
            for i in all_Cplx.index:
                if simp in all_Cplx.loc[i,'Cplx'] and not found_birth:
                    b = all_Cplx.loc[i,'Time']
                    found_birth = True
                if simp not in all_Cplx.loc[i,'Cplx'] and found_birth:
                    d = all_Cplx.loc[i,'Time']
                    found_death = True

                if found_birth and found_death:
                    break

            if np.isfinite(d):
                times_df.loc[timesind,'B,D'] = [b ,d]
            else:
                times_df.loc[timesind,'B,D'] = [b]

            timesind = timesind+1

        self.filtration = f
#         self.times_df = times_df
        self.times_list = list(times_df['B,D'])
    
    
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


def from_DistMat_to_Cpx(D,start_ind=0, delta = .1):
    '''
    Using a distance matrix, compute the flag complex.

    :param D: pairwise distance matrix
    :param start_ind: starting index for vertices
    :param delta: Parameter for Flag or Rips complex.

    '''

    numVerts = D.shape[0]
    listAllVerts = [dio.Simplex([k+start_ind]) for k in range(numVerts)] 

    PossibleEdges = (D<= delta) - np.eye(D.shape[0])
    listAllEdges = np.array(np.where(PossibleEdges)).T
    listAllEdges = [dio.Simplex(e+start_ind) for e in listAllEdges]
    Cpx = dio.closure(listAllVerts + listAllEdges ,2)
    return Cpx

def fix_dio_vert_nums(D, vertind):
    '''
    Helper function for `get_All_Cplx` to adjust indexing of vertices in Dionysus from `fill_rips` function.

    :param vertind: starting index for vertices

    '''
    C = []
    for s in D:
        s.data = 0
        vlist = []
        for v in s:
            vlist.append(v+vertind)
        s = dio.Simplex(vlist)
        C.append(s)
    return C 