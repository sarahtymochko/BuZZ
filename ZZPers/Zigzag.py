import numpy as np
import dionysus as dio
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from ZZPers.ZZPers.PersDgm import PD
import time

class ComputeZigzag(object):
    def __init__(self, ptclouds, delta, k=2,cplx_type ='Rips', verbose=False):
        self.ptclouds = ptclouds
        self.delta = delta
        self.k = k
        self.cplx_type = cplx_type
        
        self.Run_Zigzag_from_PC(cplx_type = cplx_type, verbose = verbose)
            
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
    
    def Run_Zigzag_from_PC(self, cplx_type = 'Rips', verbose=False):
        
        st_getallcplx = time.time()
        if cplx_type == 'Rips':
            self.get_All_Cplx_RIPS()
        else:
            self.get_All_Cplx()
        end_getallcplx = time.time()
        
        if verbose:
            print('Time to build all complexes: ', str(end_getallcplx-st_getallcplx))
        
        st_getfilt = time.time()
        if cplx_type == 'Rips':
            self.all_Cplx_to_Filtration_RIPS()
        else:
            self.all_Cplx_to_Filtration()
        end_getfilt = time.time()
        
        if verbose:
            print('Time to build filtration: ', str(end_getfilt-st_getfilt))
        
        st_zz = time.time()
        zz, dgms, cells = dio.zigzag_homology_persistence(self.filtration,self.times_list)
        end_zz = time.time()
        
        if verbose:
            print('Time to compute zigzag: ', str(end_zz-st_zz))
                  
        if verbose:
            print('\nTotal Time: ', str(end_zz - st_getallcplx), '\n')
            
        self.zz = zz
        self.dgms = to_PD_Class(dgms)
        self.cells = cells
        
    def get_All_Cplx(self,verbose = False):
        # Input: list of point clouds for each time 
        #        Threshold delta at which to construct the flag complex at each time 
        # 
        # Returns: dionysus diagrams list
        
        AllPtClds = self.ptclouds
        delta = self.delta

        # Initialize empty dataframe to store all complexes
        allCplx = pd.DataFrame(columns=['Time','Cplx','Label'])

        # Initialize first complex
        D = distance_matrix(AllPtClds[0],AllPtClds[0])
        C = from_DistMat_to_Cpx(D,delta = delta)

        allCplx.loc[0] = [0, C, 'PC']

        dfind = 1
        vertind = 0
        for i in range(0,len(AllPtClds)-1):    
            D_union = distance_matrix(np.concatenate([AllPtClds[i],AllPtClds[i+1]]), np.concatenate([AllPtClds[i],AllPtClds[i+1]]) )
            C_union = from_DistMat_to_Cpx(D_union, start_ind = vertind, delta = delta)

            allCplx.loc[dfind] = [dfind,C_union,'Union']

            dfind = dfind + 1
            vertind = vertind + len(AllPtClds[i])

            D_next = distance_matrix(AllPtClds[i+1],AllPtClds[i+1])
            C_next = from_DistMat_to_Cpx(D_next, start_ind = vertind, delta = delta)

            allCplx.loc[dfind] = [dfind,C_next,'PC']

            dfind = dfind + 1

            D = D_next
            C = C_next

        self.all_Cplx = allCplx
    
    
    def all_Cplx_to_Filtration(self):
        
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
        
    def get_All_Cplx_RIPS(self,verbose = False):
        # Input: list of point clouds for each time 
        #        Threshold delta at which to construct the flag complex at each time 
        # 
        # Returns: dionysus diagrams list
        
        AllPtClds = self.ptclouds
        delta = self.delta

        # Initialize empty dataframe to store all complexes
        allCplx = pd.DataFrame(columns=['Time','Cplx','Label'])

        # Initialize first complex
        D = dio.fill_rips(AllPtClds[0], self.k, delta)
        C = []
        for s in D:
            s.data = 0
            C.append(s)

        allCplx.loc[0] = [0, C, 'PC']

        dfind = 1
        vertind = 0
        for i in range(0,len(AllPtClds)-1):    
            D_union = dio.fill_rips(np.concatenate([AllPtClds[i],AllPtClds[i+1]]), self.k, delta)
            C_union = fix_dio_vert_nums(D_union,vertind)

            allCplx.loc[dfind] = [dfind,C_union,'Union']

            dfind = dfind + 1
            vertind = vertind + len(AllPtClds[i])


            D_next = dio.fill_rips(AllPtClds[i+1], self.k, delta)
            C_next = fix_dio_vert_nums(D_next,vertind)
                
            allCplx.loc[dfind] = [dfind,C_next,'PC']

            dfind = dfind + 1

            D = D_next
            C = C_next

        self.all_Cplx = allCplx
    
    def all_Cplx_to_Filtration_RIPS(self):
        
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

def to_PD_Class(dio_dgms):
    dgms_list = []
    for i,dgm in enumerate(dio_dgms):
        dgms_list.append([[[p.birth,p.death] for p in dgm],i])

    all_dgms = [PD(dgms_list[i][0],dgms_list[i][1]) for i in range(len(dgms_list))]

    return all_dgms


def from_DistMat_to_Cpx(D,start_ind=0, delta = .1):
    

    numVerts = D.shape[0]
    listAllVerts = [dio.Simplex([k+start_ind]) for k in range(numVerts)] 

    PossibleEdges = (D<= delta) - np.eye(D.shape[0])
    listAllEdges = np.array(np.where(PossibleEdges)).T
    listAllEdges = [dio.Simplex(e+start_ind) for e in listAllEdges]
    Cpx = dio.closure(listAllVerts + listAllEdges ,2)
    return Cpx

def fix_dio_vert_nums(D, vertind):
    C = []
    for s in D:
        s.data = 0
        vlist = []
        for v in s:
            vlist.append(v+vertind)
        s = dio.Simplex(vlist)
        C.append(s)
    return C 