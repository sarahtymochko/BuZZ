import numpy as np
import dionysus as dio
import pandas as pd
import matplotlib.pyplot as plt
# from scipy.spatial import distance_matrix
from ZZPers.ZZPers.PD import PD
from ZZPers.ZZPers.Extras import MinMaxSubsample
import time
from ripser import ripser
# from scipy.spatial.distance import euclidean
import gudhi


class PtClouds(object):
    '''
    Class to hold collection of point clouds.

    Parameters
    ----------
    ptclouds: list
        List of point clouds
    cplx_type: string (not case sensitive)
        Type of complex you want to use. Options are Rips, Witness, Landmark. Defauly is Rips.
    verbose: bool
        If true, prints updates when running code. Default is False.

    '''

    def __init__(self, ptclouds, cplx_type='Rips', num_landmarks=None, verbose=False):
        '''
        Initialize.
        
        '''

        if isinstance(ptclouds, pd.DataFrame):
            self.ptclouds_full = ptclouds
        else:
            self.ptclouds_full = pd.DataFrame(columns=['PtCloud'])
            self.ptclouds_full['PtCloud'] = ptclouds
            
        self.use_Landmarks(num_landmarks = num_landmarks)
        
        self.num_landmarks = num_landmarks
        
        self.cplx_type = cplx_type.lower()
            
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
    
    def use_Landmarks(self, num_landmarks = None):  
        '''
        Uses landmark complex by subsampling each point cloud using a max min approach.
        
        Parameters
        ----------
        
        num_landmarks: int
            Number of points to subsample (default is None, meaning no subsampling)
        '''
        
        # No subsampling, use full point clouds
        if num_landmarks == None:
            self.ptclouds = self.ptclouds_full
            return
        
        # Subsample num_landmarks points from each point cloud using MinMax approach
        self.ptclouds = pd.DataFrame(columns=['PtCloud'])
        for i, pc in enumerate(self.ptclouds_full['PtCloud']):
            self.ptclouds.loc[i,'PtCloud'] = MinMaxSubsample(pc, num_landmarks, seed=None)
        

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
        
        # Error checking for rips and landmark complex
        if self.cplx_type == 'rips' or self.cplx_type == 'landmark':
            if r == None:
                print('Parameter r is required to use rips or landmark complex')
                print('Quitting...')
                return
            else:
                self.r = r
                
        # Error checking for witness complex      
        if self.cplx_type == 'witness':
            print('Uh oh... Witness complex not set up yet... Quitting...')
            return
            if alpha == None:
                print('Parameter alpha is required to use witness complex')
                print('Quitting...')
                return
            else:
                self.alpha = alpha

        # Set up inputs for dionysus zigzag persistence
        ft_st = time.time()
        if self.cplx_type == 'rips' or self.cplx_type == 'landmark':
            if type(r) == float or type(r) == int:
                filtration, times = self.setup_Zigzag_fixed(r = self.r, k = self.k, verbose = self.verbose)
            else:
                filtration, times = self.setup_Zigzag_changing(r = self.r, k = self.k, verbose = self.verbose)
        elif self.cplx_type == 'witness':
            filtration, times = self.setup_Zigzag_witness(alpha = self.alpha, num_landmarks = self.num_landmarks, k = self.k, verbose = self.verbose)
        else:
            print("Complex type not recognized...")
            print("Options are: 'Rips', 'Landmark', 'Witness'")
            print("Try again...")
            return
        ft_end = time.time()

        self.filtration = filtration
        self.times_list = times

        if self.verbose:
            print('Time to build filtration, times: ', str(ft_end-ft_st))

        # Run zigzag presistence using dionysus
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


    def run_Ripser(self,maxdim=1,thresh=np.inf):
        '''
        Function to run Ripser on all point clouds in the list and save them in the 'Dgms' column of the `ptclouds` attribute as a column of the DataFrame.

        Parameters
        ----------

        maxdim: int, optional
            Maximum homology dimension computed (default is 1)

        thresh: float, optional
            Maximum distances considered when constructing filtration (default is np.inf, meaning whole filtration is calculated)

        '''
        st_ripser = time.time()
        dgms_list = []
        for PC in list(self.ptclouds['PtCloud']):
            diagrams = ripser(PC,maxdim=maxdim, thresh=thresh)['dgms']
            dgms_dict = {i: diagrams[i] for i in range(len(diagrams))}

            dgms_list.append(dgms_dict)

        end_ripser = time.time()

        if self.verbose:
            print('Time to compute persistence: ', str(end_ripser-st_ripser))
        self.ptclouds['Dgms'] = dgms_list

    def setup_Zigzag_fixed(self, r, k=2, verbose=False):
        '''
        Helper function for ``run_Zigzag`` that sets up inputs needed for Dionysus' zigzag persistence function.
        This only works for a fixed radius r.

        Parameters
        ----------

        r: float
            Radius for rips complex

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

        lst = list(self.ptclouds['PtCloud'])

        # simps_df = pd.DataFrame(columns = ['Simp','B,D'])
        simps_list = []; times_list = []

        # Vertex counter
        vertind = 0

        init_st = time.time()
        # Initialize A with R(X_0)
        rips = dio.fill_rips(lst[0], k=2, r=r)
        rips.sort()
        rips_set = set(rips)

        # Initialize A with set of simplices with verts in X_0
        A = rips_set

        # # Add all simps to the list with birth,death=[0,1]
        simps_list =  simps_list + [s for s in A]
        times_list = times_list + [[0,1] for j in range(len(A))]

        # Initialize with vertices for X_0
        verts = set([dio.Simplex([j+vertind],0) for j,pc in enumerate(lst[0])])

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
                bdy = getVerts(simp)

                # If it has no boundary and its in B, its a vertex in B and has been handled
                if not bdy:
                    continue

                # If all of its verts are in A, it's been handled in the initialization or the previous iteration
                if bdy.intersection(A) == bdy:
                    continue

                # If all of its verts are in B, add it to B
                elif bdy.intersection(B) == bdy:
                    B.add(simp)

                # If it has some verts in A and some in B, it only exists in the union
                # Add it to M
                else:
                    M.add(simp)


            # Add simplices in B with the corresponding birth,death times
            simps_list = simps_list + [s for s in B]
            times_list = times_list + [ [i-0.5,i+1] for j in range(len(B)) ]

            # Add simplicies in M with corresponding birth,death times
            simps_list = simps_list + [s for s in M]
            times_list = times_list + [ [i-0.5,i] for j in range(len(M)) ]

            # Reinitialize for next iteration
            verts = verts_next
            A = B

        loop_end = time.time()
        if verbose:
            print(f'Loop done in {loop_end-loop_st} seconds...')

        f_st = time.time()
        filtration = dio.Filtration(simps_list)
        f_end = time.time()

        return filtration, times_list


    def setup_Zigzag_changing(self, r, k=2, verbose=False):
        '''
        Helper function for ``run_Zigzag`` that sets up inputs needed for Dionysus' zigzag persistence function.
        This one allows r to be a list of radii, rather than one fixed value

        Parameters
        ----------

        r: list
            List of radii for rips complex of each X_i

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

        lst = list(self.ptclouds['PtCloud'])

        # If you haven't input enough r values, just append extra of the last
        # list entry to make it the right number
        if len(r) < len(lst):
            if verbose:
                print('Warning: too few radii given, duplicating last entry')
            r = r + ([r[-1]]* (len(lst)- len(r)))
            self.r = r
        elif len(r) > len(lst):
            if verbose:
                print('Warning: too many radii given, only using first ', len(lst))
            r = r[:len(lst)]
            self.r = r

        # simps_df = pd.DataFrame(columns = ['Simp','B,D'])
        simps_list = []; times_list = []

        # Vertex counter
        vertind = 0

        init_st = time.time()

        # Initialize A with R(X_0)
        rips = dio.fill_rips(lst[0], k=2, r=r[0])
        rips.sort()
        rips_set = set(rips)

        # Initialize A with set of simplices with verts in X_0
        # In the loop, this will store simplices with verts in X_{i-1}
        A = rips_set

        # Add all simps to the list with birth,death=[0,1]
        simps_list =  simps_list + [s for s in A]
        times_list = times_list + [[0,1] for j in range(len(A))]

        # Initialize with vertices for X_0
        # In the loop, this will store vertices in X_{i-1}
        verts = set([dio.Simplex([j+vertind],0) for j,pc in enumerate(lst[0])])

        init_end = time.time()
        if verbose:
            print(f'Initializing done in {init_end-init_st} seconds...')

        loop_st = time.time()
        # Loop over the rest of the point clouds
        for i in range(1,len(lst)):

            # Calculate rips of X_{i-1} \cup X_i
            rips = dio.fill_rips(np.vstack([lst[i-1],lst[i]]), k=2, r=max(r[i-1],r[i]))

            # Adjust vertex numbers, sort, and make into a set
            rips = fix_dio_vert_nums(rips, vertind)
            rips.sort()
            rips_set = set(rips)

            # Increment vertex counter
            vertind = vertind+len(verts)

            # Set of vertices in X_{i}
            verts_next = set([dio.Simplex([j+vertind],0) for j,pc in enumerate(lst[i])])

            # Set of simplices with verts in X_{i}
            B = set(verts_next.intersection(rips_set))

            # Set of simplices with verts in X_{i-1} AND X_{i}
            # And simplicies in X_{i-1} \cup X_{i} that are not in X_{i-1} or X_i
            M = set()

            # Loop over vertices in R(X_{i-1} \cup R_i)
            for simp in rips:

                # Get list of vertices of simp
                bdy = getVerts(simp) #set([s for s in simp.boundary()])

                # If it has no boundary and its in B, its a vertex in B and has been handled
                if not bdy:
                    continue

                # If all of its verts are in A, it's been handled in the initialization or the previous iteration
                if bdy.intersection(A) == bdy:
                    if r[i-1] < r[i]:
                        if simp.data > r[i-1]:
                            # If we haven't seen it before, add it to M
                            if simp not in simps_list:
                                M.add(simp)

                            # If we've already added it to the list...
                            else:
                                # Edit the lists to include new birth,death times
                                simps_list, times_list = edit_Simp_Times(simp,[i-0.5,i],simps_list,times_list)


                # If all of its verts are in B...
                elif bdy.intersection(B) == bdy:

                    # If r[i-1] <= r[i], anything with verts in B should also be in B
                    if r[i-1] <= r[i]:
                        B.add(simp)

                    # If r[i-1] > r[i], we need to check if it should go in M or B
                    else:
                        # If simplex's birth time is greater than the radius, it goes in M
                        if simp.data > r[i]:
                            M.add(simp)

                        # If it's <= then it goes in B
                        else:
                            B.add(simp)

                # If it has some verts in A and some in B, it only exists in the union
                else:
                    # If we haven't seen it before, add it to M
                    if simp not in simps_list:
                        M.add(simp)

                    # If we've already added it to the list...
                    else:
                        # Edit the lists to include new birth,death times
                        simps_list, times_list = edit_Simp_Times(simp,[i-0.5,i],simps_list,times_list)


            # Add simps and times that are in B
            simps_list = simps_list + [simp for simp in B]
            times_list = times_list + [[i-0.5,i+1] for j in range(len(B))]

            # Add simps and times that are in M
            simps_list = simps_list + [simp for simp in M]
            times_list = times_list + [[i-0.5,i] for j in range(len(M))]

            # print('A', A)
            # print('B', B)
            # print('M', M, '\n')

            # Reinitialize for next iteration
            verts = verts_next
            A = B

        loop_end = time.time()
        if verbose:
            print(f'Loop done in {loop_end-loop_st} seconds...')

        # Put list of simplices into Filtration format
        filtration = dio.Filtration(simps_list)

        return filtration, times_list


    def setup_Zigzag_witness(self, alpha, num_landmarks, k=2, verbose=False):
        '''
        Helper function for ``run_Zigzag`` that sets up inputs needed for Dionysus' zigzag persistence function.
        This only works for a fixed radius r.

        Parameters
        ----------

        alpha: float
            Alpha value for witness complex
            
        num_landmarks: int
            Number of landmark points you want (number of vertices in the witness complex)

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
        
        lst = list(self.ptclouds['PtCloud'])

        # simps_df = pd.DataFrame(columns = ['Simp','B,D'])
        simps_list = []; times_list = []

        # Vertex counter
        vertind = 0

        init_st = time.time()
        # Initialize A with W(X_0)
        
        landmarks_1 = MinMaxSubsample(lst[0], num_landmarks)
        self.ptclouds.loc[0,'PtCloud'] = landmarks_1
        
        wit = get_Witness_Complex(lst[0], landmarks_1, alpha=alpha, k=self.k)
        wit.sort()
        wit_set = set(wit)
#         print(wit_set)
        

        # Initialize A with set of simplices with verts in X_0
        A = wit_set

        # # Add all simps to the list with birth,death=[0,1]
        simps_list =  simps_list + [s for s in A]
        times_list = times_list + [[0,1] for j in range(len(A))]

        # Initialize with vertices for X_0
        verts = set([dio.Simplex([j+vertind],0) for j,pc in enumerate(landmarks_1)])

        init_end = time.time()
        if verbose:
            print(f'Initializing done in {init_end-init_st} seconds...')

        loop_st = time.time()
        # Loop over the rest of the point clouds
        for i in range(1,len(lst)):

            landmarks_2 = MinMaxSubsample(lst[i], num_landmarks)
            self.ptclouds.loc[i,'PtCloud'] = landmarks_2
            
            # Calculate witness complex of X_{i-1} \cup X_i
            wit = get_Witness_Complex(np.vstack([lst[i-1],lst[i]]), np.vstack([landmarks_1, landmarks_2]), alpha=alpha, k=self.k)

            # Adjust vertex numbers, sort, and make into a set
            wit = fix_dio_vert_nums(wit, vertind)
            wit.sort()
            wit_set = set(wit)
            
#             print(wit_set)
            
            # Increment vertex counter
            vertind = vertind+len(verts)

            # Set of vertices in R(X_i)
            verts_next = set([dio.Simplex([j+vertind],0) for j,pc in enumerate(landmarks_2)])

            # Set of simplices with verts in X_{i}
            B = set(verts_next.intersection(wit_set))

            # Set of simplices with verts in X_{i-1} AND X_{i}
            M = set()

            # Loop over simplices in R(X_{i-1} \cup R_i)
            for simp in wit:

                # Get list of vertices of simp
                bdy = getVerts(simp)

                # If it has no boundary and its in B, its a vertex in B and has been handled
                if not bdy:
                    continue

                # If all of its verts are in A, it's been handled in the initialization or the previous iteration
                if bdy.intersection(A) == bdy:
                    continue

                # If all of its verts are in B, add it to B
                elif bdy.intersection(B) == bdy:
                    B.add(simp)

                # If it has some verts in A and some in B, it only exists in the union
                # Add it to M
                else:
                    M.add(simp)


            # Add simplices in B with the corresponding birth,death times
            simps_list = simps_list + [s for s in B]
            times_list = times_list + [ [i-0.5,i+1] for j in range(len(B)) ]

            # Add simplices in M with corresponding birth,death times
            simps_list = simps_list + [s for s in M]
            times_list = times_list + [ [i-0.5,i] for j in range(len(M)) ]

            # Reinitialize for next iteration
            verts = verts_next
            A = B
            landmarks_1 = landmarks_2

        loop_end = time.time()
        if verbose:
            print(f'Loop done in {loop_end-loop_st} seconds...')

        f_st = time.time()
        filtration = dio.Filtration(simps_list)
        f_end = time.time()

        return filtration, times_list
    
    
    
    def plot_ZZ_PtClouds(self, save=False, savename='PC.png'):
        '''
        Plot the point clouds used
        
        Parameters
        ----------
        save: bool
            Set to true if you want to save the figure (default is False)
            
        savename: string
            Path to save figure (default is 'PC.png' in current directory)
        
        '''
        
        if not hasattr(self,'ptclouds'):
            print('No point clouds found...')
            print('Quitting...')
            return
        
        PC_list = list(self.ptclouds['PtCloud'])
        All_PC = np.vstack(PC_list)
        
        xmin = min(All_PC[:,0]) - 2
        xmax = max(All_PC[:,0]) + 2

        plt.rcParams.update({'font.size': 18})

        fig, axs = plt.subplots(1, int((2*len(PC_list)-1)),sharey=True, figsize=[20,2])

        # Make color list
        cs = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # Make sure color list is long enough
        if len(PC_list) > 10:
            cs = cs * int(np.ceil(len(PCs)/10))

        # Plotting...
        for i,ax in enumerate(fig.axes):
            # Plot point clouds alone
            if i%2==0:
                ax.scatter(PC_list[int(i/2)][:,0], PC_list[int(i/2)][:,1], c=cs[int(i/2)])
                ax.set_title( f'$X_{str(int(i/2))}$' ) #int(i/2))
                ax.set_xlim([xmin,xmax])
                
            # Plot union of point clouds
            else:
                ax.scatter(PC_list[int((i-1)/2)][:,0], PC_list[int((i-1)/2)][:,1], c=cs[int((i-1)/2)])
                ax.scatter(PC_list[int((i+1)/2)][:,0], PC_list[int((i+1)/2)][:,1], c=cs[int((i+1)/2)])
                ax.set_title( f'$X_{str(int((i-1)/2))} \cup X_{str(int((i+1)/2))}$' ) 
                ax.set_xlim([xmin,xmax])

        # Save figure
        if save:
            print('Saving fig at ', savename, '...')
            plt.savefig(savename, dpi=500, bbox_inches='tight')
            
    
    def plot_ZZ_Full_PtClouds(self, save=False, savename='PC_Full.png'):
        '''
        Plot the point clouds used
        
        Parameters
        ----------
        save: bool
            Set to true if you want to save the figure (default is False)
            
        savename: string
            Path to save figure (default is 'PC_Full.png' in current directory)
        
        '''
        
        if not hasattr(self,'ptclouds_full'):
            print('No point clouds found...')
            print('Quitting...')
            return
        
        PC_list = list(self.ptclouds_full['PtCloud'])
        All_PC = np.vstack(PC_list)
        
        xmin = min(All_PC[:,0]) - 2
        xmax = max(All_PC[:,0]) + 2

        plt.rcParams.update({'font.size': 18})

        fig, axs = plt.subplots(1, int((2*len(PC_list)-1)),sharey=True, figsize=[20,2])

        # Make color list
        cs = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # Make sure color list is long enough
        if len(PC_list) > 10:
            cs = cs * int(np.ceil(len(PCs)/10))

        # Plotting...
        for i,ax in enumerate(fig.axes):
            # Plot point clouds alone
            if i%2==0:
                ax.scatter(PC_list[int(i/2)][:,0], PC_list[int(i/2)][:,1], c=cs[int(i/2)])
                ax.set_title( f'$X_{str(int(i/2))}$' ) #int(i/2))
                ax.set_xlim([xmin,xmax])
                
            # Plot union of point clouds
            else:
                ax.scatter(PC_list[int((i-1)/2)][:,0], PC_list[int((i-1)/2)][:,1], c=cs[int((i-1)/2)])
                ax.scatter(PC_list[int((i+1)/2)][:,0], PC_list[int((i+1)/2)][:,1], c=cs[int((i+1)/2)])
                ax.set_title( f'$X_{str(int((i-1)/2))} \cup X_{str(int((i+1)/2))}$' ) 
                ax.set_xlim([xmin,xmax])

        # Save figure
        if save:
            print('Saving fig at ', savename, '...')
            plt.savefig(savename, dpi=500, bbox_inches='tight')
            
    
    def plot_ZZ_Cplx(self, save=False, savename='Cplx.png'):
        '''
        Plot the complexes used for the zigzag
        
        Parameters
        ----------
        save: bool
            Set to true if you want to save the figure (default is False)
            
        savename: string
            Path to save figure (default is 'Cplx.png' in current directory)
        
        '''
        
        if not hasattr(self,'filtration'):
            print('No filtration calculated yet...')
            print('Use run_Zigzag first then you can use this function...')
            print('Quitting...')
            return
            
        PC_list = list(self.ptclouds['PtCloud'])
        All_PC = np.vstack(PC_list)
        
        xmin = min(All_PC[:,0]) - 2
        xmax = max(All_PC[:,0]) + 2

        plt.rcParams.update({'font.size': 18})

        fig, axs = plt.subplots(1, int((2*len(PC_list)-1)),sharey=True, figsize=[20,2])

        # Make color list
        cs = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # Make sure color list is long enough
        if len(PC_list) > 10:
            cs = cs * int(np.ceil(len(PCs)/10))

        # Plotting vertices
        for i,ax in enumerate(fig.axes):
            # Plot point clouds alone
            if i%2==0:
                ax.scatter(PC_list[int(i/2)][:,0], PC_list[int(i/2)][:,1], c=cs[int(i/2)])
                if self.cplx_type == 'rips' or self.cplx_type == 'landmark':
                    ax.set_title( f'$R(X_{str(int(i/2))},{self.r[int(i/2)]})$' )
                elif self.cplx_type == 'witness':
                    ax.set_title( f'$W(X_{str(int(i/2))})$' )
                ax.set_xlim([xmin,xmax])
                
            # Plot union of point clouds
            else:
                ax.scatter(PC_list[int((i-1)/2)][:,0], PC_list[int((i-1)/2)][:,1], c=cs[int((i-1)/2)])
                ax.scatter(PC_list[int((i+1)/2)][:,0], PC_list[int((i+1)/2)][:,1], c=cs[int((i+1)/2)])
                if self.cplx_type == 'rips' or self.cplx_type == 'landmark':
                    ax.set_title( f'$R(X_{str(int((i-1)/2))} \cup X_{str(int((i+1)/2))}, {max(self.r[int((i-1)/2)], self.r[int((i+1)/2)])})$' ) 
                elif self.cplx_type == 'witness':
                    ax.set_title( f'$W(X_{str(int((i-1)/2))} \cup X_{str(int((i+1)/2))})$' ) 
                ax.set_xlim([xmin,xmax])


        # Plottinng edges
        for i, s in enumerate(self.filtration):
            if s.dimension() == 1: 
                vs = [v for v in s]
                ts = [int(t*2) for t in self.times_list[i]]
                for j in range(ts[0], ts[1]):
                    if j == int((2*len(PC_list)-1)):
                        break
                    else:
                        axs[j].plot( [All_PC[vs[0]][0], All_PC[vs[1]][0]], [All_PC[vs[0]][1], All_PC[vs[1]][1]], c='k' )
        # Save figure
        if save:
            print('Saving fig at ', savename, '...')
            plt.savefig(savename, dpi=500, bbox_inches='tight')

            
            
    def plot_ZZ_dgm(self, dims=[0,1], save=False, savename='Dgm.png'):
        '''
        Plot the zigzag persistence diagram
        
        Parameters
        ----------
        
        dims: list of ints
            List of integers representing dimensions you want plotted (default is [0,1])
        save: bool
            Set to true if you want to save the figure (default is False)
            
        savename: string
            Path to save figure (default is 'Cplx.png' in current directory)
        
        '''
        
        # Check diagrams have been calculated
        if not hasattr(self,'zz_dgms'):
            print('No diagrams calculated yet...')
            print('Use run_Zigzag first then you can use this function...')
            print('Quitting...')
            return
        
        # Check dimensions selected are calculated
        if not set(dims).issubset(range(len(self.zz_dgms))):
            print('Those dimensions arent available')
            print('Max dimension calculated is ', len(self.zz_dgms)-1)
            print('Quitting...')
            return
       
        # Plot zigzag diagram
        fig,ax = plt.subplots(figsize=[6,6])

        for d in dims:
            self.zz_dgms[d].drawDgm()

        plt.legend(loc=4,fontsize=25)
        
        # Save figure
        if save:
            print('Saving fig at ', savename, '...')
            plt.savefig(savename, dpi=500, bbox_inches='tight')
            
def get_Witness_Complex(witnesses, landmarks, alpha, k=2):# , num_landmarks):
    '''
    Computes the witness complex and reformats it to match dionysus formatting.
    
    Parameters
    ----------
    witnesses: np.array
        Full point cloud
    
    num_landmarks: int
        Number of landmark points you want (number of vertices in the witness complex)
        
    alpha: float or int
        Parameter alpha for the witness complex
        
    Returns
    -------
    filtration: dio.Filtration
        Dionysus filtration representing the witness complex
    
    landmarks: np.array
        Array containing locations of landmark points

    '''
        
    witness_complex = gudhi.EuclideanWitnessComplex(witnesses=witnesses, landmarks=landmarks)
    simplex_tree = witness_complex.create_simplex_tree(max_alpha_square = alpha, limit_dimension=k)
    simplex_tree.initialize_filtration()
    filt = simplex_tree.get_filtration()
    
    filtration = dio.Filtration([ dio.Simplex(f[0], f[1]) for f in filt ])
    
    return filtration
    
    

def edit_Simp_Times(simp,new_bd_times,simps_list,times_list):

    # Find where it is in the list
    simp_ind = simps_list.index(simp)

    # Remove the simplex and its birth,death time from each list
    simps_list.pop(simp_ind)
    curr_times = times_list.pop(simp_ind)

    # Add to lists and concatenate birth,death times we've added before with new ones
    simps_list.append(simp)
    times_list.append(curr_times + new_bd_times)

    return simps_list, times_list

def to_PD_Class(dio_dgms):
    '''
    Helper function to convert Dionysus diagram class into my PD class.

    Parameters
    ----------

    dio_dgms: list of dio.Diagrams
        Persistence diagrams from Dionysus

    Returns
    -------
    all_dgms: list
        List of persistence diagrams formatted as instances of PD class.

    '''
    dgms_list = []
    for i,dgm in enumerate(dio_dgms):
        dgms_list.append([[[p.birth,p.death] for p in dgm],i])

    all_dgms = [PD(dgms_list[i][0],dgms_list[i][1]) for i in range(len(dgms_list))]

    return all_dgms


def fix_dio_vert_nums(Cplx, vertind):
    '''
    Helper function to adjust indexing of vertices in Dionysus from ``fill_rips`` function.

    Cplx: List of dio.Simplex's or a dio.Filtration
        Starting complex you want to adjust indices of

    vertind: int
        Starting index for vertices

    Returns
    -------
    New complex with vertex indices adjusted by vertind.

    '''
    New_Cplx = []
    for s in Cplx:
        # s.data = data
        vlist = []
        for v in s:
            vlist.append(v+vertind)
        s = dio.Simplex(vlist,s.data)

        New_Cplx.append(s)
    return New_Cplx

def getVerts(simp):
    '''
    Helper function to get all vertices of the input simplex

    Parameters
    -----------
    simp: dio.Simplex
        Instance of Dionysus simplex class

    Returns
    -------
    List of vertices in the simplex

    '''
    if simp.dimension == 2:
        return [dio.Simplex([v],0) for v in t]
    else:
        return set([s for s in simp.boundary()])

    
# def plot_Zigzag(PC_list, delta, save=False, savename=[], verbose=False):
#     '''
#     Plots the rips complexes including unions for zigzag complex.

#     '''

#     st_time = time.time()

#     fig, axs = plt.subplots(1, 2*len(PC_list)-1,sharey=True, figsize=[20,2])

#     if type(delta) == int or type(delta) == float:
#         delta = [delta]

#     if len(delta) == 1:
#         delta = delta * (2*len(PC_list)-1)
#     else:
#         delta_new = []
#         delta_new.append(delta[0])
#         for i in range(0,len(delta)-1):
#             delta_new.append( max(delta[i],delta[i+1]) )
#             delta_new.append(delta[i+1])
#         delta = delta_new

#     pt_clouds = []
#     pt_clouds.append(PC_list[0])
#     for j in range(1,len(PC_list)):
#         pt_clouds.append(np.concatenate([PC_list[j-1],PC_list[j]]))
#         pt_clouds.append(PC_list[j])

#     i=0

#     colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#     cs = [val for val in colors for _ in (0, 1)]

#     xmax = max(np.concatenate(PC_list)[:,0])
#     xmin = min(np.concatenate(PC_list)[:,0])
#     ymax = max(np.concatenate(PC_list)[:,1])
#     ymin = min(np.concatenate(PC_list)[:,1])

#     for i in range(2*len(PC_list)-1):
#         if i%2 == 0:
#             axs[i].scatter(pt_clouds[i][:,0],pt_clouds[i][:,1],c=cs[i])
#             axs[i].set_title('$'+ str(int(i/2)) +'$'+ '\n' + '$r = ' + str(delta[i]) + '$')
#             axs[i].set_xlim([xmin-0.2,xmax+0.2])

#             for j in range(len(pt_clouds[i])):
#                 for k in range(j):
#                     if euclidean(pt_clouds[i][j] ,pt_clouds[i][k]) <= delta[i]:
#                         axs[i].plot([pt_clouds[i][j][0],pt_clouds[i][k][0]],
#                                     [pt_clouds[i][j][1],pt_clouds[i][k][1]],
#                                     c='k')
#                     if i > 0:
#                         if euclidean(pt_clouds[i][j] ,pt_clouds[i][k]) <= delta[i-1]:
#                             axs[i-1].plot([pt_clouds[i][j][0],pt_clouds[i][k][0]],
#                                         [pt_clouds[i][j][1],pt_clouds[i][k][1]],
#                                         c='k')
#                     if i < 2*len(PC_list)-2:
#                         if euclidean(pt_clouds[i][j] ,pt_clouds[i][k]) <= delta[i+1]:
#                             axs[i+1].plot([pt_clouds[i][j][0],pt_clouds[i][k][0]],
#                                         [pt_clouds[i][j][1],pt_clouds[i][k][1]],
#                                         c='k')
#         else:
#             axs[i].scatter(pt_clouds[i-1][:,0],pt_clouds[i-1][:,1],c=cs[i-1])
#             axs[i].scatter(pt_clouds[i+1][:,0],pt_clouds[i+1][:,1],c=cs[i+1])
#             axs[i].set_title('$'+ str(i/2) +'$'+ '\n' + '$r = ' + str(delta[i]) + '$')

#             axs[i].set_xlim([xmin-0.2,xmax+0.2])

#             for p1 in pt_clouds[i-1]:
#                 for p2 in pt_clouds[i+1]:
#                     if euclidean(p1,p2) <= delta[i]:
#                         axs[i].plot([p1[0],p2[0]],[p1[1],p2[1]],c='k')
#         if verbose:
#             print('Done with', i)

#     end_time = time.time()
#     print('Time to plot:', end_time-st_time)

#     if save:
#         plt.savefig(savename, dpi=500, bbox_inches='tight')

#     # for ax in fig.axes:
#     #     if i%2==0:
#     #         ax.scatter(pt_clouds[i][:,0],pt_clouds[i][:,1],c=cs[i])
#     #         ax.set_title(str(int(i/2)) + '\n' + 'r = ' + str(delta[i]))
#     #         for p1 in pt_clouds[i]:
#     #             for p2 in pt_clouds[i]:
#     #                 if euclidean(p1,p2) <= delta[i]:
#     #                     ax.plot([p1[0],p2[0]],[p1[1],p2[1]],c='k')
#     #                     ax.set_xlim([xmin-2,xmax+2])
#     #     else:
#     #         ax.scatter(pt_clouds[i-1][:,0],pt_clouds[i-1][:,1],c=cs[i-1])
#     #         ax.scatter(pt_clouds[i+1][:,0],pt_clouds[i+1][:,1],c=cs[i+1])
#     #         ax.set_title(str(int(i/2)) + '\n' + 'r = ' + str(delta[i]))
#     #         for p1 in np.concatenate([pt_clouds[i-1],pt_clouds[i+1]]):
#     #             for p2 in np.concatenate([pt_clouds[i-1],pt_clouds[i+1]]):
#     #                 if euclidean(p1,p2) <= delta[i]:
#     #                     ax.plot([p1[0],p2[0]],[p1[1],p2[1]],c='k')
#     #                     ax.set_xlim([xmin-2,xmax+2])
#     #     i=i+1
