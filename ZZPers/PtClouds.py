import numpy as np
import dionysus as dio
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from ZZPers.ZZPers.PD import PD
import time
from ripser import ripser
from scipy.spatial.distance import euclidean

## Using a fixed radius is working... Need to add changing radius

class PtClouds(object):
    '''
    Class to hold collection of point clouds.

    Parameters
    ----------
    ptclouds: list
        List of point clouds
    verbose: bool
        If true, prints updates when running code

    '''

    def __init__(self, ptclouds, verbose=False):
        '''
        Initialize.

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


    def run_Zigzag(self, r, k=2):
        '''
        Runs zigzag persistence on collections of point clouds, taking unions between each adjacent pair of point clouds.
        Adds attributes to class ``zz``, ``zz_dgms``, ``zz_cells`` which are the outputs of dionysus' zigzag persistence function.
        Also adds attributes ``setup_time`` and ``zigzag_time`` with timing data.

        Parameters
        ----------

        r: float or list of floats
            Parameter for Rips complex on each point cloud. Can be a single number or a list with one value per point cloud.

        k: int, optional
            Maximum dimension simplex desired (default is 2)

        '''

        self.r = r
        self.k = k

        # Set up inputs for dionysus zigzag persistence
        ft_st = time.time()
        if type(r) == float or type(r) == int:
            filtration, times = self.setup_Zigzag_fixed(r = self.r, k = self.k, verbose = self.verbose)
        else:
            filtration, times = self.setup_Zigzag_changing(r = self.r, k = self.k, verbose = self.verbose)
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

        if verbose:
            print(f'Calculated filtration in {f_end-f_st} seconds...\n')

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


    def plot_Zigzag(self):
        '''
        Plots the rips complexes including unions for zigzag complex.

        '''

        PC_list = list(self.ptclouds['PtCloud'])
        delta = self.r

        fig, axs = plt.subplots(1, 2*len(PC_list)-1,sharey=True, figsize=[20,2])

        if type(delta) == int or type(delta) == float:
            delta = [delta]

        if len(delta) == 1:
            delta = delta * (2*len(PC_list)-1)
        else:
            delta_new = []
            delta_new.append(delta[0])
            for i in range(0,len(delta)-1):
                delta_new.append( max(delta[i],delta[i+1]) )
                delta_new.append(delta[i+1])
            delta = delta_new

        pt_clouds = []
        pt_clouds.append(PC_list[0])
        for j in range(1,len(PC_list)):
            pt_clouds.append(np.concatenate([PC_list[j-1],PC_list[j]]))
            pt_clouds.append(PC_list[j])

        i=0

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        cs = [val for val in colors for _ in (0, 1)]

        xmax = max(np.concatenate(PC_list)[:,0])
        xmin = min(np.concatenate(PC_list)[:,0])
        ymax = max(np.concatenate(PC_list)[:,1])
        ymin = min(np.concatenate(PC_list)[:,1])

        for ax in fig.axes:
            if i%2==0:
                ax.scatter(pt_clouds[i][:,0],pt_clouds[i][:,1],c=cs[i])
                ax.set_title(str(int(i/2)) + '\n' + 'r = ' + str(delta[i]))
                for p1 in pt_clouds[i]:
                    for p2 in pt_clouds[i]:
                        if euclidean(p1,p2) <= delta[i]:
                            ax.plot([p1[0],p2[0]],[p1[1],p2[1]],c='k')
                            ax.set_xlim([xmin-2,xmax+2])
            else:
                ax.scatter(pt_clouds[i-1][:,0],pt_clouds[i-1][:,1],c=cs[i-1])
                ax.scatter(pt_clouds[i+1][:,0],pt_clouds[i+1][:,1],c=cs[i+1])
                ax.set_title(str(int(i/2)) + '\n' + 'r = ' + str(delta[i]))
                for p1 in np.concatenate([pt_clouds[i-1],pt_clouds[i+1]]):
                    for p2 in np.concatenate([pt_clouds[i-1],pt_clouds[i+1]]):
                        if euclidean(p1,p2) <= delta[i]:
                            ax.plot([p1[0],p2[0]],[p1[1],p2[1]],c='k')
                            ax.set_xlim([xmin-2,xmax+2])
            i=i+1




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

    Cplx: List of dio.Simplex's or adio.Filtration
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
