
import numpy as np
import dionysus as dio
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

class PD(object):
    '''
    Class to hold persistence diagram.

    Parameters
    ----------
    dgm: np.array
        A kx2 numpy array containing birth death coordinates of the diagram
    dim: int
        Dimension of the diagram


    '''
    def __init__(self,dgm,dimension):
        '''
        Initialize.

        '''
        self.dgm = np.array(dgm)
        self.dim = dimension
        self.numPts = len(dgm)

    def __len__(self):
        '''
        Returns
        --------

        Number of points in the diagram

        '''
        return self.numPts

    def __str__(self):
        '''
        Nicely prints.

        '''
        attrs = vars(self)
        output = ''
        output += 'Persistence Diagram\n'
        output += '---\n'
        for key in attrs.keys():
            output += str(key) + ' : '
            output += str(attrs[key])+ '\n'
        output += '---\n'
        return output

    def drawDgm(self,boundary=None,epsilon = .5, color=None):
        '''
        Plots persistence diagram.

        Parameters
        ----------
        boundary: float
            The diagram is drawn on [0,boundary]x[0,boundary].
        epsilon: float
            If boundary not given, then it is determined to be the max death time from the input diagram plus epsilon.
        color:
            Color for plotting points in diagram.

        '''

        D = np.copy(self.dgm)

        if not self.dgm.size:
            print('Uh oh.. Diagram ' + str(self.dim) + ' is empty...')
            print('Quitting...')
            return


        # Separate out the infinite classes if they exist
        includesInfPts = np.inf in D
        if includesInfPts:
            Dinf = D[np.isinf(D[:,1]),:]
            D = D[np.isfinite(D[:,1]),:]

        # Get the max birth/death time if it's not already specified
        if not boundary:
            boundary = np.amax(D)+epsilon

        # if fig is None:
        #     fig = plt.figure()
        # ax = fig.gca()
        # Plot the diagonal
        plt.plot([0,boundary],[0,boundary],color='gray')

        # Plot the diagram points
        if color is None:
            plt.scatter(D[:,0],D[:,1], label='Dim '+ str(self.dim))
        else:
            plt.scatter(D[:,0],D[:,1], c=color, label='Dim '+ str(self.dim))

        if includesInfPts:
            for i in range(len(Dinf[:,0])):
                if color is None:
                    plt.scatter(Dinf[i,0], .98*boundary, marker='s', color='red')
                else:
                    plt.scatter(Dinf[i,0], .98*boundary, marker='s', color=color)

            plt.axis([-.01*boundary,boundary,-.01*boundary,boundary])

        plt.ylabel('Death')
        plt.xlabel('Birth')
        # plt.title('Persistence Diagram\nDimension '+str(self.dim))


    def drawDgm_BL(self,boundary=None,epsilon = .5, color=None):

        '''
        Plots persistence diagram in birth-lifetime coordinates.

        Parameters
        ----------
        boundary: float
            The diagram is drawn on [0,boundary]x[0,boundary].
        epsilon: float
            If boundary not given, then it is determined to be the max death time from the input diagram plus epsilon.
        color:
            Color for plotting points in diagram.

        '''

        D = np.copy(self.dgm)
        D[:,1] = D[:,1] - D[:,0]

        # Separate out the infinite classes if they exist
        includesInfPts = np.inf in D
        if includesInfPts:
            Dinf = D[np.isinf(D[:,1]),:]
            D = D[np.isfinite(D[:,1]),:]

        # Get the max birth/death time if it's not already specified
        if not boundary:
            boundary = np.amax(D)+epsilon

        # Plot the diagram points
        if color is None:
            plt.scatter(D[:,0],D[:,1])
        else:
            plt.scatter(D[:,0],D[:,1], c=color)

        if includesInfPts:
            for i in range(len(Dinf[:,0])):
                if color is None:
                    plt.scatter(Dinf[i,0], .98*boundary, marker='s', color='red')
                else:
                    plt.scatter(Dinf[i,0], .98*boundary, marker='s', color=color)

        plt.ylabel('Lifetime')
        plt.xlabel('Birth')
        # plt.title('Persistence Diagram\nDimension '+str(self.dim))


    def removeInfiniteClasses(self):
        '''
        Simply deletes classes that have infinite lifetimes.

        '''
        keepRows = np.isfinite(self.dgm[:,1])
        return self.dgm[keepRows,:]

    def maxPers(self):
        '''
        Calculates maximum persistence of the diagram.

        '''

        Dgm = np.copy(self.dgm)

        try:
            lifetimes = Dgm[:,1] - Dgm[:,0]
            m = max(lifetimes)
            if m == np.inf:
                # Get rid of rows with death time infinity
                numRows = Dgm.shape[0]
                rowsWithoutInf = list(set(np.where(Dgm[:,1] != np.inf)[0]))
                m = max(lifetimes[rowsWithoutInf])
            return m
        except:
            return 0

    def toBirthLifetime(self):
        '''
        Coverts diagram to birth-lifetime coordinates.

        '''

        Dgm = np.copy(self.dgm)
        Dgm[:,1] = Dgm[:,1] - Dgm[:,0]

        return Dgm
