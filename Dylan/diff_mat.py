# Externel Packages #
import numpy as np

# Finite difference matrices #

def forward_Dif(size, dy, periodic=False):
    """
    Produces a square forward difference matrix.
    :param size: width of square matrix
    :param dy: step between grid points
    :param periodic: toggle for periodic boundary conditions
    :return: Square forward difference matrix D.
    """
    dy_inv = 1/dy
    D = np.zeros((size,size))
    for i in range(size - 1):
        D[i, i:i + 2] = [-dy_inv, dy_inv]
    D[-1][-1] = -dy_inv
    if periodic:
        D[-1][0] = dy_inv
    return D

def central_Dif(size, dy, periodic=False):
    """
    Produces a square central difference matrix.
    :param size: width of square matrix
    :param dy: step between grid points
    :param periodic: toggle for periodic boundary conditions
    :return: Square central difference matrix D.
    """
    dy_inv = 1 / (2*dy)
    D = np.zeros((size,size))
    for i in range(1, size - 1):
        D[i, i-1:i + 2] = [-dy_inv, 0, dy_inv]
    D[0][1] = dy_inv
    D[-1][-2] = -dy_inv
    if periodic:
        D[0][-1] = -dy_inv
        D[-1][0] = dy_inv
    return D

def upwind_Dif(size, dy, positive=True, periodic=False):
    """
        Produces a square central difference matrix.
        :param size: width of square matrix
        :param dy: step between grid points
        :param positive: toggle for positive/negative use
        :param periodic: toggle for periodic boundary conditions
        :return: Upwind 2nd Order difference matrix D.
    """
    dy_inv = 1 / (2*dy)
    D = np.zeros((size, size))
    if not positive:
        for i in range(size-2):
            D[i, i:i+3] = [-3, 4, -1]
        D[-2, -2:size] = [-3, 4]
        D[-1, -1] = -3
        if periodic:
            D[-2][0] = -1
            D[-1][0:2] = [4, -1]
    else:
        D[0, 0] = 3
        D[1, 0:2] = [-4, 3]
        for i in range(2, size):
            D[i, i-2:i+1] = [1, -4, 3]
        if periodic:
            D[0][-2:size] = [1, -4]
            D[1][-1] = 1
    return D*dy_inv
