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
    :return: Square central difference matrix D
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
