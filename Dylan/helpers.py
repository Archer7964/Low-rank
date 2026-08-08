# Externel Packages #
import numpy as np

# Rank Selection (Archer) #

def select_rank(singular_values, MAX_RANK, RANK_TOL):
    limit = RANK_TOL * np.linalg.norm(singular_values)
    for rank in range(1, len(singular_values) + 1):
        if np.linalg.norm(singular_values[rank:]) <= limit:
            return min(rank, MAX_RANK)
    return min(len(singular_values), MAX_RANK)