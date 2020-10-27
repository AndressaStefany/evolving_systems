import numpy as np
import pandas as pd

# For production
from algoritms.sostream.utils import dist

# For test
from utils import dist


def find_neighbors(win_microcluster, min_pts, micro_list, all_microclusters):
    if len(micro_list) >= min_pts:
        win_dist = []
        all_win_dist = []
        for microcluster in micro_list:
            # calculating all distances from the winner to the neighbors
            win_dist.append(dist(microcluster.centroid,
                                 win_microcluster.centroid))

        for microcluster in all_microclusters:
            # calculating all distances from the winner to the neighbors
            all_win_dist.append(dist(microcluster.centroid,
                                     win_microcluster.centroid))
        # order from the smallest to the largest distance, references the winner
        win_dist.sort()
        all_win_dist.sort()
        # indexes that sort the win_dist array
        idx_microclusters = np.argsort(win_dist)

        # k_dist is half the distance between the winner
        # and the nearest microcluster
        k_dist = all_win_dist[min_pts-1]
        win_microcluster.radius = k_dist
        win_nn = [micro_list[idx] for idx in idx_microclusters[0:(min_pts)]]
        return win_nn
    else:
        return []
