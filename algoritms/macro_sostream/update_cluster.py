from math import exp
import numpy as np

# For production
from algoritms.macro_sostream.utils import dist, max_dist

# For test
# from utils import dist, max_dist


def update_microcluster(macro_cluster, win_micro_cluster, vt, alpha, winner_neighborhood):

    win_micro_cluster.centroid = (win_micro_cluster.number_points *
                                  win_micro_cluster.centroid + vt) / (win_micro_cluster.number_points+1)
    win_micro_cluster.number_points += 1
    width_neighbor = win_micro_cluster.radius ** 2

    for neighbor_micro_cluster in winner_neighborhood:
        influence = exp(-(dist(neighbor_micro_cluster.centroid,
                               win_micro_cluster.centroid)/(2 * width_neighbor)))
        neighbor_micro_cluster.centroid = neighbor_micro_cluster.centroid + alpha * \
            influence*(win_micro_cluster.centroid -
                       neighbor_micro_cluster.centroid)

    centroids_micro = np.array([win_micro_cluster.centroid])
    macro_cluster.micros.remove(win_micro_cluster)
    for micro in macro_cluster.micros:
        centroids_micro = np.concatenate(
            (centroids_micro, np.array([micro.centroid.copy()])))

    macro_cluster.micros.append(win_micro_cluster)

    # update macro
    macro_cluster.centroid = np.median(centroids_micro, axis=0)
    far_micro_cluster = max_dist(macro_cluster.centroid, macro_cluster.micros)
    aux_radius = dist(macro_cluster.centroid,
                      far_micro_cluster.centroid) + far_micro_cluster.radius
    macro_cluster.radius = aux_radius if aux_radius > macro_cluster.radius else macro_cluster.radius


def update_macrocluster(macro_cluster, new_micro_cluster):
    macro_cluster.centroid = (macro_cluster.number_micro_clusters * macro_cluster.centroid +
                              new_micro_cluster.centroid) / (macro_cluster.number_micro_clusters+1)
    macro_cluster.number_micro_clusters += 1
    macro_cluster.micros.append(new_micro_cluster)

    # find far micro cluster
    far_micro_cluster = max_dist(macro_cluster.centroid, macro_cluster.micros)
    aux_radius = dist(macro_cluster.centroid,
                      far_micro_cluster.centroid) + far_micro_cluster.radius
    macro_cluster.radius = aux_radius if aux_radius > macro_cluster.radius else macro_cluster.radius
