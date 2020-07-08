import numpy as np
import pandas as pd
from algoritms.sostream.utils import dist


def find_neighbors(win_microcluster, min_pts, model_t, all_mclusters):
    if len(model_t) >= min_pts:
        win_dist = []
        all_win_dist = []
        for microcluster in model_t:
            # calculando todas as distancias do vencedor para os vizinhos
            win_dist.append(dist(microcluster.centroid,
                                 win_microcluster.centroid))

        for microcluster in all_mclusters:
            # calculando todas as distancias do vencedor para os vizinhos
            all_win_dist.append(dist(microcluster.centroid,
                                     win_microcluster.centroid))
        # ordem da menor para a maior distancia, referencia o vencedor
        win_dist.sort()
        all_win_dist.sort()
        # indeces que ordenam o array win_dist
        idx_microclusters = np.argsort(win_dist)

        # k_dist é a metade da distância entre o vencedor e
        # um microcluster mais proximo
        k_dist = all_win_dist[min_pts-1]  # mudado
        win_microcluster.radius = k_dist
        win_nn = [model_t[idx] for idx in idx_microclusters[0:(min_pts)]]
        return win_nn
    else:
        return []
