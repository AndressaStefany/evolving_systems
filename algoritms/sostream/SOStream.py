'''
Code downloaded from:
https://github.com/ruteee/SOStream
'''

from algoritms.sostream.find_neighbors import find_neighbors
from algoritms.sostream.find_overlap import find_overlap
from algoritms.sostream.merge_clusters import merge_clusters
from algoritms.sostream.new_cluster import newCluster
from algoritms.sostream.update_cluster import updateCluster
from algoritms.sostream.utils import min_dist, dist


class SOStream:

    def __init__(self, alpha=0.1, min_pts=10, merge_threshold=27000):
        """
        SOStream - Self Organizing Density-Based Clustering over Data Stream.

        Parameters
        ----------
        alpha: float, optional
            The scale factor that brings neighbors closer to the winning
            cluster.
        min_pts : int, optional
            The minimum number of neighbors.
        merge_threshold : float
            The fusion threshold.

        Attributes
        ----------
        M : array, shape = n_samples x n_clusters
            List composed of lists of microclusters that are created each
            time t.

        Notes
        -----

        References
        ----------
        Isaksson, C., Dunham, M. H., & Hahsler, M. (2012). SOStream: Self 
        organizing density-based clustering over data stream. Lecture Notes 
        in Computer Science (Including Subseries Lecture Notes in Artificial 
        Intelligence and Lecture Notes in Bioinformatics), 7376 LNAI, 264–278.
        https://doi.org/10.1007/978-3-642-31537-4_21
        """
        self.alpha = alpha
        self.min_pts = min_pts
        self.M = [[]]
        self.merge_threshold = merge_threshold

    def fit_predict(self, X):
        for r in X.values:
            self.process(r)

    def process(self, vt):
        winner_micro_cluster = min_dist(vt, self.M[-1])
        new_M = self.M[-1].copy()
        if len(new_M) >= self.min_pts:
            winner_neighborhood = find_neighbors(
                winner_micro_cluster, self.min_pts, new_M)
            if dist(vt, winner_micro_cluster.centroid) < winner_micro_cluster.radius:
                updateCluster(winner_micro_cluster, vt,
                              self.alpha, winner_neighborhood)
            else:
                new_M.append(newCluster(vt))
            overlap = find_overlap(winner_micro_cluster, winner_neighborhood)
            if len(overlap) > 0:
                merged_cluster, deleted_clusters = merge_clusters(
                    winner_micro_cluster, overlap, self.merge_threshold)
                for deleted_cluster in deleted_clusters:
                    new_M.remove(deleted_cluster)
                if merged_cluster is not None:
                    new_M.append(merged_cluster)
        else:
            new_M.append(newCluster(vt))
        self.M.append(new_M)
    pass
