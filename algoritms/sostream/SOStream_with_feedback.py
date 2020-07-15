'''
Code downloaded from:
https://github.com/ruteee/SOStream
'''
import numpy as np
import pandas as pd
from algoritms.sostream.find_neighbors import find_neighbors
from algoritms.sostream.find_overlap import find_overlap
from algoritms.sostream.merge_clusters import merge_clusters
from algoritms.sostream.new_cluster import newCluster
from algoritms.sostream.update_cluster import updateCluster
from algoritms.sostream.utils import min_dist, dist
from sklearn.preprocessing import LabelEncoder


class SOStream_feedback:

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
        self.feedback = []

    def fit_predict(self, X):
        for r in X:
            self.process(r)

    def get_predict(self):
        labelencoder_X = LabelEncoder()

        df_y_pred = pd.DataFrame(self.feedback, columns=['x', 'y'])
        df_y_pred['x+y'] = df_y_pred['x'].astype(str) + \
            '_'+df_y_pred['y'].astype(str)
        df_y_pred['CLASS'] = labelencoder_X.fit_transform(
            df_y_pred.values[:, 2])

        return df_y_pred['CLASS']

    def process(self, vt):
        winner_micro_cluster = min_dist(vt, self.M[-1])
        new_M = self.M[-1].copy()

        if len(new_M) >= self.min_pts:
            winner_neighborhood = find_neighbors(
                winner_micro_cluster, self.min_pts, new_M)

            if dist(vt, winner_micro_cluster.centroid) < winner_micro_cluster.radius:
                old_centroid_winner = winner_micro_cluster.centroid
                old_centroids = [c.centroid for c in winner_neighborhood]

                updateCluster(winner_micro_cluster, vt,
                              self.alpha, winner_neighborhood)

                # update centroid in self.feedback
                self.feedback.append(winner_micro_cluster.centroid.tolist())
                self.feedback = np.where(np.equal(self.feedback, old_centroid_winner),
                                         winner_micro_cluster.centroid.tolist(),
                                         self.feedback)
                new_centroids = [c.centroid for c in winner_neighborhood]
                for i in range(0, len(old_centroids)):
                    self.feedback = np.where(self.feedback == old_centroids[i],
                                             new_centroids[i],
                                             self.feedback)
                try:
                    self.feedback = self.feedback.tolist()
                except:
                    pass

            else:
                new_M.append(newCluster(vt))
                self.feedback.append(vt.tolist())

            overlap = find_overlap(winner_micro_cluster, winner_neighborhood)

            if len(overlap) > 0:
                merged_cluster, deleted_clusters = merge_clusters(
                    winner_micro_cluster, overlap, self.merge_threshold)

                for deleted_cluster in deleted_clusters:
                    new_M.remove(deleted_cluster)

                    if merged_cluster is not None:
                        self.feedback = np.where(np.equal(self.feedback, deleted_cluster.centroid),
                                                 merged_cluster.centroid.tolist(),
                                                 self.feedback)
                        self.feedback = self.feedback.tolist()

                if merged_cluster is not None:
                    new_M.append(merged_cluster)

        else:
            new_M.append(newCluster(vt))
            self.feedback.append(vt.tolist())

        self.M.append(new_M)
    pass
