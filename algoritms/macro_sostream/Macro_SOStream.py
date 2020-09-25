import numpy as np
import pandas as pd
from algoritms.macro_sostream.find_neighbors import find_neighbors
from algoritms.macro_sostream.find_overlap import find_overlap
from algoritms.macro_sostream.merge_clusters import merge_clusters
from algoritms.macro_sostream.new_cluster import newCluster
from algoritms.macro_sostream.update_cluster import updateCluster
from algoritms.macro_sostream.utils import min_dist, dist
from sklearn.preprocessing import LabelEncoder


class Macro_SOStream:

    def __init__(self, alpha=0.1, min_pts=10, merge_threshold=27000, p=1):
        """
        Macro SOStream - Modification of SOStream, Self Organizing 
        Density-Based Clustering over Data Stream.

        Parameters
        ----------
        alpha: float, optional
            The scale factor that brings neighbors closer to the winning
            cluster.
        min_pts : int, optional
            The minimum number of neighbors.
        merge_threshold : float
            The fusion threshold.
        p : float
            p-radius.

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
        self.merge_threshold = merge_threshold
        self.p = p
        self.lists = [
            [[]]
        ]
        self.class_index = []
        self.class_centroids = []
        self.class_lists = []
        self.list_number = 0

    def fit_predict(self, X):
        for r in X.values:
            self.process(r)

    def get_micro_predict(self):
        labelencoder_X = LabelEncoder()

        df_y_pred = pd.DataFrame(self.class_centroids, columns=['x', 'y'])
        df_y_pred['x+y'] = df_y_pred['x'].astype(str) + \
            '_'+df_y_pred['y'].astype(str)
        df_y_pred['CLASS'] = labelencoder_X.fit_transform(
            df_y_pred.values[:, 2])

        return df_y_pred['CLASS']

    def get_macro_predict(self):
        macroclusters = [c for c in self.class_lists]
        return macroclusters

    def get_centroid_count(self):
        list_count = 0
        for l in self.lists:
            list_count = list_count + len(l[-1])
        return list_count

    def get_all_mclusters(self):
        ans = []
        for l in self.lists:
            ans.extend((l[-1]))
        return ans

    def check_overlap(self, winner_micro_cluster, winner_neighborhood, new_X, merge_threshold):
        overlap = find_overlap(winner_micro_cluster, winner_neighborhood)

        if len(overlap) > 0:
            merged_cluster, deleted_clusters = merge_clusters(winner_micro_cluster,
                                                              overlap,
                                                              merge_threshold)
            if merged_cluster is not None:
                new_X.append(merged_cluster)
            if deleted_clusters is not None:
                for deleted_cluster in deleted_clusters:
                    self.class_centroids = np.where(np.equal(self.class_centroids, deleted_cluster.centroid),
                                                    merged_cluster.centroid,
                                                    self.class_centroids)
                    new_X.remove(deleted_cluster)
        try:
            self.class_centroids = self.class_centroids.tolist()
        except:
            pass
        return new_X

    def check_point_cluster(self, vt, new_X):
        winner_micro_cluster = min_dist(vt, new_X)
        all_mclusters = self.get_all_mclusters()

        winner_neighborhood = find_neighbors(winner_micro_cluster,
                                             self.min_pts,
                                             new_X,
                                             all_mclusters)

        idx = self.list_number
        new_cluster_list_aux = self.lists[idx].copy()
        new_cluster_list = new_cluster_list_aux[-1]

        if dist(vt, winner_micro_cluster.centroid) < winner_micro_cluster.radius:
            self.class_index = np.append(self.class_index, 0)

            old_centroid_winner = winner_micro_cluster.centroid
            old_centroids = [c.centroid for c in winner_neighborhood]
            updateCluster(winner_micro_cluster,
                          vt,
                          self.alpha,
                          winner_neighborhood)

            # update centroid
            new_centroids = [c.centroid for c in winner_neighborhood]
            self.class_centroids.append(winner_micro_cluster.centroid)
            self.class_centroids = np.where(np.equal(self.class_centroids, old_centroid_winner),
                                            np.array(
                                                winner_micro_cluster.centroid),
                                            self.class_centroids)

            for i in range(0, len(old_centroids)):
                self.class_centroids = np.where(self.class_centroids == old_centroids[i],
                                                new_centroids[i],
                                                self.class_centroids)
            new_X = self.check_overlap(winner_micro_cluster,
                                       winner_neighborhood,
                                       new_X,
                                       self.merge_threshold)
            try:
                self.class_centroids = self.class_centroids.tolist()
            except:
                pass

        elif (dist(vt, winner_micro_cluster.centroid) > self.p*(winner_micro_cluster.radius)) & (self.list_number < 10) & (len(new_cluster_list) >= self.min_pts):
            self.class_index = np.append(self.class_index, 1)
            self.list_number = self.list_number+1

            try:
                new_X_aux = self.lists[self.list_number].copy()
                new_X = new_X_aux[-1]
                new_X = self.check_point_cluster(vt, new_X)
            except:
                # create new empty list
                self.lists.append([[]])
                new_X_aux = self.lists[self.list_number].copy()
                new_X = new_X_aux[-1]
                new_X.append(newCluster(vt))
                self.class_centroids.append(vt)
                pass

            self.lists[self.list_number].append(new_X)
            new_winner_micro_cluster = min_dist(vt, new_X)
            new_winner_neighborhood = find_neighbors(new_winner_micro_cluster,
                                                     self.min_pts,
                                                     new_X,
                                                     all_mclusters)
            new_X = self.check_overlap(new_winner_micro_cluster,
                                       new_winner_neighborhood,
                                       new_X,
                                       self.merge_threshold)
        else:
            self.class_centroids.append(vt)
            self.class_index = np.append(self.class_index, 0)
            new_X.append(newCluster(vt))

        return new_X

    def process(self, vt):  # type vt = numpy array
        self.list_number = 0
        idx = self.list_number
        new_cluster_list_aux = self.lists[idx].copy()
        new_cluster_list = new_cluster_list_aux[-1]

        if len(new_cluster_list) >= self.min_pts:
            new_cluster_list = self.check_point_cluster(vt, new_cluster_list)
        else:
            self.class_index = np.append(self.class_index, 0)
            self.class_centroids.append(vt)
            new_cluster_list.append(newCluster(vt))

        self.lists[self.list_number].append(new_cluster_list)
        self.class_lists.append(self.list_number)
    pass
