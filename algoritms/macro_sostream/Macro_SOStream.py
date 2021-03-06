from sklearn.metrics.cluster import adjusted_rand_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# For production
from algoritms.macro_sostream.find_neighbors import find_neighbors
from algoritms.macro_sostream.find_overlap import find_overlap
from algoritms.macro_sostream.merge_clusters import merge_microclusters, merge_macroclusters
from algoritms.macro_sostream.new_cluster import new_microcluster, new_macrocluster
from algoritms.macro_sostream.update_cluster import update_microcluster, update_macrocluster
from algoritms.macro_sostream.utils import min_dist, dist

# For test
# from find_neighbors import find_neighbors
# from find_overlap import find_overlap
# from merge_clusters import merge_microclusters, merge_macroclusters
# from new_cluster import new_microcluster, new_macrocluster
# from update_cluster import update_microcluster, update_macrocluster
# from utils import min_dist, dist


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
        self.macro_list = [[]]
        self.class_micro_centroids = []
        self.class_macro_lists = []
        self.idx_macro = 0
        self.idx_macro_unique = [0]

    def fit_predict(self, X):
        for r in X.values:
            self.process(r)

    def get_micro_predict(self):
        labelencoder_X = LabelEncoder()

        df_y_pred = pd.DataFrame(
            self.class_micro_centroids, columns=['x', 'y'])
        df_y_pred['x+y'] = df_y_pred['x'].astype(str) + \
            '_'+df_y_pred['y'].astype(str)
        df_y_pred['CLASS'] = labelencoder_X.fit_transform(
            df_y_pred.values[:, 2])

        return df_y_pred['CLASS']

    def get_macro_predict(self):
        macroclusters = [c for c in self.class_macro_lists]
        return macroclusters

    def get_all_microclusters(self):
        ans = []
        for macro in self.macro_list[-1]:
            ans.extend(macro.micros)
        return ans

    def check_micro_overlap(self, winner_microcluster, winner_neighborhood, micro_list):
        """
        Check Overlap function.

        Parameters
        ----------
        winner_microcluster
        winner_neighborhood
        micro_list
        """

        overlap = find_overlap(winner_microcluster, winner_neighborhood)

        if len(overlap) > 0:
            merged_cluster, deleted_clusters = merge_microclusters(winner_microcluster,
                                                                   overlap,
                                                                   self.merge_threshold)
            if merged_cluster is not None:
                micro_list.append(merged_cluster)
            if deleted_clusters is not None:
                for deleted_cluster in deleted_clusters:
                    self.class_micro_centroids = np.where(np.equal(self.class_micro_centroids, deleted_cluster.centroid),
                                                          merged_cluster.centroid,
                                                          self.class_micro_centroids)
                    micro_list.remove(deleted_cluster)
        try:
            self.class_micro_centroids = self.class_micro_centroids.tolist()
        except:
            pass
        pass

    def check_macro_overlap(self, microcluster, macro_list):
        merged_cluster = None
        deleted_idx = None
        other_micro = None

        for i,m in enumerate(macro_list):
            if (m is not macro_list[self.idx_macro]) and (i in self.idx_macro_unique):
                # verify everything microclusters
                other_micro = min_dist(microcluster.centroid, m.micros)
                overlap = dist(microcluster.centroid, other_micro.centroid) - (microcluster.radius + other_micro.radius) <= self.merge_threshold

                if overlap:
                    # merged macroclusters
                    merged_cluster = merge_macroclusters(macro_list[self.idx_macro], macro_list[i])
                    deleted_idx = i
                    break

        if merged_cluster is not None:
            macro_list[self.idx_macro].valid = False
            macro_list[self.idx_macro].micros = []
            self.idx_macro_unique.remove(macro_list.index(macro_list[self.idx_macro]))

            macro_list[deleted_idx].valid = False
            macro_list[deleted_idx].micros = []
            self.idx_macro_unique.remove(macro_list.index(macro_list[deleted_idx]))

            self.idx_macro_unique.append(len(macro_list))
            macro_list.append(merged_cluster)
            

            self.class_macro_lists = np.where(np.equal(self.class_macro_lists, deleted_idx),
                                                macro_list.index(merged_cluster),
                                                self.class_macro_lists)
            self.class_macro_lists = np.where(np.equal(self.class_macro_lists, self.idx_macro),
                                                macro_list.index(merged_cluster),
                                                self.class_macro_lists)
            try:
                self.class_macro_lists = self.class_macro_lists.tolist()
            except:
                pass

            self.idx_macro = self.idx_macro_unique[-1]
            
        return macro_list

    def check_point_cluster(self, vt, macro_list):

        micro_list = macro_list[self.idx_macro].micros

        winner_microcluster = min_dist(vt, micro_list)
        all_microclusters = self.get_all_microclusters()

        winner_neighborhood = find_neighbors(winner_microcluster,
                                             self.min_pts,
                                             micro_list,
                                             all_microclusters)

        if dist(vt, winner_microcluster.centroid) < winner_microcluster.radius:

            old_centroid_winner = winner_microcluster.centroid
            old_centroids = [c.centroid for c in winner_neighborhood]
            update_microcluster(macro_list[self.idx_macro],
                                winner_microcluster,
                                vt,
                                self.alpha,
                                winner_neighborhood)

            # update centroid
            new_centroids = [c.centroid for c in winner_neighborhood]
            self.class_micro_centroids.append(winner_microcluster.centroid)
            self.class_micro_centroids = np.where(np.equal(self.class_micro_centroids, old_centroid_winner),
                                                  np.array(
                                                      winner_microcluster.centroid),
                                                  self.class_micro_centroids)

            for i in range(0, len(old_centroids)):
                self.class_micro_centroids = np.where(self.class_micro_centroids == old_centroids[i],
                                                      new_centroids[i],
                                                      self.class_micro_centroids)
            self.check_micro_overlap(winner_microcluster,
                                     winner_neighborhood,
                                     micro_list)
            try:
                self.class_micro_centroids = self.class_micro_centroids.tolist()
            except:
                pass

        elif (dist(vt, winner_microcluster.centroid) > self.p*(winner_microcluster.radius)) and (self.idx_macro < 10) and (macro_list[self.idx_macro].number_micro_clusters >= self.min_pts):
            try:
                self.idx_macro = self.idx_macro_unique[self.idx_macro_unique.index(self.idx_macro)+1]
                self.check_point_cluster(vt, macro_list)
            except:
                # create new macrocluster
                self.idx_macro_unique.append(len(macro_list))
                self.idx_macro = self.idx_macro_unique[-1]

                macro_list.append(new_macrocluster(new_microcluster(vt)))
                self.class_micro_centroids.append(vt)
                pass
        else:
            self.class_micro_centroids.append(vt)
            update_macrocluster(
                macro_list[self.idx_macro], new_microcluster(vt))
        
        
        if len(macro_list) > 1:
            winner_microcluster = min_dist(vt, macro_list[self.idx_macro].micros)
            macro_list = self.check_macro_overlap(winner_microcluster, macro_list)
            
        return macro_list

    def process(self, vt):
        """
        Process function.

        Parameters
        ----------
        vt : numpy array, required
        """
        self.idx_macro = self.idx_macro_unique[0]
        new_macrocluster_list = self.macro_list[-1].copy()

        if (new_macrocluster_list) and (new_macrocluster_list[self.idx_macro].number_micro_clusters >= self.min_pts):
            new_macrocluster_list = self.check_point_cluster(vt, new_macrocluster_list)
        else:
            self.class_micro_centroids.append(vt)
            if new_macrocluster_list:
                update_macrocluster(
                    new_macrocluster_list[self.idx_macro], new_microcluster(vt))
            else:
                # create a new microcluster and macrocluster
                new_macrocluster_list.append(new_macrocluster(new_microcluster(vt)))

        self.class_macro_lists.append(self.idx_macro)
        self.macro_list.append(new_macrocluster_list)
    pass
