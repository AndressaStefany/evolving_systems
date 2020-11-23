# For production
from algoritms.macro_sostream.utils import dist, weighted_mean
from algoritms.macro_sostream.micro_cluster import MicroCluster
from algoritms.macro_sostream.macro_cluster import MacroCluster

# For test
# from utils import dist, weighted_mean
# from micro_cluster import MicroCluster
# from macro_cluster import MacroCluster


def merge_microclusters(win_micro_cluster, overlaping_micro_clusters, merge_threshold):
    merged_cluster = None
    deleted_clusters = list()
    for micro_cluster in overlaping_micro_clusters:
        if dist(micro_cluster.centroid, win_micro_cluster.centroid) < merge_threshold:
            if len(deleted_clusters) == 0:
                deleted_clusters.append(win_micro_cluster)
                merged_cluster = MicroCluster(win_micro_cluster.centroid,
                                              number_points=win_micro_cluster.number_points,
                                              radius=win_micro_cluster.radius)
            merged_cluster = merge_micro(micro_cluster, merged_cluster)
            deleted_clusters.append(micro_cluster)
    return merged_cluster, deleted_clusters


def merge_micro(cluster_a, cluster_b):
    new_cluster_centroid = weighted_mean(
        cluster_a.centroid, cluster_b.centroid, cluster_a.number_points, cluster_b.number_points)
    new_cluster_radius = dist(
        cluster_a.centroid, cluster_b.centroid) + max(cluster_a.radius, cluster_b.radius)
    new_cluster = MicroCluster(centroid=new_cluster_centroid,
                               number_points=cluster_a.number_points + cluster_b.number_points,
                               radius=new_cluster_radius)
    return new_cluster


def merge_macroclusters(cluster_a, cluster_b):
    new_cluster_centroid = weighted_mean(cluster_a.centroid, 
                                         cluster_b.centroid, 
                                         cluster_a.number_micro_clusters, 
                                         cluster_b.number_micro_clusters)
    new_cluster_radius = dist(
        cluster_a.centroid, cluster_b.centroid) + max(cluster_a.radius, cluster_b.radius)
    
    micro_list = cluster_a.micros + cluster_b.micros

    new_cluster = MacroCluster(centroid=new_cluster_centroid,
                               number_micro_clusters=cluster_a.number_micro_clusters + cluster_b.number_micro_clusters,
                               radius=new_cluster_radius,
                               micros=micro_list)
    
    return new_cluster
