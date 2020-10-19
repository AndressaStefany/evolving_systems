# from algoritms.sostream.micro_cluster import MicroCluster
from micro_cluster import MicroCluster
# from algoritms.sostream.macro_cluster import MacroCluster
from macro_cluster import MacroCluster


def new_cluster(vt):  # microcluster
    return MicroCluster(vt)


def new_macro_cluster(microcluster):
    centroid = microcluster.centroid
    number_micro_clusters = 1
    radius = microcluster.radius
    micros = [microcluster]

    return MacroCluster(centroid, number_micro_clusters, radius, micros)
