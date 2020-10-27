# For production
from algoritms.sostream.micro_cluster import MicroCluster
from algoritms.sostream.macro_cluster import MacroCluster

# For test
# from micro_cluster import MicroCluster
# from macro_cluster import MacroCluster


def new_microcluster(vt):
    return MicroCluster(vt)


def new_macrocluster(microcluster):
    centroid = microcluster.centroid
    number_micro_clusters = 1
    radius = microcluster.radius
    micros = [microcluster]

    return MacroCluster(centroid, number_micro_clusters, radius, micros)
