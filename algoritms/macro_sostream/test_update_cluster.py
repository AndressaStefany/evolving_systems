import unittest
import numpy as np
from update_cluster import update_macrocluster
from micro_cluster import MicroCluster
from macro_cluster import MacroCluster


class TestUpdate(unittest.TestCase):
    # def test_update_micro_cluster(self):
    # Write here the test of update micro cluster

    def test_update_macro_cluster(self):
        micro_cluster = MicroCluster(
            centroid=np.array([1.2, 3]),
            number_points=3,
            radius=1)
        macro_cluster = MacroCluster(
            centroid=micro_cluster.centroid,
            number_micro_clusters=1,
            radius=micro_cluster.radius,
            micros=[micro_cluster])

        micro_cluster_1 = MicroCluster(
            centroid=np.array([2.2, 2]),
            number_points=3,
            radius=6)

        update_macrocluster(macro_cluster, micro_cluster_1)

        expected_macrocluster_1 = MacroCluster(
            centroid=np.array([1.7, 2.5]),
            number_micro_clusters=2,
            radius=6.7071,
            micros=[micro_cluster, micro_cluster_1])

        np.testing.assert_array_equal(np.round(macro_cluster.centroid, decimals=2),
                                      expected_macrocluster_1.centroid)
        self.assertEqual(round(macro_cluster.radius, 4),
                         expected_macrocluster_1.radius)
        self.assertEqual(macro_cluster.number_micro_clusters,
                         expected_macrocluster_1.number_micro_clusters)
        np.testing.assert_array_equal(macro_cluster.micros,
                                      expected_macrocluster_1.micros)

        micro_cluster_2 = MicroCluster(
            centroid=np.array([2.5, 1.5]),
            number_points=3,
            radius=2)

        update_macrocluster(macro_cluster, micro_cluster_2)

        expected_macrocluster_2 = MacroCluster(
            centroid=np.array([1.97, 2.17]),
            number_micro_clusters=3,
            radius=6.7071,
            micros=[micro_cluster, micro_cluster_1, micro_cluster_2])

        np.testing.assert_array_equal(np.round(macro_cluster.centroid, decimals=2),
                                      expected_macrocluster_2.centroid)
        self.assertEqual(round(macro_cluster.radius, 4),
                         expected_macrocluster_2.radius)
        self.assertEqual(macro_cluster.number_micro_clusters,
                         expected_macrocluster_2.number_micro_clusters)
        np.testing.assert_array_equal(macro_cluster.micros,
                                      expected_macrocluster_2.micros)


if __name__ == '__main__':
    unittest.main()
