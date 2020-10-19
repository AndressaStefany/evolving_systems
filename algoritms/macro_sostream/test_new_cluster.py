import unittest
import numpy as np
from new_cluster import new_cluster, new_macro_cluster
from micro_cluster import MicroCluster
from macro_cluster import MacroCluster


class TestCreate(unittest.TestCase):
    def test_create_micro_cluster(self):
        microcluster = new_cluster(np.array([2.2, 7]))
        expected_microcluster = MicroCluster(np.array([2.2, 7]))

        np.testing.assert_array_equal(microcluster.centroid,
                                      expected_microcluster.centroid)
        self.assertEqual(microcluster.radius, expected_microcluster.radius)
        self.assertEqual(microcluster.number_points,
                         expected_microcluster.number_points)

    def test_create_macro_cluster(self):
        micro_cluster = MicroCluster(centroid=np.array(
            [2.2, 7]), number_points=3, radius=6)

        expected_macrocluster = MacroCluster(
            np.array([2.2, 7]), 1, 6, [micro_cluster])

        macro_cluster = new_macro_cluster(micro_cluster)

        np.testing.assert_array_equal(macro_cluster.centroid,
                                      expected_macrocluster.centroid)
        self.assertEqual(macro_cluster.radius, expected_macrocluster.radius)
        self.assertEqual(macro_cluster.number_micro_clusters,
                         expected_macrocluster.number_micro_clusters)
        self.assertListEqual(macro_cluster.micros,
                             expected_macrocluster.micros)


if __name__ == '__main__':
    unittest.main()
