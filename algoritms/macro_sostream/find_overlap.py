import pandas as pd

# For production
from algoritms.macro_sostream.utils import dist

# For test
# from utils import dist


def find_overlap(win, win_nn):
    overlap = []
    for cluster in win_nn:
        if win is not cluster:
            if dist(win.centroid, cluster.centroid) - (win.radius + cluster.radius) < 0:
                overlap.append(cluster)
    return overlap
