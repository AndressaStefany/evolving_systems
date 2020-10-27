import pandas as pd

# For production
from algoritms.sostream.utils import dist

# For test
# from utils import dist


def find_overlap(win, win_nn):
    overlap = []
    for microcluster in win_nn:
        if win is not microcluster:
            if dist(win.centroid, microcluster.centroid) - (win.radius + microcluster.radius) < 0:
                overlap.append(microcluster)
    return overlap
