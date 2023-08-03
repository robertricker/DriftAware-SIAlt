import numpy as np


def get_neighbor_dyn_range(row, data, var, tree):
    # Create a KDTree for spatial indexing
    neighbors_indices = tree.query_ball_point((row['geometry'].x, row['geometry'].y), row[var+'_drift_unc'], p=2.0)
    neighbor_thickness = data.loc[neighbors_indices][var]
    return np.percentile(neighbor_thickness, 75)-np.percentile(neighbor_thickness, 25)
