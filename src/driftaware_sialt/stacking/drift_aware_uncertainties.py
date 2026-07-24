import numpy as np


def get_neighbor_dyn_range(row, data, var, tree, dist_threshold):
    point = (row['geometry'].x, row['geometry'].y)
    if np.sqrt(2) * row[var + '_drift_unc'] < dist_threshold:
        return 0.0

    neighbors_indices = tree.query_ball_point(point, np.sqrt(2)*row[var+'_drift_unc'], p=2.0)

    if not neighbors_indices:
        return 0.0  # Handle the case when there are no neighbors

    neighbor_thickness = data.loc[neighbors_indices, var].values
    return np.percentile(neighbor_thickness, 75) - np.percentile(neighbor_thickness, 25)
