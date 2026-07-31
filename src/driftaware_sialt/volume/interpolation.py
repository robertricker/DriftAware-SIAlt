"""Interpolation of missing gridded sea-ice variables."""

import numpy as np
from scipy.interpolate import RBFInterpolator


def interpolate_missing_values(
        data, target_var, ice_conc, sic_threshold=0.15):
    """Interpolate a missing variable where ice concentration is sufficient."""
    if target_var not in data:
        return data

    if isinstance(ice_conc, dict):
        ice_conc_arr = ice_conc.get("ice_conc")
    else:
        ice_conc_arr = (
            ice_conc.values
            if hasattr(ice_conc, "values")
            else np.array(ice_conc)
        )

    if ice_conc_arr.ndim == 3 and ice_conc_arr.shape[0] == 1:
        ice_conc_arr = ice_conc_arr[0]

    variable = data[target_var]

    if "time" in variable.dims and variable.sizes.get("time", 1) > 1:
        for time_index in range(variable.sizes["time"]):
            variable_at_time = variable.isel(time=time_index)
            if "xc" in variable_at_time.dims and "yc" in variable_at_time.dims:
                data[target_var].values[time_index] = _interpolate_2d_rbf(
                    variable_at_time.values,
                    ice_conc_arr,
                    variable_at_time["xc"].values,
                    variable_at_time["yc"].values,
                    sic_threshold,
                )
    elif "xc" in variable.dims and "yc" in variable.dims:
        if variable.sizes.get("time") == 1:
            values = variable.values[0]
        else:
            values = variable.values

        if values.ndim == 3 and values.shape[0] == 1:
            values = values[0]

        filled_values = _interpolate_2d_rbf(
            values,
            ice_conc_arr,
            variable["xc"].values,
            variable["yc"].values,
            sic_threshold,
        )

        if "time" in variable.dims and variable.sizes.get("time", 1) == 1:
            data[target_var].values[0] = filled_values
        else:
            data[target_var].values = filled_values

    return data


def _interpolate_2d_rbf(
        variable_values, ice_conc_arr, xc, yc, sic_threshold):
    """Apply Gaussian RBF interpolation to a two-dimensional array."""
    x_grid, y_grid = np.meshgrid(xc, yc)
    dx = np.abs(np.diff(xc))
    dy = np.abs(np.diff(yc))
    if (
        not dx.size or not dy.size
        or dx[0] == 0 or dy[0] == 0
        or not np.allclose(dx, dx[0])
        or not np.allclose(dy, dy[0])
    ):
        raise ValueError('xc and yc must have regular, non-zero spacing')
    grid_spacing = np.sqrt(dx[0] * dy[0])

    valid_mask = ~np.isnan(variable_values.squeeze())
    known_points = np.column_stack(
        (x_grid[valid_mask], y_grid[valid_mask]))
    known_values = variable_values.squeeze()[valid_mask]

    try:
        interpolator = RBFInterpolator(
            known_points,
            known_values,
            kernel="gaussian",
            epsilon=1.8 / grid_spacing,
            neighbors=20,
            smoothing=0.05,
        )
        interpolation_mask = ice_conc_arr.squeeze() > sic_threshold
        target_points = np.column_stack(
            (x_grid[interpolation_mask], y_grid[interpolation_mask]))
        interpolated_values = interpolator(target_points)

        filled = np.full_like(variable_values.squeeze(), np.nan)
        filled[interpolation_mask] = interpolated_values
        return np.expand_dims(filled, axis=0)

    except Exception as error:
        print(
            f"RBF interpolation failed: {error}. Returning original data.")
        return variable_values.copy()
