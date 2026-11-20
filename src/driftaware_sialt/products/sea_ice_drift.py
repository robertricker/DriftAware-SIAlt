import netCDF4
import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt, sobel
from driftaware_sialt.products.sea_ice_concentration import SeaIceConcentrationProducts
from driftaware_sialt.io_tools import transform_coords
from driftaware_sialt.products.drift_timing import (
    DRIFT_PRODUCTS, EPOCH, filename_interval, time_bounds, vector_duration)
import datetime


def _velocity_gradient(field, x, y):
    """Sobel derivatives with signed spacing and linearly extrapolated edges."""
    spacing = [np.diff(axis) for axis in (x, y)]
    if any(not len(d) or not np.isfinite(d).all() or d[0] == 0
           or not np.allclose(d, d[0]) for d in spacing):
        raise ValueError('drift derivatives require a regular grid with nonzero spacing')
    padded = np.pad(field, 1, mode='edge')
    padded[1:-1, 0] = 2 * field[:, 0] - field[:, 1]
    padded[1:-1, -1] = 2 * field[:, -1] - field[:, -2]
    padded[0] = 2 * padded[1] - padded[2]
    padded[-1] = 2 * padded[-2] - padded[-3]
    return tuple(sobel(padded, axis=axis, mode='nearest')[1:-1, 1:-1] / (8 * d[0])
                 for axis, d in zip((1, 0), spacing))


def select_drift_product(products, start, end, max_time_difference_days=7):
    """Prefer native interval coverage, then the closest interval midpoint.

    Product priority wins for covering files; temporal fallback prefers the
    closest midpoint, then product priority. Ties prefer the earlier interval.
    """
    if start >= end:
        raise ValueError('requested drift interval must be increasing')
    midpoint = start + (end - start) / 2
    limit = datetime.timedelta(days=max_time_difference_days)
    exact, nearby = [], []
    for priority, product in enumerate(products):
        product.target_files = None
        for path, (nominal_start, nominal_end) in zip(product.file_list, product.file_intervals):
            nominal_mid = nominal_start + (nominal_end - nominal_start) / 2
            if abs(nominal_mid - midpoint) > limit + (nominal_end - nominal_start):
                continue
            first, last = product.time_interval(path)
            centre = first + (last - first) / 2
            delta = abs(centre - midpoint)
            candidate = (priority, delta, centre > midpoint, centre, path, product, first, last)
            if first <= start and last >= end:
                exact.append(candidate)
            elif delta <= limit:
                nearby.append((delta, priority) + candidate[2:])
    if exact:
        priority, delta, _, centre, path, product, first, last = min(exact)
        fallback = False
    elif nearby:
        delta, priority, _, centre, path, product, first, last = min(nearby)
        fallback = True
    else:
        return None
    product.target_files = path
    product.target_file_date = last
    product.target_native_interval = (first, last)
    product.target_requested_interval = (start, end)
    product.target_time_offset_days = (centre - midpoint).total_seconds() / 86400.
    product.used_temporal_fallback = fallback
    return product


def _coastal_drift_taper(valid_mask, land_mask):
    """Return a linear taper from valid drift cells to adjacent land."""
    valid_mask = np.asarray(valid_mask, dtype=bool)
    land_mask = np.asarray(land_mask, dtype=bool)
    taper = np.ones(valid_mask.shape, dtype=float)
    taper[land_mask] = 0.0

    if not valid_mask.any() or not land_mask.any():
        return taper

    distance_to_valid = distance_transform_edt(~valid_mask)
    distance_to_land = distance_transform_edt(~land_mask)
    invalid_ocean = ~valid_mask & ~land_mask
    distance_sum = distance_to_valid + distance_to_land
    taper[invalid_ocean] = (
        distance_to_land[invalid_ocean]
        / distance_sum[invalid_ocean])
    return taper


class SeaIceDriftProducts(SeaIceConcentrationProducts):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.ice_drift = None

        self.function_map = {
            'osi405': self.get_ice_drift,
            'osi455': self.get_ice_drift,
            'osi435': self.get_ice_drift
        }

        self.config = {name: {
            'hem_nh': '_nh_', 'hem_sh': '_sh_', 'date_str': '{12}',
            'date_pt': '%Y%m%d%H%M', 'date_offset': datetime.timedelta(0),
        } for name in DRIFT_PRODUCTS}
        self._interval_cache = {}

    def get_file_dates(self):
        self.file_intervals = [filename_interval(path, self.product_id) for path in self.file_list]
        self.file_dates = [end for _, end in self.file_intervals]

    def time_interval(self, path):
        if path not in self._interval_cache:
            with netCDF4.Dataset(path) as data:
                bounds = time_bounds(data)
            self._interval_cache[path] = tuple(EPOCH + datetime.timedelta(seconds=float(t)) for t in bounds)
        return self._interval_cache[path]

    def get_target_files(self, t0, t1, max_time_difference_days=7):
        selected = select_drift_product([self], t0, t1, max_time_difference_days)
        return selected.target_files if selected else None

    def get_ice_drift(self, target_files, ice_conc):
        with netCDF4.Dataset(target_files) as data:
            def values(name):
                return np.ma.filled(np.ma.asarray(data[name][:], dtype=float), np.nan).ravel()

            uncertainty = data.variables['uncert_dX_and_dY']
            units = getattr(uncertainty, 'units', None)
            if units not in ('km', 'm'):
                raise ValueError(f'unsupported drift uncertainty units: {units}')
            dx_dy_unc = values('uncert_dX_and_dY') * (1000. if units == 'km' else 1.)
            lon0, lat0, lon1, lat1 = [values(name) for name in ('lon', 'lat', 'lon1', 'lat1')]
            time_bnds = time_bounds(data)
            duration = np.ma.filled(np.ma.asarray(vector_duration(data, time_bnds)), np.nan)
            duration = np.broadcast_to(duration.ravel(), lon0.shape)
            status_flag = None
            if getattr(self, 'coastal_taper', False):
                status_flag = np.ma.getdata(
                    data.variables['status_flag'][0, :, :])

        valid_drift = (np.isfinite(dx_dy_unc) & (dx_dy_unc >= 0) & np.isfinite(duration)
                       & (duration > 0) & np.isfinite(lon0) & np.isfinite(lat0)
                       & np.isfinite(lon1) & np.isfinite(lat1))
        x_all, y_all = transform_coords(
            lon0, lat0, 'epsg:4326', self.out_epsg)

        x0, y0 = x_all[valid_drift], y_all[valid_drift]

        x1, y1 = transform_coords(
            lon1[valid_drift], lat1[valid_drift],
            'epsg:4326', self.out_epsg)

        # Interpolate rates, not mixed 24/48-hour displacement magnitudes.
        hours = duration[valid_drift] / 3600.
        dx = (x1 - x0) / hours
        dy = (y1 - y0) / hours
        dx_dy_unc[valid_drift] /= hours

        coords = np.transpose(np.vstack((x0, y0)))

        xc, yc = ice_conc["xc"], ice_conc["yc"]
        ice_conc = ice_conc["ice_conc"]

        if coords.shape[0] == 0:
            dx_i = np.zeros_like(xc)
            dy_i = np.zeros_like(xc)
            dx_dy_unc_i = np.zeros_like(xc)
            dx_fill = np.zeros_like(xc)
            dy_fill = np.zeros_like(xc)
            dx_dy_unc_fill = np.zeros_like(xc)
        else:
            if coords.shape[0] < 4:
                # Too few valid drift points for a Delaunay triangulation.
                # Use nearest neighbour interpolation instead of linear.
                interp_method = 'nearest'
            else:
                interp_method = 'linear'

            dx_i = griddata(coords, dx, (xc, yc), method=interp_method)
            dy_i = griddata(coords, dy, (xc, yc), method=interp_method)
            dx_dy_unc_i = griddata(
                coords, dx_dy_unc[valid_drift],
                (xc, yc), method=interp_method)

            dx_fill = griddata(coords, dx, (xc, yc), method='nearest')
            dy_fill = griddata(coords, dy, (xc, yc), method='nearest')
            dx_dy_unc_fill = griddata(
                coords, dx_dy_unc[valid_drift],
                (xc, yc), method='nearest')

        invalid = np.isnan(dx_i) | np.isnan(dy_i)
        dx_i[invalid] = dx_fill[invalid]
        dy_i[invalid] = dy_fill[invalid]
        dx_dy_unc_i[invalid] = dx_dy_unc_fill[invalid]

        if getattr(self, 'coastal_taper', False) and coords.shape[0] > 0:
            source_taper = _coastal_drift_taper(
                valid_drift.reshape(status_flag.shape),
                status_flag == 1)
            source_coords = np.column_stack((x_all, y_all))
            finite_source = np.isfinite(source_coords).all(axis=1)
            taper_i = griddata(
                source_coords[finite_source],
                source_taper.ravel()[finite_source],
                (xc, yc), method='linear')
            taper_fill = griddata(
                source_coords[finite_source],
                source_taper.ravel()[finite_source],
                (xc, yc), method='nearest')
            taper_invalid = np.isnan(taper_i)
            taper_i[taper_invalid] = taper_fill[taper_invalid]
            taper_i = np.clip(taper_i, 0.0, 1.0)
            dx_i *= taper_i
            dy_i *= taper_i

        dx_i[ice_conc == 0] = 0
        dy_i[ice_conc == 0] = 0
        dx_dy_unc_i[ice_conc == 0] = 0

        self.ice_drift = {"xc": xc, "yc": yc, "u": dx_i, "v": dy_i, "velocity_unc": dx_dy_unc_i,
                          "time_bnds": time_bnds}

    def drift_correction(self, x, y):
        """Interpolate velocity and component uncertainty in metres per hour."""
        xc, yc = self.ice_drift["xc"][0, :], self.ice_drift["yc"][:, 0]
        dx, dy = self.ice_drift['u'], self.ice_drift['v']
        dx_dy_unc = self.ice_drift['velocity_unc']
        # Check if xc and yc are in descending order
        if xc[0] > xc[-1]:
            xc = xc[::-1]
            dx = dx[:, ::-1]
            dy = dy[:, ::-1]
            dx_dy_unc = dx_dy_unc[:, ::-1]
        if yc[0] > yc[-1]:
            yc = yc[::-1]
            dx = dx[::-1, :]
            dy = dy[::-1, :]
            dx_dy_unc = dx_dy_unc[::-1, :]

        f_dx = RegularGridInterpolator((xc, yc), dx.T, method='linear')
        f_dy = RegularGridInterpolator((xc, yc), dy.T, method='linear')
        f_dx_dy_unc = RegularGridInterpolator((xc, yc), dx_dy_unc.T, method='linear')

        dx_proj = f_dx((x, y))
        dy_proj = f_dy((x, y))
        dx_dy_unc_proj = f_dx_dy_unc((x, y))

        # output in m/h
        return dx_proj.flatten(), dy_proj.flatten(), dx_dy_unc_proj

    def deformation(self, x, y):
        """
        This function compute the shear and the ice deformation from the drift derivative, 
        the derivatives are computed using a sobel filter see :
        https://tc.copernicus.org/articles/15/2167/2021/tc-15-2167-2021.pdf
        """
        xc, yc = self.ice_drift["xc"][0, :], self.ice_drift["yc"][:, 0]
        u, v = self.ice_drift['u'] / 3600., self.ice_drift['v'] / 3600.
        dudx, dudy = _velocity_gradient(u, xc, yc)
        dvdx, dvdy = _velocity_gradient(v, xc, yc)

        div = (dudx + dvdy) * 86400.0  # 1/day
        she = np.sqrt((dudx - dvdy)**2 + (dudy + dvdx)**2) * 86400.0  # 1/day

        # Check if xc and yc are in descending order
        if xc[0] > xc[-1]:
            xc = xc[::-1]
            div = div[:, ::-1]
            she = she[:, ::-1]
        if yc[0] > yc[-1]:
            yc = yc[::-1]
            div = div[::-1, :]
            she = she[::-1, :]

        f_div = RegularGridInterpolator((xc, yc), div.T, method='linear')
        f_she = RegularGridInterpolator((xc, yc), she.T, method='linear')

        div_proj = f_div((x, y))
        she_proj = f_she((x, y))

        return np.around(div_proj.flatten(), 4), np.around(she_proj.flatten(), 4)
