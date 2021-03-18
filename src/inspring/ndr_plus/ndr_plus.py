"""Primary script for NDR plus."""
import logging
import os
import shutil
import tempfile
import warnings

from osgeo import gdal
from osgeo import osr
import pygeoprocessing
import numpy

from inspring.ndr_plus import ndr_plus_cython

LOGGER = logging.getLogger(__name__)
NODATA = -9999
USE_AG_LOAD_ID = 999


def mult_arrays(
        target_raster_path, gdal_type, target_nodata, raster_path_list):
    """Multiply arrays and be careful of nodata values."""
    nodata_array = numpy.array([
        pygeoprocessing.get_raster_info(path)['nodata'][0]
        for path in raster_path_list])

    def _mult_arrays(*array_list):
        """Multiply arrays in array list but block out stacks with NODATA."""
        try:
            stack = numpy.stack(array_list)
            valid_mask = (numpy.bitwise_and.reduce(
                [~numpy.isclose(nodata, array)
                 for nodata, array in zip(nodata_array, stack)], axis=0))
            n_valid = numpy.count_nonzero(valid_mask)
            broadcast_valid_mask = numpy.broadcast_to(valid_mask, stack.shape)
            valid_stack = stack[broadcast_valid_mask].reshape(
                len(array_list), n_valid)
            result = numpy.empty(array_list[0].shape, dtype=numpy.float64)
            result[:] = NODATA
            result[valid_mask] = numpy.prod(valid_stack, axis=0)
            return result
        except Exception:
            LOGGER.exception(
                'values: %s', array_list)
            raise

    pygeoprocessing.raster_calculator(
        [(path, 1) for path in raster_path_list], _mult_arrays,
        target_raster_path, gdal_type, target_nodata)


def calculate_ndr(downstream_ret_eff_path, ic_path, k_val, target_ndr_path):
    """Calculate NDR raster.

    Parameters:
        downstream_ret_eff_path (string): path to downstream retention
            raster.
        ic_path (string): path to IC raster
        k_val (float): value of k in Eq. 4.
        target_ndr_path (string): path to NDR raster calculated by this func.

    Returns:
        None.

    """
    # calculate ic_0
    ic_raster = gdal.OpenEx(ic_path, gdal.OF_RASTER)
    ic_min, ic_max, _, _ = ic_raster.GetRasterBand(1).GetStatistics(0, 1)
    ic_0 = (ic_max + ic_min) / 2.0

    def ndr_op(downstream_ret_eff_array, ic_array):
        """Calculate NDR from Eq. (4)."""
        with numpy.errstate(invalid='raise'):
            try:
                result = numpy.empty_like(downstream_ret_eff_array)
                result[:] = NODATA
                valid_mask = (
                    downstream_ret_eff_array != NODATA) & (
                        ic_array != NODATA)
                if numpy.count_nonzero(valid_mask) > 0:
                    result[valid_mask] = (
                        1 - downstream_ret_eff_array[valid_mask]) / (
                            1 + numpy.exp(
                                (ic_array[valid_mask] - ic_0) / k_val))
                return result
            except FloatingPointError:
                LOGGER.debug(
                    'bad values: %s %s %s', ic_array[valid_mask], ic_0,
                    ic_path)
                raise

    pygeoprocessing.raster_calculator(
        [(downstream_ret_eff_path, 1), (ic_path, 1)], ndr_op, target_ndr_path,
        gdal.GDT_Float32, NODATA)


def modified_load(
        load_raster_path, runoff_proxy_path, target_modified_load_path):
    """Calculate Modified load (eq 1).

    Args:
        load_raster_path (str): path to load raster.
        runoff_proxy_path (str): path to runoff index.
        target_modified_load_path (str): path to calculated modified load
            raster.

    Return:
        None
    """
    load_raster_info = pygeoprocessing.get_raster_info(load_raster_path)
    runoff_nodata = pygeoprocessing.get_raster_info(
        runoff_proxy_path)['nodata'][0]
    runoff_sum = 0.0
    runoff_count = 0

    for _, raster_block in pygeoprocessing.iterblocks(
            (runoff_proxy_path, 1)):
        # this complicated call ensures we don't end up with some garbage
        # precipitation value like what we're getting with
        # he26pr50.
        valid_mask = (
            ~numpy.isclose(raster_block, runoff_nodata) &
            (raster_block >= 0) &
            (raster_block < 1e7))
        runoff_sum += numpy.sum(raster_block[valid_mask])
        runoff_count += numpy.count_nonzero(raster_block)
    avg_runoff = 1.0
    if runoff_count > 0 and runoff_sum > 0:
        avg_runoff = runoff_sum / runoff_count

    load_nodata = load_raster_info['nodata'][0]
    cell_area_ha = abs(
        load_raster_info['pixel_size'][0] *
        load_raster_info['pixel_size'][1]) * 0.0001

    def _modified_load_op(load_array, runoff_array):
        """Multiply arrays and divide by average runoff."""
        result = numpy.empty(load_array.shape, dtype=numpy.float32)
        result[:] = NODATA
        valid_mask = (
            ~numpy.isclose(load_array, load_nodata) &
            ~numpy.isclose(runoff_array, runoff_nodata))
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                result[valid_mask] = (
                    cell_area_ha * load_array[valid_mask] *
                    runoff_array[valid_mask] / avg_runoff)
                if any((result[valid_mask] > 1e25) |
                       (result[valid_mask] < -1e25)):
                    raise ValueError("bad result")
            except Exception:
                LOGGER.error(
                    "warning or error in modified load %s %s %s %s %s %s",
                    avg_runoff, cell_area_ha, runoff_array[valid_mask],
                    result[valid_mask], load_raster_path, runoff_proxy_path)
        return result

    pygeoprocessing.raster_calculator(
        [(load_raster_path, 1), (runoff_proxy_path, 1)], _modified_load_op,
        target_modified_load_path, gdal.GDT_Float32, NODATA)


def calculate_ag_load(
        load_n_per_ha_raster_path, ag_load_raster_path, target_ag_load_path):
    """Add the agricultural load onto the base load.

    Args:
        load_n_per_ha_raster_path (string): path to a base load raster with
            `USE_AG_LOAD_ID` where the pixel should be replaced with the
            managed ag load.
        ag_load_raster_path (string): path to a raster that indicates
            what the ag load is at `USE_AG_LOAD_ID` pixels
        target_ag_load_path (string): generated raster that has the base
            values from `load_n_per_ha_raster_path` but with the
            USE_AG_LOAD_IDs replaced by `ag_load_raster_path`.

    Return:
        None
    """
    load_raster_info = pygeoprocessing.get_raster_info(
        ag_load_raster_path)
    load_nodata = load_raster_info['nodata'][0]

    def ag_load_op(base_load_n_array, ag_load_array):
        """Raster calculator replace USE_AG_LOAD_ID with ag loads."""
        result = numpy.copy(base_load_n_array)
        if load_nodata is not None:
            nodata_load_mask = numpy.isclose(ag_load_array, load_nodata)
        else:
            nodata_load_mask = numpy.zeros(ag_load_array.shape, dtype=bool)
        ag_mask = (base_load_n_array == USE_AG_LOAD_ID)
        result[ag_mask & ~nodata_load_mask] = (
            ag_load_array[ag_mask & ~nodata_load_mask])
        result[ag_mask & nodata_load_mask] = 0.0
        return result

    nodata = pygeoprocessing.get_raster_info(
        load_n_per_ha_raster_path)['nodata'][0]

    pygeoprocessing.raster_calculator(
        [(load_n_per_ha_raster_path, 1), (ag_load_raster_path, 1)],
        ag_load_op, target_ag_load_path,
        gdal.GDT_Float32, nodata)


def calc_ic(d_up_array, d_dn_array):
    """Calculate log_10(d_up/d_dn) unless nodata or 0."""
    result = numpy.empty_like(d_up_array)
    result[:] = NODATA
    zero_mask = (d_dn_array == 0) | (d_up_array == 0)
    valid_mask = (
        ~numpy.isclose(d_up_array, NODATA) &
        ~numpy.isclose(d_dn_array, NODATA) &
        (d_up_array > 0) & (d_dn_array > 0) &
        ~zero_mask)
    result[valid_mask] = numpy.log10(
        d_up_array[valid_mask] / d_dn_array[valid_mask])
    result[zero_mask] = 0.0
    return result


def div_arrays(num_array, denom_array):
    """Calculate num / denom except when denom = 0 or nodata."""
    result = numpy.empty_like(num_array)
    result[:] = NODATA
    valid_mask = (
        (num_array != NODATA) & (denom_array != NODATA) & (denom_array != 0))
    result[valid_mask] = num_array[valid_mask] / denom_array[valid_mask]
    return result


def _mult_by_scalar_op(array, scalar, nodata, target_nodata):
    """Multiply non-nodta values by self.scalar."""
    result = numpy.empty_like(array)
    result[:] = target_nodata
    valid_mask = array != nodata
    result[valid_mask] = array[valid_mask] * scalar
    return result


def mult_by_scalar_func(
        raster_path, scalar, target_nodata, target_path):
    """Multiply raster by scalar."""
    nodata = pygeoprocessing.get_raster_info(raster_path)['nodata'][0]
    pygeoprocessing.raster_calculator(
        [(raster_path, 1), (scalar, 'raw'), (nodata, 'raw'),
         (target_nodata, 'raw')], _mult_by_scalar_op, target_path,
        gdal.GDT_Float32, target_nodata)


def clamp_op(array, threshold_val, nodata):
    """Clamp non-nodata in array to >= threshold_val."""
    result = numpy.empty_like(array)
    result[:] = array
    threshold_mask = (array != nodata) & (array <= threshold_val)
    result[threshold_mask] = threshold_val
    return result


def clamp_func(raster_path, threshold_val, target_path):
    """Clamp values that exeed ``threshold val`` in ``raster_path```."""
    nodata = pygeoprocessing.get_raster_info(raster_path)['nodata'][0]
    pygeoprocessing.raster_calculator(
        [(raster_path, 1), (threshold_val, 'raw'), (nodata, 'raw')],
        clamp_op, target_path, gdal.GDT_Float32, nodata)


def _d_up_op(
        slope_accum_array, flow_accmulation_array, pixel_area,
        flow_accum_nodata):
    """Mult average upslope by sqrt of upslope area."""
    result = numpy.empty_like(slope_accum_array)
    result[:] = NODATA
    valid_mask = flow_accmulation_array != flow_accum_nodata
    result[valid_mask] = (
        slope_accum_array[valid_mask] /
        flow_accmulation_array[valid_mask]) * numpy.sqrt(
            flow_accmulation_array[valid_mask] * pixel_area)
    return result


def d_up_op_func(
        pixel_area, slope_accum_raster_path,
        flow_accum_raster_path, target_d_up_raster_path):
    """Calculate the DUp equation from NDR.

    Args:
        pixel_area (float): area of input raster pixel in m^2.
        slope_accum_raster_path (string): path to slope accumulation
            raster.
        flow_accum_raster_path (string): path to flow accumulation raster.
        target_d_up_raster_path (string): path to target d_up raster path
            created by a call to __call__.
    Return:
        None
    """
    flow_accum_nodata = pygeoprocessing.get_raster_info(
        flow_accum_raster_path)['nodata'][0]
    pygeoprocessing.raster_calculator(
        [(slope_accum_raster_path, 1),
         (flow_accum_raster_path, 1), (pixel_area, 'raw'),
         (flow_accum_nodata, 'raw')], _d_up_op,
        target_d_up_raster_path, gdal.GDT_Float32, NODATA)


def add_outlets_to_channel_raster(d8_flow_dir_path, channel_raster_path):
    """Add outlets to an existing channel raster path."""
    channel_info = pygeoprocessing.get_raster_info(channel_raster_path)
    workspace_dir = tempfile.mkdtemp(
        dir=os.path.dirname(channel_raster_path))
    outlet_vector_path = os.path.join(workspace_dir, 'outlets.gpkg')
    pygeoprocessing.routing.detect_outlets(
        (d8_flow_dir_path, 1), outlet_vector_path)

    inv_gt = gdal.InvGeoTransform(channel_info['geotransform'])
    outlet_vector = gdal.OpenEx(outlet_vector_path, gdal.OF_VECTOR)
    outlet_layer = outlet_vector.GetLayer()
    channel_raster = gdal.OpenEx(
        channel_raster_path, gdal.OF_RASTER | gdal.GA_Update)
    channel_band = channel_raster.GetRasterBand(1)
    LOGGER.debug(f"found {outlet_layer.GetFeatureCount()} outlets")
    for outlet_feature in outlet_layer:
        outlet_point = outlet_feature.GetGeometryRef()
        i, j = gdal.ApplyGeoTransform(
            inv_gt, outlet_point.GetX(), outlet_point.GetY())
        LOGGER.debug(f'setting channel at {i} {j}')
        channel_band.WriteArray(numpy.array([[1]]), int(i), int(j))
    channel_band = None
    channel_raster = None
    outlet_layer = None
    outlet_vector = None


def threshold_flow_accumulation(
        flow_accum_path, flow_threshold, target_channel_path):
    """Calculate channel raster by thresholding flow accumulation.

    Args:
        flow_accum_path (str): path to a single band flow accumulation raster.
        flow_threshold (float): if the value in `flow_accum_path` is less
            than or equal to this value, the pixel will be classified as a
            channel.
        target_channel_path (str): path to target raster that will contain
            pixels set to 1 if they are a channel, 0 if not, and possibly
            between 0 and 1 if a partial channel. (to be defined).

    Return:
        None
    """
    nodata = pygeoprocessing.get_raster_info(flow_accum_path)['nodata'][0]
    channel_nodata = 2

    def threshold_op(flow_val, threshold_val):
        valid_mask = ~numpy.isclose(flow_val, nodata)
        result = numpy.empty(flow_val.shape, dtype=numpy.byte)
        result[:] = channel_nodata
        result[valid_mask] = flow_val[valid_mask] >= threshold_val
        return result

    pygeoprocessing.raster_calculator(
        [(flow_accum_path, 1), (flow_threshold, 'raw')], threshold_op,
        target_channel_path, gdal.GDT_Byte, channel_nodata)


def get_utm_code(lng, lat):
    """Return the UTM zone code that contains the given coordinate.

    Args:
        lng (float): longitude coordinate.
        lat (float): latitude coordinate.

    Return:
        An EPSG code representing the UTM zone containing the given
        coordinate. This value is in a form that can be passed to
        gdal's `ImportFromEPSG`.
    """
    utm_code = int(numpy.floor((lng + 180)/6) % 60 + 1)
    lat_code = 6 if lat > 0 else 7
    epsg_code = int(f'32{lat_code:d}{utm_code:02d}')
    return epsg_code


def ndr_plus(
        watershed_path, watershed_fid,
        target_cell_length_m,
        retention_length_m,
        k_val,
        flow_threshold,
        max_pixel_fill_count,
        routing_algorithm,
        dem_path,
        lulc_path,
        precip_path,
        custom_load_path,
        eff_n_lucode_map,
        load_n_lucode_map,
        target_export_raster_path,
        target_modified_load_raster_path,
        workspace_dir):
    """Process a watershed for NDR analysis.

    Args:
        watershed_path (str): path to watershed vector
        watershed_fid (str): watershed FID to run the analysis on.
        target_cell_length_m (float): length of target cell size to process
            in meters.
        retention_length_m (float): NDR retention length in meters.
        k_val (float): K parameter in NDR calculation.
        routing_algorithm (str): one of 'D8' or 'DINF' for D8 or D-infinity
            routing.
        flow_threshold (int): D8 flow accumulation threshold to cutoff
        max_pixel_fill_count (int): maximum distance to search when filling
            pits.
        dem_path (str): path to base DEM raster.
        lulc_path (str): path to LULC raster.
        precip_path (str): path to precipitation raster.
        custom_load_path (str): path to load raster to use on given values
            TODO: say what these values are.
        eff_n_lucode_map (dict): maps lucodes to NDR efficiency values.
        load_n_lucode_map (dict): maps lucodes to NDR load values.
        workspace_dir (str): path to workspace to create working files.
        aligned_file_set (set): set of all output files generated by align
            and resize raster stack.

    Return:
        None
    """
    watershed_vector = gdal.OpenEx(watershed_path, gdal.OF_VECTOR)
    watershed_layer = watershed_vector.GetLayer()
    watershed_feature = watershed_layer.GetFeature(watershed_fid)

    os.makedirs(workspace_dir, exist_ok=True)

    watershed_geometry = watershed_feature.GetGeometryRef()
    centroid_geom = watershed_geometry.Centroid()
    utm_code = get_utm_code(centroid_geom.GetX(), centroid_geom.GetY())
    utm_srs = osr.SpatialReference()
    utm_srs.ImportFromEPSG(utm_code)

    # swizzle so it's xmin, ymin, xmax, ymax
    watershed_bb = [
        watershed_geometry.GetEnvelope()[i] for i in [0, 2, 1, 3]]

    # make sure the bounding coordinates snap to pixel grid in global coords
    dem_info = pygeoprocessing.get_raster_info(dem_path)
    base_cell_length_deg = dem_info['pixel_size'][0]
    LOGGER.debug(f'base watershed_bb: {watershed_bb}')
    watershed_bb[0] -= watershed_bb[0] % base_cell_length_deg
    watershed_bb[1] -= watershed_bb[1] % base_cell_length_deg
    watershed_bb[2] += watershed_bb[2] % base_cell_length_deg
    watershed_bb[3] += watershed_bb[3] % base_cell_length_deg

    target_bounding_box = [
        round(v) for v in pygeoprocessing.transform_bounding_box(
            watershed_bb, watershed_layer.GetSpatialRef().ExportToWkt(),
            utm_srs.ExportToWkt())]

    # make sure the bounding coordinates snap to pixel grid
    LOGGER.debug(f'base watershed_bb: {target_bounding_box}')
    target_bounding_box[0] -= target_bounding_box[0] % target_cell_length_m
    target_bounding_box[1] -= target_bounding_box[1] % target_cell_length_m
    target_bounding_box[2] += target_bounding_box[2] % target_cell_length_m
    target_bounding_box[3] += target_bounding_box[3] % target_cell_length_m

    watershed_geometry = None
    watershed_layer = None
    watershed_vector = None

    LOGGER.debug(f'base bb {watershed_bb} to target {target_bounding_box}')

    base_raster_path_list = [
        dem_path, lulc_path, precip_path, custom_load_path]
    interpolation_mode_list = ['near', 'mode', 'near', 'near']
    aligned_path_list = [
        os.path.join(workspace_dir, os.path.basename(path))
        for path in base_raster_path_list]
    (aligned_dem_path, aligned_lulc_path, aligned_precip_path,
     aligned_custom_load_path) = aligned_path_list
    LOGGER.info(f'algining raster stack of {base_raster_path_list} to cell size {target_cell_length_m} and bounding box {target_bounding_box}'
        f'\naligned_path_list: {aligned_path_list}'
        f'\nwatershed_path: {watershed_path} {watershed_fid}')
    try:
        pygeoprocessing.align_and_resize_raster_stack(
            base_raster_path_list, aligned_path_list,
            interpolation_mode_list,
            (target_cell_length_m, -target_cell_length_m),
            target_bounding_box,
            target_projection_wkt=utm_srs.ExportToWkt(),
            vector_mask_options={
                'mask_vector_path': watershed_path,
                'mask_vector_where_filter': f'"fid"={watershed_fid}'
            })
    except Exception:
        LOGGER.exception(
            f'base_raster_path_list: {base_raster_path_list}\naligned_path_list: {aligned_path_list}\nwatershed_path: {watershed_path} {watershed_fid}\ntarget bounding box: {target_bounding_box}\nbase bounding box {watershed_bb}\nprojection: {utm_srs.ExportToWkt()}')
        raise

    # fill and route dem
    filled_dem_path = os.path.join(workspace_dir, 'dem_filled.tif')
    flow_dir_path = os.path.join(workspace_dir, 'flow_dir.tif')

    pygeoprocessing.routing.fill_pits(
        (aligned_dem_path, 1), filled_dem_path,
        working_dir=workspace_dir,
        max_pixel_fill_count=max_pixel_fill_count)

    pygeoprocessing.routing.flow_dir_d8(
        (filled_dem_path, 1), flow_dir_path,
        working_dir=workspace_dir)

    # flow accum dem
    flow_accum_path = os.path.join(
        workspace_dir, 'flow_accum.tif')
    pygeoprocessing.routing.flow_accumulation_d8(
        (flow_dir_path, 1), flow_accum_path)

    # calculate slope
    slope_raster_path = os.path.join(workspace_dir, 'slope.tif')
    pygeoprocessing.calculate_slope(
        (filled_dem_path, 1), slope_raster_path)

    clamp_slope_raster_path = os.path.join(workspace_dir, 'clamp_slope.tif')
    clamp_func(slope_raster_path, 0.005, clamp_slope_raster_path)

    # calculate D_up
    slope_accum_watershed_dem_path = os.path.join(
        workspace_dir, 's_accum.tif')
    pygeoprocessing.routing.flow_accumulation_d8(
        (flow_dir_path, 1), slope_accum_watershed_dem_path,
        weight_raster_path_band=(clamp_slope_raster_path, 1))

    d_up_raster_path = os.path.join(workspace_dir, 'd_up.tif')
    d_up_op_func(
        target_cell_length_m**2, slope_accum_watershed_dem_path,
        flow_accum_path, d_up_raster_path)

    # calculate the flow channels
    channel_path = os.path.join(workspace_dir, f'channel_{flow_threshold}.tif')
    threshold_flow_accumulation(
        flow_accum_path, flow_threshold, channel_path)

    # add in outlets just in case
    LOGGER.debug('adding outlets to channel raster')
    add_outlets_to_channel_raster(flow_dir_path, channel_path)

    # calculate flow path in pixels length down to stream
    pixel_flow_length_raster_path = os.path.join(
        workspace_dir, 'pixel_flow_length.tif')
    pygeoprocessing.routing.distance_to_channel_d8(
        (flow_dir_path, 1), (channel_path, 1), pixel_flow_length_raster_path)

    # calculate real flow_path (flow length * pixel size)
    downstream_flow_distance_path = os.path.join(
        workspace_dir, 'm_flow_length.tif')
    mult_by_scalar_func(
        pixel_flow_length_raster_path, target_cell_length_m, NODATA,
        downstream_flow_distance_path)

    # calculate downstream distance / downstream slope
    d_dn_per_pixel_path = os.path.join(
        workspace_dir, 'd_dn_per_pixel.tif')
    pygeoprocessing.raster_calculator(
        [(downstream_flow_distance_path, 1), (clamp_slope_raster_path, 1)],
        div_arrays, d_dn_per_pixel_path, gdal.GDT_Float32, NODATA)

    # calculate D_dn: downstream sum of distance / downstream slope
    d_dn_raster_path = os.path.join(
        workspace_dir, 'd_dn.tif')
    pygeoprocessing.routing.distance_to_channel_d8(
        (flow_dir_path, 1), (channel_path, 1), d_dn_raster_path,
        weight_raster_path_band=(d_dn_per_pixel_path, 1))

    # calculate IC
    ic_path = os.path.join(workspace_dir, 'ic.tif')
    pygeoprocessing.raster_calculator(
        [(d_up_raster_path, 1), (d_dn_raster_path, 1)], calc_ic, ic_path,
        gdal.GDT_Float32, NODATA)

    eff_n_raster_path = os.path.join(workspace_dir, 'eff_n.tif')
    pygeoprocessing.reclassify_raster(
        (aligned_lulc_path, 1), eff_n_lucode_map,
        eff_n_raster_path, gdal.GDT_Float32, NODATA)

    load_n_per_ha_raster_path = os.path.join(
        workspace_dir, 'load_n_per_ha.tif')
    pygeoprocessing.reclassify_raster(
        (aligned_lulc_path, 1), load_n_lucode_map,
        load_n_per_ha_raster_path, gdal.GDT_Float32, NODATA)

    ag_load_per_ha_path = os.path.join(workspace_dir, 'ag_load_n_per_ha.tif')
    calculate_ag_load(
        load_n_per_ha_raster_path, aligned_custom_load_path,
        ag_load_per_ha_path)

    modified_load(
        ag_load_per_ha_path, aligned_precip_path,
        target_modified_load_raster_path)

    downstream_ret_eff_path = os.path.join(
        workspace_dir, 'downstream_ret_eff.tif')
    ndr_plus_cython.calculate_downstream_ret_eff(
        (flow_dir_path, 1), (channel_path, 1), (eff_n_raster_path, 1),
        retention_length_m, downstream_ret_eff_path,
        temp_dir_path=workspace_dir)

    # calculate NDR specific values
    ndr_path = os.path.join(workspace_dir, 'ndr.tif')
    calculate_ndr(downstream_ret_eff_path, ic_path, k_val, ndr_path)

    mult_arrays(
        target_export_raster_path, gdal.GDT_Float32, NODATA,
        [target_modified_load_raster_path, ndr_path])
