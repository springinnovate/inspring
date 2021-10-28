"""NDR PLUS -- single drain option, custom load map."""
import itertools
import logging
import os
import pickle

import numpy
import ecoshard.geoprocessing as geoprocessing
import ecoshard.geoprocessing.routing as routing
from osgeo import gdal, ogr
import taskgraph

from .. import utils
from . import ndr_mfd_plus_core

LOGGER = logging.getLogger(__name__)

_OUTPUT_BASE_FILES = {
    'n_export_path': 'n_export.tif',
    }

_INTERMEDIATE_BASE_FILES = {
    'ic_factor_path': 'ic_factor.tif',
    'load_n_path': 'load_n.tif',
    'modified_load_n_path': 'modified_load_n.tif',
    'ndr_n_path': 'ndr_n.tif',
    'runoff_proxy_index_path': 'runoff_proxy_index.tif',
    's_accumulation_path': 's_accumulation.tif',
    's_bar_path': 's_bar.tif',
    's_factor_inverse_path': 's_factor_inverse.tif',
    'stream_path': 'stream.tif',
    'crit_len_n_path': 'crit_len_n.tif',
    'd_dn_path': 'd_dn.tif',
    'd_up_path': 'd_up.tif',
    'eff_n_path': 'eff_n.tif',
    'effective_retention_n_path': 'effective_retention_n.tif',
    'flow_accumulation_path': 'flow_accumulation.tif',
    'flow_direction_path': 'flow_direction.tif',
    'thresholded_slope_path': 'thresholded_slope.tif',
    'dist_to_channel_path': 'dist_to_channel.tif',
    }

_CACHE_BASE_FILES = {
    'filled_dem_path': 'filled_dem.tif',
    'aligned_dem_path': 'aligned_dem.tif',
    'slope_path': 'slope.tif',
    'aligned_lulc_path': 'aligned_lulc.tif',
    'aligned_runoff_proxy_path': 'aligned_runoff_proxy.tif',
    }

_TARGET_NODATA = -1


def execute(args):
    """Nutrient Delivery Ratio.

    Args:
        args['workspace_dir'] (string):  path to current workspace
        args['dem_path'] (string): path to digital elevation map raster
        args['lulc_path'] (string): a path to landcover map raster
        args['runoff_proxy_path'] (string): a path to a runoff proxy raster
        args['watersheds_path'] (string): path to the watershed shapefile
        args['biophysical_table_path'] (string): path to csv table on disk
            containing nutrient retention values.

            Must contain the following headers:
            'load_n', 'eff_n', 'crit_len_n'
        args['biophyisical_lucode_fieldname'] (str): field in biophysical
            table that is used to reference the lucode.
        args['fertilizer_path'] (string): path to raster to use for fertlizer
            rates when biophysical table uses a 'use raster' value for the
            biophysical table field.
        args['results_suffix'] (string): (optional) a text field to append to
            all output files
        args['threshold_flow_accumulation']: a number representing the flow
            accumulation in terms of upstream pixels.
        args['k_param'] (number): The Borselli k parameter. This is a
            calibration parameter that determines the shape of the
            relationship between hydrologic connectivity.
        args['target_pixel_size'] (2-tuple): optional, requested target pixel
            size in local projection coordinate system. If not provided the
            pixel size is the smallest of all the input rasters.
        args['target_projection_wkt'] (str): optional, if provided the
            model is run in this target projection. Otherwise runs in the DEM
            projection.
        args['single_outlet'] (str): if True only one drain is modeled, either
            a large sink or the lowest pixel on the edge of the dem.

    Returns:
        None

    """
    output_dir = os.path.join(args['workspace_dir'])
    intermediate_output_dir = os.path.join(
        args['workspace_dir'], 'intermediate_outputs')
    output_dir = os.path.join(args['workspace_dir'])
    cache_dir = os.path.join(intermediate_output_dir, 'cache_dir')
    utils.make_directories([output_dir, intermediate_output_dir, cache_dir])

    task_graph = taskgraph.TaskGraph(cache_dir, -1)

    file_suffix = utils.make_suffix_string(args, 'results_suffix')
    f_reg = utils.build_file_registry(
        [(_OUTPUT_BASE_FILES, output_dir),
         (_INTERMEDIATE_BASE_FILES, intermediate_output_dir),
         (_CACHE_BASE_FILES, cache_dir)], file_suffix)

    # Build up a list of nutrients to process based on what's checked on
    nutrients_to_process = []
    for nutrient_id in ['n', 'p']:
        if args['calc_' + nutrient_id]:
            nutrients_to_process.append(nutrient_id)

    lucode_to_parameters = utils.build_lookup_from_csv(
        args['biophysical_table_path'], args['biophyisical_lucode_fieldname'])

    dem_raster_info geoprocessing.get_raster_info(args['dem_path'])
    min_pixel_size = numpy.min(numpy.abs(dem_raster_info['pixel_size']))

    if 'target_pixel_size' in args:
        target_pixel_size = args['target_pixel_size']
    else:
        target_pixel_size = (min_pixel_size, -min_pixel_size)

    if 'target_projection_wkt' in args:
        target_projection_wkt = args['target_projection_wkt']
    else:
        target_projection_wkt = dem_raster_info['projection_wkt']

    base_raster_list = [
        args['dem_path'], args['lulc_path'], args['runoff_proxy_path']]
    aligned_raster_list = [
        f_reg['aligned_dem_path'], f_reg['aligned_lulc_path'],
        f_reg['aligned_runoff_proxy_path']]

    align_raster_task = task_graph.add_task(
        func=geoprocessing.align_and_resize_raster_stack,
        args=(
            base_raster_list, aligned_raster_list,
            ['near']*len(base_raster_list), target_pixel_size,
            'intersection'),
        kwargs={
            'target_projection_wkt': target_projection_wkt,
            'base_vector_path_list': [args['watersheds_path']],
            'raster_align_index': 0,
            'vector_mask_options': {
                'mask_vector_path': args['watersheds_path']}},
        target_path_list=aligned_raster_list,
        task_name='align rasters')

    if 'single_outlet' in args and args['single_outlet'] is True:
        get_drain_sink_pixel_task = task_graph.add_task(
            func=routing.detect_lowest_drain_and_sink,
            args=((f_reg['aligned_dem_path'], 1),),
            store_result=True,
            dependent_task_list=[align_raster_task],
            task_name=f"get drain/sink pixel for {f_reg['aligned_dem_path']}")

        edge_pixel, edge_height, pit_pixel, pit_height = (
            get_drain_sink_pixel_task.get())

        if pit_height < edge_height - 20:
            # if the pit is 20 m lower than edge it's probably a big sink
            single_outlet_tuple = pit_pixel
        else:
            single_outlet_tuple = edge_pixel
    else:
        single_outlet_tuple = None

    fill_pits_task = task_graph.add_task(
        func=routing.fill_pits,
        args=(
            (f_reg['aligned_dem_path'], 1), f_reg['filled_dem_path']),
        kwargs={
            'working_dir': cache_dir,
            'single_outlet_tuple': single_outlet_tuple},
        dependent_task_list=[align_raster_task],
        target_path_list=[f_reg['filled_dem_path']],
        task_name='fill pits')

    flow_dir_task = task_graph.add_task(
        func=routing.flow_dir_mfd,
        args=(
            (f_reg['filled_dem_path'], 1), f_reg['flow_direction_path']),
        kwargs={'working_dir': cache_dir},
        dependent_task_list=[fill_pits_task],
        target_path_list=[f_reg['flow_direction_path']],
        task_name='flow dir')

    flow_accum_task = task_graph.add_task(
        func=routing.flow_accumulation_mfd,
        args=(
            (f_reg['flow_direction_path'], 1),
            f_reg['flow_accumulation_path']),
        target_path_list=[f_reg['flow_accumulation_path']],
        dependent_task_list=[flow_dir_task],
        task_name='flow accum')

    stream_extraction_task = task_graph.add_task(
        func=routing.extract_streams_mfd,
        args=(
            (f_reg['flow_accumulation_path'], 1),
            (f_reg['flow_direction_path'], 1),
            float(args['threshold_flow_accumulation']), f_reg['stream_path']),
        target_path_list=[f_reg['stream_path']],
        dependent_task_list=[flow_accum_task],
        task_name='stream extraction')

    calculate_slope_task = task_graph.add_task(
        func=geoprocessing.calculate_slope,
        args=((f_reg['filled_dem_path'], 1), f_reg['slope_path']),
        target_path_list=[f_reg['slope_path']],
        dependent_task_list=[stream_extraction_task],
        task_name='calculate slope')

    threshold_slope_task = task_graph.add_task(
        func=_slope_proportion_and_threshold,
        args=(f_reg['slope_path'], f_reg['thresholded_slope_path']),
        target_path_list=[f_reg['thresholded_slope_path']],
        dependent_task_list=[calculate_slope_task],
        task_name='threshold slope')

    runoff_proxy_index_task = task_graph.add_task(
        func=_normalize_raster,
        args=((f_reg['aligned_runoff_proxy_path'], 1),
              f_reg['runoff_proxy_index_path']),
        target_path_list=[f_reg['runoff_proxy_index_path']],
        dependent_task_list=[align_raster_task],
        task_name='runoff proxy mean')

    s_task = task_graph.add_task(
        func=routing.flow_accumulation_mfd,
        args=((f_reg['flow_direction_path'], 1), f_reg['s_accumulation_path']),
        kwargs={
            'weight_raster_path_band': (f_reg['thresholded_slope_path'], 1)},
        target_path_list=[f_reg['s_accumulation_path']],
        dependent_task_list=[flow_dir_task, threshold_slope_task],
        task_name='route s')

    s_bar_task = task_graph.add_task(
        func=s_bar_calculate,
        args=(f_reg['s_accumulation_path'], f_reg['flow_accumulation_path'],
              f_reg['s_bar_path']),
        target_path_list=[f_reg['s_bar_path']],
        dependent_task_list=[s_task, flow_accum_task],
        task_name='calculate s bar')

    d_up_task = task_graph.add_task(
        func=d_up_calculation,
        args=(f_reg['s_bar_path'], f_reg['flow_accumulation_path'],
              f_reg['d_up_path']),
        target_path_list=[f_reg['d_up_path']],
        dependent_task_list=[s_bar_task, flow_accum_task],
        task_name='d up')

    s_inv_task = task_graph.add_task(
        func=invert_raster_values,
        args=(f_reg['thresholded_slope_path'], f_reg['s_factor_inverse_path']),
        target_path_list=[f_reg['s_factor_inverse_path']],
        dependent_task_list=[threshold_slope_task],
        task_name='s inv')

    d_dn_task = task_graph.add_task(
        func=routing.distance_to_channel_mfd,
        args=(
            (f_reg['flow_direction_path'], 1), (f_reg['stream_path'], 1),
            f_reg['d_dn_path']),
        kwargs={'weight_raster_path_band': (
            f_reg['s_factor_inverse_path'], 1)},
        dependent_task_list=[stream_extraction_task, s_inv_task],
        target_path_list=[f_reg['d_dn_path']],
        task_name='d dn')

    dist_to_channel_task = task_graph.add_task(
        func=routing.distance_to_channel_mfd,
        args=(
            (f_reg['flow_direction_path'], 1), (f_reg['stream_path'], 1),
            f_reg['dist_to_channel_path']),
        dependent_task_list=[stream_extraction_task],
        target_path_list=[f_reg['dist_to_channel_path']],
        task_name='dist to channel')

    ic_task = task_graph.add_task(
        func=calculate_ic,
        args=(
            f_reg['d_up_path'], f_reg['d_dn_path'], f_reg['ic_factor_path']),
        target_path_list=[f_reg['ic_factor_path']],
        dependent_task_list=[d_dn_task, d_up_task],
        task_name='calc ic')

    load_path = f_reg['load_n_path']
    modified_load_path = f_reg['modified_load_n_path']
    # Perrine says that 'n' is the only case where we could consider a
    # prop subsurface component.  So there's a special case for that.
    load_task = task_graph.add_task(
        func=_calculate_load,
        args=(
            f_reg['aligned_lulc_path'],
            f_reg['aligned_fertilizer_path'],
            lucode_to_parameters,
            'load_%s' % nutrient, load_path),
        dependent_task_list=[align_raster_task],
        target_path_list=[load_path],
        task_name='%s load' % nutrient)

    modified_load_task = task_graph.add_task(
        func=_multiply_rasters,
        args=([load_path, f_reg['runoff_proxy_index_path']],
              _TARGET_NODATA, modified_load_path),
        target_path_list=[modified_load_path],
        dependent_task_list=[load_task, runoff_proxy_index_task],
        task_name='modified load %s' % nutrient)

    eff_path = f_reg['eff_%s_path' % nutrient]
    eff_task = task_graph.add_task(
        func=_map_lulc_to_val_mask_stream,
        args=(
            f_reg['aligned_lulc_path'], f_reg['stream_path'],
            lucode_to_parameters, 'eff_%s' % nutrient, eff_path),
        target_path_list=[eff_path],
        dependent_task_list=[align_raster_task, stream_extraction_task],
        task_name='ret eff %s' % nutrient)

    crit_len_path = f_reg['crit_len_%s_path' % nutrient]
    crit_len_task = task_graph.add_task(
        func=_map_lulc_to_val_mask_stream,
        args=(
            f_reg['aligned_lulc_path'], f_reg['stream_path'],
            lucode_to_parameters, 'crit_len_%s' % nutrient, crit_len_path),
        target_path_list=[crit_len_path],
        dependent_task_list=[align_raster_task, stream_extraction_task],
        task_name='ret eff %s' % nutrient)

    effective_retention_path = (
        f_reg['effective_retention_%s_path' % nutrient])
    ndr_eff_task = task_graph.add_task(
        func=ndr_mfd_plus_core.ndr_eff_calculation,
        args=(
            f_reg['flow_direction_path'], f_reg['stream_path'], eff_path,
            crit_len_path, effective_retention_path),
        target_path_list=[effective_retention_path],
        dependent_task_list=[
            stream_extraction_task, eff_task, crit_len_task],
        task_name='eff ret %s' % nutrient)

    ndr_path = f_reg['ndr_%s_path' % nutrient]
    ndr_task = task_graph.add_task(
        func=_calculate_ndr,
        args=(
            effective_retention_path, f_reg['ic_factor_path'],
            float(args['k_param']), ndr_path),
        target_path_list=[ndr_path],
        dependent_task_list=[ndr_eff_task, ic_task],
        task_name='calc ndr %s' % nutrient)

    export_path = f_reg['%s_export_path' % nutrient]
    calculate_export_task = task_graph.add_task(
        func=_calculate_export,
        args=(
            modified_load_path, ndr_path, export_path),
        target_path_list=[export_path],
        dependent_task_list=[load_task, ndr_task],
        task_name='export %s' % nutrient)

    task_graph.close()
    task_graph.join()

    LOGGER.info(r'NDR complete!')


def _slope_proportion_and_threshold(slope_path, target_threshold_slope_path):
    """Rescale slope to proportion and threshold to between 0.005 and 1.0.

    Args:
        slope_path (string): a raster with slope values in percent.
        target_threshold_slope_path (string): generated raster with slope
            values as a proportion (100% is 1.0) and thresholded to values
            between 0.005 and 1.0.

    Returns:
        None.

    """
    slope_nodata = geoprocessing.get_raster_info(slope_path)['nodata'][0]

    def _slope_proportion_and_threshold_op(slope):
        """Rescale and threshold slope between 0.005 and 1.0."""
        valid_mask = slope != slope_nodata
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = slope_nodata
        slope_fraction = slope[valid_mask] / 100
        slope_fraction[slope_fraction < 0.005] = 0.005
        slope_fraction[slope_fraction > 1.0] = 1.0
        result[valid_mask] = slope_fraction
        return result

    geoprocessing.raster_calculator(
        [(slope_path, 1)], _slope_proportion_and_threshold_op,
        target_threshold_slope_path, gdal.GDT_Float32, slope_nodata)


def _add_fields_to_shapefile(
        field_pickle_map, field_header_order, target_vector_path):
    """Add fields and values to an OGR layer open for writing.

    Args:
        field_pickle_map (dict): maps field name to a pickle file that is a
            result of geoprocessing.zonal_stats with FIDs that match
            `target_vector_path`.
        field_header_order (list of string): a list of field headers in the
            order to appear in the output table.
        target_vector_path (string): path to target vector file.

    Returns:
        None.

    """
    target_vector = gdal.OpenEx(
        target_vector_path, gdal.OF_VECTOR | gdal.GA_Update)
    target_layer = target_vector.GetLayer()
    field_summaries = {}
    for field_name in field_header_order:
        field_def = ogr.FieldDefn(field_name, ogr.OFTReal)
        field_def.SetWidth(24)
        field_def.SetPrecision(11)
        target_layer.CreateField(field_def)
        with open(field_pickle_map[field_name], 'rb') as pickle_file:
            field_summaries[field_name] = pickle.load(pickle_file)

    for feature in target_layer:
        fid = feature.GetFID()
        for field_name in field_header_order:
            feature.SetField(
                field_name, float(field_summaries[field_name][fid]['sum']))
        # Save back to datasource
        target_layer.SetFeature(feature)
    target_layer = None
    target_vector = None


def _normalize_raster(base_raster_path_band, target_normalized_raster_path):
    """Calculate normalize raster by dividing by the mean value.

    Args:
        base_raster_path_band (tuple): raster path/band tuple to calculate
            mean.
        target_normalized_raster_path (string): path to target normalized
            raster from base_raster_path_band.

    Returns:
        None.

    """
    value_sum = 0.0
    value_count = 0.0
    base_nodata = geoprocessing.get_raster_info(
        base_raster_path_band[0])['nodata'][base_raster_path_band[1]-1]
    for _, raster_block in geoprocessing.iterblocks(
            base_raster_path_band):
        valid_mask = slice(None)
        if base_nodata is not None:
            valid_mask = ~numpy.isclose(raster_block, base_nodata)

        valid_block = raster_block[valid_mask]
        value_sum += numpy.sum(valid_block)
        value_count += valid_block.size

    value_mean = value_sum
    if value_count > 0.0:
        value_mean /= value_count

    def _normalize_raster_op(array):
        """Divide values by mean."""
        result = numpy.empty(array.shape, dtype=numpy.float32)
        result[:] = numpy.float32(base_nodata)

        valid_mask = slice(None)
        if base_nodata is not None:
            valid_mask = ~numpy.isclose(array, base_nodata)
        result[valid_mask] = array[valid_mask]
        if value_mean != 0:
            result[valid_mask] /= value_mean
        return result

    # It's possible for base_nodata to extend outside what can be represented
    # in a float32, yet GDAL expects a python float.  Casting to numpy.float32
    # and back to a python float allows for the nodata value to reflect the
    # actual nodata pixel values.
    target_nodata = float(numpy.float32(base_nodata))
    geoprocessing.raster_calculator(
        [base_raster_path_band], _normalize_raster_op,
        target_normalized_raster_path, gdal.GDT_Float32,
        target_nodata)


def _calculate_load(
        lulc_raster_path, fertilizer_path, lucode_to_parameters,
        load_type, target_load_raster):
    """Calculate load raster by mapping landcover and multiplying by area.

    Args:
        lulc_raster_path (string): path to integer landcover raster.
        fertilizer_path (str): path to fertilizer path.
        lucode_to_parameters (dict): a mapping of landcover IDs to a
            dictionary indexed by the value of `load_n` that
            represents a per-area nutrient load. If contains a non integer
            field use value from `fertilizer_path` instead.
        load_type (string): represent nutrient to map, either 'load_n' or
            'load_p'.
        target_load_raster (string): path to target raster that will have
            total load per pixel.

    Returns:
        None.

    """
    lulc_raster_info = geoprocessing.get_raster_info(lulc_raster_path)
    nodata_landuse = lulc_raster_info['nodata'][0]
    cell_area_ha = abs(numpy.prod(lulc_raster_info['pixel_size'])) * 0.0001

    # TODO: add fert replacement here

    def _map_load_op(lucode_array):
        """Convert unit load to total load & handle nodata."""
        result = numpy.empty(lucode_array.shape)
        result[:] = _TARGET_NODATA
        for lucode in numpy.unique(lucode_array):
            if lucode != nodata_landuse:
                try:
                    result[lucode_array == lucode] = (
                        lucode_to_parameters[lucode][load_type] *
                        cell_area_ha)
                except KeyError:
                    raise KeyError(
                        'lucode: %d is present in the landuse raster but '
                        'missing from the biophysical table' % lucode)
        return result

    geoprocessing.raster_calculator(
        [(lulc_raster_path, 1)], _map_load_op, target_load_raster,
        gdal.GDT_Float32, _TARGET_NODATA)


def _multiply_rasters(raster_path_list, target_nodata, target_result_path):
    """Multiply the rasters in `raster_path_list`.

    Args:
        raster_path_list (list): list of single band raster paths.
        target_nodata (float): desired target nodata value.
        target_result_path (string): path to float 32 target raster
            multiplied where all rasters are not nodata.

    Returns:
        None.

    """
    def _mult_op(*array_nodata_list):
        """Multiply non-nodata stacks."""
        result = numpy.empty(array_nodata_list[0].shape)
        result[:] = target_nodata
        valid_mask = numpy.full(result.shape, True)
        for array, nodata in zip(*[iter(array_nodata_list)]*2):
            if nodata is not None:
                valid_mask &= ~numpy.isclose(array, nodata)
        result[valid_mask] = array_nodata_list[0][valid_mask]
        for array in array_nodata_list[2::2]:
            result[valid_mask] *= array[valid_mask]
        return result

    # make a list of (raster_path_band, nodata) tuples, then flatten it
    path_nodata_list = list(itertools.chain(*[
        ((path, 1),
         (geoprocessing.get_raster_info(path)['nodata'][0], 'raw'))
        for path in raster_path_list]))
    geoprocessing.raster_calculator(
        path_nodata_list, _mult_op, target_result_path,
        gdal.GDT_Float32, target_nodata)


def _map_lulc_to_val_mask_stream(
        lulc_raster_path, stream_path, lucode_to_parameters, map_id,
        target_eff_path):
    """Make retention efficiency raster from landcover.

    Args:
        lulc_raster_path (string): path to landcover raster.
        stream_path (string) path to stream layer 0, no stream 1 stream.
        lucode_to_parameters (dict) mapping of landcover code to a dictionary
            that contains the key in `map_id`
        map_id (string): the id in the lookup table with values to map
            landcover to efficiency.
        target_eff_path (string): target raster that contains the mapping of
            landcover codes to retention efficiency values except where there
            is a stream in which case the retention efficiency is 0.

    Returns:
        None.

    """
    keys = sorted(numpy.array(list(lucode_to_parameters)))
    values = numpy.array(
        [lucode_to_parameters[x][map_id] for x in keys])

    nodata_landuse = geoprocessing.get_raster_info(
        lulc_raster_path)['nodata'][0]
    nodata_stream = geoprocessing.get_raster_info(stream_path)['nodata'][0]

    def _map_eff_op(lucode_array, stream_array):
        """Map efficiency from LULC and handle nodata/streams."""
        valid_mask = (
            (lucode_array != nodata_landuse) &
            (stream_array != nodata_stream))
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        index = numpy.digitize(
            lucode_array[valid_mask].ravel(), keys, right=True)
        result[valid_mask] = (
            values[index] * (1 - stream_array[valid_mask]))
        return result

    geoprocessing.raster_calculator(
        ((lulc_raster_path, 1), (stream_path, 1)), _map_eff_op,
        target_eff_path, gdal.GDT_Float32, _TARGET_NODATA)


def s_bar_calculate(
        s_accumulation_path, flow_accumulation_path, target_s_bar_path):
    """Calculate bar op which is s/flow."""
    s_nodata = geoprocessing.get_raster_info(
        s_accumulation_path)['nodata'][0]
    flow_nodata = geoprocessing.get_raster_info(
        flow_accumulation_path)['nodata'][0]

    def _bar_op(s_accumulation, flow_accumulation):
        """Calculate bar operation of s_accum / flow_accum."""
        valid_mask = (
            (s_accumulation != s_nodata) &
            (flow_accumulation != flow_nodata))
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        result[valid_mask] = (
            s_accumulation[valid_mask] / flow_accumulation[valid_mask])
        return result

    geoprocessing.raster_calculator(
        ((s_accumulation_path, 1), (flow_accumulation_path, 1)), _bar_op,
        target_s_bar_path, gdal.GDT_Float32, _TARGET_NODATA)


def d_up_calculation(s_bar_path, flow_accum_path, target_d_up_path):
    """Calculate d_up = s_bar * sqrt(upstream area)."""
    s_bar_info = geoprocessing.get_raster_info(s_bar_path)
    s_bar_nodata = s_bar_info['nodata'][0]
    flow_accum_nodata = geoprocessing.get_raster_info(
        flow_accum_path)['nodata'][0]
    cell_area_m2 = abs(numpy.prod(s_bar_info['pixel_size']))

    def _d_up_op(s_bar, flow_accumulation):
        """Calculate d_up index."""
        valid_mask = (
            (s_bar != s_bar_nodata) &
            (flow_accumulation != flow_accum_nodata))
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        result[valid_mask] = (
            s_bar[valid_mask] * numpy.sqrt(
                flow_accumulation[valid_mask] * cell_area_m2))
        return result

    geoprocessing.raster_calculator(
        [(s_bar_path, 1), (flow_accum_path, 1)], _d_up_op,
        target_d_up_path, gdal.GDT_Float32, _TARGET_NODATA)


def invert_raster_values(base_raster_path, target_raster_path):
    """Invert (1/x) the values in `base`.

    Args:
        base_raster_path (string): path to floating point raster.
        target_raster_path (string): path to created output raster whose
            values are 1/x of base.

    Returns:
        None.

    """
    base_nodata = geoprocessing.get_raster_info(
        base_raster_path)['nodata'][0]

    def _inverse_op(base_val):
        """Calculate inverse of S factor."""
        result = numpy.empty(base_val.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        valid_mask = slice(None)
        if base_nodata is not None:
            valid_mask = ~numpy.isclose(base_val, base_nodata)

        zero_mask = base_val == 0.0
        result[valid_mask & ~zero_mask] = (
            1.0 / base_val[valid_mask & ~zero_mask])
        result[zero_mask] = 0.0
        return result

    geoprocessing.raster_calculator(
        ((base_raster_path, 1),), _inverse_op,
        target_raster_path, gdal.GDT_Float32, _TARGET_NODATA)


def calculate_ic(d_up_path, d_dn_path, target_ic_path):
    """Calculate IC as log_10(d_up/d_dn)."""
    ic_nodata = float(numpy.finfo(numpy.float32).min)
    d_up_nodata = geoprocessing.get_raster_info(d_up_path)['nodata'][0]
    d_dn_nodata = geoprocessing.get_raster_info(d_dn_path)['nodata'][0]

    def _ic_op(d_up, d_dn):
        """Calculate IC0."""
        valid_mask = (
            (d_up != d_up_nodata) & (d_dn != d_dn_nodata) & (d_up != 0) &
            (d_dn != 0))
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = ic_nodata
        result[valid_mask] = numpy.log10(d_up[valid_mask] / d_dn[valid_mask])
        return result

    geoprocessing.raster_calculator(
        [(d_up_path, 1), (d_dn_path, 1)], _ic_op,
        target_ic_path, gdal.GDT_Float32, ic_nodata)


def _calculate_ndr(
        effective_retention_path, ic_factor_path, k_param, target_ndr_path):
    """Calculate NDR as a function of Equation 4 in the user's guide."""
    ic_factor_raster = gdal.OpenEx(ic_factor_path, gdal.OF_RASTER)
    ic_factor_band = ic_factor_raster.GetRasterBand(1)
    ic_min, ic_max, _, _ = ic_factor_band.GetStatistics(0, 1)
    ic_factor_band = None
    ic_factor_raster = None
    ic_0_param = (ic_min + ic_max) / 2.0
    effective_retention_nodata = geoprocessing.get_raster_info(
        effective_retention_path)['nodata'][0]
    ic_nodata = geoprocessing.get_raster_info(ic_factor_path)['nodata'][0]

    def _calculate_ndr_op(effective_retention_array, ic_array):
        """Calculate NDR."""
        valid_mask = (
            (effective_retention_array != effective_retention_nodata) &
            (ic_array != ic_nodata))
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        result[valid_mask] = (
            (1.0 - effective_retention_array[valid_mask]) /
            (1.0 + numpy.exp(
                (ic_0_param - ic_array[valid_mask]) / k_param)))
        return result

    geoprocessing.raster_calculator(
        [(effective_retention_path, 1), (ic_factor_path, 1)],
        _calculate_ndr_op, target_ndr_path, gdal.GDT_Float32, _TARGET_NODATA)


def _calculate_sub_ndr(
        eff_sub, crit_len_sub, dist_to_channel_path, target_sub_ndr_path):
    """Calculate subsurface: subndr = eff_sub(1-e^(-5*l/crit_len)."""
    dist_to_channel_nodata = geoprocessing.get_raster_info(
        dist_to_channel_path)['nodata'][0]

    def _sub_ndr_op(dist_to_channel_array):
        """Calculate subsurface NDR."""
        # nodata value from this ntermediate output should always be
        # defined by pygeoprocessing, not None
        valid_mask = ~numpy.isclose(
            dist_to_channel_array, dist_to_channel_nodata)
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        result[valid_mask] = 1.0 - eff_sub * (
            1-numpy.exp(-5*dist_to_channel_array[valid_mask]/crit_len_sub))
        return result

    geoprocessing.raster_calculator(
        [(dist_to_channel_path, 1)], _sub_ndr_op, target_sub_ndr_path,
        gdal.GDT_Float32, _TARGET_NODATA)


def _calculate_export(
        surface_load_path, ndr_path, target_export_path):
    """Calculate export."""
    load_nodata = geoprocessing.get_raster_info(
        surface_load_path)['nodata'][0]
    ndr_nodata = geoprocessing.get_raster_info(
        ndr_path)['nodata'][0]

    def _calculate_export_op(modified_load_array, ndr_array):
        """Combine NDR and subsurface NDR."""
        # these intermediate outputs should always have defined nodata
        # values assigned by pygeoprocessing
        valid_mask = ~(
            numpy.isclose(modified_load_array, load_nodata) |
            numpy.isclose(ndr_array, ndr_nodata))
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        result[valid_mask] = (
            modified_load_array[valid_mask] * ndr_array[valid_mask])
        return result

    geoprocessing.raster_calculator(
        [(surface_load_path, 1), (ndr_path, 1)],
        _calculate_export_op, target_export_path, gdal.GDT_Float32,
        _TARGET_NODATA)


def _aggregate_and_pickle_total(
        base_raster_path_band, aggregate_vector_path, target_pickle_path):
    """Aggregate base raster path to vector path FIDs and pickle result.

    Args:
        base_raster_path_band (tuple): raster/path band to aggregate over.
        aggregate_vector_path (string): path to vector to use geometry to
            aggregate over.
        target_pickle_path (string): path to a file that will contain the
            result of a geoprocessing.zonal_statistics call over
            base_raster_path_band from aggregate_vector_path.

    Returns:
        None.

    """
    result = geoprocessing.zonal_statistics(
        base_raster_path_band, aggregate_vector_path,
        working_dir=os.path.dirname(target_pickle_path))

    with open(target_pickle_path, 'wb') as target_pickle_file:
        pickle.dump(result, target_pickle_file)


def create_vector_copy(base_vector_path, target_vector_path):
    """Create a copy of base vector."""
    if os.path.isfile(target_vector_path):
        os.remove(target_vector_path)
    base_vector = gdal.OpenEx(base_vector_path, gdal.OF_VECTOR)
    driver = gdal.GetDriverByName('ESRI Shapefile')
    target_vector = driver.CreateCopy(
        target_vector_path, base_vector)
    target_vector = None  # seemingly uncessary but gdal seems to like it.
