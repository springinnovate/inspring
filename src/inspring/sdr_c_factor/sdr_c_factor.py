"""InVEST Sediment Delivery Ratio (SDR) module.

The SDR method in this model is based on:
    Winchell, M. F., et al. "Extension and validation of a geographic
    information system-based method for calculating the Revised Universal
    Soil Loss Equation length-slope factor for erosion risk assessments in
    large watersheds." Journal of Soil and Water Conservation 63.3 (2008):
    105-111.
"""
import os
import logging

from osgeo import gdal
import numpy

from ecoshard import taskgraph
import ecoshard.geoprocessing as geoprocessing
import ecoshard.geoprocessing.routing as routing
from .. import utils
from . import sdr_c_factor_core

LOGGER = logging.getLogger(__name__)

_DEFAULT_L_CAP = 122

_OUTPUT_BASE_FILES = {
    'rkls_path': 'rkls.tif',
    'sed_export_path': 'sed_export.tif',
    'sed_retention_index_path': 'sed_retention_index.tif',
    'sed_retention_path': 'sed_retention.tif',
    'sed_deposition_path': 'sed_deposition.tif',
    'stream_and_drainage_path': 'stream_and_drainage.tif',
    'stream_path': 'stream.tif',
    'usle_path': 'usle.tif',
    }

_INTERMEDIATE_BASE_FILES = {
    'cp_factor_path': 'cp.tif',
    'd_dn_bare_soil_path': 'd_dn_bare_soil.tif',
    'd_dn_path': 'd_dn.tif',
    'd_up_bare_soil_path': 'd_up_bare_soil.tif',
    'd_up_path': 'd_up.tif',
    'dem_offset_path': 'dem_offset.tif',
    'f_path': 'f.tif',
    'flow_accumulation_path': 'flow_accumulation.tif',
    'flow_direction_path': 'flow_direction.tif',
    'ic_bare_soil_path': 'ic_bare_soil.tif',
    'ic_path': 'ic.tif',
    'ls_path': 'ls.tif',
    'pit_filled_dem_path': 'pit_filled_dem.tif',
    's_accumulation_path': 's_accumulation.tif',
    's_bar_path': 's_bar.tif',
    's_inverse_path': 's_inverse.tif',
    'sdr_bare_soil_path': 'sdr_bare_soil.tif',
    'sdr_path': 'sdr_factor.tif',
    'slope_path': 'slope.tif',
    'thresholded_slope_path': 'slope_threshold.tif',
    'thresholded_c_path': 'c_threshold.tif',
    'c_accumulation_path': 'c_accumulation.tif',
    'c_bar_path': 'c_bar.tif',
    'c_path': 'w.tif',
    'cs_inverse_path': 'ws_inverse.tif',
    'e_prime_path': 'e_prime.tif',
    }

_TMP_BASE_FILES = {
    'aligned_dem_path': 'aligned_dem.tif',
    'aligned_drainage_path': 'aligned_drainage.tif',
    'aligned_erodibility_path': 'aligned_erodibility.tif',
    'aligned_erosivity_path': 'aligned_erosivity.tif',
    'aligned_lulc_path': 'aligned_lulc.tif',
    'aligned_c_factor_path': 'aligned_c_factor.tif',
    'usle_c_path': 'usle_c.tif',
    'usle_p_path': 'usle_p.tif',
    }

# Target nodata is for general rasters that are positive, and _IC_NODATA are
# for rasters that are any range
_TARGET_NODATA = -1.0
_IC_NODATA = float(numpy.finfo('float32').min)

# This dictionary translates headers in the biophysical table to potential
# hard-coded raster paths that can be passed in
_BIOPHYSICAL_TABLE_FIELDS_PATH_MAP = {
    'usle_c': 'usle_c_path',
    'usle_p': 'usle_p_path',
    }


def _reclassify_or_clip(
        key_field, biophysical_table_path, lulc_raster_path, args,
        target_raster_info, f_reg):
    """Either reclassify lulc with key_field, or reference base input.

    Args:
        key_field (str): a field that is either a column in
            `biophysical_table_path`, or the prefix to `{key_field}_path` in
            `args` dictionary.
        base_raster_path (str): path to a raster to use to frame how large
            a potential target raster should be
        aoi_path (str): if not None, a vector to further limit target bounding
            box size
        biophysical_table_path (str): if not None, lookup table containing
            reference to `key_field` and lulc_field for reclassification.
        lulc_field (str): column in `biophysical_table_path` to use as a
            lookup from the `lulc_raster_path` values to `key_field` values
        lulc_raster_path (str): if not None, reference to raster that is used
            to reclassify from
        args (dict): base model argument dictionary, used to look up _path
        target_raster_info (dict): dictionary to use for desired target
            projection, bounding box, pixel size, etc, in case raster must
            be warped.
        f_reg (dict): file registry, index into to find the desired target
            path for a file

    Returns:
        path to raster to use for biophysical component.
    """
    key_path = f'{key_field}_path'
    if key_path in args and args[key_path] is not None:
        geoprocessing.warp_raster(
            args[key_path], target_raster_info['pixel_size'], f_reg[key_path],
            'bilinear', target_bb=target_raster_info['bounding_box'],
            target_projection_wkt=target_raster_info['projection_wkt'],
            working_dir=os.path.dirname(f_reg[key_path]))
        return f_reg[key_path]

    if args['biophysical_table_path'] is None:
        raise ValueError(
            f'Neither {key_field} or "biophysical_table_path" were defined in '
            f'args, one must be defined. value of args: {args}')

    lufield_id = args.get('biophysical_table_lucode_field', 'lucode')
    biophysical_table = utils.build_lookup_from_csv(
        args['biophysical_table_path'], lufield_id)
    LOGGER.warn(f'****************************** {biophysical_table}')

    lulc_to_val = dict(
        [(lulc_code, float(table[key_field])) for
         (lulc_code, table) in biophysical_table.items()])

    geoprocessing.reclassify_raster(
        (lulc_raster_path, 1), lulc_to_val,
        f_reg[_BIOPHYSICAL_TABLE_FIELDS_PATH_MAP[key_field]], gdal.GDT_Float32,
        _TARGET_NODATA)
    return f_reg[_BIOPHYSICAL_TABLE_FIELDS_PATH_MAP[key_field]]


def execute(args):
    """Sediment Delivery Ratio.

    This function calculates the sediment export and retention of a landscape
    using the sediment delivery ratio model described in the InVEST user's
    guide.

    Parameters:
        args['workspace_dir'] (string): output directory for intermediate,
            temporary, and final files
        args['results_suffix'] (string): (optional) string to append to any
            output file names
        args['dem_path'] (string): path to a digital elevation raster
        args['erosivity_path'] (string): path to rainfall erosivity index
            raster
        args['erodibility_path'] (string): a path to soil erodibility raster
        args['lulc_path'] (string): path to land use/land cover raster
        args['watersheds_path'] (string): path to vector of the watersheds
        args['biophysical_table_path'] (string): path to CSV file with
            biophysical information of each land use classes.  contain the
            fields 'usle_c' and 'usle_p'
        args['usle_c_path'], args['usle_p_path'] (string): if either of these
            are passed in they are used instead of their equivalent field in
            the biophysical table. If BOTH are passed in then 'lulc_path' and
            args['biophysical_table_path'] are not required.
        args['threshold_flow_accumulation'] (number): number of upstream pixels
            on the dem to threshold to a stream.
        args['k_param'] (number): k calibration parameter
        args['sdr_max'] (number): max value the SDR
        args['ic_0_param'] (number): ic_0 calibration parameter
        args['drainage_path'] (string): (optional) path to drainage raster that
            is used to add additional drainage areas to the internally
            calculated stream layer
        args['biophysical_table_lucode_field'] (str): optional, if exists
            use this instead of 'lucode'.
        args['c_factor_path'] (str): optional, if present this is a
            raster with values ranging 0..1 which represents the C factor to
            be used in the USLE calculation. The presence of this raster will
            override any C values in the biophysical table.
        args['l_cap'] (float): optional, if present sets the upstream flow
            length cap (square of the upstream area) to this value, otherwise
            default is 122.
        args['target_pixel_size'] (2-tuple): optional, requested target pixel
            size in local projection coordinate system. If not provided the
            pixel size is the smallest of all the input rasters.
        args['target_projection_wkt'] (str): optional, if provided the
            model is run in this target projection. Otherwise runs in the DEM
            projection.
        args['single_outlet'] (str): if True only one drain is modeled, either
            a large sink or the lowest pixel on the edge of the dem.
        args['prealigned'] (bool): if true, input rasters are already aligned
            and projected.
        args['reuse_dem'] (bool): if true, attempts to reuse a DEM from a
            previous run if it exists based off its filename.

    Returns:
        None.

    """
    file_suffix = utils.make_suffix_string(args, 'results_suffix')
    l_cap = _DEFAULT_L_CAP
    if 'l_cap' in args:
        l_cap = args['l_cap']

    intermediate_output_dir = os.path.join(
        args['workspace_dir'], 'intermediate_outputs')
    output_dir = os.path.join(args['workspace_dir'])
    churn_dir = os.path.join(
        intermediate_output_dir, 'churn_dir_not_for_humans')
    utils.make_directories([output_dir, intermediate_output_dir, churn_dir])

    f_reg = utils.build_file_registry(
        [(_OUTPUT_BASE_FILES, output_dir),
         (_INTERMEDIATE_BASE_FILES, intermediate_output_dir),
         (_TMP_BASE_FILES, churn_dir)], file_suffix)

    # scrub the file suffix from some files if reusing the DEM
    if 'reuse_dem' in args and args['reuse_dem']:
        for key, target_path in [
                ('stream_and_drainage_path', 'stream_and_drainage.tif'),
                ('stream_path', 'stream.tif'),
                ('dem_offset_path', 'dem_offset.tif'),
                ('flow_accumulation_path', 'flow_accumulation.tif'),
                ('flow_direction_path', 'flow_direction.tif'),
                ('pit_filled_dem_path', 'pit_filled_dem.tif'),
                ('s_accumulation_path', 's_accumulation.tif'),
                ('s_bar_path', 's_bar.tif'),
                ('s_inverse_path', 's_inverse.tif'),
                ('slope_path', 'slope.tif'),
                ('thresholded_slope_path', 'slope_threshold.tif'),
                ('aligned_dem_path', 'aligned_dem.tif'),
                ('aligned_drainage_path', 'aligned_drainage.tif')]:
            f_reg[key] = os.path.join(
                os.path.dirname(f_reg[key]), target_path)

    task_graph = taskgraph.TaskGraph(churn_dir, -1)

    base_list = []
    aligned_list = []
    aligned_key_list = []
    for file_key in ['dem', 'erosivity', 'erodibility']:
        base_list.append(args[file_key + "_path"])
        aligned_list.append(f_reg["aligned_" + file_key + "_path"])
        aligned_key_list.append(
            (file_key + "_path", "aligned_" + file_key + "_path"))
    # all continuous rasters can use bilinear, but lulc should be mode
    interpolation_list = ['bilinear', 'bilinear', 'bilinear']
    lulc_path = args.get('lulc_path', None)
    if lulc_path is not None:
        base_list.append(lulc_path)
        aligned_list.append(f_reg['aligned_lulc_path'])
        aligned_key_list.append(('lulc_path', 'aligned_lulc_path'))

    drainage_present = False
    if 'drainage_path' in args and args['drainage_path'] != '':
        drainage_present = True
        base_list.append(args['drainage_path'])
        aligned_list.append(f_reg['aligned_drainage_path'])
        aligned_key_list.append(('drainage_path', 'aligned_drainage_path'))
        interpolation_list.append('near')

    dem_raster_info = geoprocessing.get_raster_info(args['dem_path'])
    min_pixel_size = numpy.min(numpy.abs(dem_raster_info['pixel_size']))
    target_pixel_size = args.get(
        'target_pixel_size', (min_pixel_size, -min_pixel_size))
    target_projection_wkt = args.get(
        'target_projection_wkt', dem_raster_info['projection_wkt'])

    # determine bounding box, and then create c/p factor rasters

    if 'prealigned' not in args or not args['prealigned']:
        vector_mask_options = {'mask_vector_path': args['watersheds_path']}
        align_task = task_graph.add_task(
            func=geoprocessing.align_and_resize_raster_stack,
            args=(
                base_list, aligned_list, interpolation_list,
                target_pixel_size, 'intersection'),
            kwargs={
                'target_projection_wkt': target_projection_wkt,
                'base_vector_path_list': (args['watersheds_path'],),
                'raster_align_index': 0,
                'vector_mask_options': vector_mask_options,
                },
            target_path_list=aligned_list,
            task_name='align input rasters')
        align_task.join()
    else:
        # the aligned stuff is the base stuff
        for base_key, aligned_key in aligned_key_list:
            f_reg[aligned_key] = args[base_key]
        align_task = task_graph.add_task()

    if 'single_outlet' in args and args['single_outlet'] is True:
        get_drain_sink_pixel_task = task_graph.add_task(
            func=geoprocessing.routing.detect_lowest_drain_and_sink,
            args=((f_reg['aligned_dem_path'], 1),),
            store_result=True,
            dependent_task_list=[align_task],
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

    pit_fill_task = task_graph.add_task(
        func=routing.fill_pits,
        args=(
            (f_reg['aligned_dem_path'], 1),
            f_reg['pit_filled_dem_path']),
        kwargs={
            'working_dir': args['workspace_dir'],
            'max_pixel_fill_count': -1,
            'single_outlet_tuple': single_outlet_tuple},
        target_path_list=[f_reg['pit_filled_dem_path']],
        dependent_task_list=[align_task],
        task_name='fill pits')

    slope_task = task_graph.add_task(
        func=geoprocessing.calculate_slope,
        args=(
            (f_reg['pit_filled_dem_path'], 1),
            f_reg['slope_path']),
        dependent_task_list=[pit_fill_task],
        target_path_list=[f_reg['slope_path']],
        task_name='calculate slope')

    threshold_slope_task = task_graph.add_task(
        func=_threshold_slope,
        args=(f_reg['slope_path'], f_reg['thresholded_slope_path']),
        target_path_list=[f_reg['thresholded_slope_path']],
        dependent_task_list=[slope_task],
        task_name='threshold slope')

    flow_dir_task = task_graph.add_task(
        func=routing.flow_dir_mfd,
        args=(
            (f_reg['pit_filled_dem_path'], 1),
            f_reg['flow_direction_path']),
        target_path_list=[f_reg['flow_direction_path']],
        dependent_task_list=[pit_fill_task],
        task_name='flow direction calculation')

    flow_accumulation_task = task_graph.add_task(
        func=routing.flow_accumulation_mfd,
        args=(
            (f_reg['flow_direction_path'], 1),
            f_reg['flow_accumulation_path']),
        target_path_list=[f_reg['flow_accumulation_path']],
        dependent_task_list=[flow_dir_task],
        task_name='flow accumulation calculation')

    ls_factor_task = task_graph.add_task(
        func=_calculate_ls_factor,
        args=(
            f_reg['flow_accumulation_path'], f_reg['slope_path'],
            f_reg['flow_direction_path'],
            l_cap, f_reg['ls_path']),
        target_path_list=[f_reg['ls_path']],
        dependent_task_list=[flow_accumulation_task, slope_task],
        task_name='ls factor calculation')

    stream_task = task_graph.add_task(
        func=routing.extract_streams_mfd,
        args=(
            (f_reg['flow_accumulation_path'], 1),
            (f_reg['flow_direction_path'], 1),
            float(args['threshold_flow_accumulation']),
            f_reg['stream_path']),
        kwargs={'trace_threshold_proportion': 0.7},
        target_path_list=[f_reg['stream_path']],
        dependent_task_list=[flow_accumulation_task],
        task_name='extract streams')

    if drainage_present:
        drainage_task = task_graph.add_task(
            func=_add_drainage(
                f_reg['stream_path'],
                f_reg['aligned_drainage_path'],
                f_reg['stream_and_drainage_path']),
            hash_algorithm='md5',
            copy_duplicate_artifact=True,
            target_path_list=[f_reg['stream_and_drainage_path']],
            dependent_task_list=[stream_task, align_task],
            task_name='add drainage')
        drainage_raster_path_task = (
            f_reg['stream_and_drainage_path'], drainage_task)
    else:
        drainage_raster_path_task = (f_reg['stream_path'], stream_task)

    # TODO: calculate c_path here
    raster_info = geoprocessing.get_raster_info(f_reg['aligned_dem_path'])
    usle_factor_dict = {}
    for usle_key in ['usle_c', 'usle_p']:
        usle_task = task_graph.add_task(
            func=_reclassify_or_clip,
            args=(
                usle_key, args.get('biophysical_table_path', None),
                f_reg.get('aligned_lulc_path', None), args, raster_info,
                f_reg),
            store_result=True,
            task_name=f'reclassify or warp {usle_key}')
        usle_factor_dict[usle_key] = usle_task

    threshold_c_task = task_graph.add_task(
        func=_threshold_c,
        args=(
            usle_factor_dict['usle_c'].get(),
            f_reg['thresholded_c_path']),
        target_path_list=[f_reg['thresholded_c_path']],
        dependent_task_list=[align_task],
        task_name='calculate thresholded C')

    cp_task = task_graph.add_task(
        func=_calculate_cp,
        args=(
            f_reg['thresholded_c_path'],
            usle_factor_dict['usle_p'].get(),
            f_reg['cp_factor_path']),
        target_path_list=[f_reg['cp_factor_path']],
        dependent_task_list=[align_task],
        task_name='calculate CP')

    rkls_task = task_graph.add_task(
        func=_calculate_rkls,
        args=(
            f_reg['ls_path'],
            f_reg['aligned_erosivity_path'],
            f_reg['aligned_erodibility_path'],
            drainage_raster_path_task[0],
            f_reg['rkls_path']),
        target_path_list=[f_reg['rkls_path']],
        dependent_task_list=[
            align_task, ls_factor_task, drainage_raster_path_task[1]],
        task_name='calculate RKLS')

    usle_task = task_graph.add_task(
        func=_calculate_usle,
        args=(
            f_reg['rkls_path'],
            f_reg['cp_factor_path'],
            drainage_raster_path_task[0],
            f_reg['usle_path']),
        target_path_list=[f_reg['usle_path']],
        dependent_task_list=[
            rkls_task, cp_task, drainage_raster_path_task[1]],
        task_name='calculate USLE')

    bar_task_map = {}
    for factor_path, factor_task, accumulation_path, out_bar_path, bar_id in [
            (f_reg['thresholded_c_path'], threshold_c_task,
             f_reg['c_accumulation_path'],
             f_reg['c_bar_path'],
             'c_bar'),
            (f_reg['thresholded_slope_path'], threshold_slope_task,
             f_reg['s_accumulation_path'],
             f_reg['s_bar_path'],
             's_bar')]:
        bar_task = task_graph.add_task(
            func=_calculate_bar_factor,
            args=(
                f_reg['flow_direction_path'], factor_path,
                f_reg['flow_accumulation_path'],
                accumulation_path, out_bar_path),
            hash_algorithm='md5',
            copy_duplicate_artifact=True,
            target_path_list=[accumulation_path, out_bar_path],
            dependent_task_list=[
                align_task, factor_task, flow_accumulation_task,
                flow_dir_task],
            task_name='calculate %s' % bar_id)
        bar_task_map[bar_id] = bar_task

    d_up_task = task_graph.add_task(
        func=_calculate_d_up,
        args=(
            f_reg['c_bar_path'], f_reg['s_bar_path'],
            f_reg['flow_accumulation_path'], f_reg['d_up_path']),
        target_path_list=[f_reg['d_up_path']],
        dependent_task_list=[
            bar_task_map['s_bar'], bar_task_map['c_bar'],
            flow_accumulation_task],
        task_name='calculate Dup')

    inverse_cs_factor_task = task_graph.add_task(
        func=_calculate_inverse_cs_factor,
        args=(
            f_reg['thresholded_slope_path'], f_reg['thresholded_c_path'],
            f_reg['cs_inverse_path']),
        target_path_list=[f_reg['cs_inverse_path']],
        dependent_task_list=[threshold_slope_task, threshold_c_task],
        task_name='calculate inverse ws factor')

    d_dn_task = task_graph.add_task(
        func=routing.distance_to_channel_mfd,
        args=(
            (f_reg['flow_direction_path'], 1),
            (drainage_raster_path_task[0], 1),
            f_reg['d_dn_path']),
        kwargs={'weight_raster_path_band': (f_reg['cs_inverse_path'], 1)},
        target_path_list=[f_reg['d_dn_path']],
        dependent_task_list=[
            flow_dir_task, drainage_raster_path_task[1],
            inverse_cs_factor_task],
        task_name='calculating d_dn')

    ic_task = task_graph.add_task(
        func=_calculate_ic,
        args=(
            f_reg['d_up_path'], f_reg['d_dn_path'], f_reg['ic_path']),
        target_path_list=[f_reg['ic_path']],
        dependent_task_list=[d_up_task, d_dn_task],
        task_name='calculate ic')

    sdr_task = task_graph.add_task(
        func=_calculate_sdr,
        args=(
            float(args['k_param']), float(args['ic_0_param']),
            float(args['sdr_max']), f_reg['ic_path'],
            drainage_raster_path_task[0], f_reg['sdr_path']),
        target_path_list=[f_reg['sdr_path']],
        dependent_task_list=[ic_task],
        task_name='calculate sdr')

    _ = task_graph.add_task(
        func=_calculate_sed_export,
        args=(
            f_reg['usle_path'], f_reg['sdr_path'], f_reg['sed_export_path']),
        target_path_list=[f_reg['sed_export_path']],
        dependent_task_list=[usle_task, sdr_task],
        task_name='calculate sed export')

    e_prime_task = task_graph.add_task(
        func=_calculate_e_prime,
        args=(
            f_reg['usle_path'], f_reg['sdr_path'], f_reg['e_prime_path']),
        target_path_list=[f_reg['e_prime_path']],
        dependent_task_list=[usle_task, sdr_task],
        task_name='calculate export prime')

    _ = task_graph.add_task(
        func=sdr_c_factor_core.calculate_sediment_deposition,
        args=(
            f_reg['flow_direction_path'], f_reg['e_prime_path'],
            f_reg['f_path'], f_reg['sdr_path'],
            f_reg['sed_deposition_path']),
        dependent_task_list=[e_prime_task, sdr_task, flow_dir_task],
        target_path_list=[f_reg['sed_deposition_path']],
        task_name='sediment deposition')

    _ = task_graph.add_task(
        func=_calculate_sed_retention_index,
        args=(
            f_reg['rkls_path'], f_reg['usle_path'], f_reg['sdr_path'],
            float(args['sdr_max']), f_reg['sed_retention_index_path']),
        target_path_list=[f_reg['sed_retention_index_path']],
        dependent_task_list=[rkls_task, usle_task, sdr_task],
        task_name='calculate sediment retention index')

    # This next section is for calculating the bare soil part.
    s_inverse_task = task_graph.add_task(
        func=_calculate_inverse_s_factor,
        args=(f_reg['thresholded_slope_path'], f_reg['s_inverse_path']),
        target_path_list=[f_reg['s_inverse_path']],
        dependent_task_list=[threshold_slope_task],
        task_name='calculate S factor')

    d_dn_bare_task = task_graph.add_task(
        func=routing.distance_to_channel_mfd,
        args=(
            (f_reg['flow_direction_path'], 1),
            (drainage_raster_path_task[0], 1),
            f_reg['d_dn_bare_soil_path']),
        kwargs={'weight_raster_path_band': (f_reg['s_inverse_path'], 1)},
        target_path_list=[f_reg['d_dn_bare_soil_path']],
        dependent_task_list=[
            flow_dir_task, drainage_raster_path_task[1], s_inverse_task],
        task_name='calculating d_dn soil')

    d_up_bare_task = task_graph.add_task(
        func=_calculate_d_up_bare,
        args=(
            f_reg['s_bar_path'], f_reg['flow_accumulation_path'],
            f_reg['d_up_bare_soil_path']),
        target_path_list=[f_reg['d_up_bare_soil_path']],
        dependent_task_list=[bar_task_map['s_bar'], flow_accumulation_task],
        task_name='calculating d_up bare soil')

    ic_bare_task = task_graph.add_task(
        func=_calculate_ic,
        args=(
            f_reg['d_up_bare_soil_path'], f_reg['d_dn_bare_soil_path'],
            f_reg['ic_bare_soil_path']),
        target_path_list=[f_reg['ic_bare_soil_path']],
        dependent_task_list=[d_up_bare_task, d_dn_bare_task],
        task_name='calculate bare soil ic')

    sdr_bare_task = task_graph.add_task(
        func=_calculate_sdr,
        args=(
            float(args['k_param']), float(args['ic_0_param']),
            float(args['sdr_max']), f_reg['ic_bare_soil_path'],
            drainage_raster_path_task[0], f_reg['sdr_bare_soil_path']),
        target_path_list=[f_reg['sdr_bare_soil_path']],
        dependent_task_list=[ic_bare_task, drainage_raster_path_task[1]],
        task_name='calculate bare SDR')

    _ = task_graph.add_task(
        func=_calculate_sed_retention,
        args=(
            f_reg['rkls_path'], f_reg['usle_path'],
            drainage_raster_path_task[0], f_reg['sdr_path'],
            f_reg['sdr_bare_soil_path'], f_reg['sed_retention_path']),
        target_path_list=[f_reg['sed_retention_path']],
        dependent_task_list=[
            rkls_task, usle_task, drainage_raster_path_task[1], sdr_task,
            sdr_bare_task],
        task_name='calculate sediment retention')

    task_graph.close()
    task_graph.join()


def _calculate_ls_factor(
        flow_accumulation_path, slope_path, flow_direction_path, l_cap,
        out_ls_factor_path):
    """Calculate LS factor.

    LS factor as Equation 3 from "Extension and validation
    of a geographic information system-based method for calculating the
    Revised Universal Soil Loss Equation length-slope factor for erosion
    risk assessments in large watersheds"

    Parameters:
        flow_accumulation_path (string): path to raster, pixel values are the
            contributing upstream area at that cell. Pixel size is square.
        slope_path (string): path to slope raster as a percent
        flow_direction_path (string): path to a 32 bit in raster representing 8 MFD
            intensities as a 4 bit int. where the first direction is mask
            0xF the second 0xF << 4 etc.
        l_cap (float): set the upstream area to be no greater than the
            square of this number. This is the "McCool l factor cap".
        out_ls_factor_path (string): path to output ls_factor raster

    Returns:
        None

    """
    slope_nodata = geoprocessing.get_raster_info(slope_path)['nodata'][0]

    flow_accumulation_info = geoprocessing.get_raster_info(
        flow_accumulation_path)
    flow_accumulation_nodata = flow_accumulation_info['nodata'][0]
    cell_size = abs(flow_accumulation_info['pixel_size'][0])
    cell_area = cell_size ** 2

    def ls_factor_function(
            percent_slope, flow_direction_mfd, flow_accumulation, l_cap):
        """Calculate the LS factor.

        Parameters:
            percent_slope (numpy.ndarray): slope in percent
            flow_direction_mfd (numpy.ndarray): 32 bit ints representing
                MFD direction
            flow_accumulation (numpy.ndarray): upstream pixels
            l_cap (float): set the upstream area to be no greater than the
                square of this number. This is the "McCool l factor cap".

        Returns:
            ls_factor

        """
        valid_mask = (
            (percent_slope != slope_nodata) &
            (flow_accumulation != flow_accumulation_nodata))
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA

        # take the weighted mean of the aspect angles using the MFD angles
        mfd_weights = numpy.array(
            [1, numpy.sqrt(2), 1, numpy.sqrt(2),
             1, numpy.sqrt(2), 1, numpy.sqrt(2)])
        mfd_masks = 0xF << (numpy.arange(0, 32, 4)).astype(numpy.uint32)
        shifted = (flow_direction_mfd[valid_mask][..., None] & mfd_masks) >> numpy.arange(
            0, 32, 4)
        total_value = numpy.sum(shifted * mfd_weights, axis=-1)
        total_weight = numpy.sum(mfd_weights)
        xij = total_value / total_weight

        contributing_flow_length_area = numpy.sqrt(
            (flow_accumulation[valid_mask]-1) * cell_area)
        slope_in_radians = numpy.arctan(percent_slope[valid_mask] / 100.0)

        # From Equation 4 in "Extension and validation of a geographic
        # information system ..."
        slope_factor = numpy.where(
            percent_slope[valid_mask] < 9.0,
            10.8 * numpy.sin(slope_in_radians) + 0.03,
            16.8 * numpy.sin(slope_in_radians) - 0.5)

        beta = (
            (numpy.sin(slope_in_radians) / 0.0896) /
            (3 * numpy.sin(slope_in_radians)**0.8 + 0.56))

        # Set m value via lookup table: Table 1 in
        # InVEST Sediment Model_modifications_10-01-2012_RS.docx
        # note slope_table in percent
        slope_table = numpy.array([1., 3.5, 5., 9.])
        m_table = numpy.array([0.2, 0.3, 0.4, 0.5])
        # mask where slopes are larger than lookup table
        big_slope_mask = percent_slope[valid_mask] > slope_table[-1]
        m_indexes = numpy.digitize(
            percent_slope[valid_mask][~big_slope_mask], slope_table,
            right=True)
        m_exp = numpy.empty(big_slope_mask.shape, dtype=numpy.float32)
        m_exp[big_slope_mask] = (
            beta[big_slope_mask] / (1 + beta[big_slope_mask]))
        m_exp[~big_slope_mask] = m_table[m_indexes]

        l_factor = (
            ((contributing_flow_length_area + cell_area)**(m_exp+1) -
             contributing_flow_length_area ** (m_exp+1)) /
            ((cell_size ** (m_exp + 2)) * (xij**m_exp) * (22.13**m_exp)))

        # ensure l_factor is no larger than l_cap
        l_factor[l_factor > l_cap] = l_cap
        result[valid_mask] = l_factor * slope_factor
        return result

    geoprocessing.raster_calculator(
        [(path, 1) for path in [
            slope_path, flow_direction_path, flow_accumulation_path]] +
        [(l_cap, 'raw')], ls_factor_function, out_ls_factor_path,
        gdal.GDT_Float32, _TARGET_NODATA)


def _calculate_rkls(
        ls_factor_path, erosivity_path, erodibility_path, stream_path,
        rkls_path):
    """Calculate per-pixel potential soil loss using the RKLS.

    (revised universal soil loss equation with no C or P).

    Parameters:
        ls_factor_path (string): path to LS raster that has square pixels in
            meter units.
        erosivity_path (string): path to per pixel erosivity raster
        erodibility_path (string): path to erodibility raster
        stream_path (string): path to drainage raster
            (1 is drainage, 0 is not)
        rkls_path (string): path to RKLS raster

    Returns:
        None

    """
    erosivity_nodata = geoprocessing.get_raster_info(
        erosivity_path)['nodata'][0]
    erodibility_nodata = geoprocessing.get_raster_info(
        erodibility_path)['nodata'][0]
    stream_nodata = geoprocessing.get_raster_info(
        stream_path)['nodata'][0]

    cell_size = abs(
        geoprocessing.get_raster_info(ls_factor_path)['pixel_size'][0])
    cell_area_ha = cell_size**2 / 10000.0

    def rkls_function(ls_factor, erosivity, erodibility, stream):
        """Calculate the RKLS equation.

        Parameters:
            ls_factor (numpy.ndarray): length/slope factor
        erosivity (numpy.ndarray): related to peak rainfall events
        erodibility (numpy.ndarray): related to the potential for soil to
            erode
        stream (numpy.ndarray): stream mask (1 stream, 0 no stream)

        Returns:
            ls_factor * erosivity * erodibility * usle_c_p or nodata if
            any values are nodata themselves.

        """
        rkls = numpy.empty(ls_factor.shape, dtype=numpy.float32)
        nodata_mask = (
            (ls_factor != _TARGET_NODATA) & (stream != stream_nodata))
        if erosivity_nodata is not None:
            nodata_mask &= ~numpy.isclose(erosivity, erosivity_nodata)
        if erodibility_nodata is not None:
            nodata_mask &= ~numpy.isclose(erodibility, erodibility_nodata)

        valid_mask = nodata_mask & (stream == 0)
        rkls[:] = _TARGET_NODATA

        rkls[valid_mask] = (
            ls_factor[valid_mask] * erosivity[valid_mask] *
            erodibility[valid_mask] * cell_area_ha)

        # rkls is 1 on the stream
        rkls[nodata_mask & (stream == 1)] = 1
        return rkls

    # aligning with index 3 that's the stream and the most likely to be
    # aligned with LULCs
    geoprocessing.raster_calculator(
        [(path, 1) for path in [
            ls_factor_path, erosivity_path, erodibility_path, stream_path]],
        rkls_function, rkls_path, gdal.GDT_Float32, _TARGET_NODATA)


def _threshold_slope(slope_path, out_thresholded_slope_path):
    """Threshold the slope between 0.005 and 1.0.

    Parameters:
        slope_path (string): path to a raster of slope in percent
        out_thresholded_slope_path (string): path to output raster of
            thresholded slope between 0.005 and 1.0

    Returns:
        None

    """
    slope_nodata = geoprocessing.get_raster_info(slope_path)['nodata'][0]

    def threshold_slope(slope):
        """Convert slope to m/m and clamp at 0.005 and 1.0.

        As desribed in Cavalli et al., 2013.
        """
        valid_slope = slope != slope_nodata
        slope_m = slope[valid_slope] / 100.0
        slope_m[slope_m < 0.005] = 0.005
        slope_m[slope_m > 1.0] = 1.0
        result = numpy.empty(valid_slope.shape, dtype=numpy.float32)
        result[:] = slope_nodata
        result[valid_slope] = slope_m
        return result

    geoprocessing.raster_calculator(
        [(slope_path, 1)], threshold_slope, out_thresholded_slope_path,
        gdal.GDT_Float32, slope_nodata)


def _add_drainage(stream_path, drainage_path, out_stream_and_drainage_path):
    """Combine stream and drainage masks into one raster mask.

    Parameters:
        stream_path (string): path to stream raster mask where 1 indicates
            a stream, and 0 is a valid landscape pixel but not a stream.
        drainage_raster_path (string): path to 1/0 mask of drainage areas.
            1 indicates any water reaching that pixel drains to a stream.
        out_stream_and_drainage_path (string): output raster of a logical
            OR of stream and drainage inputs

    Returns:
        None

    """
    def add_drainage_op(stream, drainage):
        """Add drainage mask to stream layer."""
        return numpy.where(drainage == 1, 1, stream)

    stream_nodata = geoprocessing.get_raster_info(stream_path)['nodata'][0]
    geoprocessing.raster_calculator(
        [(path, 1) for path in [stream_path, drainage_path]], add_drainage_op,
        out_stream_and_drainage_path, gdal.GDT_Byte, stream_nodata)


def _threshold_c(
        usle_c_path, target_threshold_c_factor_path):
    """W factor: map C values from LULC and lower threshold to 0.001.

    W is a factor in calculating d_up accumulation for SDR.

    Parameters:
        usle_c_path (str): path to usle raster
        target_threshold_c_factor_path (str): W factor from `w_factor_path`
            thresholded to be no less than 0.001.

    Returns:
        None

    """
    def threshold_c(c_val):
        """Threshold c to 0.001."""
        result = c_val.copy()
        valid_mask = c_val != _TARGET_NODATA
        result[(c_val < 0.001) & valid_mask] = 0.001
        return result

    geoprocessing.raster_calculator(
        [(usle_c_path, 1)], threshold_c, target_threshold_c_factor_path,
        gdal.GDT_Float32, _TARGET_NODATA)


def _calculate_cp(
        usle_c_path, usle_p_path, target_cp_factor_path):
    """Map LULC to C*P value.

    Parameters:
        c_factor_path (str): path to c factor raster
        p_factor_path (str): path to p factor raster
        target_cp_factor_path (str): target C*P raster

    Returns:
        None

    """
    def _local_mult_op(usle_c, usle_p):
        result = usle_c.copy()
        valid_mask = usle_p != _TARGET_NODATA
        result[valid_mask] *= usle_p[valid_mask]
        return result
    geoprocessing.raster_calculator(
        [(usle_c_path, 1), (usle_p_path, 1)],
        _local_mult_op, target_cp_factor_path, gdal.GDT_Float32,
        _TARGET_NODATA)


def _multiply_op(array_a, array_b, nodata_a, nodata_b, result_nodata):
    """Mult a by b."""
    result = numpy.empty_like(array_a)
    result[:] = result_nodata
    valid_mask = numpy.ones(array_a.shape, dtype=numpy.bool)
    if nodata_a is not None:
        valid_mask &= ~numpy.isclose(array_a, nodata_a)
    if nodata_b is not None:
        valid_mask &= ~numpy.isclose(array_b, nodata_b)
    result[valid_mask] = array_a[valid_mask] * array_b[valid_mask]
    return result


def _calculate_usle(
        rkls_path, cp_factor_path, drainage_raster_path, out_usle_path):
    """Calculate USLE, multiply RKLS by CP and set to 1 on drains."""
    def usle_op(rkls, cp_factor, drainage):
        """Calculate USLE."""
        result = numpy.empty(rkls.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        valid_mask = (rkls != _TARGET_NODATA) & (cp_factor != _TARGET_NODATA)
        result[valid_mask] = rkls[valid_mask] * cp_factor[valid_mask] * (
            1 - drainage[valid_mask])
        return result

    geoprocessing.raster_calculator(
        [(path, 1) for path in [
            rkls_path, cp_factor_path, drainage_raster_path]], usle_op,
        out_usle_path, gdal.GDT_Float32, _TARGET_NODATA)


def _calculate_bar_factor(
        flow_direction_path, factor_path, flow_accumulation_path,
        accumulation_path, out_bar_path):
    """Route user defined source across DEM.

    Used for calculating S and W bar in the SDR operation.

    Parameters:
        dem_path (string): path to DEM raster
        factor_path (string): path to arbitrary factor raster
        flow_accumulation_path (string): path to flow accumulation raster
        flow_direction_path (string): path to flow direction path (in radians)
        accumulation_path (string): path to a raster that can be used to
            save the accumulation of the factor.  Temporary file.
        out_bar_path (string): path to output raster that is the result of
            the factor accumulation raster divided by the flow accumulation
            raster.

    Returns:
        None.

    """
    flow_accumulation_nodata = geoprocessing.get_raster_info(
        flow_accumulation_path)['nodata'][0]

    LOGGER.debug("doing flow accumulation mfd on %s", factor_path)
    # manually setting compression to DEFLATE because we got some LZW
    # errors when testing with large data.
    routing.flow_accumulation_mfd(
        (flow_direction_path, 1), accumulation_path,
        weight_raster_path_band=(factor_path, 1),
        raster_driver_creation_tuple=('GTIFF', [
            'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=DEFLATE',
            'PREDICTOR=3']))

    def bar_op(base_accumulation, flow_accumulation):
        """Aggregate accumulation from base divided by the flow accum."""
        result = numpy.empty(base_accumulation.shape, dtype=numpy.float32)
        valid_mask = (
            ~numpy.isclose(base_accumulation, _TARGET_NODATA) &
            ~numpy.isclose(flow_accumulation, flow_accumulation_nodata))
        result[:] = _TARGET_NODATA
        result[valid_mask] = (
            base_accumulation[valid_mask] / flow_accumulation[valid_mask])
        return result
    geoprocessing.raster_calculator(
        [(accumulation_path, 1), (flow_accumulation_path, 1)], bar_op,
        out_bar_path, gdal.GDT_Float32, _TARGET_NODATA)


def _calculate_d_up(
        c_bar_path, s_bar_path, flow_accumulation_path, out_d_up_path):
    """Calculate c_bar * s_bar * sqrt(flow accumulation * cell area)."""
    cell_area = abs(
        geoprocessing.get_raster_info(c_bar_path)['pixel_size'][0])**2
    flow_accumulation_nodata = geoprocessing.get_raster_info(
        flow_accumulation_path)['nodata'][0]

    def d_up_op(c_bar, s_bar, flow_accumulation):
        """Calculate the d_up index.

        c_bar * s_bar * sqrt(upstream area)

        """
        valid_mask = (
            (c_bar != _TARGET_NODATA) & (s_bar != _TARGET_NODATA) &
            (flow_accumulation != flow_accumulation_nodata))
        d_up_array = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        d_up_array[:] = _TARGET_NODATA
        d_up_array[valid_mask] = (
            c_bar[valid_mask] * s_bar[valid_mask] * numpy.sqrt(
                flow_accumulation[valid_mask] * cell_area))
        return d_up_array

    geoprocessing.raster_calculator(
        [(path, 1) for path in [
            c_bar_path, s_bar_path, flow_accumulation_path]], d_up_op,
        out_d_up_path, gdal.GDT_Float32, _TARGET_NODATA)


def _calculate_d_up_bare(
        s_bar_path, flow_accumulation_path, out_d_up_bare_path):
    """Calculate s_bar * sqrt(flow accumulation * cell area)."""
    cell_area = abs(
        geoprocessing.get_raster_info(s_bar_path)['pixel_size'][0])**2
    flow_accumulation_nodata = geoprocessing.get_raster_info(
        flow_accumulation_path)['nodata'][0]

    def d_up_op(s_bar, flow_accumulation):
        """Calculate the bare d_up index.

        s_bar * sqrt(upstream area)

        """
        valid_mask = (
            (flow_accumulation != flow_accumulation_nodata) &
            (s_bar != _TARGET_NODATA))
        d_up_array = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        d_up_array[:] = _TARGET_NODATA
        d_up_array[valid_mask] = (
            numpy.sqrt(flow_accumulation[valid_mask] * cell_area) *
            s_bar[valid_mask])
        return d_up_array

    geoprocessing.raster_calculator(
        [(s_bar_path, 1), (flow_accumulation_path, 1)], d_up_op,
        out_d_up_bare_path, gdal.GDT_Float32, _TARGET_NODATA)


def _calculate_inverse_cs_factor(
        thresholded_slope_path, thresholded_w_factor_path,
        out_ws_factor_inverse_path):
    """Calculate 1/(w*s)."""
    slope_nodata = geoprocessing.get_raster_info(
        thresholded_slope_path)['nodata'][0]

    def ws_op(w_factor, s_factor):
        """Calculate the inverse ws factor."""
        valid_mask = (w_factor != _TARGET_NODATA) & (s_factor != slope_nodata)
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        result[valid_mask] = (
            1.0 / (w_factor[valid_mask] * s_factor[valid_mask]))
        return result

    geoprocessing.raster_calculator(
        [(thresholded_w_factor_path, 1), (thresholded_slope_path, 1)], ws_op,
        out_ws_factor_inverse_path, gdal.GDT_Float32, _TARGET_NODATA)


def _calculate_inverse_s_factor(
        thresholded_slope_path, out_s_factor_inverse_path):
    """Calculate 1/s."""
    slope_nodata = geoprocessing.get_raster_info(
        thresholded_slope_path)['nodata'][0]

    def s_op(s_factor):
        """Calculate the inverse s factor."""
        valid_mask = (s_factor != slope_nodata)
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        result[valid_mask] = 1.0 / s_factor[valid_mask]
        return result

    geoprocessing.raster_calculator(
        [(thresholded_slope_path, 1)], s_op,
        out_s_factor_inverse_path, gdal.GDT_Float32, _TARGET_NODATA)


def _calculate_ic(d_up_path, d_dn_path, out_ic_factor_path):
    """Calculate log10(d_up/d_dn)."""
    # ic can be positive or negative, so float.min is a reasonable nodata value
    d_dn_nodata = geoprocessing.get_raster_info(d_dn_path)['nodata'][0]

    def ic_op(d_up, d_dn):
        """Calculate IC factor."""
        valid_mask = (
            (d_up != _TARGET_NODATA) & (d_dn != d_dn_nodata) & (d_dn != 0) &
            (d_up != 0))
        ic_array = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        ic_array[:] = _IC_NODATA
        ic_array[valid_mask] = numpy.log10(
            d_up[valid_mask] / d_dn[valid_mask])
        return ic_array

    geoprocessing.raster_calculator(
        [(d_up_path, 1), (d_dn_path, 1)], ic_op, out_ic_factor_path,
        gdal.GDT_Float32, _IC_NODATA)


def _calculate_sdr(
        k_factor, ic_0, sdr_max, ic_path, stream_path, out_sdr_path):
    """Derive SDR from k, ic0, ic; 0 on the stream and clamped to sdr_max."""
    def sdr_op(ic_factor, stream):
        """Calculate SDR factor."""
        valid_mask = (
            (ic_factor != _IC_NODATA) & (stream != 1))
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        result[valid_mask] = (
            sdr_max / (1+numpy.exp((ic_0-ic_factor[valid_mask])/k_factor)))
        result[stream == 1] = 0.0
        return result

    geoprocessing.raster_calculator(
        [(ic_path, 1), (stream_path, 1)], sdr_op, out_sdr_path,
        gdal.GDT_Float32, _TARGET_NODATA)


def _calculate_sed_export(usle_path, sdr_path, target_sed_export_path):
    """Calculate USLE * SDR."""
    def sed_export_op(usle, sdr):
        """Sediment export."""
        valid_mask = (usle != _TARGET_NODATA) & (sdr != _TARGET_NODATA)
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        result[valid_mask] = usle[valid_mask] * sdr[valid_mask]
        return result

    geoprocessing.raster_calculator(
        [(usle_path, 1), (sdr_path, 1)], sed_export_op,
        target_sed_export_path, gdal.GDT_Float32, _TARGET_NODATA)


def _calculate_e_prime(usle_path, sdr_path, target_e_prime):
    """Calculate USLE * (1-SDR)."""
    def e_prime_op(usle, sdr):
        """Wash that does not reach stream."""
        valid_mask = (usle != _TARGET_NODATA) & (sdr != _TARGET_NODATA)
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        result[valid_mask] = usle[valid_mask] * (1-sdr[valid_mask])
        return result

    geoprocessing.raster_calculator(
        [(usle_path, 1), (sdr_path, 1)], e_prime_op, target_e_prime,
        gdal.GDT_Float32, _TARGET_NODATA)


def _calculate_sed_retention_index(
        rkls_path, usle_path, sdr_path, sdr_max,
        out_sed_retention_index_path):
    """Calculate real index (usle/rkls)."""
    def sediment_index_op(rkls, usle, sdr_factor):
        """Calculate sediment retention index."""
        valid_mask = (
            (rkls != _TARGET_NODATA) & (usle != _TARGET_NODATA) &
            (sdr_factor != _TARGET_NODATA))
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        result[valid_mask] = (usle[valid_mask] / rkls[valid_mask])
        return result

    geoprocessing.raster_calculator(
        [(path, 1) for path in [rkls_path, usle_path, sdr_path]],
        sediment_index_op, out_sed_retention_index_path, gdal.GDT_Float32,
        _TARGET_NODATA)


def _calculate_sed_retention(
        rkls_path, usle_path, stream_path, sdr_path, sdr_bare_soil_path,
        out_sed_ret_bare_soil_path):
    """Difference in exported sediments on basic and bare watershed.

    Calculates the difference of sediment export on the real landscape and
    a bare soil landscape given that SDR has been calculated for bare soil.
    Essentially:

        RKLS * SDR_bare - USLE * SDR

    Parameters:
        rkls_path (string): path to RKLS raster
        usle_path (string): path to USLE raster
        stream_path (string): path to stream/drainage mask
        sdr_path (string): path to SDR raster
        sdr_bare_soil_path (string): path to SDR raster calculated for a bare
            watershed
        out_sed_ret_bare_soil_path (string): path to output raster indicating
            where sediment is retained

    Returns:
        None

    """
    stream_nodata = geoprocessing.get_raster_info(stream_path)['nodata'][0]

    def sediment_retention_bare_soil_op(
            rkls, usle, stream_factor, sdr_factor, sdr_factor_bare_soil):
        """Subtract bare soil export from real landcover."""
        valid_mask = (
            (rkls != _TARGET_NODATA) &
            (usle != _TARGET_NODATA) &
            (stream_factor != stream_nodata) &
            (sdr_factor != _TARGET_NODATA) &
            (sdr_factor_bare_soil != _TARGET_NODATA))
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        result[valid_mask] = (
            rkls[valid_mask] * sdr_factor_bare_soil[valid_mask] -
            usle[valid_mask] * sdr_factor[valid_mask]) * (
                1 - stream_factor[valid_mask])
        return result

    geoprocessing.raster_calculator(
        [(path, 1) for path in [
            rkls_path, usle_path, stream_path, sdr_path, sdr_bare_soil_path]],
        sediment_retention_bare_soil_op, out_sed_ret_bare_soil_path,
        gdal.GDT_Float32, _TARGET_NODATA)
