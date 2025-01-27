"""InVEST Seasonal Water Yield Model."""
import fractions
from pathlib import Path
import glob
import logging
import os
import re
import warnings

import numpy
from ecoshard import geoprocessing
from ecoshard.geoprocessing import routing
from ecoshard import taskgraph
import scipy.special
from osgeo import gdal
from osgeo import ogr

from .. import utils
from . import seasonal_water_yield_core

gdal.SetCacheMax(2**26)

LOGGER = logging.getLogger(__name__)

TARGET_NODATA = -1
N_MONTHS = 12
MONTH_ID_TO_LABEL = [
    'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct',
    'nov', 'dec']


_OUTPUT_BASE_FILES = {
    'aggregate_vector_path': 'aggregated_results_swy.shp',
    'annual_precip_path': 'P.tif',
    'cn_path': 'CN.tif',
    'l_avail_path': 'L_avail.tif',
    'l_path': 'L.tif',
    'l_sum_path': 'L_sum.tif',
    'l_sum_avail_path': 'L_sum_avail.tif',
    'qf_path': 'QF.tif',
    'b_sum_path': 'B_sum.tif',
    'b_path': 'B.tif',
    'vri_path': 'Vri.tif',
}

_INTERMEDIATE_BASE_FILES = {
    'aet_path': 'aet.tif',
    'aetm_path_list': ['aetm_%d.tif' % (x+1) for x in range(N_MONTHS)],
    'flow_dir_mfd_path': 'flow_dir_mfd.tif',
    'qfm_path_list': ['qf_%d.tif' % (x+1) for x in range(N_MONTHS)],
    'stream_path': 'stream.tif',
}

_TMP_BASE_FILES = {
    'outflow_direction_path': 'outflow_direction.tif',
    'outflow_weights_path': 'outflow_weights.tif',
    'kc_path': 'kc.tif',
    'si_path': 'Si.tif',
    'lulc_aligned_path': 'lulc_aligned.tif',
    'dem_aligned_path': 'dem_aligned.tif',
    'dem_pit_filled_path': 'pit_filled_dem.tif',
    'loss_path': 'loss.tif',
    'zero_absorption_source_path': 'zero_absorption.tif',
    'soil_group_aligned_path': 'soil_group_aligned.tif',
    'flow_accum_path': 'flow_accum.tif',
    'precip_path_aligned_list': ['prcp_a%d.tif' % x for x in range(N_MONTHS)],
    'n_events_path_list': ['n_events%d.tif' % x for x in range(N_MONTHS)],
    'et0_path_aligned_list': ['et0_a%d.tif' % x for x in range(N_MONTHS)],
    'kc_1': 'kc_1.tif',
    'kc_2': 'kc_2.tif',
    'kc_3': 'kc_3.tif',
    'kc_4': 'kc_4.tif',
    'kc_5': 'kc_5.tif',
    'kc_6': 'kc_6.tif',
    'kc_7': 'kc_7.tif',
    'kc_8': 'kc_8.tif',
    'kc_9': 'kc_9.tif',
    'kc_10': 'kc_10.tif',
    'kc_11': 'kc_11.tif',
    'kc_12': 'kc_12.tif',
    'root_depth': 'root_depth.tif',
    'cn_a': 'CN_A.tif',
    'cn_b': 'CN_B.tif',
    'cn_c': 'CN_C.tif',
    'cn_d': 'CN_D.tif',
    'l_aligned_path': 'l_aligned.tif',
    'cz_aligned_raster_path': 'cz_aligned.tif',
    'l_sum_pre_clamp': 'l_sum_pre_clamp.tif'
}

_TABLE_BASED_BIOPHYSICAL_FACTORS = [
    'root_depth', 'cn_a', 'cn_b', 'cn_c', 'cn_d', 'kc_1', 'kc_2',
    'kc_3', 'kc_4', 'kc_5', 'kc_6', 'kc_7', 'kc_8', 'kc_9', 'kc_10',
    'kc_11', 'kc_12']


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
    key_field = key_field.lower()
    key_path = f'{key_field}_path'
    if key_path in args and args[key_path] not in ['', None]:
        geoprocessing.warp_raster(
            args[key_path], target_raster_info['pixel_size'], f_reg[key_field],
            'bilinear', target_bb=target_raster_info['bounding_box'],
            target_projection_wkt=target_raster_info['projection_wkt'],
            working_dir=os.path.dirname(f_reg[key_field]))
        return f_reg[key_field]

    if args['biophysical_table_path'] is None:
        raise ValueError(
            f'Neither {key_field} or "biophysical_table_path" were defined in '
            f'args, one must be defined. value of args: {args}')

    lufield_id = args.get('lucode_field', 'lucode')
    biophysical_table = utils.build_lookup_from_csv(
        args['biophysical_table_path'], lufield_id)
    lulc_to_val = dict(
        [(lulc_code, float(table[key_field])) for
         (lulc_code, table) in biophysical_table.items()])

    geoprocessing.reclassify_raster(
        (lulc_raster_path, 1), lulc_to_val, f_reg[key_field],
        gdal.GDT_Float32, TARGET_NODATA)
    return f_reg[key_field]


def execute(args):
    """Seasonal Water Yield.

    Args:
        args['workspace_dir'] (string): output directory for intermediate,
            temporary, and final files
        args['results_suffix'] (string): (optional) string to append to any
            output files
            stream pixels from the DEM by thresholding the number of upslope
            cells that must flow into a cell before it's considered
            part of a stream.
        args['et0_dir'] (string): required if
            args['user_defined_local_recharge'] is False.  Path to a directory
            that contains rasters of monthly reference evapotranspiration;
            units in mm.
        args['precip_dir'] (string): required if
            args['user_defined_local_recharge'] is False. A path to a directory
            that contains rasters of monthly precipitation; units in mm.
        args['dem_raster_path'] (string): a path to a digital elevation raster
        args['lulc_raster_path'] (string): a path to a land cover raster used
            to classify biophysical properties of pixels.
        args['lucode_field'] (string): fieldname in lucode lookup table for the
            lucode.
        args['soil_group_path'] (string): required if
            args['user_defined_local_recharge'] is  False. A path to a raster
            indicating SCS soil groups where integer values are mapped to soil
            types
        args['aoi_path'] (string): path to a vector that indicates the area
            over which the model should be run, as well as the area in which to
            aggregate over when calculating the output Qb.
        args['biophysical_table_path'] (string): path to a CSV table that maps
            landcover codes paired with soil group types to curve numbers as
            well as Kc values.  Headers must include 'lucode', 'CN_A', 'CN_B',
            'CN_C', 'CN_D', 'Kc_1', 'Kc_2', 'Kc_3', 'Kc_4', 'Kc_5', 'Kc_6',
            'Kc_7', 'Kc_8', 'Kc_9', 'Kc_10', 'Kc_11', 'Kc_12'.
        args['rain_events_table_path'] (string): Not required if
            args['user_defined_local_recharge'] is True or
            args['user_defined_climate_zones'] is True.  Path to a CSV table
            that has headers 'month' (1-12) and 'events' (int >= 0) that
            indicates the number of rain events per month
        args['alpha_m'] (float or string): required if args['monthly_alpha'] is
            false.  Is the proportion of upslope annual available local
            recharge that is available in month m.
        args['beta_i'] (float or string): is the fraction of the upgradient
            subsidy that is available for downgradient evapotranspiration.
        args['gamma'] (float or string): is the fraction of pixel local
            recharge that is available to downgradient pixels.
        args['guser_defined_local_recharge'] (boolean): if True, indicates user
            will provide pre-defined local recharge raster layer
        args['l_path'] (string): required if
            args['user_defined_local_recharge'] is True.  If provided pixels
            indicate the amount of local recharge; units in mm.
        args['user_defined_climate_zones'] (boolean): if True, user provides
            a climate zone rain events table and a climate zone raster map in
            lieu of a global rain events table.
        args['climate_zone_table_path'] (string): required if
            args['user_defined_climate_zones'] is True. Contains monthly
            precipitation events per climate zone.  Fields must be
            "cz_id", "jan", "feb", "mar", "apr", "may", "jun", "jul",
            "aug", "sep", "oct", "nov", "dec".
        args['climate_zone_raster_path'] (string): required if
            args['user_defined_climate_zones'] is True, pixel values correspond
            to the "cz_id" values defined in args['climate_zone_table_path']
        args['user_defined_rain_events_dir'] (string): path to directory of n_event rasters
        args['monthly_alpha'] (boolean): if True, use the alpha
        args['monthly_alpha_path'] (string): required if args['monthly_alpha']
            is True. A CSV file.
        args['prealigned'] (bool): if true, input rasters are already aligned
            and projected.
        args['max_pixel_fill_count'] (int): (optional), if provided limits the
            flood fill pixel count to this value when determining if the DEM
            has a pit. Useful on landscapes that have natural depressions.

    Returns:
        None.
    """
    # This upgrades warnings to exceptions across this model.
    # I found this useful to catch all kinds of weird inputs to the model
    # during debugging and think it makes sense to have in production of this
    # model too.
    try:
        warnings.filterwarnings('error')
        _execute(args)
    finally:
        warnings.resetwarnings()


def _execute(args):
    """Execute the seasonal water yield model.

    Args:
        See the parameters for
        `natcap.invest.seasonal_water_yield.seasonal_wateryield.execute`.

    Returns:
        None
    """
    LOGGER.info('prepare and test inputs for common errors')

    # fail early on a missing required rain events table
    for key in ['user_defined_local_recharge', 'user_defined_climate_zones', 'user_defined_rain_events_dir']:
        if key not in args:
            args[key] = None
    if (not args['user_defined_local_recharge'] and
            not args['user_defined_climate_zones'] and
            not args['user_defined_rain_events_dir']):
        rain_events_lookup = (
            utils.build_lookup_from_csv(
                args['rain_events_table_path'], 'month'))

    if 'monthly_alpha' in args and args['monthly_alpha']:
        # parse out the alpha lookup table of the form (month_id: alpha_val)
        alpha_month_map = dict(
            (key, val['alpha']) for key, val in
            utils.build_lookup_from_csv(
                args['monthly_alpha_path'], 'month').items())
    else:
        # make all 12 entries equal to args['alpha_m']
        alpha_m = float(fractions.Fraction(args['alpha_m']))
        alpha_month_map = dict(
            (month_index+1, alpha_m) for month_index in range(N_MONTHS))

    beta_i = float(fractions.Fraction(args['beta_i']))
    gamma = float(fractions.Fraction(args['gamma']))
    threshold_flow_accumulation = float(args['threshold_flow_accumulation'])
    pixel_size = geoprocessing.get_raster_info(
        args['dem_raster_path'])['pixel_size']
    file_suffix = utils.make_suffix_string(args, 'results_suffix')
    intermediate_output_dir = os.path.join(
        args['workspace_dir'], 'intermediate_outputs')
    cache_dir = os.path.join(args['workspace_dir'], 'cache_dir')
    output_dir = args['workspace_dir']
    utils.make_directories([intermediate_output_dir, cache_dir, output_dir])

    task_graph = taskgraph.TaskGraph(cache_dir, -1)

    LOGGER.info('Building file registry')
    file_registry = utils.build_file_registry(
        [(_OUTPUT_BASE_FILES, output_dir),
         (_INTERMEDIATE_BASE_FILES, intermediate_output_dir),
         (_TMP_BASE_FILES, cache_dir)], file_suffix)

    LOGGER.info('Checking that the AOI is not the output aggregate vector')
    if (os.path.normpath(args['aoi_path']) ==
            os.path.normpath(file_registry['aggregate_vector_path'])):
        raise ValueError(
            "The input AOI is the same as the output aggregate vector, "
            "please choose a different workspace or move the AOI file "
            "out of the current workspace %s" %
            file_registry['aggregate_vector_path'])

    LOGGER.info('Aligning and clipping dataset list')
    input_align_list = [args['lulc_raster_path'], args['dem_raster_path']]
    output_align_list = [
        file_registry['lulc_aligned_path'], file_registry['dem_aligned_path']]
    aligned_key_list = [
        ('lulc_raster_path', 'lulc_aligned_path'),
        ('dem_raster_path', 'dem_aligned_path')]
    if not args['user_defined_local_recharge']:
        et0_path_list = sorted(str(path) for path in Path(args['et0_dir']).glob('*.tif'))
        precip_path_list = sorted(str(path) for path in Path(args['precip_dir']).glob('*.tif'))

        # for month_index in range(1, N_MONTHS + 1):
        #     month_file_match = re.compile(r'.*[^\d]0?%d\.tif$' % month_index)
        #     LOGGER.debug(month_file_match)
        #     try:
        #         for data_type, dir_list, path_list in [
        #                 ('et0', et0_dir_list, et0_path_list),
        #                 ('Precip', precip_dir_list, precip_path_list)]:
        #             LOGGER.debug(dir_list)
        #             file_list = [
        #                 month_file_path for month_file_path in dir_list
        #                 if month_file_match.match(month_file_path)]
        #             if len(file_list) == 0:
        #                 raise ValueError(
        #                     "No %s found for month %d" % (data_type, month_index))
        #             if len(file_list) > 1:
        #                 raise ValueError(
        #                     "Ambiguous set of files found for month %d: %s" %
        #                     (month_index, file_list))
        #             path_list.append(file_list[0])
        #     except ValueError:
        #         path_list = sorted(dir_list)

        input_align_list = (
            precip_path_list + [args['soil_group_path']] + et0_path_list +
            input_align_list)
        output_align_list = (
            file_registry['precip_path_aligned_list'] +
            [file_registry['soil_group_aligned_path']] +
            file_registry['et0_path_aligned_list'] + output_align_list)

        aligned_key_list.append(
            ('soil_group_path', 'soil_group_aligned_path'))

    align_index = len(input_align_list) - 1  # this aligns with the DEM
    if args['user_defined_local_recharge']:
        input_align_list.append(args['l_path'])
        output_align_list.append(file_registry['l_aligned_path'])
    elif args['user_defined_climate_zones']:
        input_align_list.append(args['climate_zone_raster_path'])
        output_align_list.append(
            file_registry['cz_aligned_raster_path'])
    interpolate_list = ['near'] * len(input_align_list)

    reclassify_n_events_task_list = []

    if args['user_defined_rain_events_dir']:
        n_events_path_list = sorted(os.listdir(args['user_defined_rain_events_dir']))
        input_align_list.extend(n_events_path_list)
        file_registry['n_events_path_list'] = n_events_path_list
        output_align_list.extend(n_events_path_list)

    if 'prealigned' not in args or not args['prealigned']:
        LOGGER.debug(f'**************** about to align:\n\n\t{input_align_list}\n\n\t{output_align_list}\n\n\t{interpolate_list}')
        align_task = task_graph.add_task(
            func=geoprocessing.align_and_resize_raster_stack,
            args=(
                input_align_list, output_align_list, interpolate_list,
                pixel_size, 'intersection'),
            kwargs={
                'base_vector_path_list': (args['aoi_path'],),
                'raster_align_index': align_index},
            target_path_list=output_align_list,
            task_name='align rasters')
    else:
        # the aligned stuff is the base stuff
        for base_key, aligned_key in aligned_key_list:
            file_registry[aligned_key] = args[base_key]
        file_registry['precip_path_aligned_list'] = precip_path_list
        file_registry['et0_path_aligned_list'] = et0_path_list
        if args['user_defined_rain_events_dir']:
            n_events_path_list = sorted(
                str(path) for path in Path(args['user_defined_rain_events_dir']).glob('*.tif')
            )
            input_align_list.extend(n_events_path_list)
            file_registry['n_events_path_list'] = n_events_path_list
        align_task = task_graph.add_task()
        reclassify_n_events_task_list = [align_task]*12  # it's empty though on purpose

    raster_info = geoprocessing.get_raster_info(
        file_registry['dem_aligned_path'])
    biophysical_factor_dict = {}
    for biophysical_key in _TABLE_BASED_BIOPHYSICAL_FACTORS:
        reclassify_clip_task = task_graph.add_task(
            func=_reclassify_or_clip,
            args=(
                biophysical_key, args.get('biophysical_table_path', None),
                file_registry.get('lulc_aligned_path', None), args,
                raster_info, file_registry),
            store_result=True,
            task_name=f'reclassify or warp {biophysical_key}')
        biophysical_factor_dict[biophysical_key] = reclassify_clip_task
    # load all the results
    for key, task in list(biophysical_factor_dict.items()):
        biophysical_factor_dict[key] = task.get()

    if 'single_outlet' in args and args['single_outlet'] is True:
        get_drain_sink_pixel_task = task_graph.add_task(
            func=geoprocessing.routing.detect_lowest_drain_and_sink,
            args=((file_registry['dem_aligned_path'], 1),),
            store_result=True,
            dependent_task_list=[align_task],
            task_name=f"get drain/sink pixel for {file_registry['dem_aligned_path']}")

        edge_pixel, edge_height, pit_pixel, pit_height = (
            get_drain_sink_pixel_task.get())

        if pit_height < edge_height - 20:
            # if the pit is 20 m lower than edge it's probably a big sink
            single_outlet_tuple = pit_pixel
        else:
            single_outlet_tuple = edge_pixel
    else:
        single_outlet_tuple = None

    fill_pit_task = task_graph.add_task(
        func=routing.fill_pits,
        args=(
            (file_registry['dem_aligned_path'], 1),
            file_registry['dem_pit_filled_path']),
        kwargs={
            'working_dir': cache_dir,
            'single_outlet_tuple': single_outlet_tuple,
            'max_pixel_fill_count': (
                -1 if 'max_pixel_fill_count' not in args
                else int(args['max_pixel_fill_count'])),
            },
        target_path_list=[file_registry['dem_pit_filled_path']],
        dependent_task_list=[align_task],
        task_name='fill dem pits')

    flow_dir_task = task_graph.add_task(
        func=routing.flow_dir_mfd,
        args=(
            (file_registry['dem_pit_filled_path'], 1),
            file_registry['flow_dir_mfd_path']),
        kwargs={'working_dir': cache_dir},
        target_path_list=[file_registry['flow_dir_mfd_path']],
        dependent_task_list=[fill_pit_task],
        task_name='flow dir mfd')

    flow_accum_task = task_graph.add_task(
        func=routing.flow_accumulation_mfd,
        args=(
            (file_registry['flow_dir_mfd_path'], 1),
            file_registry['flow_accum_path']),
        target_path_list=[file_registry['flow_accum_path']],
        dependent_task_list=[flow_dir_task],
        task_name='flow accum task')

    stream_threshold_task = task_graph.add_task(
        func=routing.extract_streams_mfd,
        args=(
            (file_registry['flow_accum_path'], 1),
            (file_registry['flow_dir_mfd_path'], 1),
            threshold_flow_accumulation,
            file_registry['stream_path']),
        target_path_list=[file_registry['stream_path']],
        dependent_task_list=[flow_accum_task],
        task_name='stream threshold')

    LOGGER.info('quick flow')
    if args['user_defined_local_recharge']:
        file_registry['l_path'] = file_registry['l_aligned_path']

        l_avail_task = task_graph.add_task(
            func=_calculate_l_avail,
            args=(
                file_registry['l_path'], gamma,
                file_registry['l_avail_path']),
            target_path_list=[file_registry['l_avail_path']],
            dependent_task_list=[align_task],
            task_name='l avail task')
    else:
        # user didn't predefine local recharge or montly events so calculate it
        if 'user_defined_rain_events_dir' not in args or not args['user_defined_rain_events_dir']:
            LOGGER.info('loading number of monthly events')
            reclass_error_details = {
                'raster_name': 'Climate Zone', 'column_name': 'cz_id',
                'table_name': 'Climate Zone'}
            # TODO: don't do if already defined
            for month_id in range(N_MONTHS):
                if args['user_defined_climate_zones']:
                    cz_rain_events_lookup = (
                        utils.build_lookup_from_csv(
                            args['climate_zone_table_path'], 'cz_id'))
                    month_label = MONTH_ID_TO_LABEL[month_id]
                    climate_zone_rain_events_month = dict([
                        (cz_id, cz_rain_events_lookup[cz_id][month_label]) for
                        cz_id in cz_rain_events_lookup])
                    n_events_nodata = -1
                    n_events_task = task_graph.add_task(
                        func=geoprocessing.reclassify_raster,
                        args=(
                            (file_registry['cz_aligned_raster_path'], 1),
                            climate_zone_rain_events_month,
                            file_registry['n_events_path_list'][month_id],
                            gdal.GDT_Float32, n_events_nodata,
                            reclass_error_details),
                        target_path_list=[
                            file_registry['n_events_path_list'][month_id]],
                        dependent_task_list=[align_task],
                        task_name='n_events for month %d' % month_id)
                    reclassify_n_events_task_list.append(n_events_task)
                else:
                    # rain_events_lookup defined near entry point of execute
                    n_events = rain_events_lookup[month_id+1]['events']
                    n_events_task = task_graph.add_task(
                        func=geoprocessing.new_raster_from_base,
                        args=(
                            file_registry['dem_aligned_path'],
                            file_registry['n_events_path_list'][month_id],
                            gdal.GDT_Float32, [TARGET_NODATA]),
                        kwargs={'fill_value_list': (n_events,)},
                        target_path_list=[
                            file_registry['n_events_path_list'][month_id]],
                        dependent_task_list=[align_task],
                        task_name=(
                            'n_events as a constant raster month %d' % month_id))
                    reclassify_n_events_task_list.append(n_events_task)

        curve_number_task = task_graph.add_task(
            func=_calculate_curve_number_raster,
            args=(
                file_registry['soil_group_aligned_path'],
                biophysical_factor_dict, file_registry['cn_path']),
            target_path_list=[file_registry['cn_path']],
            dependent_task_list=[align_task],
            task_name='calculate curve number')

        si_task = task_graph.add_task(
            func=_calculate_si_raster,
            args=(
                file_registry['cn_path'], file_registry['stream_path'],
                file_registry['si_path']),
            target_path_list=[file_registry['si_path']],
            dependent_task_list=[curve_number_task, stream_threshold_task],
            task_name='calculate Si raster')

        quick_flow_task_list = []
        # LOGGER.warning(
        #     f"for workspace {args['workspace_dir']}\n"
        #     f"resulting rain events path list: {file_registry['n_events_path_list']}\n"
        #     f"args['user_defined_rain_events_path']: {args['user_defined_rain_events_path']}\n"
        #     f"prealinged {args['prealigned']}\n")
        for month_index in range(N_MONTHS):
            LOGGER.info('calculate quick flow for month %d', month_index+1)
            monthly_quick_flow_task = task_graph.add_task(
                func=_calculate_monthly_quick_flow,
                args=(
                    file_registry['precip_path_aligned_list'][month_index],
                    file_registry['cn_path'],
                    file_registry['n_events_path_list'][month_index],
                    file_registry['stream_path'],
                    file_registry['si_path'],
                    file_registry['qfm_path_list'][month_index]),
                target_path_list=[
                    file_registry['qfm_path_list'][month_index]],
                dependent_task_list=[
                    align_task, reclassify_n_events_task_list[month_index],
                    si_task, stream_threshold_task],
                task_name='calculate quick flow for month %d' % (
                    month_index+1))
            quick_flow_task_list.append(monthly_quick_flow_task)

        qf_task = task_graph.add_task(
            func=_calculate_annual_qfi,
            args=(file_registry['qfm_path_list'], file_registry['qf_path']),
            target_path_list=[file_registry['qf_path']],
            dependent_task_list=quick_flow_task_list,
            task_name='calculate QFi')

        LOGGER.info('calculate local recharge')
        kc_task_list = []
        reclass_error_details = {
            'raster_name': 'LULC', 'column_name': 'lucode',
            'table_name': 'Biophysical'}

        # biophysical_factor_dict[kc_n] contains the kc rasters

        # call through to a cython function that does the necessary routing
        # between AET and L.sum.avail in equation [7], [4], and [3]
        kc_path_list = [
            biophysical_factor_dict[f'kc_{index}']
            for index in range(1, 13)]
        calculate_local_recharge_task = task_graph.add_task(
            func=seasonal_water_yield_core.calculate_local_recharge,
            args=(
                file_registry['precip_path_aligned_list'],
                file_registry['et0_path_aligned_list'],
                file_registry['qfm_path_list'],
                file_registry['flow_dir_mfd_path'],
                kc_path_list,
                alpha_month_map,
                beta_i, gamma, file_registry['stream_path'],
                file_registry['l_path'],
                file_registry['l_avail_path'],
                file_registry['l_sum_avail_path'],
                file_registry['aet_path'],
                file_registry['annual_precip_path']),
            target_path_list=[
                file_registry['l_path'],
                file_registry['l_avail_path'],
                file_registry['l_sum_avail_path'],
                file_registry['aet_path'],
                file_registry['annual_precip_path'],
            ],
            dependent_task_list=[
                align_task, flow_dir_task, stream_threshold_task,
                fill_pit_task, qf_task] + quick_flow_task_list,
            task_name='calculate local recharge')

    # calculate Qb as the sum of local_recharge_avail over the AOI, Eq [9]
    if args['user_defined_local_recharge']:
        vri_dependent_task_list = [l_avail_task]
    else:
        vri_dependent_task_list = [calculate_local_recharge_task]

    vri_task = task_graph.add_task(
        func=_calculate_vri,
        args=(file_registry['l_path'], file_registry['vri_path']),
        target_path_list=[file_registry['vri_path']],
        dependent_task_list=vri_dependent_task_list,
        task_name='calculate vri')

    aggregate_recharge_task = task_graph.add_task(
        func=_aggregate_recharge,
        args=(
            args['aoi_path'], file_registry['l_path'],
            file_registry['vri_path'],
            file_registry['aggregate_vector_path']),
        target_path_list=[file_registry['aggregate_vector_path']],
        dependent_task_list=[vri_task],
        task_name='aggregate recharge')

    LOGGER.info('calculate L_sum')  # Eq. [12]
    l_sum_task = task_graph.add_task(
        func=routing.flow_accumulation_mfd,
        args=(
            (file_registry['flow_dir_mfd_path'], 1),
            file_registry['l_sum_path']),
        kwargs={'weight_raster_path_band': (file_registry['l_path'], 1)},
        target_path_list=[file_registry['l_sum_path']],
        dependent_task_list=vri_dependent_task_list + [
            fill_pit_task, flow_dir_task, stream_threshold_task],
        task_name='calculate l sum')

    if args['user_defined_local_recharge']:
        b_sum_dependent_task_list = [l_avail_task]
    else:
        b_sum_dependent_task_list = [calculate_local_recharge_task]

    b_sum_task = task_graph.add_task(
        func=seasonal_water_yield_core.route_baseflow_sum,
        args=(
            file_registry['flow_dir_mfd_path'],
            file_registry['l_path'],
            file_registry['l_avail_path'],
            file_registry['l_sum_path'],
            file_registry['stream_path'],
            file_registry['b_path'],
            file_registry['b_sum_path']),

        target_path_list=[
            file_registry['b_sum_path'], file_registry['b_path']],
        dependent_task_list=b_sum_dependent_task_list + [l_sum_task],
        task_name='calculate B_sum')

    task_graph.close()
    task_graph.join()

    LOGGER.info('  (\\w/)  SWY Complete!')
    LOGGER.info('  (..  \\ ')
    LOGGER.info(' _/  )  \\______')
    LOGGER.info('(oo /\'\\        )`,')
    LOGGER.info(' `--\' (v  __( / ||')
    LOGGER.info('       |||  ||| ||')
    LOGGER.info('      //_| //_|')


def _calculate_vri(l_path, target_vri_path):
    """Calculate VRI as li_array / qb_sum.

    Args:
        l_path (str): path to L raster.
        target_vri_path (str): path to output Vri raster.

    Returns:
        None.

    """
    qb_sum = 0.0
    qb_valid_count = 0
    l_nodata = geoprocessing.get_raster_info(l_path)['nodata'][0]

    for _, block in geoprocessing.iterblocks((l_path, 1)):
        valid_mask = (
            ~numpy.isclose(block, l_nodata) &
            (~numpy.isinf(block)))
        qb_sum += numpy.sum(block[valid_mask])
        qb_valid_count += numpy.count_nonzero(valid_mask)
    li_nodata = geoprocessing.get_raster_info(l_path)['nodata'][0]

    def vri_op(li_array):
        """Calculate vri index [Eq 10]."""
        result = numpy.empty_like(li_array)
        result[:] = li_nodata
        if qb_sum > 0:
            valid_mask = ~numpy.isclose(li_array, li_nodata)
            try:
                result[valid_mask] = li_array[valid_mask] / qb_sum
            except RuntimeWarning:
                LOGGER.exception(qb_sum)
                raise
        return result
    geoprocessing.raster_calculator(
        [(l_path, 1)], vri_op, target_vri_path, gdal.GDT_Float32,
        li_nodata)


def _calculate_annual_qfi(qfm_path_list, target_qf_path):
    """Calculate annual quickflow.

    Args:
        qfm_path_list (list): list of monthly quickflow raster paths.
        target_qf_path (str): path to target annual quickflow raster.

    Returns:
        None.

    """
    qf_nodata = -1

    def qfi_sum_op(*qf_values):
        """Sum the monthly qfis."""
        qf_sum = numpy.zeros(qf_values[0].shape)
        running_valid_mask = numpy.zeros(qf_sum.shape, dtype=bool)
        for local_qf_array in qf_values:
            local_valid_mask = ~numpy.isclose(local_qf_array, qf_nodata)
            qf_sum[local_valid_mask] += local_qf_array[local_valid_mask]
            running_valid_mask |= local_valid_mask
        qf_sum[~running_valid_mask] = qf_nodata
        return qf_sum

    geoprocessing.raster_calculator(
        [(path, 1) for path in qfm_path_list],
        qfi_sum_op, target_qf_path, gdal.GDT_Float32, qf_nodata)


def _calculate_monthly_quick_flow(
        precip_path, cn_path, n_events_raster_path,
        stream_path, si_path, qf_monthly_path):
    """Calculate quick flow for a month.

    Args:
        precip_path (string): path to file that correspond to monthly
            precipitation
        cn_path (string): path to curve number raster
        n_events_raster_path (string): a path to a raster where each pixel
            indicates the number of rain events.
        stream_path (string): path to stream mask raster where 1 indicates a
            stream pixel, 0 is a non-stream but otherwise valid area from the
            original DEM, and nodata indicates areas outside the valid DEM.
        si_path (string): path to raster that has potential maximum retention
        qf_monthly_path_list (list of string): list of paths to output monthly
            rasters.

    Returns:
        None
    """
    si_nodata = geoprocessing.get_raster_info(si_path)['nodata'][0]

    qf_nodata = -1
    p_nodata = geoprocessing.get_raster_info(precip_path)['nodata'][0]
    n_events_nodata = geoprocessing.get_raster_info(
        n_events_raster_path)['nodata'][0]
    stream_nodata = geoprocessing.get_raster_info(stream_path)['nodata'][0]

    def qf_debug_op(p_im, s_i, n_events, stream_array):
        """Calculate quick flow as in Eq [1] in user's guide.

        Args:
            p_im (numpy.array): precipitation at pixel i on month m
            s_i (numpy.array): factor that is 1000/CN_i - 10
                (Equation 1b from user's guide)
            n_events (numpy.array): number of rain events on the pixel
            stream_mask (numpy.array): 1 if stream, otherwise not a stream
                pixel.

        Returns:
            quick flow (numpy.array)

        """
        # s_i is an intermediate output which will always have a defined
        # nodata value
        valid_mask = ((p_im != 0.0) &
                      (stream_array != 1) &
                      (n_events > 0) &
                      ~numpy.isclose(s_i, si_nodata))
        if p_nodata is not None:
            valid_mask &= ~numpy.isclose(p_im, p_nodata)
        valid_mask &= numpy.isfinite(p_im)
        if n_events_nodata is not None:
            valid_mask &= ~numpy.isclose(n_events, n_events_nodata)
        valid_mask &= numpy.isfinite(n_events)
        # stream_nodata is the only input that carry over nodata values from
        # the aligned DEM.
        if stream_nodata is not None:
            valid_mask &= ~numpy.isclose(
                stream_array, stream_nodata)
        valid_mask &= numpy.isfinite(stream_array)

        valid_n_events = n_events[valid_mask]
        valid_si = s_i[valid_mask]

        # a_im is the mean rain depth on a rainy day at pixel i on month m
        # the 25.4 converts inches to mm since Si is in inches
        a_im = numpy.empty(valid_n_events.shape)
        a_im = p_im[valid_mask] / (valid_n_events * 25.4)
        qf_im = numpy.full(p_im.shape, qf_nodata)

        # Precompute the last two terms in quickflow so we can handle a
        # numerical instability when s_i is large and/or a_im is small
        # on large valid_si/a_im this number will be zero and the latter
        # exponent will also be zero because of a divide by zero. rather than
        # raise that numerical warning, just handle it manually
        E1 = scipy.special.expn(1, valid_si / a_im)
        E1[valid_si == 0] = 0
        nonzero_e1_mask = E1 != 0
        exp_result = numpy.zeros(valid_si.shape)
        exp_result[nonzero_e1_mask] = numpy.exp(
            (0.8 * valid_si[nonzero_e1_mask]) / a_im[nonzero_e1_mask] +
            numpy.log(E1[nonzero_e1_mask]))
        qf_im[valid_mask] = a_im
        return qf_im

        # qf_im is the quickflow at pixel i on month m Eq. [1]
        try:
            qf_im[valid_mask] = (25.4 * valid_n_events * (
                (a_im - valid_si) * numpy.exp(-0.2 * valid_si / a_im) +
                valid_si ** 2 / a_im * exp_result))
        except RuntimeWarning:
            LOGGER.exception(
                f'************error on quickflow:\n'
                f'(25.4 * {valid_n_events} * ('
                f'({a_im} - {valid_si}) * numpy.exp(-0.2 * {valid_si} / {a_im}) +'
                f'{valid_si} ** 2 / {a_im} * {exp_result}))')
            LOGGER.exception(f'{valid_si[valid_si < 0]} {a_im[a_im <= 0]}')
            div_result = valid_si / a_im
            invalid_result = ~numpy.isfinite(div_result)
            LOGGER.exception(f'{div_result} {div_result[invalid_result]} ({valid_si[invalid_result]}) / ({a_im[invalid_result]}')
            for array, array_id in [(valid_n_events, 'valid_n_events'), (a_im, 'a_im'), (valid_si, 'valid_si'), (exp_result, 'exp_result')]:
                invalid_values = ~numpy.isfinite(array)
                if any(invalid_values):
                    LOGGER.error(
                        f'{array_id}: {array[~numpy.isfinite(array[invalid_values])]}')
                raise

        # if precip is 0, then QF should be zero
        qf_im[(p_im == 0) | (n_events == 0)] = 0.0
        # if we're on a stream, set quickflow to the precipitation
        valid_stream_precip_mask = (stream_array == 1) & numpy.isfinite(p_im)
        if p_nodata is not None:
            valid_stream_precip_mask &= ~numpy.isclose(
                p_im, p_nodata)
        qf_im[valid_stream_precip_mask] = p_im[valid_stream_precip_mask]

        # this handles some user cases where they don't have data defined on
        # their landcover raster. It otherwise crashes later with some NaNs.
        # more intermediate outputs with nodata values guaranteed to be defined
        qf_im[numpy.isclose(qf_im, qf_nodata) &
              ~numpy.isclose(stream_array, stream_nodata)] = 0.0
        qf_im[~valid_mask] = qf_nodata
        return qf_im

    def qf_op(p_im, s_i, n_events, stream_array):
        """Calculate quick flow as in Eq [1] in user's guide.

        Args:
            p_im (numpy.array): precipitation at pixel i on month m
            s_i (numpy.array): factor that is 1000/CN_i - 10
                (Equation 1b from user's guide)
            n_events (numpy.array): number of rain events on the pixel
            stream_mask (numpy.array): 1 if stream, otherwise not a stream
                pixel.

        Returns:
            quick flow (numpy.array)

        """
        # s_i is an intermediate output which will always have a defined
        # nodata value
        valid_mask = ((p_im != 0.0) &
                      (stream_array != 1) &
                      (n_events > 0) &
                      ~numpy.isclose(s_i, si_nodata))
        if p_nodata is not None:
            valid_mask &= ~numpy.isclose(p_im, p_nodata)
        valid_mask &= numpy.isfinite(p_im)
        if n_events_nodata is not None:
            valid_mask &= ~numpy.isclose(n_events, n_events_nodata)
        valid_mask &= numpy.isfinite(n_events)
        # stream_nodata is the only input that carry over nodata values from
        # the aligned DEM.
        if stream_nodata is not None:
            valid_mask &= ~numpy.isclose(
                stream_array, stream_nodata)
        valid_mask &= numpy.isfinite(stream_array)

        valid_n_events = n_events[valid_mask]
        valid_si = s_i[valid_mask]

        # a_im is the mean rain depth on a rainy day at pixel i on month m
        # the 25.4 converts inches to mm since Si is in inches
        a_im = numpy.empty(valid_n_events.shape)
        a_im = p_im[valid_mask] / (valid_n_events * 25.4)
        qf_im = numpy.full(p_im.shape, qf_nodata)

        # Precompute the last two terms in quickflow so we can handle a
        # numerical instability when s_i is large and/or a_im is small
        # on large valid_si/a_im this number will be zero and the latter
        # exponent will also be zero because of a divide by zero. rather than
        # raise that numerical warning, just handle it manually
        E1 = scipy.special.expn(1, valid_si / a_im)
        E1[valid_si == 0] = 0
        nonzero_e1_mask = E1 != 0
        exp_result = numpy.zeros(valid_si.shape)
        exp_result[nonzero_e1_mask] = numpy.exp(
            (0.8 * valid_si[nonzero_e1_mask]) / a_im[nonzero_e1_mask] +
            numpy.log(E1[nonzero_e1_mask]))

        # qf_im is the quickflow at pixel i on month m Eq. [1]
        try:
            qf_im[valid_mask] = (25.4 * valid_n_events * (
                (a_im - valid_si) * numpy.exp(-0.2 * valid_si / a_im) +
                valid_si ** 2 / a_im * exp_result))
        except RuntimeWarning:
            LOGGER.exception(
                f'************error on quickflow:\n'
                f'(25.4 * {valid_n_events} * ('
                f'({a_im} - {valid_si}) * numpy.exp(-0.2 * {valid_si} / {a_im}) +'
                f'{valid_si} ** 2 / {a_im} * {exp_result}))')
            LOGGER.exception(f'{valid_si[valid_si < 0]} {a_im[a_im <= 0]}')
            div_result = valid_si / a_im
            invalid_result = ~numpy.isfinite(div_result)
            LOGGER.exception(f'{div_result} {div_result[invalid_result]} ({valid_si[invalid_result]}) / ({a_im[invalid_result]}')
            for array, array_id in [(valid_n_events, 'valid_n_events'), (a_im, 'a_im'), (valid_si, 'valid_si'), (exp_result, 'exp_result')]:
                invalid_values = ~numpy.isfinite(array)
                if any(invalid_values):
                    LOGGER.error(
                        f'{array_id}: {array[~numpy.isfinite(array[invalid_values])]}')
                raise

        # if precip is 0, then QF should be zero
        qf_im[(p_im == 0) | (n_events == 0)] = 0.0
        # if we're on a stream, set quickflow to the precipitation
        valid_stream_precip_mask = (stream_array == 1) & numpy.isfinite(p_im)
        if p_nodata is not None:
            valid_stream_precip_mask &= ~numpy.isclose(
                p_im, p_nodata)
        qf_im[valid_stream_precip_mask] = p_im[valid_stream_precip_mask]

        # this handles some user cases where they don't have data defined on
        # their landcover raster. It otherwise crashes later with some NaNs.
        # more intermediate outputs with nodata values guaranteed to be defined
        qf_im[numpy.isclose(qf_im, qf_nodata) &
              ~numpy.isclose(stream_array, stream_nodata)] = 0.0
        qf_im[~valid_mask] = qf_nodata
        return qf_im

    geoprocessing.raster_calculator(
        [(path, 1) for path in [
            precip_path, si_path, n_events_raster_path, stream_path]], qf_debug_op,
        '%s_aim%s' % os.path.splitext(qf_monthly_path), gdal.GDT_Float32, qf_nodata)

    geoprocessing.raster_calculator(
        [(path, 1) for path in [
            precip_path, si_path, n_events_raster_path, stream_path]], qf_op,
        qf_monthly_path, gdal.GDT_Float32, qf_nodata)


def _calculate_curve_number_raster(
        soil_group_path, biophysical_factor_dict, cn_path):
    """Calculate the CN raster from the landcover and soil group rasters.

    Args:
        soil_group_path (string): path to raster indicating soil group where
            pixel values are in [1,2,3,4]
        biophysical_factor_dict (dict): dictionary that indexes the paths for
            'cn_a', 'cn_b', 'cn_c', 'cn_d', rasters.
        cn_path (string): path to output curve number raster to be output.

    Returns:
        None
    """
    # curve numbers are always positive so -1 a good nodata choice
    cn_nodata = -1

    def cn_op(cn_a, cn_b, cn_c, cn_d, soil_group_array):
        """Map soil type to a curve number."""
        cn_result = numpy.empty(soil_group_array.shape)
        cn_result[:] = cn_nodata
        cn_lookup = {
            1: cn_a,
            2: cn_b,
            3: cn_c,
            4: cn_d
        }
        for soil_group_id in numpy.unique(soil_group_array):
            if not (1 <= soil_group_id <=4):
                continue
            current_soil_mask = (soil_group_array == soil_group_id)
            cn_result[current_soil_mask] = (
                cn_lookup[soil_group_id][current_soil_mask])
        return cn_result

    geoprocessing.raster_calculator(
        [(biophysical_factor_dict[index], 1)
         for index in ['cn_a', 'cn_b', 'cn_c', 'cn_d']] + [
         (soil_group_path, 1)], cn_op, cn_path, gdal.GDT_Float32, cn_nodata)


def _calculate_si_raster(cn_path, stream_path, si_path):
    """Calculate the S factor of the quickflow equation [1].

    Args:
        cn_path (string): path to curve number raster
        stream_path (string): path to a stream raster (0, 1)
        si_path (string): path to output s_i raster

    Returns:
        None
    """
    si_nodata = -1
    cn_nodata = geoprocessing.get_raster_info(cn_path)['nodata'][0]

    def si_op(ci_factor, stream_mask):
        """Calculate si factor."""
        valid_mask = (
            ~numpy.isclose(ci_factor, cn_nodata) &
            (ci_factor > 0))
        si_array = numpy.empty(ci_factor.shape)
        si_array[:] = si_nodata
        # multiply by the stream mask != 1 so we get 0s on the stream and
        # unaffected results everywhere else
        si_array[valid_mask] = (
            (1000.0 / ci_factor[valid_mask] - 10) * (
                stream_mask[valid_mask] != 1))
        si_array[si_array < 0] = -1  # guard against a negative something
        return si_array

    geoprocessing.raster_calculator(
        [(cn_path, 1), (stream_path, 1)], si_op, si_path, gdal.GDT_Float32,
        si_nodata)


def _aggregate_recharge(
        aoi_path, l_path, vri_path, aggregate_vector_path):
    """Aggregate recharge values for the provided watersheds/AOIs.

    Generates a new shapefile that's a copy of 'aoi_path' in sum values from L
    and Vri.

    Args:
        aoi_path (string): path to shapefile that will be used to
            aggregate rasters
        l_path (string): path to (L) local recharge raster
        vri_path (string): path to Vri raster
        aggregate_vector_path (string): path to shapefile that will be created
            by this function as the aggregating output.  will contain fields
            'l_sum' and 'vri_sum' per original feature in `aoi_path`.  If this
            file exists on disk prior to the call it is overwritten with
            the result of this call.

    Returns:
        None
    """
    if os.path.exists(aggregate_vector_path):
        LOGGER.warning(
            '%s exists, deleting and writing new output',
            aggregate_vector_path)
        os.remove(aggregate_vector_path)

    original_aoi_vector = gdal.OpenEx(aoi_path, gdal.OF_VECTOR)

    driver = gdal.GetDriverByName('ESRI Shapefile')
    driver.CreateCopy(aggregate_vector_path, original_aoi_vector)
    gdal.Dataset.__swig_destroy__(original_aoi_vector)
    original_aoi_vector = None
    aggregate_vector = gdal.OpenEx(aggregate_vector_path, 1)
    aggregate_layer = aggregate_vector.GetLayer()

    for raster_path, aggregate_field_id, op_type in [
            (l_path, 'qb', 'mean'), (vri_path, 'vri_sum', 'sum')]:

        # aggregate carbon stocks by the new ID field
        aggregate_stats = geoprocessing.zonal_statistics(
            (raster_path, 1), aggregate_vector_path)

        aggregate_field = ogr.FieldDefn(aggregate_field_id, ogr.OFTReal)
        aggregate_field.SetWidth(24)
        aggregate_field.SetPrecision(11)
        aggregate_layer.CreateField(aggregate_field)

        aggregate_layer.ResetReading()
        for poly_index, poly_feat in enumerate(aggregate_layer):
            if op_type == 'mean':
                pixel_count = aggregate_stats[poly_index]['count']
                if pixel_count != 0:
                    value = (aggregate_stats[poly_index]['sum'] / pixel_count)
                else:
                    LOGGER.warning(
                        "no coverage for polygon %s", ', '.join(
                            [str(poly_feat.GetField(_)) for _ in range(
                                poly_feat.GetFieldCount())]))
                    value = 0.0
            elif op_type == 'sum':
                value = aggregate_stats[poly_index]['sum']
            poly_feat.SetField(aggregate_field_id, float(value))
            aggregate_layer.SetFeature(poly_feat)

    aggregate_layer.SyncToDisk()
    aggregate_layer = None
    gdal.Dataset.__swig_destroy__(aggregate_vector)
    aggregate_vector = None


def _calculate_l_avail(l_path, gamma, target_l_avail_path):
    """l avail = l * gamma."""
    li_nodata = geoprocessing.get_raster_info(l_path)['nodata'][0]

    def l_avail_op(l_array):
        """Calculate equation [8] L_avail = min(gamma*L, L)."""
        result = numpy.empty(l_array.shape)
        result[:] = li_nodata
        valid_mask = ~numpy.isclose(l_array, li_nodata)
        result[valid_mask] = numpy.min(numpy.stack(
            (gamma*l_array[valid_mask], l_array[valid_mask])), axis=0)
        return result

    geoprocessing.raster_calculator(
        [(l_path, 1)], l_avail_op, target_l_avail_path, gdal.GDT_Float32,
        li_nodata)
