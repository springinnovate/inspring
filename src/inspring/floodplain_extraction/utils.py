"""Floodplain extraction utilities."""
import logging
import os
import numpy

from ecoshard import geoprocessing
from ecoshard import taskgraph
from ecoshard.utils import scrub_invalid_values

LOGGER = logging.getLogger(__name__)


def prep_floodplain_data(
    working_dir, dem_path, min_flow_accum_threshold, target_stream_vector_path,
        target_watershed_boundary_vector_path):
    """Prep the DEM and extract stream and subwatershed vectors.

    Args:
        working_dir (str): existing path that can be used to create prepared
            floodplain files
        dem_path (str): path to dem
        min_flow_accum_threshold (int): minimum flow accumulation threshold to
            start defining streams.
        target_stream_vector_path (str): path to created stream vector
        target_watershed_boundary_vector_path (str) path to created
            subwatershed vector

    Returns:
        None
    """
    dem_info = geoprocessing.get_raster_info(dem_path)
    dem_type = dem_info['numpy_type']
    nodata = dem_info['nodata'][0]
    if isinstance(dem_type, float):
        new_nodata = float(numpy.finfo(dem_type).min)
    else:
        new_nodata = int(numpy.iinfo(dem_type).min)

    scrubbed_dem_path = os.path.join(working_dir, 'scrubbed_dem.tif')
    task_graph = taskgraph.TaskGraph(working_dir, -1)

    scrub_dem_task = task_graph.add_task(
        func=geoprocessing.raster_calculator,
        args=(
            [(dem_path, 1), (nodata, 'raw'), (new_nodata, 'raw')],
            scrub_invalid_values, scrubbed_dem_path,
            dem_info['datatype'], new_nodata),
        target_path_list=[scrubbed_dem_path],
        task_name='scrub dem')

    LOGGER.info('fill pits')
    filled_pits_path = os.path.join(working_dir, 'filled_pits_dem.tif')
    fill_pits_task = task_graph.add_task(
        func=geoprocessing.routing.fill_pits,
        args=((scrubbed_dem_path, 1), filled_pits_path),
        kwargs={'max_pixel_fill_count': 1000000},
        target_path_list=[filled_pits_path],
        dependent_task_list=[scrub_dem_task],
        task_name='fill pits')

    LOGGER.info('flow dir d8')
    flow_dir_d8_path = os.path.join(working_dir, 'flow_dir_d8.tif')
    flow_dir_task = task_graph.add_task(
        func=geoprocessing.routing.flow_dir_d8,
        args=((filled_pits_path, 1), flow_dir_d8_path),
        kwargs={'working_dir': working_dir},
        target_path_list=[flow_dir_d8_path],
        dependent_task_list=[fill_pits_task],
        task_name='flow dir d8')

    LOGGER.info('flow accum d8')
    flow_accum_d8_path = os.path.join(working_dir, 'flow_accum_d8.tif')
    flow_accum_task = task_graph.add_task(
        func=geoprocessing.routing.flow_accumulation_d8,
        args=((flow_dir_d8_path, 1), flow_accum_d8_path),
        target_path_list=[flow_accum_d8_path],
        dependent_task_list=[flow_dir_task],
        task_name='flow accum d8')

    extract_stream_task = task_graph.add_task(
        func=geoprocessing.routing.extract_strahler_streams_d8,
        args=(
            (flow_dir_d8_path, 1), (flow_accum_d8_path, 1),
            (filled_pits_path, 1), target_stream_vector_path),
        kwargs={
            'min_flow_accum_threshold': min_flow_accum_threshold,
            'river_order': 7},
        target_path_list=[target_stream_vector_path],
        ignore_path_list=[target_stream_vector_path],
        dependent_task_list=[flow_accum_task],
        task_name='stream extraction')

    if not os.path.exists(target_watershed_boundary_vector_path):
        calculate_watershed_boundary_task = task_graph.add_task(
            func=geoprocessing.routing.calculate_subwatershed_boundary,
            args=(
                (flow_dir_d8_path, 1), target_stream_vector_path,
                target_watershed_boundary_vector_path),
            kwargs={'outlet_at_confluence': False},
            target_path_list=[target_watershed_boundary_vector_path],
            ignore_path_list=[target_watershed_boundary_vector_path],
            dependent_task_list=[extract_stream_task],
            task_name='watershed boundary')
        calculate_watershed_boundary_task.join()
