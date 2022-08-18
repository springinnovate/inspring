"""Earth Observation Driven Pollinator service model for inspring."""
import multiprocessing
import logging
import os

from ecoshard import geoprocessing
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from inspring.floodplain_extraction import floodplain_extraction
import numpy
from ecoshard import taskgraph

LOGGER = logging.getLogger(__name__)



def execute(args):
    """Flood model.

    Args:
        args['workspace_dir'] (str): a path to the output workspace folder.
            Will overwrite any files that exist if the path already exists.
        args['results_suffix'] (str): string appended to each output
            file path.
        args['dem_path'] (str): path to DEM raster.

    Returns:
        None
    """
    intermediate_dir = os.path.join(args['workspace_dir'], 'intermediate')
    filled_dem_path = os.path.join(intermediate_dir, 'filled_dem.tif')
    flow_accum_path = os.path.join(intermediate_dir, 'flow_accum_d8.tif')
    flow_dir_path = os.path.join(intermediate_dir, 'flow_dir_d8.tif')
    stream_path = os.path.join(intermediate_dir, 'streams.gpkg')
    subwatershed_path = os.path.join(intermediate_dir, 'subwatershed.gpkg')

    for dir_path in [args['workspace_dir'], intermediate_dir]:
        os.makedirs(dir_path, exist_ok=True)

    task_graph = taskgraph.TaskGraph(
        args['workspace_dir'], multiprocessing.cpu_count(), 15.0)

    floodplain_extraction(
        t_return_parameter,
        min_flow_accum_threshold,
        dem_path,
        stream_gauge_vector_path,
        stream_gauge_table_path,
        stream_gauge_id_field,
        table_field_prefix,
        target_stream_vector_path,
        target_watershed_boundary_vector_path,
        target_floodplain_raster_path,
        target_snap_point_vector_path)



    """

    def flow_accumulation_d8(
        flow_dir_raster_path_band, target_flow_accum_raster_path,
        weight_raster_path_band=None,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):

    extract_strahler_streams_d8(
        flow_dir_d8_raster_path_band, flow_accum_raster_path_band,
        dem_raster_path_band,
        target_stream_vector_path,
        long min_flow_accum_threshold=100,
        int river_order=5,
        float min_p_val=0.05,
        autotune_flow_accumulation=False,
        osr_axis_mapping_strategy=DEFAULT_OSR_AXIS_MAPPING_STRATEGY):

    calculate_subwatershed_boundary(
        d8_flow_dir_raster_path_band,
        strahler_stream_vector_path, target_watershed_boundary_vector_path,
        max_steps_per_watershed=1000000,
        outlet_at_confluence=False)

    """