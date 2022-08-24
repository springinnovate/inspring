"""Test framework for floodplain development."""
import logging
import os
import sys

from ecoshard import geoprocessing
from scipy.ndimage import gaussian_filter
from osgeo import gdal
from inspring.floodplain_extraction.utils import prep_floodplain_data
import numpy


gdal.SetCacheMax(2**26)

LOGGER = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(processName)s %(levelname)s '
        '%(name)s [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)


def main():
    """Entry point."""
    # flow_dir_d8_path = './test_floodplain_workspace/flow_dir_d8.tif'
    # target_stream_vector_path = './test_floodplain_workspace/stream.gpkg'
    # target_watershed_boundary_vector_path = 'test_watershed.gpkg'
    # geoprocessing.routing.calculate_subwatershed_boundary(
    #     (flow_dir_d8_path, 1), target_stream_vector_path,
    #     target_watershed_boundary_vector_path,
    #     outlet_at_confluence=False)
    # return



    side_length = 1000
    dem_array = numpy.zeros((side_length, side_length))

    n_peaks = 10
    x = numpy.random.randint(0, side_length, n_peaks)
    y = numpy.random.randint(0, side_length, n_peaks)
    peak_height = numpy.random.randint(0, 10*side_length**2, n_peaks)
    coord_list = [x, y]
    print(coord_list)

    #numpy.take(dem_array, numpy.ravel_multi_index(coord_list, dem_array.shape))
    dem_array.flat[numpy.ravel_multi_index(coord_list, dem_array.shape)] = peak_height

    dem_array = gaussian_filter(dem_array, side_length / 10)
    dem_array += numpy.random.randint(0, 5, (side_length, side_length))

    working_dir = 'test_floodplain_workspace'
    os.makedirs(working_dir, exist_ok=True)
    min_flow_accum_threshold = 200

    target_stream_vector_path = os.path.join(working_dir, 'stream.gpkg')
    target_watershed_boundary_vector_path = os.path.join(
        working_dir, 'watershed_boundary.gpkg')

    dem_path = os.path.join(working_dir, 'dem.tif')

    geoprocessing.numpy_array_to_raster(
        dem_array, None, (30, -30), (0, 0), None,
        dem_path)

    prep_floodplain_data(
        working_dir, dem_path, min_flow_accum_threshold,
        target_stream_vector_path, target_watershed_boundary_vector_path)


if __name__ == '__main__':
    main()
