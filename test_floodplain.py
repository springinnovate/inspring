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
    side_length = 1000
    dem_array = numpy.zeros((side_length, side_length))

    dem_array[side_length // 4, side_length // 2] = side_length**2
    dem_array[side_length // 2, side_length // 2] = side_length**2
    dem_array[-1, side_length // 5] = side_length**2
    dem_array += numpy.random.random((side_length, side_length))*side_length
    dem_array = gaussian_filter(dem_array, side_length // 5)

    working_dir = 'test_floodplain_workspace'
    os.makedirs(working_dir, exist_ok=True)
    min_flow_accum_threshold = 10

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
