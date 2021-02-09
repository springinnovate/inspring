# cython: profile=False
# cython: language_level=3
import collections
import logging
import os
import shutil
import taskgraph
import tempfile
import time

cimport cython
cimport numpy
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc
from libc.time cimport time as ctime
from libc.time cimport time_t
from libcpp.deque cimport deque
from libcpp.list cimport list as clist
from libcpp.pair cimport pair
from libcpp.queue cimport queue
from libcpp.set cimport set as cset
from libcpp.stack cimport stack
from libcpp.vector cimport vector
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import numpy
import pandas
import shapely
import shapely.wkb
import shapely.ops
from shapely.ops import unary_union
import scipy.optimize
import scipy.stats

import pygeoprocessing
from pygeoprocessing.geoprocessing_core import DEFAULT_OSR_AXIS_MAPPING_STRATEGY

LOGGER = logging.getLogger(__name__)

# Number of raster blocks to hold in memory at once per Managed Raster
cdef int MANAGED_RASTER_N_BLOCKS = 2**6

# this is a least recently used cache written in C++ in an external file,
# exposing here so _ManagedRaster can use it
cdef extern from "LRUCache.h" nogil:
    cdef cppclass LRUCache[KEY_T, VAL_T]:
        LRUCache(int)
        void put(KEY_T&, VAL_T&, clist[pair[KEY_T,VAL_T]]&)
        clist[pair[KEY_T,VAL_T]].iterator begin()
        clist[pair[KEY_T,VAL_T]].iterator end()
        bint exist(KEY_T &)
        VAL_T get(KEY_T &)
        void clean(clist[pair[KEY_T,VAL_T]]&, int n_items)
        size_t size()


# this ctype is used to store the block ID and the block buffer as one object
# inside Managed Raster
ctypedef pair[int, double*] BlockBufferPair

# a class to allow fast random per-pixel access to a raster for both setting
# and reading pixels.
cdef class _ManagedRaster:
    cdef LRUCache[int, double*]* lru_cache
    cdef cset[int] dirty_blocks
    cdef int block_xsize
    cdef int block_ysize
    cdef int block_xmod
    cdef int block_ymod
    cdef int block_xbits
    cdef int block_ybits
    cdef int raster_x_size
    cdef int raster_y_size
    cdef int block_nx
    cdef int block_ny
    cdef int write_mode
    cdef bytes raster_path
    cdef int band_id
    cdef int closed

    def __cinit__(self, raster_path, band_id, write_mode):
        """Create new instance of Managed Raster.

        Parameters:
            raster_path (char*): path to raster that has block sizes that are
                powers of 2. If not, an exception is raised.
            band_id (int): which band in `raster_path` to index. Uses GDAL
                notation that starts at 1.
            write_mode (boolean): if true, this raster is writable and dirty
                memory blocks will be written back to the raster as blocks
                are swapped out of the cache or when the object deconstructs.

        Returns:
            None.
        """
        if not os.path.isfile(raster_path):
            LOGGER.error("%s is not a file.", raster_path)
            return
        raster_info = pygeoprocessing.get_raster_info(raster_path)
        self.raster_x_size, self.raster_y_size = raster_info['raster_size']
        self.block_xsize, self.block_ysize = raster_info['block_size']
        self.block_xmod = self.block_xsize-1
        self.block_ymod = self.block_ysize-1

        if not (1 <= band_id <= raster_info['n_bands']):
            err_msg = (
                "Error: band ID (%s) is not a valid band number. "
                "This exception is happening in Cython, so it will cause a "
                "hard seg-fault, but it's otherwise meant to be a "
                "ValueError." % (band_id))
            print(err_msg)
            raise ValueError(err_msg)
        self.band_id = band_id

        if (self.block_xsize & (self.block_xsize - 1) != 0) or (
                self.block_ysize & (self.block_ysize - 1) != 0):
            # If inputs are not a power of two, this will at least print
            # an error message. Unfortunately with Cython, the exception will
            # present itself as a hard seg-fault, but I'm leaving the
            # ValueError in here at least for readability.
            err_msg = (
                "Error: Block size is not a power of two: "
                "block_xsize: %d, %d, %s. This exception is happening"
                "in Cython, so it will cause a hard seg-fault, but it's"
                "otherwise meant to be a ValueError." % (
                    self.block_xsize, self.block_ysize, raster_path))
            print(err_msg)
            raise ValueError(err_msg)

        self.block_xbits = numpy.log2(self.block_xsize)
        self.block_ybits = numpy.log2(self.block_ysize)
        self.block_nx = (
            self.raster_x_size + (self.block_xsize) - 1) // self.block_xsize
        self.block_ny = (
            self.raster_y_size + (self.block_ysize) - 1) // self.block_ysize

        self.lru_cache = new LRUCache[int, double*](MANAGED_RASTER_N_BLOCKS)
        self.raster_path = <bytes> raster_path
        self.write_mode = write_mode
        self.closed = 0

    def __dealloc__(self):
        """Deallocate _ManagedRaster.

        This operation manually frees memory from the LRUCache and writes any
        dirty memory blocks back to the raster if `self.write_mode` is True.
        """
        self.close()

    def close(self):
        """Close the _ManagedRaster and free up resources.

            This call writes any dirty blocks to disk, frees up the memory
            allocated as part of the cache, and frees all GDAL references.

            Any subsequent calls to any other functions in _ManagedRaster will
            have undefined behavior.
        """
        if self.closed:
            return
        self.closed = 1
        cdef int xi_copy, yi_copy
        cdef numpy.ndarray[double, ndim=2] block_array = numpy.empty(
            (self.block_ysize, self.block_xsize))
        cdef double *double_buffer
        cdef int block_xi
        cdef int block_yi
        # initially the win size is the same as the block size unless
        # we're at the edge of a raster
        cdef int win_xsize
        cdef int win_ysize

        # we need the offsets to subtract from global indexes for cached array
        cdef int xoff
        cdef int yoff

        cdef clist[BlockBufferPair].iterator it = self.lru_cache.begin()
        cdef clist[BlockBufferPair].iterator end = self.lru_cache.end()
        if not self.write_mode:
            while it != end:
                # write the changed value back if desired
                PyMem_Free(deref(it).second)
                inc(it)
            return

        raster = gdal.OpenEx(
            self.raster_path, gdal.GA_Update | gdal.OF_RASTER)
        raster_band = raster.GetRasterBand(self.band_id)

        # if we get here, we're in write_mode
        cdef cset[int].iterator dirty_itr
        while it != end:
            double_buffer = deref(it).second
            block_index = deref(it).first

            # write to disk if block is dirty
            dirty_itr = self.dirty_blocks.find(block_index)
            if dirty_itr != self.dirty_blocks.end():
                self.dirty_blocks.erase(dirty_itr)
                block_xi = block_index % self.block_nx
                block_yi = block_index / self.block_nx

                # we need the offsets to subtract from global indexes for
                # cached array
                xoff = block_xi << self.block_xbits
                yoff = block_yi << self.block_ybits

                win_xsize = self.block_xsize
                win_ysize = self.block_ysize

                # clip window sizes if necessary
                if xoff+win_xsize > self.raster_x_size:
                    win_xsize = win_xsize - (
                        xoff+win_xsize - self.raster_x_size)
                if yoff+win_ysize > self.raster_y_size:
                    win_ysize = win_ysize - (
                        yoff+win_ysize - self.raster_y_size)

                for xi_copy in range(win_xsize):
                    for yi_copy in range(win_ysize):
                        block_array[yi_copy, xi_copy] = (
                            double_buffer[
                                (yi_copy << self.block_xbits) + xi_copy])
                raster_band.WriteArray(
                    block_array[0:win_ysize, 0:win_xsize],
                    xoff=xoff, yoff=yoff)
            PyMem_Free(double_buffer)
            inc(it)
        raster_band.FlushCache()
        raster_band = None
        raster = None

    cdef inline void set(self, int xi, int yi, double value):
        """Set the pixel at `xi,yi` to `value`."""
        if xi < 0 or xi >= self.raster_x_size:
            LOGGER.error("x out of bounds %s" % xi)
        if yi < 0 or yi >= self.raster_y_size:
            LOGGER.error("y out of bounds %s" % yi)
        cdef int block_xi = xi >> self.block_xbits
        cdef int block_yi = yi >> self.block_ybits
        # this is the flat index for the block
        cdef int block_index = block_yi * self.block_nx + block_xi
        if not self.lru_cache.exist(block_index):
            self._load_block(block_index)
        self.lru_cache.get(
            block_index)[
                ((yi & (self.block_ymod)) << self.block_xbits) +
                (xi & (self.block_xmod))] = value
        if self.write_mode:
            dirty_itr = self.dirty_blocks.find(block_index)
            if dirty_itr == self.dirty_blocks.end():
                self.dirty_blocks.insert(block_index)

    cdef inline double get(self, int xi, int yi):
        """Return the value of the pixel at `xi,yi`."""
        if xi < 0 or xi >= self.raster_x_size:
            LOGGER.error("x out of bounds %s" % xi)
        if yi < 0 or yi >= self.raster_y_size:
            LOGGER.error("y out of bounds %s" % yi)
        cdef int block_xi = xi >> self.block_xbits
        cdef int block_yi = yi >> self.block_ybits
        # this is the flat index for the block
        cdef int block_index = block_yi * self.block_nx + block_xi
        if not self.lru_cache.exist(block_index):
            self._load_block(block_index)
        return self.lru_cache.get(
            block_index)[
                ((yi & (self.block_ymod)) << self.block_xbits) +
                (xi & (self.block_xmod))]

    cdef void _load_block(self, int block_index) except *:
        cdef int block_xi = block_index % self.block_nx
        cdef int block_yi = block_index // self.block_nx

        # we need the offsets to subtract from global indexes for cached array
        cdef int xoff = block_xi << self.block_xbits
        cdef int yoff = block_yi << self.block_ybits

        cdef int xi_copy, yi_copy
        cdef numpy.ndarray[double, ndim=2] block_array
        cdef double *double_buffer
        cdef clist[BlockBufferPair] removed_value_list

        # determine the block aligned xoffset for read as array

        # initially the win size is the same as the block size unless
        # we're at the edge of a raster
        cdef int win_xsize = self.block_xsize
        cdef int win_ysize = self.block_ysize

        # load a new block
        if xoff+win_xsize > self.raster_x_size:
            win_xsize = win_xsize - (xoff+win_xsize - self.raster_x_size)
        if yoff+win_ysize > self.raster_y_size:
            win_ysize = win_ysize - (yoff+win_ysize - self.raster_y_size)

        raster = gdal.OpenEx(self.raster_path, gdal.OF_RASTER)
        raster_band = raster.GetRasterBand(self.band_id)
        block_array = raster_band.ReadAsArray(
            xoff=xoff, yoff=yoff, win_xsize=win_xsize,
            win_ysize=win_ysize).astype(numpy.float64)
        raster_band = None
        raster = None
        double_buffer = <double*>PyMem_Malloc(
            (sizeof(double) << self.block_xbits) * win_ysize)
        for xi_copy in range(win_xsize):
            for yi_copy in range(win_ysize):
                double_buffer[(yi_copy << self.block_xbits)+xi_copy] = (
                    block_array[yi_copy, xi_copy])
        self.lru_cache.put(
            <int>block_index, <double*>double_buffer, removed_value_list)

        if self.write_mode:
            n_attempts = 5
            while True:
                raster = gdal.OpenEx(
                    self.raster_path, gdal.GA_Update | gdal.OF_RASTER)
                if raster is None:
                    if n_attempts == 0:
                        raise RuntimeError(
                            f'could not open {self.raster_path} for writing')
                    LOGGER.warning(
                        f'opening {self.raster_path} resulted in null, '
                        f'trying {n_attempts} more times.')
                    n_attempts -= 1
                    time.sleep(0.5)
                raster_band = raster.GetRasterBand(self.band_id)
                break

        block_array = numpy.empty(
            (self.block_ysize, self.block_xsize), dtype=numpy.double)
        while not removed_value_list.empty():
            # write the changed value back if desired
            double_buffer = removed_value_list.front().second

            if self.write_mode:
                block_index = removed_value_list.front().first

                # write back the block if it's dirty
                dirty_itr = self.dirty_blocks.find(block_index)
                if dirty_itr != self.dirty_blocks.end():
                    self.dirty_blocks.erase(dirty_itr)

                    block_xi = block_index % self.block_nx
                    block_yi = block_index // self.block_nx

                    xoff = block_xi << self.block_xbits
                    yoff = block_yi << self.block_ybits

                    win_xsize = self.block_xsize
                    win_ysize = self.block_ysize

                    if xoff+win_xsize > self.raster_x_size:
                        win_xsize = win_xsize - (
                            xoff+win_xsize - self.raster_x_size)
                    if yoff+win_ysize > self.raster_y_size:
                        win_ysize = win_ysize - (
                            yoff+win_ysize - self.raster_y_size)

                    for xi_copy in range(win_xsize):
                        for yi_copy in range(win_ysize):
                            block_array[yi_copy, xi_copy] = double_buffer[
                                (yi_copy << self.block_xbits) + xi_copy]
                    raster_band.WriteArray(
                        block_array[0:win_ysize, 0:win_xsize],
                        xoff=xoff, yoff=yoff)
            PyMem_Free(double_buffer)
            removed_value_list.pop_front()

        if self.write_mode:
            raster_band = None
            raster = None

    cdef void flush(self) except *:
        cdef clist[BlockBufferPair] removed_value_list
        cdef double *double_buffer
        cdef cset[int].iterator dirty_itr
        cdef int block_index, block_xi, block_yi
        cdef int xoff, yoff, win_xsize, win_ysize

        self.lru_cache.clean(removed_value_list, self.lru_cache.size())

        raster_band = None
        if self.write_mode:
            max_retries = 5
            while max_retries > 0:
                raster = gdal.OpenEx(
                    self.raster_path, gdal.GA_Update | gdal.OF_RASTER)
                if raster is None:
                    max_retries -= 1
                    LOGGER.error(
                        f'unable to open {self.raster_path}, retrying...')
                    time.sleep(0.2)
                    continue
                break
            if max_retries == 0:
                raise ValueError(
                    f'unable to open {self.raster_path} in '
                    'ManagedRaster.flush')
            raster_band = raster.GetRasterBand(self.band_id)

        block_array = numpy.empty(
            (self.block_ysize, self.block_xsize), dtype=numpy.double)
        while not removed_value_list.empty():
            # write the changed value back if desired
            double_buffer = removed_value_list.front().second

            if self.write_mode:
                block_index = removed_value_list.front().first

                # write back the block if it's dirty
                dirty_itr = self.dirty_blocks.find(block_index)
                if dirty_itr != self.dirty_blocks.end():
                    self.dirty_blocks.erase(dirty_itr)

                    block_xi = block_index % self.block_nx
                    block_yi = block_index // self.block_nx

                    xoff = block_xi << self.block_xbits
                    yoff = block_yi << self.block_ybits

                    win_xsize = self.block_xsize
                    win_ysize = self.block_ysize

                    if xoff+win_xsize > self.raster_x_size:
                        win_xsize = win_xsize - (
                            xoff+win_xsize - self.raster_x_size)
                    if yoff+win_ysize > self.raster_y_size:
                        win_ysize = win_ysize - (
                            yoff+win_ysize - self.raster_y_size)

                    for xi_copy in range(win_xsize):
                        for yi_copy in range(win_ysize):
                            block_array[yi_copy, xi_copy] = double_buffer[
                                (yi_copy << self.block_xbits) + xi_copy]
                    raster_band.WriteArray(
                        block_array[0:win_ysize, 0:win_xsize],
                        xoff=xoff, yoff=yoff)
            PyMem_Free(double_buffer)
            removed_value_list.pop_front()

        if self.write_mode:
            raster_band = None
            raster = None


def _scrub_invalid_values(base_array, nodata, new_nodata):
    result = numpy.copy(base_array)
    invalid_mask = (
        ~numpy.isfinite(base_array) |
        numpy.isclose(result, nodata))
    result[invalid_mask] = new_nodata
    return result


def build_flood_height():
    """floo dnight."""
    pass


def func_powerlaw(x, b, a):
    return a*x**b


def snap_points(
        point_vector_path, key_field, line_vector_path,
        target_snap_point_path):
    """Snap points to nearest line."""
    line_vector = gdal.OpenEx(line_vector_path, gdal.OF_VECTOR)
    line_layer = line_vector.GetLayer()
    line_srs = line_layer.GetSpatialRef()

    point_vector = gdal.OpenEx(point_vector_path, gdal.OF_VECTOR)
    point_layer = point_vector.GetLayer()
    point_layer_defn = point_layer.GetLayerDefn()
    key_field_defn = point_layer_defn.GetFieldDefn(
        point_layer_defn.GetFieldIndex(key_field))
    point_srs = point_layer.GetSpatialRef()

    snap_basename = os.path.basename(
        os.path.splitext(target_snap_point_path)[0])
    gpkg_driver = gdal.GetDriverByName('GPKG')
    snap_point_vector = gpkg_driver.Create(
        target_snap_point_path, 0, 0, 0, gdal.GDT_Unknown)
    snap_point_layer = snap_point_vector.CreateLayer(
        snap_basename, line_srs, ogr.wkbPoint)
    snap_point_layer.CreateField(key_field_defn)

    # project stream points to stream layer projection
    point_srs.SetAxisMappingStrategy(DEFAULT_OSR_AXIS_MAPPING_STRATEGY)
    line_srs.SetAxisMappingStrategy(DEFAULT_OSR_AXIS_MAPPING_STRATEGY)

    # Create a coordinate transformation
    coord_trans = osr.CreateCoordinateTransformation(point_srs, line_srs)

    line_segment_geom = unary_union([
        shapely.wkb.loads(seg.GetGeometryRef().ExportToWkb())
        for seg in line_layer])
    for point_feature in point_layer:
        point_geom_ref = point_feature.GetGeometryRef()
        point_geom_ref.Transform(coord_trans)
        point_geom = shapely.wkb.loads(point_geom_ref.ExportToWkb())
        points = shapely.ops.nearest_points(
            point_geom, line_segment_geom)
        _, snapped_geom = points
        snapped_feature = ogr.Feature(snap_point_layer.GetLayerDefn())
        snapped_feature.SetField(
            key_field, point_feature.GetField(key_field))
        snapped_feature.SetGeometry(ogr.CreateGeometryFromWkb(
            snapped_geom.wkb))
        snap_point_layer.CreateFeature(snapped_feature)


def build_gauge_stats(
        flood_level_year, gauge_table_path, table_field_prefix,
        gauge_id_field, flow_accum_path, gauge_vector_path):
    LOGGER.info('build gauge stats')
    gauge_df = pandas.read_csv(gauge_table_path)
    gauge_vector = gdal.OpenEx(gauge_vector_path, gdal.OF_VECTOR)
    gauge_layer = gauge_vector.GetLayer()
    water_level_params_per_gauge = {}
    flow_accum_raster_info = pygeoprocessing.get_raster_info(
        flow_accum_path)
    flow_accum_raster = gdal.OpenEx(flow_accum_path)
    flow_accum_band = flow_accum_raster.GetRasterBand(1)
    inv_gt = gdal.InvGeoTransform(flow_accum_raster_info['geotransform'])
    water_level_list = []
    water_level_bankfull_list = []
    upstream_area_list = []
    for gauge_feature in gauge_layer:
        station_id = gauge_feature.GetField(gauge_id_field)
        gauge_geom = gauge_feature.GetGeometryRef()
        x_pos = gauge_geom.GetX()
        y_pos = gauge_geom.GetY()
        table_station_id = f'{table_field_prefix}{station_id}'

        i, j = [int(p) for p in gdal.ApplyGeoTransform(inv_gt, x_pos, y_pos)]
        fa_val = flow_accum_band.ReadAsArray(i, j, 1, 1)[0, 0]
        if table_station_id not in gauge_df:
            LOGGER.warning(
                f'{table_station_id} is in the vector but not in the table, '
                f'skipping')
            continue
        water_level_raw = gauge_df[table_station_id]
        water_level_series = water_level_raw[water_level_raw >= 0]
        sigma = water_level_series.mean()
        mu = water_level_series.std()
        alpha = numpy.sqrt(6)/numpy.pi * sigma * water_level_series.max()

        beta = mu * water_level_series.max() - 0.5772 * alpha
        # calculate bankfull (r=1.5) water level
        wl_bf = beta-alpha*numpy.log(-numpy.log(1.-1./1.5))
        wl_r = beta-alpha*numpy.log(-numpy.log(1.-1./flood_level_year))
        water_level_bankfull_list.append(wl_bf)
        water_level_list.append(wl_r)
        upstream_area_list.append(fa_val)
        water_level_params_per_gauge[station_id] = {
            'series': water_level_series,
            'sigma': sigma,
            'mu': mu,
            'max_height': max(water_level_series),
            'fa_val': fa_val,
        }
    # fit power law for wl_r
    LOGGER.info(f'water level params: {water_level_params_per_gauge}')
    LOGGER.info('fit power law for wl_r')
    def _power_func(x, a, b):
        return a*x**b
    LOGGER.info(f'fitting these areas: {upstream_area_list}')
    LOGGER.info(f'to these water levels: {water_level_list}')
    LOGGER.info(f'and these bankfull levels: {water_level_bankfull_list}')
    wl_popt, wl_pcov = scipy.optimize.curve_fit(
        _power_func, upstream_area_list, water_level_list)
    # fit power law for wl_bf
    LOGGER.info('fit power law for wl_bf')
    wl_bf_popt, wl_bf_pcov = scipy.optimize.curve_fit(
        _power_func, upstream_area_list, water_level_bankfull_list)
    LOGGER.info(
        f'\n'
        f'1.5 at 1e6: {_power_func(1e6, *wl_bf_popt)}\n'
        f'{flood_level_year} at 1e6: {_power_func(1e6, *wl_popt)}\n')
    return {
        'wl_popt': wl_popt,
        'wl_bf_popt': wl_bf_popt,
    }


def power_func(x, a, b):
    return a*x**b


def floodplane_extraction(
        level_year_parameter,
        dem_path,
        stream_gauge_vector_path,
        gauge_table_path,
        gauge_id_field,
        start_year_id_field,
        end_year_id_field,
        year_table_column_id_field,
        table_field_prefix,
        target_floodplane_raster_path,
        target_snap_point_vector_path,
        min_flow_accum_threshold=100):
    """Entry point."""
    LOGGER.info('snap points')
    dem_info = pygeoprocessing.get_raster_info(dem_path)
    dem_type = dem_info['numpy_type']
    working_dir = os.path.join(
        os.path.dirname(target_floodplane_raster_path),
        f'''workspace_{os.path.basename(os.path.splitext(
            target_floodplane_raster_path)[0])}''')
    nodata = dem_info['nodata'][0]
    new_nodata = float(numpy.finfo(dem_type).min)

    scrubbed_dem_path = os.path.join(working_dir, 'scrubbed_dem.tif')
    task_graph = taskgraph.TaskGraph(working_dir, -1)

    scrub_dem_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=(
            [(dem_path, 1), (nodata, 'raw'), (new_nodata, 'raw')],
            _scrub_invalid_values, scrubbed_dem_path,
            dem_info['datatype'], new_nodata),
        target_path_list=[scrubbed_dem_path],
        task_name='scrub dem')

    LOGGER.info('fill pits')
    filled_pits_path = os.path.join(working_dir, 'filled_pits_dem.tif')
    fill_pits_task = task_graph.add_task(
        func=pygeoprocessing.routing.fill_pits,
        args=((scrubbed_dem_path, 1), filled_pits_path),
        kwargs={'max_pixel_fill_count': 1000000},
        target_path_list=[filled_pits_path],
        dependent_task_list=[scrub_dem_task],
        task_name='fill pits')

    LOGGER.info('flow dir d8')
    flow_dir_d8_path = os.path.join(working_dir, 'flow_dir_d8.tif')
    flow_dir_task = task_graph.add_task(
        func=pygeoprocessing.routing.flow_dir_d8,
        args=((filled_pits_path, 1), flow_dir_d8_path),
        kwargs={'working_dir': working_dir},
        target_path_list=[flow_dir_d8_path],
        dependent_task_list=[fill_pits_task],
        task_name='flow dir d8')

    LOGGER.info('flow accum d8')
    flow_accum_d8_path = os.path.join(working_dir, 'flow_accum_d8.tif')
    flow_accum_task = task_graph.add_task(
        func=pygeoprocessing.routing.flow_accumulation_d8,
        args=((flow_dir_d8_path, 1), flow_accum_d8_path),
        target_path_list=[flow_accum_d8_path],
        dependent_task_list=[flow_dir_task],
        task_name='flow accum d8')

    stream_vector_path = os.path.join(
        working_dir, f'stream_segments_{min_flow_accum_threshold}.gpkg')
    extract_stream_task = task_graph.add_task(
        func=pygeoprocessing.routing.extract_strahler_streams_d8,
        args=(
            (flow_dir_d8_path, 1), (flow_accum_d8_path, 1),
            (filled_pits_path, 1), stream_vector_path),
        kwargs={
            'min_flow_accum_threshold': min_flow_accum_threshold,
            'river_order': 7},
        target_path_list=[stream_vector_path],
        ignore_path_list=[stream_vector_path],
        dependent_task_list=[flow_accum_task],
        task_name='stream extraction')

    snap_points_task = task_graph.add_task(
        func=snap_points,
        args=(
            stream_gauge_vector_path, gauge_id_field, stream_vector_path,
            target_snap_point_vector_path),
        target_path_list=[target_snap_point_vector_path],
        dependent_task_list=[extract_stream_task],
        task_name='snap points')

    build_gauge_stats_task = task_graph.add_task(
        func=build_gauge_stats,
        args=(
            level_year_parameter, gauge_table_path, table_field_prefix,
            gauge_id_field,
            flow_accum_d8_path, target_snap_point_vector_path),
        dependent_task_list=[snap_points_task],
        store_result=True,
        transient_run=True,
        task_name='build gauge stats')

    LOGGER.debug(build_gauge_stats_task.get())

    target_watershed_boundary_vector_path = os.path.join(
        working_dir, 'watershed_boundary.gpkg')
    calculate_watershed_boundary_task = task_graph.add_task(
        func=pygeoprocessing.routing.calculate_subwatershed_boundary,
        args=(
            (flow_dir_d8_path, 1), stream_vector_path,
            target_watershed_boundary_vector_path),
        kwargs={'outlet_at_confluence': False},
        target_path_list=[target_watershed_boundary_vector_path],
        transient_run=True,
        dependent_task_list=[extract_stream_task],
        task_name='watershed boundary')

    # Fill the subwatersheds to the flow height

    task_graph.close()
    task_graph.join()
