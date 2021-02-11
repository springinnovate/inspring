# cython: profile=False
# cython: language_level=3
import collections
import logging
import multiprocessing
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
    cell_area = abs(numpy.prod(flow_accum_raster_info['pixel_size']))
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
        fa_val = flow_accum_band.ReadAsArray(i, j, 1, 1)[0, 0] * cell_area
        if table_station_id not in gauge_df:
            LOGGER.warning(
                f'{table_station_id} is in the vector but not in the table, '
                f'skipping')
            continue
        water_level_raw = gauge_df[table_station_id]
        water_level_series = water_level_raw[water_level_raw >= 0]
        sigma = water_level_series.mean()
        mu = water_level_series.std()
        alpha = numpy.sqrt(6)/numpy.pi * mu * water_level_series.max()

        beta = sigma * water_level_series.max() - 0.5772 * alpha
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

    def _power_func(x, a, b):
        return a*x**b
    # fit power law for wl_r
    wl_popt, wl_pcov = scipy.optimize.curve_fit(
        _power_func, upstream_area_list, water_level_list)
    # fit power law for wl_bf
    wl_bf_popt, wl_bf_pcov = scipy.optimize.curve_fit(
        _power_func, upstream_area_list, water_level_bankfull_list)
    return {
        'wl_pow_params': (wl_popt[0], wl_popt[1]),
        'wl_bf_pow_params': (wl_bf_popt[0], wl_bf_popt[1]),
    }


def _stitch_worker(
        stitch_raster_queue, target_stitch_raster_path, batch_size=100):
    """Stitch rasters from the queue into target."""
    n_to_stitch = 0
    current_stitch_list = []
    while True:
        payload = stitch_raster_queue.get()
        if payload is not None:
            current_stitch_list.append((payload, 1))
        if len(current_stitch_list) == batch_size or payload is None:
            LOGGER.info(
                f'stitching a batch of size {len(current_stitch_list)}')
            pygeoprocessing.stitch_rasters(
                current_stitch_list,
                ['near']*len(current_stitch_list),
                (target_stitch_raster_path, 1),
                overlap_algorithm='etch')
            for path, _ in current_stitch_list:
                os.remove(path)
            current_stitch_list = []
        if payload is None:
            LOGGER.info(f'all done stitching {target_stitch_raster_path}')
            break


def _subwatershed_worker(
        dem_raster_path, flow_accum_raster_path,
        subwatershed_vector_path, pow_params_map,
        watershed_fid_process_queue, stitch_queue, watershed_stat_queue,
        working_dir):
    """Calculate flood height for the given watershed queue."""
    subwatershed_vector = gdal.OpenEx(
        subwatershed_vector_path, gdal.OF_VECTOR)
    subwatershed_layer = subwatershed_vector.GetLayer()
    dem_raster_info = pygeoprocessing.get_raster_info(dem_raster_path)
    pixel_size = dem_raster_info['pixel_size']
    cell_area = abs(numpy.prod(pixel_size))
    dem_nodata = dem_raster_info['nodata'][0]
    dem_raster = gdal.OpenEx(dem_raster_path, gdal.OF_RASTER)
    dem_band = dem_raster.GetRasterBand(1)
    flow_accum_raster = gdal.OpenEx(flow_accum_raster_path, gdal.OF_RASTER)
    flow_accum_band = flow_accum_raster.GetRasterBand(1)

    subwatershed_vector = gdal.OpenEx(
        subwatershed_vector_path, gdal.OF_VECTOR)
    subwatershed_layer = subwatershed_vector.GetLayer()
    while True:
        payload = watershed_fid_process_queue.get()
        if payload is None:
            LOGGER.info('all done filling subwatersheds')
            watershed_fid_process_queue.put(None)
            break
        subwatershed_fid = payload
        subwatershed_feature = subwatershed_layer.GetFeature(
            subwatershed_fid)
        outlet_x = subwatershed_feature.GetField('outlet_x')
        outlet_y = subwatershed_feature.GetField('outlet_y')
        outlet_dem_val = dem_band.ReadAsArray(outlet_x, outlet_y, 1, 1)[0, 0]
        upstream_area = cell_area * flow_accum_band.ReadAsArray(
            outlet_x, outlet_y, 1, 1)[0, 0]

        water_level_val = power_func(
            upstream_area, *pow_params_map['wl_pow_params'])
        water_level_bankflow_val = power_func(
            upstream_area, *pow_params_map['wl_bf_pow_params'])

        subwatershed_geom = subwatershed_feature.GetGeometryRef()
        subwatershed_dem_path = os.path.join(
            working_dir, f'{subwatershed_fid}_dem.tif')
        subwatershed_envelope = subwatershed_geom.GetEnvelope()
        subwatershed_bounds = [
            subwatershed_envelope[i]+offset for i, offset in [
                (0, -pixel_size[0]),
                (2, -pixel_size[1]),
                (1, pixel_size[0]),
                (3, pixel_size[1])]]
        pygeoprocessing.warp_raster(
            dem_raster_path, pixel_size,
            subwatershed_dem_path, 'near', target_bb=subwatershed_bounds,
            vector_mask_options={
                'mask_vector_path': subwatershed_vector_path,
                'mask_vector_where_filter': (
                    f'"fid"={subwatershed_fid}')},
            working_dir=working_dir)

        watershed_stat_queue.put((subwatershed_fid, {
            'upstream_area': upstream_area,
            'outlet_dem_val': outlet_dem_val,
            'local_flood_height': local_flood_height,
        }))

        local_flood_height = (
            water_level_val-water_level_bankflow_val+outlet_dem_val)
        flood_height_raster_path = os.path.join(
            working_dir, f'{subwatershed_fid}_flood_height.tif')
        pygeoprocessing.raster_calculator(
            [(subwatershed_dem_path, 1),
             (local_flood_height, 'raw'),
             (dem_nodata, 'raw')], _flood_height_op,
            flood_height_raster_path, dem_raster_info['datatype'],
            dem_nodata)

        stitch_queue.put(flood_height_raster_path)
        os.remove(subwatershed_dem_path)

    dem_raster = None
    dem_band = None
    flow_accum_raster = None
    flow_accum_band = None


def calculate_floodheight(
        t_return_parameter, dem_raster_path, flow_accum_raster_path,
        subwatershed_vector_path,
        pow_params_map,
        target_floodplane_raster_path,
        working_dir):
    """Create raster that shows floodheights of dem."""
    LOGGER.debug(subwatershed_vector_path)
    subwatershed_vector = gdal.OpenEx(
        subwatershed_vector_path, gdal.OF_VECTOR | gdal.GA_Update)
    subwatershed_layer = subwatershed_vector.GetLayer()
    subwatershed_layer.CreateField(
        ogr.FieldDefn('upstream_area', ogr.OFTReal))
    subwatershed_layer.CreateField(
        ogr.FieldDefn('outlet_dem_val', ogr.OFTReal))
    subwatershed_layer.CreateField(
        ogr.FieldDefn('local_flood_height', ogr.OFTReal))
    dem_raster_info = pygeoprocessing.get_raster_info(dem_raster_path)
    dem_nodata = dem_raster_info['nodata'][0]
    dem_raster = gdal.OpenEx(dem_raster_path, gdal.OF_RASTER)
    dem_band = dem_raster.GetRasterBand(1)
    flow_accum_raster = gdal.OpenEx(flow_accum_raster_path, gdal.OF_RASTER)
    flow_accum_band = flow_accum_raster.GetRasterBand(1)
    pygeoprocessing.new_raster_from_base(
        dem_raster_path, target_floodplane_raster_path,
        dem_raster_info['datatype'], [dem_nodata])
    manager = multiprocessing.Manager()
    stitch_raster_path_queue = manager.Queue()
    watershed_fid_process_queue = manager.Queue()
    watershed_stat_queue = manager.Queue()

    stitch_worker_process = multiprocessing.Process(
        target=_stitch_worker,
        args=(stitch_raster_path_queue, target_floodplane_raster_path))
    stitch_worker_process.start()
    subwatershed_worker_process_list = []
    for _ in range(multiprocessing.cpu_count()):
        subwatershed_worker_process = multiprocessing.Process(
            target=_subwatershed_worker,
            args=(
                dem_raster_path, flow_accum_raster_path,
                subwatershed_vector_path, pow_params_map,
                watershed_fid_process_queue, stitch_raster_path_queue,
                watershed_stat_queue,
                working_dir))
        subwatershed_worker_process.start()
        subwatershed_worker_process_list.append(subwatershed_worker_process)

    subwatershed_fid_list = [
        subwatershed_feature.GetFID()
        for subwatershed_feature in subwatershed_layer]
    subwatershed_layer = None
    subwatershed_vector = None

    for subwatershed_feature in subwatershed_fid_list:
        subwatershed_fid = subwatershed_feature.GetFID()
        watershed_fid_process_queue.put(subwatershed_fid)
    watershed_fid_process_queue.put(None)

    for worker_process in subwatershed_worker_process_list:
        worker_process.join()
    stitch_raster_path_queue.put(None)

    watershed_stat_queue.put(None)
    subwatershed_vector = gdal.OpenEx(
        subwatershed_vector_path, gdal.OF_VECTOR | gdal.GA_Update)
    subwatershed_layer = subwatershed_vector.GetLayer()

    for fid, field_dict in iter(watershed_stat_queue.get, None):
        subwatershed_feature = subwatershed_layer.GetFeature(fid)
        for fieldname, val in field_dict.items():
            subwatershed_feature.SetField(fieldname, val)
        subwatershed_layer.SetFeature(subwatershed_feature)
    subwatershed_layer = None
    subwatershed_vector = None

    stitch_worker_process.join()
    LOGGER.info('all done calculating floodheight')


def _flood_height_op(dem_height_array, flood_level_height, nodata):
    result = numpy.full(
        dem_height_array.shape, nodata, dtype=dem_height_array.dtype)
    valid_mask = (~numpy.isclose(dem_height_array, nodata))
    flood_height = flood_level_height-dem_height_array[valid_mask]
    flood_height[flood_height < 0] = 0
    result[valid_mask] = flood_height
    return result


def power_func(x, a, b):
    return a*x**b


def floodplane_extraction(
        t_return_parameter,
        dem_path,
        stream_gauge_vector_path,
        gauge_table_path,
        gauge_id_field,
        table_field_prefix,
        target_stream_vector_path,
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

    target_stream_vector_path
    extract_stream_task = task_graph.add_task(
        func=pygeoprocessing.routing.extract_strahler_streams_d8,
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

    snap_points_task = task_graph.add_task(
        func=snap_points,
        args=(
            stream_gauge_vector_path, gauge_id_field, target_stream_vector_path,
            target_snap_point_vector_path),
        target_path_list=[target_snap_point_vector_path],
        dependent_task_list=[extract_stream_task],
        task_name='snap points')

    build_gauge_stats_task = task_graph.add_task(
        func=build_gauge_stats,
        args=(
            t_return_parameter, gauge_table_path, table_field_prefix,
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
            (flow_dir_d8_path, 1), target_stream_vector_path,
            target_watershed_boundary_vector_path),
        kwargs={'outlet_at_confluence': False},
        target_path_list=[target_watershed_boundary_vector_path],
        ignore_path_list=[target_watershed_boundary_vector_path],
        dependent_task_list=[extract_stream_task],
        task_name='watershed boundary')

    # Fill the subwatersheds to the flow height
    calculate_floodheight(
        t_return_parameter,
        filled_pits_path,
        flow_accum_d8_path,
        target_watershed_boundary_vector_path,
        build_gauge_stats_task.get(),
        target_floodplane_raster_path,
        working_dir)

    task_graph.close()
    task_graph.join()
