"""Earth Observation Driven Pollinator service model for inspring."""
import collections
import hashlib
import inspect
import logging
import os
import re

from ecoshard import geoprocessing
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import numpy
import scipy.optimize
from ecoshard import taskgraph

from . import utils

LOGGER = logging.getLogger(__name__)

_INDEX_NODATA = -1.0

# These are patterns expected in the guilds table re expressions are season
_FLORAL_RESOURCES_EFD_MIN_RE = 'floral_resources_efd_min'
_FLORAL_RESOURCES_EFD_SUFFICIENT_RE = 'floral_resources_efd_sufficient'
_RELATIVE_SPECIES_ABUNDANCE_FIELD = 'relative_abundance'
_FLORAL_ALPHA_HEADER = 'floral_alpha'
_NESTING_ALPHA_HEADER = 'nesting_alpha'
_EXPECTED_GUILD_HEADERS = [
    'species',
    _FLORAL_ALPHA_HEADER,
    _NESTING_ALPHA_HEADER,
    _RELATIVE_SPECIES_ABUNDANCE_FIELD,
    'nesting_suitability_efd_min',
    'nesting_suitability_efd_sufficient',
    _FLORAL_RESOURCES_EFD_MIN_RE,
    _FLORAL_RESOURCES_EFD_SUFFICIENT_RE,
    ]

# replaced by (species, file_suffix)
_NESTING_SUBSTRATE_INDEX_FILE_PATTERN = 'nesting_substrate_index_%s%s.tif'
# this is used if there is a farm polygon present
_FARM_NESTING_SUBSTRATE_INDEX_FILE_PATTERN = (
    'farm_nesting_substrate_index_%s%s.tif')
# replaced by (species, file_suffix)
_HABITAT_NESTING_INDEX_FILE_PATTERN = 'habitat_nesting_index_%s%s.tif'
# replaced by (species, file_suffix)
_RELATIVE_FLORAL_ABUNDANCE_INDEX_FILE_PATTERN = (
    'relative_floral_abundance_index_%s%s.tif')
# this is used if there's a farm polygon present replace (species, file_suffix)
_FARM_RELATIVE_FLORAL_ABUNDANCE_INDEX_FILE_PATTERN = (
    'farm_relative_floral_abundance_index_%s%s.tif')
# used as an intermediate step for floral resources calculation
# replace (species, file_suffix)
_LOCAL_FORAGING_EFFECTIVENESS_FILE_PATTERN = (
    'local_foraging_effectiveness_%s%s.tif')
# pollinator supply raster replace (species, file_suffix)
_POLLINATOR_SUPPLY_FILE_PATTERN = 'pollinator_supply_%s%s.tif'
# name of reprojected farm vector replace (file_suffix)
_PROJECTED_FARM_VECTOR_FILE_PATTERN = 'reprojected_farm_vector%s.gpkg'
# used to store the 2D decay kernel for a given distance replace
# (species, alpha_type, alpha, file suffix)
_KERNEL_FILE_PATTERN = 'kernel_%s_%s_%f%s.tif'
# PA(x,s,j) replace (species, file_suffix)
_POLLINATOR_ABUNDANCE_FILE_PATTERN = 'pollinator_abundance_%s_%s.tif'
# PAT(x,j) total pollinator abundance replace (file_suffix)
_TOTAL_POLLINATOR_ABUNDANCE_FILE_PATTERN = (
    'total_pollinator_abundance_%s.tif')
# used for RA(l(x),j)*fa(s,j) replace (species, file_suffix)
_FORAGED_FLOWERS_INDEX_FILE_PATTERN = (
    'foraged_flowers_index_%s_%s.tif')
# used for convolving PS over alpha s replace (species, file_suffix)
_CONVOLVE_PS_FILE_PATH = 'convolve_ps_%s%s.tif'
# half saturation raster replace (file_suffix)
_HALF_SATURATION_FILE_PATTERN = 'half_saturation%s.tif'
_POLLINATOR_ABUDNANCE_FARM_FIELD_ID = 'pol_abundance'
# blank raster as a basis to rasterize on replace (file_suffix)
_BLANK_RASTER_FILE_PATTERN = 'blank_raster%s.tif'
# raster to hold farm pollinator replace (file_suffix)
_FARM_POLLINATOR_FILE_PATTERN = 'farm_pollinator%s.tif'
# managed pollinator indexes replace (file_suffix)
_MANAGED_POLLINATOR_FILE_PATTERN = 'managed_pollinators%s.tif'
# total pollinator raster replace (file_suffix)
_TOTAL_POLLINATOR_YIELD_FILE_PATTERN = 'total_pollinator_yield%s.tif'
# wild pollinator raster replace (file_suffix)
_WILD_POLLINATOR_YIELD_FILE_PATTERN = 'wild_pollinator_yield%s.tif'
# final aggregate farm shapefile file pattern replace (file_suffix)
_FARM_VECTOR_RESULT_FILE_PATTERN = 'farm_results%s.gpkg'
# output field on target shapefile if farms are enabled
_TOTAL_FARM_YIELD_FIELD_ID = 'y_tot'
# output field for wild pollinators on farms if farms are enabled
_WILD_POLLINATOR_FARM_YIELD_FIELD_ID = 'y_wild'
_HALF_SATURATION_FARM_HEADER = 'half_sat'
_CROP_POLLINATOR_DEPENDENCE_FIELD = 'p_dep'
_MANAGED_POLLINATORS_FIELD = 'p_managed'
_EXPECTED_FARM_HEADERS = [
    'crop_type', _HALF_SATURATION_FARM_HEADER,
    _MANAGED_POLLINATORS_FIELD,
    _CROP_POLLINATOR_DEPENDENCE_FIELD]

# used for clipping EFT to landcover replace (file_suffix)
_EFT_CLIP_FILE_PATTERN = 'eft_clip%s.tif'
# used for clipping EFT to landcover replace (file_suffix)
_EFT_FARM_FILE_PATTERN = 'eft_clip_farm_eft%s.tif'
# used for creating EFD from EFT replace (species, file_suffix)
_EFD_FILE_PATTERN = 'efd_%s%s.tif'


def _mkdir(dir_path):
    """Create dir_path if it doesn't exist and return it.

    Returns:
        dir_path
    """
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def _weighted_average_op(nodata, *array_weight_list):
    """Calcualte weighted average of array w/ weights."""
    aw_iter = iter(array_weight_list)
    result = numpy.zeros(array_weight_list[0].shape)
    valid_mask = numpy.ones(array_weight_list[0].shape, dtype=numpy.bool)
    weight_sum = 0.0
    for array, weight in zip(aw_iter, aw_iter):
        valid_mask &= numpy.isfinite(array) & (array >= 0)
        result[valid_mask] += array[valid_mask] * weight
        weight_sum += weight
    result[~valid_mask] = nodata
    result[valid_mask] /= weight_sum
    return result


class BoundedSigmoid(object):
    """Capture bounded sigmoid properties to pass to raster_calcualtor."""
    def __init__(self, lower_bound, upper_bound):
        """Create a sigmoid bounded between 0.1 and 0.9 by these params."""
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self._sigmoid = BoundedSigmoid._sigmoid

        a_val = 0.1
        b_val = 0.9

        self.min_root_val = scipy.optimize.newton(
            lambda x: BoundedSigmoid._sigmoid(x, a_val), 0)
        self.max_root_val = scipy.optimize.newton(
            lambda x: BoundedSigmoid._sigmoid(x, b_val), 0)

    @staticmethod
    def _sigmoid(x, root=0.0):
        """Calcualte sigmoid - root."""
        return 1/(1+numpy.exp(-x)) - root

    def __name__(self):
        """Return source code of object for taskgraph."""
        return inspect.getsource(BoundedSigmoid)

    def __call__(self, array):
        """Evaluate half sigmoid with given bounds.

        We want lower_bound to be when sigmoid == a_val
        We want upper_bound to be when sigmoid == b_val

        Args:
            array (numpy.array): arbitrary positive array

        Returns:
            when this value is 0, do a, when 1 do b
        """
        result = numpy.empty(array.shape)
        result[:] = _INDEX_NODATA
        valid_mask = (array >= 0) & numpy.isfinite(array)
        val_interp = (
            (array[valid_mask] - self.lower_bound) /
            (self.upper_bound - self.lower_bound))
        ab_interp = (
            self.min_root_val*(1-val_interp)+self.max_root_val*(val_interp))
        result[valid_mask] = BoundedSigmoid._sigmoid(ab_interp)
        return result


def _calculate_utm_wkt_region(base_raster_path):
    """Calculate which UTM zone the raster lies, return the WKT for it."""
    raster_info = geoprocessing.get_raster_info(base_raster_path)
    width, height = raster_info['raster_size']
    base_srs = osr.SpatialReference()
    base_srs.ImportFromWkt(raster_info['projection_wkt'])
    LOGGER.debug(base_srs)
    if base_srs.EPSGTreatsAsLatLong():
        LOGGER.debug('LAT/LNG base')
        centroid_x, centroid_y = gdal.ApplyGeoTransform(
            raster_info['geotransform'],
            raster_info['raster_size'][0]/2,
            raster_info['raster_size'][1]/2)

        utm_code = (numpy.floor((centroid_x + 180)/6) % 60) + 1
        lat_code = 6 if centroid_y > 0 else 7
        epsg_code = int('32%d%02d' % (lat_code, utm_code))
        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(epsg_code)
    else:
        target_srs = base_srs
    return target_srs.ExportToWkt()


def _resample_to_utm(base_raster_path, target_raster_path, pixel_scale=1.0):
    """Resample base to a square utm raster."""
    target_srs_wkt = _calculate_utm_wkt_region(base_raster_path)
    raster_info = geoprocessing.get_raster_info(base_raster_path)
    width, height = raster_info['raster_size']

    transformed_bb = geoprocessing.transform_bounding_box(
        raster_info['bounding_box'],
        raster_info['projection_wkt'],
        target_srs_wkt)

    pixel_x = (transformed_bb[2]-transformed_bb[0])/width
    pixel_y = (transformed_bb[3]-transformed_bb[1])/height
    pixel_length = min(pixel_x, pixel_y)
    target_pixel_size = (pixel_length*pixel_scale, -pixel_length*pixel_scale)

    geoprocessing.warp_raster(
        base_raster_path, target_pixel_size, target_raster_path, 'mode',
        target_projection_wkt=target_srs_wkt,
        working_dir=os.path.dirname(target_raster_path))


def _create_wddi(weighted_eft_raster_list, target_wddi_raster_path):
    """Create WDDI from given list of rasters.

    Create -> 1/sum((weighted_raster^2)) for all weighted rasters

    Args:
        weighted_eft_raster_list (list): path to 0/1 rasters indicating the
            weighted exponential decay of a particular EFT presence for a
            given alpha for a species.

        target_wddi_raster_path (str): path to WDDI raster.

    Returns:
        None
    """
    nodata = geoprocessing.get_raster_info(
        weighted_eft_raster_list[0])['nodata'][0]
    n_types = len(weighted_eft_raster_list)

    def _wddi_op(*array_list):
        """Calculate WDDI as described above."""
        result = numpy.zeros(array_list[0].shape)
        nodata_mask = ~numpy.isfinite(array_list[0])
        if nodata is not None:
            nodata_mask |= numpy.isclose(
                array_list[0], nodata)
            result[nodata_mask] = nodata
        sum_valid = numpy.zeros(result.shape)
        # assume zeros everywhere then when we encounter a non-zero flip it off
        zero_mask = numpy.ones(result.shape, dtype=numpy.bool)
        for array in array_list:
            sum_valid[~nodata_mask] += array[~nodata_mask]
            zero_mask &= numpy.isclose(array, 0.0)

        # do this so we don't get a divide error, we'll zero it out later
        sum_valid[zero_mask] = 1.0

        for array in array_list:
            result[~nodata_mask] += (
                array[~nodata_mask]/sum_valid[~nodata_mask])**2

        result[zero_mask] = nodata
        valid_mask = ~(nodata_mask | zero_mask)
        result[valid_mask] = 1/result[valid_mask]
        # since these come from convolutions there can be numerical noise that
        # explodes values, hence if it's an order of magnitude greater than
        # the biggest expected value, we set to 0.0 because it must be
        # numerical noise
        result[result > n_types**2] = 0.0
        return result

    geoprocessing.raster_calculator(
        [(path, 1) for path in weighted_eft_raster_list], _wddi_op,
        target_wddi_raster_path, gdal.GDT_Float32, nodata)


def _mask_raster(base_raster_path, unique_value, target_raster_path):
    """Create 0/1 nodata raster based on input values.

    Args:
        base_raster_path (str): path to single band raster.
        unique_value (numeric): desired value to mask against in base.
        target_raster_path (str): path to target raster that will have
            nodata where base is nodata, 1 where unique_value is present
            or 0.

    Returns:
        None
    """
    nodata = geoprocessing.get_raster_info(base_raster_path)['nodata'][0]
    local_nodata = 2

    def _mask_op(base_array):
        """Mask base to 0/1/nodata if base == unique_value."""
        result = numpy.zeros(base_array.shape, dtype=numpy.int8)
        result[base_array == unique_value] = 1
        if nodata is not None:
            result[numpy.isclose(base_array, nodata)] = local_nodata
        result[~numpy.isfinite(base_array)] = local_nodata
        return result

    geoprocessing.raster_calculator(
        [(base_raster_path, 1)], _mask_op, target_raster_path,
        gdal.GDT_Byte, local_nodata)


def _get_unique_values(raster_path):
    """Return set of unique values in the single band raster path."""
    unique_value_set = set()
    for _, raster_block in geoprocessing.iterblocks((raster_path, 1)):
        unique_values = numpy.unique(
            raster_block[numpy.isfinite(raster_block)])
        unique_value_set.update(unique_values)
    LOGGER.info(f'unique values found in dataset: {unique_value_set}')
    return unique_value_set


def execute(args):
    """Modified InVEST Pollination Model for EO.

    Args:
        args['workspace_dir'] (string): a path to the output workspace folder.
            Will overwrite any files that exist if the path already exists.
        args['results_suffix'] (string): string appended to each output
            file path.
        args['eft_raster_path'] (string): path to ecosystem functional type
            raster containing unique IDs for each unique EFT. Used in
            conjuction with the farm polygon's 'fr_eft' field as a unique
            field ID.
        eft_clip_raster_path (string): Ecosystem Functional Types raster
            whose pixels represent unique ecosystem
            functional types at a given pixel. The value is the "id" of that
            ecosystem functional type.
        args['guild_table_path'] (string): file path to a table indicating
            the bee species to analyze in this model run.  Table headers
            must include:
                * 'species': a bee species whose column string names will
                    be referred to in other tables and the model will output
                    analyses per species.
                * one or more columns matching _NESTING_SUITABILITY_PATTERN
                    with values in the range [0.0, 1.0] indicating the
                    suitability of the given species to nest in a particular
                    substrate.
                * _FLORAL_ALPHA_HEADER/_NESTING_ALPHA_HEADER the sigma average
                    flight distance of that bee species in meters for floral
                    foraging habitat suitability and nesting suitability.
                * 'relative_abundance': a weight indicating the relative
                    abundance of the particular species with respect to the
                    sum of all relative abundance weights in the table.

        args['farm_vector_path'] (string): (optional) path to a single layer
            polygon shapefile representing farms. If present will trigger the
            farm yield component of the model.

            The layer must have at least the following fields:

            * crop_type (string): a text field to identify the crop type for
                summary statistics.
            * half_sat (float): a real in the range [0.0, 1.0] representing
                the proportion of wild pollinators to achieve a 50% yield
                of that crop.
            * p_dep (float): a number in the range [0.0, 1.0]
                representing the proportion of yield dependent on pollinators.
            * p_managed (float): proportion of pollinators that come from
                non-native/managed hives.
            * fr_eft (int): Farm ecosystem functional type ID. This is the same
                ID set as those used in
            * n_[substrate] (float): One or more fields that match this
                pattern such that `substrate` also matches the nesting
                substrate headers in the biophysical and guild table.  Any
                areas that overlap the landcover map will replace nesting
                substrate suitability with this value.  Ranges from 0..1.
        args['n_workers'] (int): (optional) The number of worker processes to
            use for processing this model.  If omitted, computation will take
            place in the current process.

    Returns:
        None
    """
    rand_gen = numpy.random.RandomState(
        numpy.random.MT19937(numpy.random.SeedSequence(123456789)))

    # create initial working directories and determine file suffixes
    intermediate_output_dir = os.path.join(
        args['workspace_dir'], 'intermediate_outputs')
    work_token_dir = os.path.join(
        intermediate_output_dir, '_taskgraph_working_dir')
    output_dir = os.path.join(args['workspace_dir'])
    utils.make_directories(
        [output_dir, intermediate_output_dir])
    file_suffix = utils.make_suffix_string(args, 'results_suffix')

    if 'farm_vector_path' in args and args['farm_vector_path'] != '':
        # we set the vector path to be the projected vector that we'll create
        # later
        farm_vector_path = os.path.join(
            intermediate_output_dir,
            _PROJECTED_FARM_VECTOR_FILE_PATTERN % file_suffix)
    else:
        farm_vector_path = None

    # parse out the scenario variables from a complicated set of two tables
    # and possibly a farm polygon.  This function will also raise an exception
    # if any of the inputs are malformed.
    scenario_variables = _parse_scenario_variables(args)

    LOGGER.debug(f'these are the scenario_variables: {scenario_variables}')

    try:
        n_workers = int(args['n_workers'])
    except (KeyError, ValueError, TypeError):
        # KeyError when n_workers is not present in args
        # ValueError when n_workers is an empty string.
        # TypeError when n_workers is None.
        n_workers = -1  # Synchronous mode.
    task_graph = taskgraph.TaskGraph(work_token_dir, n_workers)
    eft_raster_info = geoprocessing.get_raster_info(
        args['eft_raster_path'])

    # process the EFT raster so pixels are square and desired resolution
    # clip the EFT raster to be the same size/projection as landcover map
    eft_clip_raster_path = os.path.join(
        intermediate_output_dir, _EFT_CLIP_FILE_PATTERN % file_suffix)
    eft_clip_task = task_graph.add_task(
        func=_resample_to_utm,
        args=(args['eft_raster_path'], eft_clip_raster_path),
        kwargs={'pixel_scale': args['pixel_scale']},
        target_path_list=[eft_clip_raster_path],
        task_name='clip EFT raster')

    if farm_vector_path is not None:
        # ensure farm vector is in the same projection as the eft map
        reproject_farm_task = task_graph.add_task(
            task_name='reproject_farm_task',
            func=geoprocessing.reproject_vector,
            args=(
                args['farm_vector_path'],
                _calculate_utm_wkt_region(args['eft_raster_path']),
                farm_vector_path),
            kwargs={'driver_name': 'GPKG'},
            target_path_list=[farm_vector_path])

        # rasterize farm polygon fr_eft to eft raster path
        eft_farm_raster_path = os.path.join(
            intermediate_output_dir, _EFT_FARM_FILE_PATTERN % file_suffix)
        rasterize_farm_task = task_graph.add_task(
            func=_rasterize_vector_onto_base,
            args=(
                eft_clip_raster_path, farm_vector_path,
                args['farm_floral_eft_field'],
                eft_farm_raster_path),
            target_path_list=[eft_farm_raster_path],
            dependent_task_list=[reproject_farm_task, eft_clip_task],
            task_name=f'rasterize EFTs from farm on top of global EFT')
        # sneak the rasterized farm eft as the primary eft raster by
        # overwriting these variables, if no farm vector then they are
        # the original clipped values
        eft_clip_task = rasterize_farm_task
        eft_clip_raster_path = eft_farm_raster_path

    # get unique eft codes
    eft_code_list = task_graph.add_task(
        func=_get_unique_values,
        args=(eft_clip_raster_path,),
        dependent_task_list=[eft_clip_task],
        store_result=True,
        task_name='get unique values from {eft_clip_raster_path}')
    eft_code_list.join()
    eft_clip_raster_info = geoprocessing.get_raster_info(
        eft_clip_raster_path)
    mean_pixel_size = utils.mean_pixel_size_and_area(
        eft_clip_raster_info['pixel_size'])[0]

    # TODO: calculate WDDI per species
    wddi_dir = os.path.join(intermediate_output_dir, 'wddi_rasters')
    try:
        os.makedirs(wddi_dir)
    except OSError:
        pass
    efd_mask_dir = _mkdir(os.path.join(intermediate_output_dir, 'eft_masks'))

    eft_to_raster_task_map = {}
    for eft_code in eft_code_list.get():
        if eft_code == eft_raster_info['nodata'][0]:
            continue
        eft_mask_raster_path = os.path.join(
            efd_mask_dir, f'eft_mask_{eft_code}{file_suffix}.tif')
        eft_mask_task = task_graph.add_task(
            func=_mask_raster,
            args=(
                eft_clip_raster_path, eft_code, eft_mask_raster_path),
            dependent_task_list=[eft_clip_task],
            target_path_list=[eft_mask_raster_path],
            task_name=f'mask {eft_code} for eft')
        eft_to_raster_task_map[eft_code] = (
            eft_mask_raster_path, eft_mask_task)

    abundance_path_weight_list = []
    abundance_task_list = []
    for species in scenario_variables['species_list']:
        resources_to_raster_task_map = {}
        for alpha_type, biophysical_type, biophysical_file_pattern in [
                ('floral', 'floral_resources',
                 _RELATIVE_FLORAL_ABUNDANCE_INDEX_FILE_PATTERN),
                ('nesting', 'nesting_suitability',
                 _NESTING_SUBSTRATE_INDEX_FILE_PATTERN)]:
            alpha_field = f'{alpha_type}_alpha'
            alpha = (
                scenario_variables[alpha_field][species] /
                mean_pixel_size)
            # this prevents us from having duplicate kernels
            alpha += rand_gen.random()*alpha/1e8
            kernel_path = os.path.join(
                efd_mask_dir, _KERNEL_FILE_PATTERN % (
                    species, alpha_type, alpha, file_suffix))
            alpha_kernel_raster_task = task_graph.add_task(
                task_name=f'decay_kernel_raster_{species}_{alpha:.8f}',
                func=utils.exponential_decay_kernel_raster,
                args=(alpha, kernel_path),
                copy_duplicate_artifact=True,
                target_path_list=[kernel_path])
            if alpha_type == 'floral':
                # this is for flying to farms later
                flight_kernel_path = kernel_path

            weighted_eft_raster_list = []
            weighted_eft_task_list = []
            for eft_code in eft_code_list.get():
                if eft_code == eft_raster_info['nodata'][0]:
                    continue
                eft_weighted_path = os.path.join(
                    efd_mask_dir,
                    f'weighted_eft_mask_{species}_{alpha_type}_'
                    f'{eft_code}{file_suffix}.tif')
                eft_mask_raster_path, eft_mask_task = \
                    eft_to_raster_task_map[eft_code]
                create_efd_weighted_task = task_graph.add_task(
                    func=geoprocessing.convolve_2d,
                    args=(
                        (eft_mask_raster_path, 1), (kernel_path, 1),
                        eft_weighted_path),
                    kwargs={
                        'ignore_nodata_and_edges': True,
                        'mask_nodata': True,
                        'normalize_kernel': False,
                        },
                    dependent_task_list=[
                        eft_mask_task, alpha_kernel_raster_task],
                    target_path_list=[eft_weighted_path],
                    task_name=f'create efd for {species}')
                weighted_eft_raster_list.append(eft_weighted_path)
                weighted_eft_task_list.append(create_efd_weighted_task)

            wddi_raster_path = os.path.join(
                wddi_dir,
                f'wddi_{species}_{alpha_type}_{file_suffix}.tif')
            create_wddi_task = task_graph.add_task(
                func=_create_wddi,
                args=(weighted_eft_raster_list, wddi_raster_path),
                dependent_task_list=weighted_eft_task_list,
                target_path_list=[wddi_raster_path],
                task_name=f'create {alpha_type} wddi for {species}')

            biophysical_raster_path = os.path.join(
                intermediate_output_dir, biophysical_file_pattern % (
                    species, file_suffix))
            efd_min = scenario_variables[
                f'{biophysical_type}_efd_min'][species]
            efd_sufficient = scenario_variables[
                f'{biophysical_type}_efd_sufficient'][species]

            biophysical_task = task_graph.add_task(
                func=geoprocessing.raster_calculator,
                args=(
                    [(wddi_raster_path, 1)],
                    BoundedSigmoid(efd_min, efd_sufficient),
                    biophysical_raster_path, gdal.GDT_Float32, _INDEX_NODATA),
                target_path_list=[biophysical_raster_path],
                dependent_task_list=[create_wddi_task],
                task_name=f'create_wddi for {species} {biophysical_type}')
            resources_to_raster_task_map[biophysical_type] = (
                biophysical_raster_path, biophysical_task)

        # calculate local pollinator supply as FR*HN
        pollinator_supply_index_path = os.path.join(
            intermediate_output_dir, _POLLINATOR_SUPPLY_FILE_PATTERN % (
                species, file_suffix))
        pollinator_supply_task = task_graph.add_task(
            task_name=f'calculate pollinator supply {species}',
            func=geoprocessing.raster_calculator,
            args=(
                [(resources_to_raster_task_map['floral_resources'][0], 1),
                 (resources_to_raster_task_map['nesting_suitability'][0], 1)],
                ps_supply_op, pollinator_supply_index_path,
                gdal.GDT_Float32, _INDEX_NODATA),
            dependent_task_list=[
                resources_to_raster_task_map['floral_resources'][1],
                resources_to_raster_task_map['nesting_suitability'][1]],
            target_path_list=[pollinator_supply_index_path])

        # fly the pollinator supply to cover farms
        pollinator_abundance_raster_path = os.path.join(
            intermediate_output_dir, _POLLINATOR_ABUNDANCE_FILE_PATTERN % (
                species, file_suffix))
        abundance_task = task_graph.add_task(
            func=geoprocessing.convolve_2d,
            args=(
                (pollinator_supply_index_path, 1),
                (flight_kernel_path, 1),
                pollinator_abundance_raster_path),
            kwargs={
                'ignore_nodata_and_edges': True,
                'mask_nodata': True,
                'normalize_kernel': False,
                },
            dependent_task_list=[
                pollinator_supply_task, alpha_kernel_raster_task],
            target_path_list=[pollinator_abundance_raster_path],
            task_name=f'create pollinator supply for {species}')

        abundance_task_list.append(abundance_task)
        abundance_path_weight_list.append(
            (pollinator_abundance_raster_path, 1))
        abundance_path_weight_list.append(
            (scenario_variables['species_abundance'][species], 'raw'))

    # combine the pollinator abundance to total abundance based on relative values
    global_pollinator_abundance_raster_path = os.path.join(
        args['workspace_dir'],
        _TOTAL_POLLINATOR_ABUNDANCE_FILE_PATTERN % file_suffix)
    global_abundance_task = task_graph.add_task(
        func=geoprocessing.raster_calculator,
        args=(
            [(_INDEX_NODATA, 'raw')] + abundance_path_weight_list,
            _weighted_average_op, global_pollinator_abundance_raster_path,
            gdal.GDT_Float32, _INDEX_NODATA),
        target_path_list=[global_pollinator_abundance_raster_path],
        dependent_task_list=abundance_task_list)

    # calculate farm yield
    blank_raster_path = os.path.join(
        intermediate_output_dir, _BLANK_RASTER_FILE_PATTERN % file_suffix)
    blank_raster_task = task_graph.add_task(
        task_name='create_blank_raster',
        func=geoprocessing.new_raster_from_base,
        args=(
            eft_clip_raster_path, blank_raster_path, gdal.GDT_Float32,
            [_INDEX_NODATA]),
        target_path_list=[blank_raster_path])

    half_saturation_raster_path = os.path.join(
        intermediate_output_dir, _HALF_SATURATION_FILE_PATTERN % file_suffix)
    half_saturation_task = task_graph.add_task(
        task_name='rasterize half saturation',
        func=_rasterize_vector_onto_base,
        args=(
            blank_raster_path, farm_vector_path,
            _HALF_SATURATION_FARM_HEADER, half_saturation_raster_path),
        dependent_task_list=[blank_raster_task],
        target_path_list=[half_saturation_raster_path])

    farm_pollinator_path = os.path.join(
        intermediate_output_dir, _FARM_POLLINATOR_FILE_PATTERN % file_suffix)
    task_graph.add_task(
        task_name='calculate on farm pollinator',
        func=geoprocessing.raster_calculator,
        args=(
            [(half_saturation_raster_path, 1),
             (global_pollinator_abundance_raster_path, 1)],
            _OnFarmPollinatorAbundance(), farm_pollinator_path,
            gdal.GDT_Float32, _INDEX_NODATA),
        dependent_task_list=[half_saturation_task, global_abundance_task],
        target_path_list=[farm_pollinator_path])

    task_graph.join()

    target_farm_result_path = os.path.join(
        output_dir, _FARM_VECTOR_RESULT_FILE_PATTERN % file_suffix)
    LOGGER.debug(f'create {target_farm_result_path} from {farm_vector_path}')
    _create_farm_result_vector(
        farm_vector_path, target_farm_result_path)

    # aggregate yield over a farm
    total_farm_results = geoprocessing.zonal_statistics(
        (global_pollinator_abundance_raster_path, 1),
        target_farm_result_path, polygons_might_overlap=False)

    LOGGER.debug(f'total_farm_results: {total_farm_results}')

    target_farm_vector = gdal.OpenEx(target_farm_result_path, 1)
    target_farm_layer = target_farm_vector.GetLayer()

    # aggregate results per farm
    for farm_feature in target_farm_layer:
        nu = float(farm_feature.GetField(_CROP_POLLINATOR_DEPENDENCE_FIELD))
        fid = farm_feature.GetFID()
        if total_farm_results[fid]['count'] > 0:
            # total pollinator farm yield is 1-*nu(1-tot_pollination_coverage)
            # this is YT from the user's guide (y_tot)
            farm_feature.SetField(
                _TOTAL_FARM_YIELD_FIELD_ID,
                1 - nu * (
                    1 - total_farm_results[fid]['sum'] /
                    float(total_farm_results[fid]['count'])))

            # this is PAT ('p_abund')
            farm_feature.SetField(
                _POLLINATOR_ABUDNANCE_FARM_FIELD_ID,
                total_farm_results[fid]['sum'] /
                float(total_farm_results[fid]['count']))

        target_farm_layer.SetFeature(farm_feature)
    target_farm_layer.SyncToDisk()
    target_farm_layer = None
    target_farm_vector = None

    task_graph.close()
    task_graph.join()
    task_graph = None
    return


def _rasterize_vector_onto_base(
        base_raster_path, base_vector_path, attribute_id,
        target_raster_path, filter_string=None):
    """Rasterize attribute from vector onto a copy of base.

    Parameters:
        base_raster_path (string): path to a base raster file
        attribute_id (string): id in `base_vector_path` to rasterize.
        target_raster_path (string): a copy of `base_raster_path` with
            `base_vector_path[attribute_id]` rasterized on top.
        filter_string (string): filtering string to select from farm layer

    Returns:
        None.
    """
    base_raster = gdal.OpenEx(base_raster_path, gdal.OF_RASTER)
    raster_driver = gdal.GetDriverByName('GTiff')
    target_raster = raster_driver.CreateCopy(target_raster_path, base_raster)
    base_raster = None

    vector = gdal.OpenEx(base_vector_path)
    layer = vector.GetLayer()

    if filter_string is not None:
        layer.SetAttributeFilter(str(filter_string))
    gdal.RasterizeLayer(
        target_raster, [1], layer,
        options=[f'ATTRIBUTE={attribute_id}', 'ALL_TOUCHED=TRUE'])
    target_raster.FlushCache()
    target_raster = None
    layer = None
    vector = None


def _create_farm_result_vector(
        base_vector_path, target_vector_path):
    """Create a copy of `base_vector_path` and add FID field to it.

    Parameters:
        base_vector_path (string): path to vector to copy
        target_vector_path (string): path to target vector that is a copy
            of the base, except for the new `fid_field_id` field that has
            unique integer IDs for each feature.  This path must not already
            exist.  It also has new entries for all the result fields:
                _TOTAL_FARM_YIELD_FIELD_ID

    Returns:
        None.

    """
    if os.path.exists(target_vector_path):
        os.remove(target_vector_path)
    base_vector = ogr.Open(base_vector_path)
    driver = ogr.GetDriverByName('GPKG')
    target_vector = driver.CopyDataSource(base_vector, target_vector_path)
    target_layer = target_vector.GetLayer()

    farm_pollinator_abundance_defn = ogr.FieldDefn(
        _POLLINATOR_ABUDNANCE_FARM_FIELD_ID, ogr.OFTReal)
    farm_pollinator_abundance_defn.SetWidth(25)
    farm_pollinator_abundance_defn.SetPrecision(11)
    target_layer.CreateField(farm_pollinator_abundance_defn)

    total_farm_yield_field_defn = ogr.FieldDefn(
        _TOTAL_FARM_YIELD_FIELD_ID, ogr.OFTReal)
    total_farm_yield_field_defn.SetWidth(25)
    total_farm_yield_field_defn.SetPrecision(11)
    target_layer.CreateField(total_farm_yield_field_defn)

    target_vector.FlushCache()
    target_layer = None
    target_vector = None


def _parse_scenario_variables(args):
    """Parse out scenario variables from input parameters.

    This function parses through the guild table, biophysical table, and
    farm polygons (if available) to generate

    Parameter:
        args (dict): this is the args dictionary passed in to the `execute`
            function, requires a 'guild_table_path', and optional
            'farm_vector_path' key.

    Returns:
        A dictionary with the keys:
            * season_list (list of string)
            * species_list (list of string)
            * alpha_value[species] (float)
            * species_abundance[species] (string->float)
            * species_foraging_activity[(species, season)] (string->float)
            * foraging_activity_index[(species, season)] (tuple->float)
    """
    guild_table_path = args['guild_table_path']
    if 'farm_vector_path' in args and args['farm_vector_path'] != '':
        farm_vector_path = args['farm_vector_path']
    else:
        farm_vector_path = None

    guild_table = utils.build_lookup_from_csv(
        guild_table_path, 'species', to_lower=True)

    LOGGER.info('Checking to make sure guild table has all expected headers')
    guild_headers = list(guild_table.values())[0].keys()
    for header in _EXPECTED_GUILD_HEADERS:
        matches = re.findall(header, " ".join(guild_headers))
        if len(matches) == 0:
            raise ValueError(
                "Expected a header in guild table that matched the pattern "
                "'%s' but was unable to find one.  Here are all the headers "
                "from %s: %s" % (
                    header, guild_table_path,
                    guild_headers))

    farm_vector = None
    if farm_vector_path is not None:
        LOGGER.info('Checking that farm polygon has expected headers')
        farm_vector = gdal.OpenEx(farm_vector_path)
        farm_layer = farm_vector.GetLayer()
        if farm_layer.GetGeomType() not in [
                ogr.wkbPolygon, ogr.wkbMultiPolygon]:
            farm_layer = None
            farm_vector = None
            raise ValueError("Farm layer not a polygon type")
        farm_layer_defn = farm_layer.GetLayerDefn()
        farm_headers = [
            farm_layer_defn.GetFieldDefn(i).GetName()
            for i in range(farm_layer_defn.GetFieldCount())]
        for header in _EXPECTED_FARM_HEADERS:
            matches = re.findall(header, " ".join(farm_headers))
            if not matches:
                raise ValueError(
                    "Missing an expected headers '%s'from %s.\n"
                    "Got these headers instead %s" % (
                        header, farm_vector_path, farm_headers))

    result = collections.defaultdict(dict)
    # * species_list (list of string)
    result['species_list'] = sorted(guild_table)
    for species in result['species_list']:
        result[_FLORAL_ALPHA_HEADER][species] = float(
            guild_table[species][_FLORAL_ALPHA_HEADER])
        result[_NESTING_ALPHA_HEADER][species] = float(
            guild_table[species][_NESTING_ALPHA_HEADER])
        result['nesting_suitability_efd_min'][species] = float(
            guild_table[species]['nesting_suitability_efd_min'])
        result['nesting_suitability_efd_sufficient'][species] = float(
            guild_table[species]['nesting_suitability_efd_sufficient'])
        result['floral_resources_efd_min'][species] = float(
            guild_table[species]['floral_resources_efd_min'])
        result['floral_resources_efd_sufficient'][species] = float(
            guild_table[species]['floral_resources_efd_sufficient'])

    # * species_abundance[species] (string->float)
    total_relative_abundance = numpy.sum([
        guild_table[species][_RELATIVE_SPECIES_ABUNDANCE_FIELD]
        for species in result['species_list']])
    result['species_abundance'] = {}
    for species in result['species_list']:
        result['species_abundance'][species] = (
            guild_table[species][_RELATIVE_SPECIES_ABUNDANCE_FIELD] /
            float(total_relative_abundance))

    return result


def ps_supply_op(
        floral_resources_array, habitat_nesting_suitability_array):
    """Calculate f_r * h_n."""
    result = numpy.empty_like(floral_resources_array)
    result[:] = _INDEX_NODATA
    valid_mask = ~numpy.isclose(floral_resources_array, _INDEX_NODATA)
    result[valid_mask] = (
        floral_resources_array[valid_mask] *
        habitat_nesting_suitability_array[valid_mask])
    return result


class _OnFarmPollinatorAbundance(object):
    """Calculate FP(x) = (PAT * (1 - h)) / (h * (1 - 2*pat)+pat))."""

    def __init__(self):
        # try to get the source code of __call__ so task graph will recompute
        # if the function has changed
        try:
            self.__name__ = hashlib.sha1(
                inspect.getsource(
                    _OnFarmPollinatorAbundance.__call__
                ).encode('utf-8')).hexdigest()
        except IOError:
            # default to the classname if it doesn't work
            self.__name__ = (
                _OnFarmPollinatorAbundance.__name__)

    def __call__(self, h_array, pat_array):
        """Return (pad * (1 - h)) / (h * (1 - 2*pat)+pat)) tolerate nodata."""
        result = numpy.empty_like(h_array)
        result[:] = _INDEX_NODATA

        valid_mask = (h_array != _INDEX_NODATA) & (pat_array != _INDEX_NODATA)

        result[valid_mask] = (
            (pat_array[valid_mask]*(1-h_array[valid_mask])) /
            (h_array[valid_mask]*(1-2*pat_array[valid_mask]) +
             pat_array[valid_mask]))
        return result
