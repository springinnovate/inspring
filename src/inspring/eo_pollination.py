"""Earth Observation Driven Pollinator service model for inspring."""
import collections
import hashlib
import inspect
import logging
import os
import re

from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import pygeoprocessing
import numpy
import scipy.optimize
import taskgraph

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
_PROJECTED_FARM_VECTOR_FILE_PATTERN = 'reprojected_farm_vector%s.shp'
# used to store the 2D decay kernel for a given distance replace
# (species, alpha_type, alpha, file suffix)
_KERNEL_FILE_PATTERN = 'kernel_%s%s%f%s.tif'
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
# blank raster as a basis to rasterize on replace (file_suffix)
_BLANK_RASTER_FILE_PATTERN = 'blank_raster%s.tif'
# raster to hold farm pollinator replace (species, file_suffix)
_FARM_POLLINATOR_FILE_PATTERN = 'farm_pollinator_%s%s.tif'
# managed pollinator indexes replace (file_suffix)
_MANAGED_POLLINATOR_FILE_PATTERN = 'managed_pollinators%s.tif'
# total pollinator raster replace (file_suffix)
_TOTAL_POLLINATOR_YIELD_FILE_PATTERN = 'total_pollinator_yield%s.tif'
# wild pollinator raster replace (file_suffix)
_WILD_POLLINATOR_YIELD_FILE_PATTERN = 'wild_pollinator_yield%s.tif'
# final aggregate farm shapefile file pattern replace (file_suffix)
_FARM_VECTOR_RESULT_FILE_PATTERN = 'farm_results%s.shp'
# output field on target shapefile if farms are enabled
_TOTAL_FARM_YIELD_FIELD_ID = 'y_tot'
# output field for wild pollinators on farms if farms are enabled
_WILD_POLLINATOR_FARM_YIELD_FIELD_ID = 'y_wild'
# output field for proportion of wild pollinators over the pollinator
# dependent part of the yield
_POLLINATOR_PROPORTION_FARM_YIELD_FIELD_ID = 'pdep_y_w'
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


def _resample_to_utm(base_raster_path, target_raster_path, pixel_scale=1.0):
    """Resample base to a square utm raster."""
    raster_info = pygeoprocessing.get_raster_info(base_raster_path)
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

    transformed_bb = pygeoprocessing.transform_bounding_box(
        raster_info['bounding_box'],
        raster_info['projection_wkt'],
        target_srs.ExportToWkt())

    pixel_x = (transformed_bb[2]-transformed_bb[0])/width
    pixel_y = (transformed_bb[3]-transformed_bb[1])/height
    pixel_length = min(pixel_x, pixel_y)
    target_pixel_size = (pixel_length*pixel_scale, -pixel_length*pixel_scale)

    pygeoprocessing.warp_raster(
        base_raster_path, target_pixel_size, target_raster_path, 'mode',
        target_projection_wkt=target_srs.ExportToWkt(),
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
    nodata = pygeoprocessing.get_raster_info(
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

    pygeoprocessing.raster_calculator(
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
    nodata = pygeoprocessing.get_raster_info(base_raster_path)['nodata'][0]
    local_nodata = 2

    def _mask_op(base_array):
        """Mask base to 0/1/nodata if base == unique_value."""
        result = numpy.zeros(base_array.shape, dtype=numpy.int8)
        result[base_array == unique_value] = 1
        if nodata is not None:
            result[numpy.isclose(base_array, nodata)] = local_nodata
        result[~numpy.isfinite(base_array)] = local_nodata
        return result

    pygeoprocessing.raster_calculator(
        [(base_raster_path, 1)], _mask_op, target_raster_path,
        gdal.GDT_Byte, local_nodata)


def _get_unique_values(raster_path):
    """Return set of unique values in the single band raster path."""
    unique_value_set = set()
    for _, raster_block in pygeoprocessing.iterblocks((raster_path, 1)):
        unique_values = numpy.unique(
            raster_block[numpy.isfinite(raster_block)])
        unique_value_set.update(unique_values)
    LOGGER.info(f'unique values found in dataset: {unique_value_set}')
    return unique_value_set


def execute(args):
    """InVEST Pollination Model.

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
    eft_raster_info = pygeoprocessing.get_raster_info(
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
            func=pygeoprocessing.reproject_vector,
            args=(
                args['farm_vector_path'],
                eft_raster_info['projection_wkt'], farm_vector_path),
            target_path_list=[farm_vector_path])

        # rasterize farm polygon fr_eft to eft raster path
        eft_farm_raster_path = os.path.join(
            intermediate_output_dir, _EFT_FARM_FILE_PATTERN % file_suffix)
        rasterize_farm_task = task_graph.add_task(
            func=_rasterize_vector_onto_base,
            args=(
                eft_clip_raster_path, farm_vector_path,
                scenario_variables['farm_floral_eft_field'],
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
    eft_clip_raster_info = pygeoprocessing.get_raster_info(
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
        wddi_alpha_raster_task_map = {}
        for alpha_type in ['floral', 'nesting']:
            alpha_field = f'{alpha_type}_alpha'
            alpha = (
                scenario_variables[alpha_field][species] /
                mean_pixel_size)
            kernel_path = os.path.join(
                efd_mask_dir, _KERNEL_FILE_PATTERN % (
                    species, alpha_type, alpha, file_suffix))
            alpha_kernel_raster_task = task_graph.add_task(
                task_name='decay_kernel_raster_%s' % alpha,
                func=utils.exponential_decay_kernel_raster,
                args=(alpha, kernel_path),
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
                    func=pygeoprocessing.convolve_2d,
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
            wddi_alpha_raster_task_map[alpha_type] = (
                wddi_raster_path, create_wddi_task)

        resources_to_raster_task_map = {}
        for (biophysical_type, biophysical_file_pattern) in [
                ('floral_resources',
                 _RELATIVE_FLORAL_ABUNDANCE_INDEX_FILE_PATTERN),
                ('nesting_suitability',
                 _NESTING_SUBSTRATE_INDEX_FILE_PATTERN)]:
            alpha_type = biophysical_type.split('_')[0]
            biophysical_raster_path = os.path.join(
                intermediate_output_dir, biophysical_file_pattern % (
                    species, file_suffix))

            efd_min = scenario_variables[
                f'{biophysical_type}_efd_min'][species]
            efd_sufficient = scenario_variables[
                f'{biophysical_type}_efd_sufficient'][species]

            biophysical_task = task_graph.add_task(
                func=pygeoprocessing.raster_calculator,
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
            func=pygeoprocessing.raster_calculator,
            args=(
                [(resources_to_raster_task_map['floral_resources'][0], 1),
                 (resources_to_raster_task_map['nesting_suitability'][0], 1),
                 (scenario_variables['species_abundance'][species], 'raw')],
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
            func=pygeoprocessing.convolve_2d,
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
        func=pygeoprocessing.raster_calculator,
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
        func=pygeoprocessing.new_raster_from_base,
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

    task_graph.close()
    task_graph.join()
    task_graph = None
    return

    pollinator_abundance_path_map = {}
    pollinator_abundance_task_map = {}
    floral_resources_index_path_map = {}
    floral_resources_index_task_map = {}
    for species in scenario_variables['species_list']:
        # calculate foraging_effectiveness[species]
        # FE(x, s) = sum_j [RA(l(x), j) * fa(s, j)]
        foraged_flowers_path_band_list = [
            (scenario_variables['foraged_flowers_index_path'][
                (species, season)], 1)
            for season in scenario_variables['season_list']]
        local_foraging_effectiveness_path = os.path.join(
            intermediate_output_dir,
            _LOCAL_FORAGING_EFFECTIVENESS_FILE_PATTERN % (
                species, file_suffix))

        local_foraging_effectiveness_task = task_graph.add_task(
            task_name='local_foraging_effectiveness_%s' % species,
            func=pygeoprocessing.raster_calculator,
            args=(
                foraged_flowers_path_band_list,
                _SumRasters(), local_foraging_effectiveness_path,
                gdal.GDT_Float32, _INDEX_NODATA),
            target_path_list=[
                local_foraging_effectiveness_path],
            dependent_task_list=[
                foraged_flowers_index_task_map[(species, season)]
                for season in scenario_variables['season_list']])

        landcover_pixel_size_tuple = eft_raster_info['pixel_size']
        try:
            landcover_mean_pixel_size = utils.mean_pixel_size_and_area(
                landcover_pixel_size_tuple)[0]
        except ValueError:
            landcover_mean_pixel_size = numpy.min(numpy.absolute(
                landcover_pixel_size_tuple))
            LOGGER.debug(
                'Land Cover Raster has unequal x, y pixel sizes: %s. Using'
                '%s as the mean pixel size.' % (
                    landcover_pixel_size_tuple, landcover_mean_pixel_size))

        # create a convolution kernel for the species flight range
        alpha = (
            scenario_variables['alpha_value'][species] /
            landcover_mean_pixel_size)
        kernel_path = os.path.join(
            intermediate_output_dir, _KERNEL_FILE_PATTERN % (
                alpha, file_suffix))

        alpha_kernel_raster_task = task_graph.add_task(
            task_name='decay_kernel_raster_%s' % alpha,
            func=utils.exponential_decay_kernel_raster,
            args=(alpha, kernel_path),
            target_path_list=[kernel_path])

        # create EFD for species
        efd_raster_path = os.path.join(
            intermediate_output_dir, _EFD_FILE_PATTERN % (
                species, file_suffix))
        create_efd_species_task = task_graph.add_task(
            func=pygeoprocessing.convolve_2d,
            args=(
                (eft_clip_raster_path, 1), (kernel_path, 1),
                efd_raster_path),
            kwargs={
                'ignore_nodata_and_edges': True,
                'mask_nodata': True,
                'normalize_kernel': False,
                },
            dependent_task_list=[eft_clip_task],
            task_name=f'create efd for {species}')

        # TODO: rather than calculating nesting habitat, use the efd for
        #       nesting habitat

        # convolve FE with alpha_s
        floral_resources_index_path = os.path.join(
            intermediate_output_dir, _FLORAL_RESOURCES_INDEX_FILE_PATTERN % (
                species, file_suffix))
        floral_resources_index_path_map[species] = floral_resources_index_path

        floral_resources_task = task_graph.add_task(
            task_name='convolve_%s' % species,
            func=pygeoprocessing.convolve_2d,
            args=(
                (local_foraging_effectiveness_path, 1), (kernel_path, 1),
                floral_resources_index_path),
            kwargs={
                'ignore_nodata_and_edges': True,
                'mask_nodata': True,
                'normalize_kernel': False,
                },
            dependent_task_list=[
                alpha_kernel_raster_task, local_foraging_effectiveness_task],
            target_path_list=[floral_resources_index_path])

        floral_resources_index_task_map[species] = floral_resources_task
        # calculate
        # pollinator_supply_index[species] PS(x,s) = FR(x,s) * HN(x,s) * sa(s)
        pollinator_supply_index_path = os.path.join(
            output_dir, _POLLINATOR_SUPPLY_FILE_PATTERN % (
                species, file_suffix))
        ps_index_op = _PollinatorSupplyIndexOp(
            scenario_variables['species_abundance'][species])
        pollinator_supply_task = task_graph.add_task(
            task_name='calculate_pollinator_supply_%s' % species,
            func=pygeoprocessing.raster_calculator,
            args=(
                [(scenario_variables['habitat_nesting_index_path'][species],
                  1),
                 (floral_resources_index_path, 1)], ps_index_op,
                pollinator_supply_index_path, gdal.GDT_Float32,
                _INDEX_NODATA),
            dependent_task_list=[
                floral_resources_task, habitat_nesting_tasks[species]],
            target_path_list=[pollinator_supply_index_path])

        # calc convolved_PS PS over alpha_s
        convolve_ps_path = os.path.join(
            intermediate_output_dir, _CONVOLVE_PS_FILE_PATH % (
                species, file_suffix))

        convolve_ps_task = task_graph.add_task(
            task_name='convolve_ps_%s' % species,
            func=pygeoprocessing.convolve_2d,
            args=(
                (pollinator_supply_index_path, 1), (kernel_path, 1),
                convolve_ps_path),
            kwargs={
                'ignore_nodata_and_edges': True,
                'mask_nodata': True,
                'normalize_kernel': False,
                },
            dependent_task_list=[
                alpha_kernel_raster_task, pollinator_supply_task],
            target_path_list=[convolve_ps_path])

        for season in scenario_variables['season_list']:
            # calculate pollinator activity as
            # PA(x,s,j)=RA(l(x),j)fa(s,j) convolve(ps, alpha_s)
            foraged_flowers_index_path = (
                scenario_variables['foraged_flowers_index_path'][
                    (species, season)])
            pollinator_abundance_path = os.path.join(
                output_dir, _POLLINATOR_ABUNDANCE_FILE_PATTERN % (
                    species, season, file_suffix))
            pollinator_abundance_task_map[(species, season)] = (
                task_graph.add_task(
                    task_name='calculate_poll_abudance_%s' % species,
                    func=pygeoprocessing.raster_calculator,
                    args=(
                        [(foraged_flowers_index_path, 1),
                         (floral_resources_index_path_map[species], 1),
                         (convolve_ps_path, 1)],
                        _PollinatorSupplyOp(), pollinator_abundance_path,
                        gdal.GDT_Float32, _INDEX_NODATA),
                    dependent_task_list=[
                        foraged_flowers_index_task_map[(species, season)],
                        floral_resources_index_task_map[species],
                        convolve_ps_task],
                    target_path_list=[pollinator_abundance_path]))
            pollinator_abundance_path_map[(species, season)] = (
                pollinator_abundance_path)

    # next step is farm vector calculation, if no farms then okay to quit
    if farm_vector_path is None:
        task_graph.close()
        task_graph.join()
        return

    # blank raster used for rasterizing all the farm parameters/fields later
    blank_raster_path = os.path.join(
        intermediate_output_dir, _BLANK_RASTER_FILE_PATTERN % file_suffix)
    blank_raster_task = task_graph.add_task(
        task_name='create_blank_raster',
        func=pygeoprocessing.new_raster_from_base,
        args=(
            args['landcover_raster_path'], blank_raster_path,
            gdal.GDT_Float32, [_INDEX_NODATA]),
        kwargs={'fill_value_list': [_INDEX_NODATA]},
        target_path_list=[blank_raster_path])

    farm_pollinator_season_path_list = []
    farm_pollinator_season_task_list = []
    total_pollinator_abundance_task = {}
    for season in scenario_variables['season_list']:
        # total_pollinator_abundance_index[season] PAT(x,j)=sum_s PA(x,s,j)
        total_pollinator_abundance_index_path = os.path.join(
            output_dir, _TOTAL_POLLINATOR_ABUNDANCE_FILE_PATTERN % (
                season, file_suffix))

        pollinator_abudnance_season_path_band_list = [
            (pollinator_abundance_path_map[(species, season)], 1)
            for species in scenario_variables['species_list']]

        total_pollinator_abundance_task[season] = task_graph.add_task(
            task_name='calculate_poll_abudnce_%s_%s' % (species, season),
            func=pygeoprocessing.raster_calculator,
            args=(
                pollinator_abudnance_season_path_band_list, _SumRasters(),
                total_pollinator_abundance_index_path, gdal.GDT_Float32,
                _INDEX_NODATA),
            dependent_task_list=[
                pollinator_abundance_task_map[(species, season)]
                for species in scenario_variables['species_list']],
            target_path_list=[total_pollinator_abundance_index_path])

        half_saturation_raster_path = os.path.join(
            intermediate_output_dir, _HALF_SATURATION_FILE_PATTERN % (
                season, file_suffix))
        half_saturation_task = task_graph.add_task(
            task_name='half_saturation_rasterize_%s' % season,
            func=_rasterize_vector_onto_base,
            args=(
                blank_raster_path, farm_vector_path,
                _HALF_SATURATION_FARM_HEADER, half_saturation_raster_path),
            kwargs={'filter_string': "%s='%s'" % (_FARM_SEASON_FIELD, season)},
            dependent_task_list=[blank_raster_task],
            target_path_list=[half_saturation_raster_path])

        # calc on farm pollinator abundance i.e. FP_season
        farm_pollinator_season_path = os.path.join(
            intermediate_output_dir, _FARM_POLLINATOR_SEASON_FILE_PATTERN % (
                season, file_suffix))
        farm_pollinator_season_task_list.append(task_graph.add_task(
            task_name='farm_pollinator_%s' % season,
            func=pygeoprocessing.raster_calculator,
            args=(
                [(half_saturation_raster_path, 1),
                 (total_pollinator_abundance_index_path, 1)],
                _OnFarmPollinatorAbundance(), farm_pollinator_season_path,
                gdal.GDT_Float32, _INDEX_NODATA),
            dependent_task_list=[
                half_saturation_task, total_pollinator_abundance_task[season]],
            target_path_list=[farm_pollinator_season_path]))
        farm_pollinator_season_path_list.append(farm_pollinator_season_path)

    # sum farm pollinators
    farm_pollinator_path = os.path.join(
        output_dir, _FARM_POLLINATOR_FILE_PATTERN % file_suffix)
    farm_pollinator_task = task_graph.add_task(
        task_name='sum_farm_pollinators',
        func=pygeoprocessing.raster_calculator,
        args=(
            [(path, 1) for path in farm_pollinator_season_path_list],
            _SumRasters(), farm_pollinator_path, gdal.GDT_Float32,
            _INDEX_NODATA),
        dependent_task_list=farm_pollinator_season_task_list,
        target_path_list=[farm_pollinator_path])

    # rasterize managed pollinators
    managed_pollinator_path = os.path.join(
        intermediate_output_dir,
        _MANAGED_POLLINATOR_FILE_PATTERN % file_suffix)
    managed_pollinator_task = task_graph.add_task(
        task_name='rasterize_managed_pollinators',
        func=_rasterize_vector_onto_base,
        args=(
            blank_raster_path, farm_vector_path, _MANAGED_POLLINATORS_FIELD,
            managed_pollinator_path),
        dependent_task_list=[reproject_farm_task, blank_raster_task],
        target_path_list=[managed_pollinator_path])

    # calculate PYT
    total_pollinator_yield_path = os.path.join(
        output_dir, _TOTAL_POLLINATOR_YIELD_FILE_PATTERN % file_suffix)
    pyt_task = task_graph.add_task(
        task_name='calculate_total_pollinators',
        func=pygeoprocessing.raster_calculator,
        args=(
            [(managed_pollinator_path, 1), (farm_pollinator_path, 1)],
            _PYTOp(), total_pollinator_yield_path, gdal.GDT_Float32,
            _INDEX_NODATA),
        dependent_task_list=[farm_pollinator_task, managed_pollinator_task],
        target_path_list=[total_pollinator_yield_path])

    # calculate PYW
    wild_pollinator_yield_path = os.path.join(
        output_dir, _WILD_POLLINATOR_YIELD_FILE_PATTERN % file_suffix)
    wild_pollinator_task = task_graph.add_task(
        task_name='calcualte_wild_pollinators',
        func=pygeoprocessing.raster_calculator,
        args=(
            [(managed_pollinator_path, 1), (total_pollinator_yield_path, 1)],
            _PYWOp(), wild_pollinator_yield_path, gdal.GDT_Float32,
            _INDEX_NODATA),
        dependent_task_list=[pyt_task, managed_pollinator_task],
        target_path_list=[wild_pollinator_yield_path])

    # aggregate yields across farms
    target_farm_result_path = os.path.join(
        output_dir, _FARM_VECTOR_RESULT_FILE_PATTERN % file_suffix)
    if os.path.exists(target_farm_result_path):
        os.remove(target_farm_result_path)
    reproject_farm_task.join()
    _create_farm_result_vector(
        farm_vector_path, target_farm_result_path)

    # aggregate wild pollinator yield over farm
    wild_pollinator_task.join()
    wild_pollinator_yield_aggregate = pygeoprocessing.zonal_statistics(
        (wild_pollinator_yield_path, 1), target_farm_result_path)

    # aggregate yield over a farm
    pyt_task.join()
    total_farm_results = pygeoprocessing.zonal_statistics(
        (total_pollinator_yield_path, 1), target_farm_result_path)

    # aggregate the pollinator abundance results over the farms
    pollinator_abundance_results = {}
    for season in scenario_variables['season_list']:
        total_pollinator_abundance_index_path = os.path.join(
            output_dir, _TOTAL_POLLINATOR_ABUNDANCE_FILE_PATTERN % (
                season, file_suffix))
        total_pollinator_abundance_task[season].join()
        pollinator_abundance_results[season] = (
            pygeoprocessing.zonal_statistics(
                (total_pollinator_abundance_index_path, 1),
                target_farm_result_path))

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

            # this is PYW ('pdep_y_w')
            farm_feature.SetField(
                _POLLINATOR_PROPORTION_FARM_YIELD_FIELD_ID,
                (wild_pollinator_yield_aggregate[fid]['sum'] /
                 float(wild_pollinator_yield_aggregate[fid]['count'])))

            # this is YW ('y_wild')
            farm_feature.SetField(
                _WILD_POLLINATOR_FARM_YIELD_FIELD_ID,
                nu * (wild_pollinator_yield_aggregate[fid]['sum'] /
                      float(wild_pollinator_yield_aggregate[fid]['count'])))

            # this is PAT ('p_abund')
            farm_season = farm_feature.GetField(_FARM_SEASON_FIELD)
            farm_feature.SetField(
                _POLLINATOR_ABUDNANCE_FARM_FIELD_ID,
                pollinator_abundance_results[farm_season][fid]['sum'] /
                float(pollinator_abundance_results[farm_season][fid]['count']))

        target_farm_layer.SetFeature(farm_feature)
    target_farm_layer.SyncToDisk()
    target_farm_layer = None
    target_farm_vector = None

    task_graph.close()
    task_graph.join()


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
        options=['ATTRIBUTE=%s' % attribute_id])
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
                _WILD_POLLINATOR_FARM_YIELD_FIELD_ID

    Returns:
        None.

    """
    base_vector = gdal.OpenEx(base_vector_path, gdal.OF_VECTOR)

    driver = gdal.GetDriverByName('ESRI Shapefile')
    target_vector = driver.CreateCopy(
        target_vector_path, base_vector)
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

    pol_proportion_farm_yield_field_defn = ogr.FieldDefn(
        _POLLINATOR_PROPORTION_FARM_YIELD_FIELD_ID, ogr.OFTReal)
    pol_proportion_farm_yield_field_defn.SetWidth(25)
    pol_proportion_farm_yield_field_defn.SetPrecision(11)
    target_layer.CreateField(pol_proportion_farm_yield_field_defn)

    wild_pol_farm_yield_field_defn = ogr.FieldDefn(
        _WILD_POLLINATOR_FARM_YIELD_FIELD_ID, ogr.OFTReal)
    wild_pol_farm_yield_field_defn.SetWidth(25)
    wild_pol_farm_yield_field_defn.SetPrecision(11)
    target_layer.CreateField(wild_pol_farm_yield_field_defn)

    target_layer = None
    target_vector.FlushCache()
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


class _CalculateHabitatNestingIndex(object):
    """Closure for HN(x, s) = max_n(N(x, n) ns(s,n)) calculation."""

    def __init__(
            self, substrate_path_map, species_substrate_index_map,
            target_habitat_nesting_index_path):
        """Define parameters necessary for HN(x,s) calculation.

        Parameters:
            substrate_path_map (dict): map substrate name to substrate index
                raster path. (N(x, n))
            species_substrate_index_map (dict): map substrate name to
                scalar value of species substrate suitability. (ns(s,n))
            target_habitat_nesting_index_path (string): path to target
                raster
        """
        # try to get the source code of __call__ so task graph will recompute
        # if the function has changed
        try:
            self.__name__ = hashlib.sha1(inspect.getsource(
                    _CalculateHabitatNestingIndex.__call__
                ).encode('utf-8')).hexdigest()
        except IOError:
            # default to the classname if it doesn't work
            self.__name__ = _CalculateHabitatNestingIndex.__name__
        self.__name__ += str([
            substrate_path_map, species_substrate_index_map,
            target_habitat_nesting_index_path])
        self.substrate_path_list = [
            substrate_path_map[substrate_id]
            for substrate_id in sorted(substrate_path_map)]

        self.species_substrate_suitability_index_array = numpy.array([
            species_substrate_index_map[substrate_id]
            for substrate_id in sorted(substrate_path_map)]).reshape(
                (len(species_substrate_index_map), 1))

        self.target_habitat_nesting_index_path = (
            target_habitat_nesting_index_path)

    def __call__(self):
        """Calculate HN(x, s) = max_n(N(x, n) ns(s,n))."""
        def max_op(*substrate_index_arrays):
            """Return the max of index_array[n] * ns[n]."""
            result = numpy.max(
                numpy.stack([x.flatten() for x in substrate_index_arrays]) *
                self.species_substrate_suitability_index_array, axis=0)
            result = result.reshape(substrate_index_arrays[0].shape)
            result[substrate_index_arrays[0] == _INDEX_NODATA] = _INDEX_NODATA
            return result

        pygeoprocessing.raster_calculator(
            [(x, 1) for x in self.substrate_path_list], max_op,
            self.target_habitat_nesting_index_path,
            gdal.GDT_Float32, _INDEX_NODATA)


class _SumRasters(object):
    """Sum all rasters where nodata is 0 unless the entire stack is nodata."""

    def __init__(self):
        # try to get the source code of __call__ so task graph will recompute
        # if the function has changed
        try:
            self.__name__ = hashlib.sha1(
                inspect.getsource(
                    _SumRasters.__call__
                ).encode('utf-8')).hexdigest()
        except IOError:
            # default to the classname if it doesn't work
            self.__name__ = (
                _SumRasters.__name__)

    def __call__(self, *array_list):
        """Calculate sum of array_list and account for nodata."""
        valid_mask = numpy.zeros(array_list[0].shape, dtype=numpy.bool)
        result = numpy.empty_like(array_list[0])
        result[:] = 0
        for array in array_list:
            local_valid_mask = array != _INDEX_NODATA
            result[local_valid_mask] += array[local_valid_mask]
            valid_mask |= local_valid_mask
        result[~valid_mask] = _INDEX_NODATA
        return result


class _PollinatorSupplyOp(object):
    """Calc PA=RA*fa/FR * convolve(PS)."""

    def __init__(self):
        # try to get the source code of __call__ so task graph will recompute
        # if the function has changed
        try:
            self.__name__ = hashlib.sha1(
                inspect.getsource(
                    _PollinatorSupplyOp.__call__
                ).encode('utf-8')).hexdigest()
        except IOError:
            # default to the classname if it doesn't work
            self.__name__ = (
                _PollinatorSupplyOp.__name__)

    def __call__(
            self, foraged_flowers_array, floral_resources_array,
            convolve_ps_array):
        """Calculating (RA*fa)/FR * convolve(PS)."""
        valid_mask = foraged_flowers_array != _INDEX_NODATA
        result = numpy.empty_like(foraged_flowers_array)
        result[:] = _INDEX_NODATA
        zero_mask = floral_resources_array == 0
        result[zero_mask & valid_mask] = 0.0
        result_mask = valid_mask & ~zero_mask
        result[result_mask] = (
            foraged_flowers_array[result_mask] /
            floral_resources_array[result_mask] *
            convolve_ps_array[result_mask])
        return result


def ps_supply_op(
        floral_resources_array, habitat_nesting_suitability_array,
        species_abundance):
    """Calculate f_r * h_n * self.species_abundance."""
    result = numpy.empty_like(floral_resources_array)
    result[:] = _INDEX_NODATA
    valid_mask = ~numpy.isclose(floral_resources_array, _INDEX_NODATA)
    result[valid_mask] = (
        species_abundance * floral_resources_array[valid_mask] *
        habitat_nesting_suitability_array[valid_mask])
    return result


class _PollinatorSupplyIndexOp(object):
    """Calculate PS(x,s) = FR(x,s) * HN(x,s) * sa(s)."""

    def __init__(self, species_abundance):
        """Create a closure for species abundance to multiply later.

        Parameters:
            species_abundance (float): value to use in `__call__` when
                calculating pollinator abundance.

        Returns:
            None.
        """
        self.species_abundance = species_abundance
        # try to get the source code of __call__ so task graph will recompute
        # if the function has changed
        try:
            self.__name__ = hashlib.sha1(
                inspect.getsource(
                    _PollinatorSupplyIndexOp.__call__
                ).encode('utf-8')).hexdigest()
        except IOError:
            # default to the classname if it doesn't work
            self.__name__ = (
                _PollinatorSupplyIndexOp.__name__)
        self.__name__ += str(species_abundance)

    def __call__(
            self, floral_resources_array, habitat_nesting_suitability_array):
        """Calculate f_r * h_n * self.species_abundance."""
        result = numpy.empty_like(floral_resources_array)
        result[:] = _INDEX_NODATA
        valid_mask = floral_resources_array != _INDEX_NODATA
        result[valid_mask] = (
            self.species_abundance * floral_resources_array[valid_mask] *
            habitat_nesting_suitability_array[valid_mask])
        return result


class _MultByScalar(object):
    """Calculate a raster * scalar.  Mask through nodata."""

    def __init__(self, scalar):
        """Create a closure for multiplying an array by a scalar.

        Parameters:
            scalar (float): value to use in `__call__` when multiplying by
                its parameter.

        Returns:
            None.
        """
        self.scalar = scalar
        # try to get the source code of __call__ so task graph will recompute
        # if the function has changed
        try:
            self.__name__ = hashlib.sha1(
                inspect.getsource(
                    _MultByScalar.__call__
                ).encode('utf-8')).hexdigest()
        except IOError:
            # default to the classname if it doesn't work
            self.__name__ = (
                _MultByScalar.__name__)
        self.__name__ += str(scalar)

    def __call__(self, array):
        """Return array * self.scalar accounting for nodata."""
        result = numpy.empty_like(array)
        result[:] = _INDEX_NODATA
        valid_mask = array != _INDEX_NODATA
        result[valid_mask] = array[valid_mask] * self.scalar
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


class _PYTOp(object):
    """Calculate PYT=min((mp+FP), 1)."""

    def __init__(self):
        # try to get the source code of __call__ so task graph will recompute
        # if the function has changed
        try:
            self.__name__ = hashlib.sha1(
                inspect.getsource(
                    _PYTOp.__call__
                ).encode('utf-8')).hexdigest()
        except IOError:
            # default to the classname if it doesn't work
            self.__name__ = (
                _PYTOp.__name__)

    def __call__(self, mp_array, FP_array):
        """Return min(mp_array+FP_array, 1) accounting for nodata."""
        valid_mask = mp_array != _INDEX_NODATA
        result = numpy.empty_like(mp_array)
        result[:] = _INDEX_NODATA
        result[valid_mask] = mp_array[valid_mask]+FP_array[valid_mask]
        min_mask = valid_mask & (result > 1.0)
        result[min_mask] = 1.0
        return result


class _PYWOp(object):
    """Calculate PYW=max(0,PYT-mp)."""

    def __init__(self):
        # try to get the source code of __call__ so task graph will recompute
        # if the function has changed
        try:
            self.__name__ = hashlib.sha1(
                inspect.getsource(
                    _PYWOp.__call__
                ).encode('utf-8')).hexdigest()
        except IOError:
            # default to the classname if it doesn't work
            self.__name__ = (
                _PYWOp.__name__)

    def __call__(self, mp_array, PYT_array):
        """Return max(0,PYT_array-mp_array) accounting for nodata."""
        valid_mask = mp_array != _INDEX_NODATA
        result = numpy.empty_like(mp_array)
        result[:] = _INDEX_NODATA
        result[valid_mask] = PYT_array[valid_mask]-mp_array[valid_mask]
        max_mask = valid_mask & (result < 0.0)
        result[max_mask] = 0.0
        return result
