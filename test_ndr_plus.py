"""Tracer for NDR watershed processing."""
import logging
import multiprocessing
import os
import subprocess
import sys
import urllib
import zipfile

from inspring.ndr_plus.ndr_plus import ndr_plus
import ecoshard
import taskgraph

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(processName)s %(levelname)s '
        '%(name)s [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)
logging.getLogger('taskgraph').setLevel(logging.INFO)
WORKSPACE_DIR = 'workspace'
ECOSHARD_DIR = os.path.join(WORKSPACE_DIR, 'ecoshards')

# All links in this dict is an ecoshard that will be downloaded to
# ECOSHARD_DIR

ECOSHARD_PREFIX = 'https://storage.googleapis.com/'
DEM_ID = 'global_dem_3s_blake2b'

ECOSHARDS = {
    DEM_ID: f'{ECOSHARD_PREFIX}ipbes-ndr-ecoshard-data/global_dem_3s_blake2b_0532bf0a1bedbe5a98d1dc449a33ef0c.zip',
    'esacci-lc-l4-lccs-map-300m-p1y-2015-v2.0.7': f'{ECOSHARD_PREFIX}ipbes-ndr-ecoshard-data/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2015-v2.0.7_md5_1254d25f937e6d9bdee5779d377c5aa4.tif',
    'extensification_bmps_irrigated': f'{ECOSHARD_PREFIX}nci-ecoshards/scenarios050420/extensification_bmps_irrigated_md5_7f5928ea3dcbcc55b0df1d47fbeec312.tif',
    'extensification_bmps_rainfed': f'{ECOSHARD_PREFIX}nci-ecoshards/scenarios050420/extensification_bmps_rainfed_md5_5350b6acebbff75bb71f27830098989f.tif',
    'extensification_current_practices': f'{ECOSHARD_PREFIX}nci-ecoshards/scenarios050420/extensification_current_practices_md5_cbe24876a57999e657b885cf58c4981a.tif',
    'extensification_intensified_irrigated': f'{ECOSHARD_PREFIX}nci-ecoshards/scenarios050420/extensification_intensified_irrigated_md5_215fe051b6bc84d3e15a4d1661b6b936.tif',
    'extensification_intensified_rainfed': f'{ECOSHARD_PREFIX}nci-ecoshards/scenarios050420/extensification_intensified_rainfed_md5_47050c834831a6bc4644060fffffb052.tif',
    'extensificationnapp_allcrops_rainfedfootprint_gapfilled_observednapprevb': f'{ECOSHARD_PREFIX}nci-ecoshards/scenarios050420/ExtensificationNapp_allcrops_rainfedfootprint_gapfilled_observedNappRevB_md5_1185e457751b672c67cc8c6bf7016d03.tif',
    'fixedarea_bmps_irrigated': f'{ECOSHARD_PREFIX}nci-ecoshards/scenarios050420/fixedarea_bmps_irrigated_md5_857517cbef7f21cd50f963b4fc9e7191.tif',
    'fixedarea_bmps_rainfed': f'{ECOSHARD_PREFIX}nci-ecoshards/scenarios050420/fixedarea_bmps_rainfed_md5_3b220e236c818a28bd3f2f5eddcc48b0.tif',
    'fixedarea_intensified_irrigated': f'{ECOSHARD_PREFIX}nci-ecoshards/scenarios050420/fixedarea_intensified_irrigated_md5_4990faf720ac68f95004635e4a2c3c74.tif',
    'fixedarea_intensified_rainfed': f'{ECOSHARD_PREFIX}nci-ecoshards/scenarios050420/fixedarea_intensified_rainfed_md5_98ac886076a35507c962263ee6733581.tif',
    'global_potential_vegetation': f'{ECOSHARD_PREFIX}nci-ecoshards/scenarios050420/global_potential_vegetation_md5_61ee1f0ffe1b6eb6f2505845f333cf30.tif',
    'intensificationnapp_allcrops_irrigated_max_model_and_observednapprevb': f'{ECOSHARD_PREFIX}nci-ecoshards/scenarios050420/IntensificationNapp_allcrops_irrigated_max_Model_and_observedNappRevB_md5_9331ed220772b21f4a2c81dd7a2d7e10.tif',
    'intensificationnapp_allcrops_irrigated_max_model_and_observednapprevb_bmps': f'{ECOSHARD_PREFIX}nci-ecoshards/scenarios050420/IntensificationNapp_allcrops_irrigated_max_Model_and_observedNappRevB_BMPs_md5_ddc000f7ce7c0773039977319bcfcf5d.tif',
    'intensificationnapp_allcrops_rainfed_max_model_and_observednapprevb': f'{ECOSHARD_PREFIX}nci-ecoshards/scenarios050420/IntensificationNapp_allcrops_rainfed_max_Model_and_observedNappRevB_md5_1df3d8463641ffc6b9321e73973f3444.tif',
    'intensificationnapp_allcrops_rainfed_max_model_and_observednapprevb_bmps': f'{ECOSHARD_PREFIX}nci-ecoshards/scenarios050420/IntensificationNapp_allcrops_rainfed_max_Model_and_observedNappRevB_BMPs_md5_fa2684c632ec2d0e0afb455b41b5d2a6.tif',
}


def unzip_and_build_vrt(
        zipfile_path, target_unzip_dir, expected_tiles_zip_path,
        target_vrt_path):
    """Build VRT of given tiles.

    Args:
        zipfile_path (str): source zip file to extract.
        target_unzip_dir (str): desired directory in which to extract
            the zipfile.
        expected_tiles_zip_path (str): the expected directory to find the
            geotiff tiles after the zipfile has been extracted to
            ``target_unzip_dir``.
        target_vrt_path (str): path to desired VRT file of those files.

    Return:
        ``None``
    """
    os.makedirs(target_unzip_dir, exist_ok=True)
    LOGGER.info(f'unzip {zipfile_path}')
    with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
        zip_ref.extractall(target_unzip_dir)

    LOGGER.info('build vrt')
    subprocess.run(
        f'gdalbuildvrt {target_vrt_path} {expected_tiles_zip_path}/*.tif',
        shell=True)
    LOGGER.info(f'all done building {target_vrt_path}')


def main():
    """Entry point."""
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    task_graph = taskgraph.TaskGraph(
        WORKSPACE_DIR, multiprocessing.cpu_count())
    os.makedirs(ECOSHARD_DIR, exist_ok=True)
    ecoshard_path_map = {}
    LOGGER.info('scheduling downloads')
    for ecoshard_id, ecoshard_url in ECOSHARDS.items():
        ecoshard_path = os.path.join(
            ECOSHARD_DIR, os.path.basename(ecoshard_url))
        LOGGER.debug(f'download {ecoshard_url}')
        LOGGER.debug(f'dlcode: {urllib.request.urlopen(ecoshard_url).getcode()}')
        download_task = task_graph.add_task(
            func=ecoshard.download_url,
            args=(ecoshard_url, ecoshard_path),
            target_path_list=[ecoshard_path])
        ecoshard_path_map[ecoshard_id] = ecoshard_path
    LOGGER.info('waiting for downloads to finish')
    task_graph.join()

    # global DEM that's used
    dem_tile_dir = os.path.join(ECOSHARD_DIR, 'global_dem_3s')
    dem_vrt_path = os.path.join(dem_tile_dir, 'global_dem_3s.vrt')
    unzip_and_build_vrt(
        ecoshard_path_map[DEM_ID],
        ECOSHARD_DIR,
        dem_tile_dir,
        dem_vrt_path)

    # ndr_plus(
    #     watershed_path, watershed_fid,
    #     target_cell_length_m,
    #     retention_length_m,
    #     k_val,
    #     flow_threshold,
    #     routing_algorithm,
    #     dem_path,
    #     lulc_path,
    #     precip_path,
    #     custom_load_path,
    #     eff_n_lucode_map,
    #     load_n_lucode_map,
    #     target_export_raster_path,
    #     target_modified_load_raster_path,
    #     workspace_dir)


if __name__ == '__main__':
    main()
