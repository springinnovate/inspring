"""Tracer for NDR watershed processing."""
import logging
import os
import sys

from inspring.ndr_plus.ndr_plus import ndr_plus


logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(processName)s %(levelname)s '
        '%(name)s [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)
logging.getLogger('taskgraph').setLevel(logging.INFO)
WORKSPACE_DIR = 'workspace'


def main():
    ndr_plus(
        watershed_path, watershed_fid,
        target_cell_length_m,
        retention_length_m,
        k_val,
        flow_threshold,
        routing_algorithm,
        dem_path,
        lulc_path,
        precip_path,
        custom_load_path,
        eff_n_lucode_map,
        load_n_lucode_map,
        target_export_raster_path,
        target_modified_load_raster_path,
        workspace_dir)
