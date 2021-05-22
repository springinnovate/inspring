# inspring
Computational Execution Environment for GIS, Hydrological, and Ecosystem Service Analysis

This execution environment that can build a Docker image that contains a Python environment with:
* pygeoprocessing==2.2.0
* taskgraph==0.10.3
* ecoshard==0.5.0
* gdal==3.1.3
* Enhancements to InVEST models 
  * NDR and SDR Plus - Modifications to the InVEST NDR and SDR models that allows users to specify a C factor raster rather than a landcover table lookup proxy.
  * Modification to InVEST SDR that allows the user to set the L factor threshold from the RKLS equation.
  * Modification to InVEST Pollination model that uses Ecosystem Functional Types (EFTs) to determine habitat suitability and abudnance.
* A model to extract floodplains from a general DEM

To run a Python script with this environment do the following:

```
docker container run therealspring/inspring:latest script_to_run.py
```
