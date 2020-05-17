# inspring
execution environment for SPRING analyses 

This is SPRING's execution environment that can build a Docker image that contains a python environment with:
* pygeoprocessing
* taskgraph
* ecoshard
* gdal
* relevant InVEST models located under "inspring", so rather that `import natcap.invest.sdr`, `import inspring.sdr`. Current models include:
  * `inspring.sdr_modifed_c_factor`
    * InVEST version of SDR that allows users to pass in a c-factor raster instead of using the one in the biophysical table

To run a Python script with this environment do the following:

```
docker run therealspring:inspring script_to_run.py
```
