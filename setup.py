"""setup.py module for inspring."""
import numpy
from setuptools.extension import Extension
from setuptools import setup

setup(
    name='inspring',
    description="InSPRING: Computational Environment for experimental InVEST",
    maintainer='Rich Sharp',
    maintainer_email='richpsharp@gmail.com',
    url='https://github.com/therealspring/inspring',
    packages=[
        'inspring',
        'inspring.sdr_c_factor',
        'inspring.ndr_mfd_plus',
    ],
    package_dir={
        'inspring': 'src/inspring'
    },
    use_scm_version={
        'version_scheme': 'post-release',
        'local_scheme': 'node-and-date'},
    setup_requires=['setuptools_scm', 'cython', 'numpy'],
    include_package_data=True,
    license='BSD',
    zip_safe=False,
    keywords='gis inspring',
    classifiers=[
        'Intended Audience :: Developers',
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft',
        'Operating System :: POSIX',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: GIS',
        'License :: OSI Approved :: BSD License'
    ],
    ext_modules=[
        Extension(
            name="inspring.sdr_c_factor.sdr_c_factor_core",
            sources=["src/inspring/sdr_c_factor/sdr_c_factor_core.pyx"],
            include_dirs=[
                numpy.get_include(),
                'src/inspring/sdr_c_factor',
                'src/inspring/lrucache'],
            language="c++",
        ),
        Extension(
            name="inspring.floodplain_extraction.floodplain_extraction",
            sources=["src/inspring/floodplain_extraction/floodplain_extraction.pyx"],
            include_dirs=[
                numpy.get_include(),
                'src/inspring/floodplain_extraction',
                'src/inspring/lrucache'],
            language="c++",
        ),
        Extension(
            name="inspring.ndr_mfd_plus.ndr_mfd_plus_cython",
            sources=["src/inspring/ndr_mfd_plus/ndr_mfd_plus_cython.pyx"],
            include_dirs=[
                numpy.get_include(),
                'src/inspring/ndr_mfd_plus',
                'src/inspring/lrucache'],
            language="c++",
        ),
        Extension(
            name="inspring.seasonal_water_yield.seasonal_water_yield_core",
            sources=["src/inspring/seasonal_water_yield/seasonal_water_yield_core.pyx"],
            include_dirs=[
                numpy.get_include(),
                'src/inspring/seasonal_water_yield',
                'src/inspring/lrucache'],
            language="c++",
        ),
    ]
)
