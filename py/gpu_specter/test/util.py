"""
Utilities to support tests
"""

import os
import importlib.resources

def find_test_file(filetype):
    """
    Find a test file of the requested type 'psf', 'preproc', or 'frame'

    Returns filepath or None if unavailable
    """
    nerscdir = '/global/cfs/cdirs/desi/spectro/redux/daily'
    nerscurl = 'https://data.desi.lbl.gov/desi/spectro/redux/daily'
    if filetype == 'psf':
        #- PSF is small enough to be included with the repo
        return importlib.resources.files('gpu_specter') / 'test/data/psf-r0-00051060.fits'
    elif filetype == 'preproc':
        if 'NERSC_HOST' in os.environ:
            return f'{nerscdir}/preproc/20200219/00051060/preproc-r0-00051060.fits'
        else:
            #- TODO: download to test/data/ and return that
            return None
    elif filetype == 'frame':
        if 'NERSC_HOST' in os.environ:
            return f'{nerscdir}/exposures/20200219/00051060/frame-r0-00051060.fits'
        else:
            #- TODO: download to test/data/ and return that
            return None
    else:
        raise ValueError(f"Unknown filetype {filetype}; expected 'psf', 'preproc', or 'frame'")


