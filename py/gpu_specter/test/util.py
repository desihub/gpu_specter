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
    ## nerscdir = '/global/cfs/cdirs/desi/spectro/redux/daily'
    ## nerscurl = 'https://data.desi.lbl.gov/desi/spectro/redux/daily'

    nerscdir = '/global/cfs/cdirs/desi/public/edr/spectro/redux/fuji'
    nerscurl = 'https://data.desi.lbl.gov/public/edr/spectro/redux/fuji'
    night = 20201214
    expid = 67784

    if filetype == 'psf':
        #- PSF is small enough to be included with the repo
        return importlib.resources.files('gpu_specter') / f'test/data/psf-r0-{expid:08d}.fits'
    elif filetype == 'preproc':
        if 'NERSC_HOST' in os.environ:
            return f'{nerscdir}/preproc/{night}/{expid:08d}/preproc-r0-{expid:08d}.fits'
        else:
            #- TODO: download to test/data/ and return that
            return None
    elif filetype == 'frame':
        if 'NERSC_HOST' in os.environ:
            return f'{nerscdir}/exposures/{night}/{expid:08d}/frame-r0-{expid:08d}.fits'
        else:
            #- TODO: download to test/data/ and return that
            return None
    else:
        raise ValueError(f"Unknown filetype {filetype}; expected 'psf', 'preproc', or 'frame'")


