import os, logging
import numpy as np
from astropy.table import Table
import fitsio

def read_psf(filename):
    """
    Read GaussHermite PSF data from input filename
    
    TODO: support old GaussHermite format with X/YTRACE in same HDU as PSF
    """
    #- Read Legendre coefficients for PSF parameters
    psf = Table.read(filename, 'PSF')
    
    #- X and Y coefficients are kept in different HDUs now
    xcoeff = fitsio.read(filename, 'XTRACE')
    ycoeff = fitsio.read(filename, 'YTRACE')

    #- X and Y coeffs could have different degree than most params
    nparams, nspec, nc = psf['COEFF'].shape
    nx = xcoeff.shape[1]
    ny = xcoeff.shape[1]
    
    #- Replace psf['COEFF'] with maximum size needed to fit all
    coeff = np.zeros((nparams, nspec, max(nc, nx, ny)))
    coeff[:,:,0:nc] = psf['COEFF']
    psf['COEFF'] = coeff
    
    #- Add X and Y to the table
    psf.add_row(('X', xcoeff[:, 0:nx], 0, 0))
    psf.add_row(('Y', ycoeff[:, 0:ny], 0, 0))
    psf.meta['LEGDEG'] = max(nc, nx, ny)-1
    
    return psf

#- subset of desiutil.log.get_logger, to avoid desiutil dependency
_loggers = dict()
def get_logger(level=None):

    if level is None:
        level = os.getenv('DESI_LOGLEVEL', 'INFO')

    level = level.upper()
    if level == 'DEBUG':
        loglevel = logging.DEBUG
    elif level == 'INFO':
        loglevel = logging.INFO
    elif level == 'WARN' or level == 'WARNING':
        loglevel = logging.WARNING
    elif level == 'ERROR':
        loglevel = logging.ERROR
    elif level == 'FATAL' or level == 'CRITICAL':
        loglevel = logging.CRITICAL
    else:
        raise ValueError('Unknown log level {}; should be DEBUG/INFO/WARNING/ERROR/CRITICAL'.format(level))

    if level not in _loggers:
        logger = logging.getLogger('desimeter.'+level)
        logger.setLevel(loglevel)

        #- handler and formatter code adapted from
        #- https://docs.python.org/3/howto/logging.html#configuring-logging

        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(loglevel)

        # create formatter
        formatter = logging.Formatter('%(levelname)s:%(filename)s:%(lineno)s:%(funcName)s:%(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

        _loggers[level] = logger

    return _loggers[level]
