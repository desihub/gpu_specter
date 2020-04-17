import os, logging
import numpy as np
from astropy.table import Table
import fitsio

def native_endian(data):
    """Temporary function, sourced from desispec.io
    Convert numpy array data to native endianness if needed.
    Returns new array if endianness is swapped, otherwise returns input data
    Context:
    By default, FITS data from astropy.io.fits.getdata() are not Intel
    native endianness and scipy 0.14 sparse matrices have a bug with
    non-native endian data.
    """
    if data.dtype.isnative:
        return data
    else:
        return data.byteswap().newbyteorder()

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

