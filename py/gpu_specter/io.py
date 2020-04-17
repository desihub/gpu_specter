import numpy as np
from astropy.table import Table
import fitsio

def read_psf(filename):
    """
    Read GaussHermite PSF data from input filename
    
    TODO: support old GaussHermite format
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