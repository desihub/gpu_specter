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
    
    Returns a dictionary of astropy Tables from the input PSF FITS file
    with keys XTRACE, YTRACE, PSF to match input file HDU EXTNAMEs
    """
    psfdata = dict()
    psfdata['PSF'] = Table.read(filename, 'PSF')
    
    with fitsio.FITS(filename, 'r') as fx:
        for extname in ('XTRACE', 'YTRACE'):
            data = fx[extname].read()
            hdr = fx[extname].read_header()
            t = Table()
            t[extname[0]] = data
            for key in ('WAVEMIN', 'WAVEMAX'):
                t.meta[key] = hdr[key]
            psfdata[extname] = t
    
    return psfdata

