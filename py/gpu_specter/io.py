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

def read_img(filename):
    """
    Read 2D image data from input filename

    Returns a dictionary numpy arrays and headers
    """

    imgdata = dict()

    with fitsio.FITS(filename, 'r') as fx:
        imgdata['image'] = native_endian(fx['IMAGE'].read().astype('f8'))
        imgdata['ivar'] = native_endian(fx['IVAR'].read().astype('f8'))
        imgdata['imagehdr'] = fx['IMAGE'].read_header()
        mask = fx['MASK'].read()
        imgdata['ivar'][mask != 0] = 0.0
        imgdata['fibermap'] = fx['FIBERMAP'].read()
        imgdata['fibermaphdr'] = fx['FIBERMAP'].read_header()

    return imgdata

def write_extract(filename, extract, dtype=np.float32):
    """
    Write extract data to output filename
    """

    #- Write output to temp file and then rename so that final file is atomic
    tmpfilename = filename + '.tmp'
    with fitsio.FITS(tmpfilename, 'rw', clobber=True) as fx:
        fx.write(extract['specflux'].astype(dtype), extname='FLUX', header=extract['imagehdr'])
        fx.write(extract['specivar'].astype(dtype), extname='IVAR')
        fx.write(extract['specmask'], extname='MASK')
        fx.write(extract['wave'], extname='WAVELENGTH')
        fx.write(extract['Rdiags'].astype(dtype), extname='RESOLUTION')
        fx.write(extract['fibermap'], extname='FIBERMAP', header=extract['fibermaphdr'])
        fx.write(extract['chi2pix'].astype(dtype), extname='CHI2PIX')
    os.rename(tmpfilename, filename)

