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

def write_frame(filename, frame, dtype=np.float32):
    """
    Write frame to output filename
    """

    #- Write output to temp file and then rename so that final file is atomic
    tmpfilename = filename + '.tmp'
    with fitsio.FITS(tmpfilename, 'rw', clobber=True) as fx:
        fx.write(frame['specflux'].astype(dtype), extname='FLUX', header=frame['imagehdr'])
        fx.write(frame['specivar'].astype(dtype), extname='IVAR')
        fx.write(frame['specmask'], extname='MASK')
        fx.write(frame['wave'], extname='WAVELENGTH')
        fx.write(frame['Rdiags'].astype(dtype), extname='RESOLUTION')
        fx.write(frame['fibermap'], extname='FIBERMAP', header=frame['fibermaphdr'])
        fx.write(frame['chi2pix'].astype(dtype), extname='CHI2PIX')
    os.rename(tmpfilename, filename)

def read_frame(filename):
    """
    Read frame data (extracted 1D spectra) from input filename

    Returns a dictionary of numpy arrays
    """
    with fitsio.FITS(filename) as fx:
        flux = fx['FLUX'].read().astype('f8')
        ivar = fx['IVAR'].read().astype('f8')
        wave = fx['WAVELENGTH'].read().astype('f8')
        resolution = fx['RESOLUTION'].read().astype('f8')
    frame = dict(
        flux=flux,
        ivar=ivar,
        wave=wave,
        resolution=resolution,
    )
    return frame
