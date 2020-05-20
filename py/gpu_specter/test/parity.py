#!/usr/bin/env python

import argparse
import os

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import fitsio

def read_frame(filename):
    """
    Simple fits file reader

    TODO: move to gpu_specter.io
    """
    with fitsio.FITS(filename) as fx:
        flux = fx['FLUX'].read().astype('f8')
        ivar = fx['IVAR'].read().astype('f8')
        wave = fx['WAVELENGTH'].read().astype('f8')
    frame = dict(
        flux=flux,
        ivar=ivar,
        wave=wave,
    )
    return frame

def format_fraction(label, fraction):
    return f'{label:>10}:  {100*fraction:6.2f}%'

def isclose_fraction_summary(diff, thresholds):
    """
    Helper function for generating a diff summary of an ndarray
    """
    
    for threshold in thresholds:
        fraction = np.average(np.abs(diff).ravel() < threshold)
        yield format_fraction(threshold, fraction)


def plot_diff_2d(diff, wmin, wmax, nspec, output):
    fig, ax = plt.subplots(figsize=(12, 4))
    extent = [wmin, wmax, nspec, 0]
    im = ax.imshow(
        np.abs(diff),
        # np.log10(np.abs(diff)),
        extent=extent,
        interpolation='none',
        norm=mcolors.LogNorm(),
        # norm=mcolors.LogNorm(vmin=1e-5, vmax=1e-2),
    )
    ax.set_ylabel('spectrum index')
    ax.set_xlabel('wavelength')
    ax.set_title('abs( (f_a - f_b)/sqrt(var_a + var_b) )')
    fig.colorbar(im, ax=ax)
    plt.savefig(output + '-2D.png', bbox_inches='tight', dpi=100)

def plot_diff_1d(diff, wave, output):
    fig, ax = plt.subplots(figsize=(8, 6))
    # pick some arbitrary spectra
    for ispec in (np.arange(20)*25 + 7):
        ax.plot(wave, 1e2*diff[ispec] + ispec, marker='.', markersize=1, lw=0)

    ax.set_ylabel('1e2*diff (offset by spectrum index)')
    ax.set_xlabel('wavelength')
    plt.savefig(output + '-1D.png', bbox_inches='tight', dpi=100)


def main():

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-a', '--filename-a', type=str,
                        help='filename a')
    parser.add_argument('-b', '--filename-b', type=str,
                        help='filename b')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='output filename base')
    args = parser.parse_args()

    a = read_frame(args.filename_a)
    b = read_frame(args.filename_b)

    assert a['wave'].shape == b['wave'].shape
    assert a['flux'].shape == b['flux'].shape
    assert a['ivar'].shape == b['ivar'].shape

    # compare wave
    assert np.all(a['wave'] == b['wave'])

    nspec = a['flux'].shape[0]
    wave = a['wave']
    wmin, wmax = wave[0], wave[-1]
    nwave = len(wave)

    # compare flux
    print('(f_a, f_b):')
    isclose = np.average(np.isclose(a['flux'], b['flux']))
    print(format_fraction('isclose', isclose))

    print('(f_a - f_b)/sqrt(var_a + var_b):')
    diff = (a['flux'] - b['flux']) / np.sqrt(1.0/a['ivar'] + 1.0/b['ivar'])
    tstart, tstop = -5, 0
    thresholds = np.logspace(tstart, tstop, (tstop - tstart) + 1, endpoint=True)
    for line in isclose_fraction_summary(diff, thresholds):
        print(line)

    if args.output is not None:
        plot_diff_2d(diff, wmin, wmax, nspec, args.output)
        plot_diff_1d(diff, wave, args.output)

    # compare ivar
    print('(ivar_a, ivar_b):')
    isclose = np.average(np.isclose(a['ivar'], b['ivar']))
    print(format_fraction('isclose', isclose))

    print('(sigma_a - sigma_b)/sqrt(var_a + var_b):')
    avar = 1.0/a['ivar']
    bvar = 1.0/b['ivar']
    diff = (np.sqrt(avar) - np.sqrt(bvar)) / np.sqrt(avar + bvar)
    tstart, tstop = -5, 0
    thresholds = np.logspace(tstart, tstop, (tstop - tstart) + 1, endpoint=True)
    for line in isclose_fraction_summary(diff, thresholds):
        print(line)


if __name__ == "__main__":
    main()
