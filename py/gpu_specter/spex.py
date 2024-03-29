import argparse

import numpy as np

from gpu_specter.util import get_logger, Timer
from gpu_specter.io import read_img, read_psf, write_frame, write_model
from gpu_specter.core import extract_frame
from gpu_specter.mpi import NoMPIIOCoordinator, SerialIOCoordinator, ParallelIOCoordinator

__all__ = ["parse", "main_gpu_specter"]

def parse(options=None):
    parser = argparse.ArgumentParser(description="Extract spectra from pre-processed raw data.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input", type=str, required=False,
                        help="input image")
    # parser.add_argument("-f", "--fibermap", type=str, required=False,
    #                     help="input fibermap file")
    parser.add_argument("-p", "--psf", type=str, required=False,
                        help="input psf file")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="output extracted spectra file")
    parser.add_argument("-m", "--model", type=str, required=False,
                        help="output 2D pixel model file")
    parser.add_argument("-w", "--wavelength", type=str, required=False,
                        help="wavemin,wavemax,dw")
    parser.add_argument("-s", "--specmin", type=int, required=False, default=0,
                        help="first spectrum to extract")
    parser.add_argument("-n", "--nspec", type=int, required=False, default=500,
                        help="number of spectra to extract")
    parser.add_argument("-r", "--regularize", type=float, required=False, default=0.0,
                        help="regularization amount (default %(default)s)")
    parser.add_argument("--bundlesize", type=int, required=False, default=25,
                        help="number of spectra per bundle")
    parser.add_argument("--nsubbundles", type=int, required=False, default=5,
                        help="number of extraction sub-bundles")
    parser.add_argument("--nwavestep", type=int, required=False, default=50,
                        help="number of wavelength steps per divide-and-conquer extraction step")
    # parser.add_argument("-v", "--verbose", action="store_true", help="print more stuff")
    parser.add_argument("--loglevel", default='info', help='log print level (debug,info,warn,error)')
    parser.add_argument("--mpi", action="store_true", help="Use MPI for parallelism")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for extraction")
    # parser.add_argument("--decorrelate-fibers", action="store_true", help="Not recommended")
    # parser.add_argument("--no-scores", action="store_true", help="Do not compute scores")
    parser.add_argument("--psferr", type=float, default=None, required=False,
                        help="fractional PSF model error used to compute chi2 and mask pixels (default = value saved in psf file)")
    # parser.add_argument("--fibermap-index", type=int, default=None, required=False,
    #                     help="start at this index in the fibermap table instead of using the spectro id from the camera")
    # parser.add_argument("--barycentric-correction", action="store_true", help="apply barycentric correction to wavelength")
    parser.add_argument("--async-io", action="store_true",
                        help="use asynchronous read/write mpi comm")
    parser.add_argument("--pixpad-frac", type=float, required=False, default=0.8,
                        help="fraction of a PSF spotsize to pad in pixels when extracting")
    parser.add_argument("--wavepad-frac", type=float, default=0.2,
                        help="fraction of a PSF spotsize to pad in wavelengths when extracting")
    parser.add_argument("--wavepad", type=int, required=False, default=10,
                        help="Number of wavelength bins to pad on boths end of extraction patch")
    parser.add_argument("--ranks-per-bundle", type=int, default=None,
                        help="number of ranks per bundle")
    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args

def check_input_options(args):
    """
    Perform pre-flight checks on input options
    
    returns ok(True/False), message
    """
    if args.bundlesize % args.nsubbundles != 0:
        msg = 'bundlesize ({}) must be evenly divisible by nsubbundles ({})'.format(
            args.bundlesize, args.nsubbundles)
        return False, msg
    
    if args.nspec % args.bundlesize != 0:
        msg = 'nspec ({}) must be evenly divisible by bundlesize ({})'.format(
            args.nspec, args.bundlesize)
        return False, msg
    
    if args.specmin % args.bundlesize != 0:
        msg = 'specmin ({}) must begin at a bundle boundary'.format(args.specmin)
        return False, msg

    if args.use_gpu:
        try:
            import cupy as cp
        except ImportError:
            return False, 'cannot import module cupy'
        if not cp.is_available():
            return False, 'gpu is not available'

    return True, 'OK'

def main_gpu_specter(args=None, comm=None, timing=None, coordinator=None):

    timer = Timer()

    if args is None:
        args = parse()

    log = get_logger(args.loglevel)

    #- Preflight checks on input arguments
    ok, message = check_input_options(args)
    if not ok:
        log.critical(message)
        raise ValueError(message)
    
    #- Load MPI only if requested and coordinator not provided
    if coordinator is not None:
        pass
    elif comm is not None or args.mpi:
        #- Use MPI if comm is provided or args.mpi is specified
        if comm is None:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        if args.async_io:
            coordinator = ParallelIOCoordinator(comm)
        else:
            coordinator = SerialIOCoordinator(comm)
    else:
        coordinator = NoMPIIOCoordinator()

    timer.split('init')

    if args.use_gpu:
        #- If using gpu, move input data to device on read using cupy
        import cupy as cp
        array = cp.array
    else:
        #- Otherwise, use numpy array for cpu
        array = np.array

    #- Load inputs
    def read():
        log.info(f'Reading image: {args.input}')
        img = read_img(args.input)
        img['image'] = array(img['image'])
        img['ivar'] = array(img['ivar'])
        log.info(f'Reading PSF: {args.psf}')
        psf = read_psf(args.psf)
        return img, psf

    img, psf = coordinator.read(read, (None, None))

    timer.split('load')

    def process():
        #- Perform extraction
        frame = extract_frame(
            img, psf, args.bundlesize,         # input data
            args.specmin, args.nspec,          # spectra to extract (specmin, specmin + nspec)
            args.wavelength,                   # wavelength range to extract
            args.nwavestep, args.nsubbundles,  # extraction algorithm parameters
            args.model,
            args.regularize,
            args.psferr,
            coordinator.work_comm,             # mpi parameters
            args.use_gpu,                      # gpu parameters
            args.loglevel,                     # log
            wavepad=args.wavepad,
            wavepad_frac=args.wavepad_frac, 
            pixpad_frac=args.pixpad_frac,
            ranks_per_bundle=args.ranks_per_bundle,
        )
        #- Pass other input data through for output
        if coordinator.is_worker_root(coordinator.rank):
            frame['imagehdr'] = img['imagehdr']
            frame['fibermap'] = img['fibermap']
            frame['fibermaphdr'] = img['fibermaphdr']
        return frame

    frame = coordinator.process(process, None)

    timer.split('extract')

    #- Write output
    def write(frame):
        if args.output is not None:
            log.info(f'Writing {args.output}')
            write_frame(args.output, frame)

        if args.model is not None:
            log.info(f'Writing model {args.model}')
            write_model(args.model, frame)
        
    coordinator.write(write, frame)

    #- Print timing summary
    timer.split('write')
    if coordinator.is_worker_root(coordinator.rank):
        timer.log_splits(log)
