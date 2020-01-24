This is a README file for the DESI gpu hackathon code.

There are two scripts:

`wrapper_specter.py` is the outer script that, does the arg parse, divides into bundles,
launches the mpi ranks

`gpu_extract.py` does most of the work, including `projection_matrix`

and the extraction kernel.

To run: 

`time srun -u -n 32 -c 2 python -u wrapper_specter.py -o test.fits`
