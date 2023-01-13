gpu_specter change log
======================

0.2.1 (unreleased)
------------------

* No changes yet

0.2.0 (2023-01-12)
------------------

* Use desi_proc mpi logic for cpu specter in benchmark (PR #64)
* Add compare specter notebook (PR #65)
* Use cupyx.lapack.posv and fix failing unit tests (PR #66)
* Scale run updates (PR #68)
* Eliminate noisy warnings (PR #71)
* Align GPU command line options with desispec (PR #72)
* Benchmark: CLI option to set subcomm size (PR #73)
* Return value from desi_mps_wrapper (PR #74)
* Return xflux in ex2d_patch and use xflux for chi2pix (PR #76)
* Update failing tests (PR #79)

0.1.0 (2021-07-06)
------------------

* First tag of gpu_specter, a gpu-enabled and cpu-faster replacement for
  [specter](https://github.com/desihub/specter).
