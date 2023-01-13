# Basic setup.py to support testing and installation.
# This does not use the desiutil installation infrastructure so that gpu_specter
# does not depend upon any of the offline desidata desi* packages.  If we
# decide to bring in those dependencies, this could be updated too.
#
# Supports:
# - python setup.py install
# - python setup.py test
#
# Does not support:
# - python setup.py version

import os, glob, re
from setuptools import setup, find_packages
from distutils.command.sdist import sdist as DistutilsSdist

def _get_version():
    for line in open('py/gpu_specter/_version.py').readlines():
        m = re.match("__version__\s*=\s*'(.*)'", line)
        if m is not None:
            version = m.groups()[0]
            break
    else:
        print('ERROR: Unable to parse version from: {}'.format(line))
        version = 'unknown'

    return version

#- Basic info
setup_keywords = dict(
    name='gpu_specter',
    version=_get_version(),
    description='Sandbox for porting specter extraction code to GPUs',
    author='DESI Collaboration',
    author_email='desi-data@desi.lbl.gov',
    license='BSD',
    url='https://github.com/sbailey/gpu_specter',
)

#- boilerplate, not sure if this is needed
setup_keywords['zip_safe'] = False
# setup_keywords['use_2to3'] = False

#- What to install
setup_keywords['packages'] = find_packages('py')
setup_keywords['package_dir'] = {'':'py'}
setup_keywords['cmdclass'] = {'sdist': DistutilsSdist}

#- Treat everything in bin/ as a script to be installed
setup_keywords['scripts'] = glob.glob(os.path.join('bin', '*'))

#- Data to include
setup_keywords['package_data'] = {
    # 'gpu_specter': ['data/*.*',],
    'gpu_specter.test': ['data/*.*',],
}

#- Testing
setup_keywords['test_suite'] = 'gpu_specter.test.test_suite'

#- Use desiutil if available; enables "python setup.py version"
try:
    from desiutil.setup import DesiVersion
    setup_keywords['cmdclass'].update({'version': DesiVersion})
except ImportError:
    pass

#- Go!
setup(**setup_keywords)
