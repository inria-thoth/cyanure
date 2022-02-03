# Copyright (c) 2019, Astropy Developers

# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice, this
#   list of conditions and the following disclaimer in the documentation and/or
#   other materials provided with the distribution.
# * Neither the name of the Astropy Team nor the names of its contributors may be
#   used to endorse or promote products derived from this software without
#   specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# This module defines functions that can be used to check whether OpenMP is
# available and if so what flags to use. To use this, import the
# add_openmp_flags_if_available function in a setup_package.py file where you
# are defining your extensions:
#
#     from extension_helpers.openmp_helpers import add_openmp_flags_if_available
#
# then call it with a single extension as the only argument:
#
#     add_openmp_flags_if_available(extension)
#
# this will add the OpenMP flags if available.

import os
import sys
import glob
import time
import datetime
import tempfile
import subprocess

from distutils import log
from distutils.errors import LinkError, CompileError
from distutils.ccompiler import new_compiler
from distutils.sysconfig import get_config_var, customize_compiler


__all__ = ['add_openmp_flags_if_available']

try:
    # Check if this has already been instantiated, only set the default once.
    _EXTENSION_HELPERS_DISABLE_OPENMP_SETUP_
except NameError:
    # It hasn't, so do so.
    _EXTENSION_HELPERS_DISABLE_OPENMP_SETUP_ = False

CCODE = """
#include <omp.h>
#include <stdio.h>
int main(void) {
  #pragma omp parallel
  printf("nthreads=%d\\n", omp_get_num_threads());
  return 0;
}
"""


def _get_flag_value_from_var(flag, var, delim=' '):
    """
    Extract flags from an environment variable.

    Parameters
    ----------
    flag : str
        The flag to extract, for example '-I' or '-L'
    var : str
        The environment variable to extract the flag from, e.g. CFLAGS or LDFLAGS.
    delim : str, optional
        The delimiter separating flags inside the environment variable

    Examples
    --------
    Let's assume the LDFLAGS is set to '-L/usr/local/include -customflag'. This
    function will then return the following:

        >>> _get_flag_value_from_var('-L', 'LDFLAGS')
        '/usr/local/include'

    Notes
    -----
    Environment variables are first checked in ``os.environ[var]``, then in
    ``distutils.sysconfig.get_config_var(var)``.

    This function is not supported on Windows.
    """

    if sys.platform.startswith('win'):
        return None

    # Simple input validation
    if not var or not flag:
        return None
    flag_length = len(flag)
    if not flag_length:
        return None

    # Look for var in os.eviron then in get_config_var
    if var in os.environ:
        flags = os.environ[var]
    else:
        try:
            flags = get_config_var(var)
        except KeyError:
            return None

    # Extract flag from {var:value}
    if flags:
        for item in flags.split(delim):
            if item.startswith(flag):
                return item[flag_length:]


def get_openmp_flags():
    """
    Utility for returning compiler and linker flags possibly needed for
    OpenMP support.

    Returns
    -------
    result : `{'compiler_flags':<flags>, 'linker_flags':<flags>}`

    Notes
    -----
    The flags returned are not tested for validity, use
    `check_openmp_support(openmp_flags=get_openmp_flags())` to do so.
    """

    compile_flags = []
    link_flags = []

    if get_compiler() == 'msvc':
        compile_flags.append('-openmp')
    else:

        include_path = _get_flag_value_from_var('-I', 'CFLAGS')
        if include_path:
            compile_flags.append('-I' + include_path)

        lib_path = _get_flag_value_from_var('-L', 'LDFLAGS')
        if lib_path:
            link_flags.append('-L' + lib_path)
            link_flags.append('-Wl,-rpath,' + lib_path)

        compile_flags.append('-fopenmp')
        link_flags.append('-fopenmp')

    return {'compiler_flags': compile_flags, 'linker_flags': link_flags}


def check_openmp_support(openmp_flags=None):
    """
    Check whether OpenMP test code can be compiled and run.

    Parameters
    ----------
    openmp_flags : dict, optional
        This should be a dictionary with keys ``compiler_flags`` and
        ``linker_flags`` giving the compiliation and linking flags respectively.
        These are passed as `extra_postargs` to `compile()` and
        `link_executable()` respectively. If this is not set, the flags will
        be automatically determined using environment variables.

    Returns
    -------
    result : bool
        `True` if the test passed, `False` otherwise.
    """

    ccompiler = new_compiler()
    customize_compiler(ccompiler)

    if not openmp_flags:
        # customize_compiler() extracts info from os.environ. If certain keys
        # exist it uses these plus those from sysconfig.get_config_vars().
        # If the key is missing in os.environ it is not extracted from
        # sysconfig.get_config_var(). E.g. 'LDFLAGS' get left out, preventing
        # clang from finding libomp.dylib because -L<path> is not passed to
        # linker. Call get_openmp_flags() to get flags missed by
        # customize_compiler().
        openmp_flags = get_openmp_flags()

    compile_flags = openmp_flags.get('compiler_flags')
    link_flags = openmp_flags.get('linker_flags')

    tmp_dir = tempfile.mkdtemp()
    start_dir = os.path.abspath('.')

    try:
        os.chdir(tmp_dir)

        # Write test program
        with open('test_openmp.c', 'w') as f:
            f.write(CCODE)

        os.mkdir('objects')

        # Compile, test program
        ccompiler.compile(['test_openmp.c'], output_dir='objects',
                          extra_postargs=compile_flags)

        # Link test program
        objects = glob.glob(os.path.join('objects', '*' + ccompiler.obj_extension))
        ccompiler.link_executable(objects, 'test_openmp',
                                  extra_postargs=link_flags)

        # Run test program
        output = subprocess.check_output('./test_openmp')
        output = output.decode(sys.stdout.encoding or 'utf-8').splitlines()

        if 'nthreads=' in output[0]:
            nthreads = int(output[0].strip().split('=')[1])
            if len(output) == nthreads:
                is_openmp_supported = True
            else:
                log.warn("Unexpected number of lines from output of test OpenMP "
                         "program (output was {0})".format(output))
                is_openmp_supported = False
        else:
            log.warn("Unexpected output from test OpenMP "
                     "program (output was {0})".format(output))
            is_openmp_supported = False
    except (CompileError, LinkError, subprocess.CalledProcessError):
        is_openmp_supported = False

    finally:
        os.chdir(start_dir)

    return is_openmp_supported


def is_openmp_supported():
    """
    Determine whether the build compiler has OpenMP support.
    """
    log_threshold = log.set_threshold(log.FATAL)
    ret = check_openmp_support()
    log.set_threshold(log_threshold)
    return ret


def add_openmp_flags_if_available(extension):
    """
    Add OpenMP compilation flags, if supported (if not a warning will be
    printed to the console and no flags will be added.)

    Returns `True` if the flags were added, `False` otherwise.
    """

    if _EXTENSION_HELPERS_DISABLE_OPENMP_SETUP_:
        log.info("OpenMP support has been explicitly disabled.")
        return False

    openmp_flags = get_openmp_flags()
    using_openmp = check_openmp_support(openmp_flags=openmp_flags)

    if using_openmp:
        compile_flags = openmp_flags.get('compiler_flags')
        link_flags = openmp_flags.get('linker_flags')
        log.info("Compiling Cython/C/C++ extension with OpenMP support")
        extension.extra_compile_args.extend(compile_flags)
        extension.extra_link_args.extend(link_flags)
    else:
        log.warn("Cannot compile Cython/C/C++ extension with OpenMP, reverting "
                 "to non-parallel code")

    return using_openmp


_IS_OPENMP_ENABLED_SRC = """
# Autogenerated by {packagename}'s setup.py on {timestamp!s}

def is_openmp_enabled():
    \"\"\"
    Determine whether this package was built with OpenMP support.
    \"\"\"
    return {return_bool}
"""[1:]


def generate_openmp_enabled_py(packagename, srcdir='.', disable_openmp=None):
    """
    Generate ``package.openmp_enabled.is_openmp_enabled``, which can then be used
    to determine, post build, whether the package was built with or without
    OpenMP support.
    """

    epoch = int(os.environ.get('SOURCE_DATE_EPOCH', time.time()))
    timestamp = datetime.datetime.utcfromtimestamp(epoch)

    if disable_openmp is not None:
        _EXTENSION_HELPERS_DISABLE_OPENMP_SETUP_ = disable_openmp
    if _EXTENSION_HELPERS_DISABLE_OPENMP_SETUP_:
        log.info("OpenMP support has been explicitly disabled.")
    openmp_support = False if _EXTENSION_HELPERS_DISABLE_OPENMP_SETUP_ else is_openmp_supported()

    src = _IS_OPENMP_ENABLED_SRC.format(packagename=packagename,
                                        timestamp=timestamp,
                                        return_bool=openmp_support)

    package_srcdir = os.path.join(srcdir, *packagename.split('.'))
    is_openmp_enabled_py = os.path.join(package_srcdir, 'openmp_enabled.py')
    with open(is_openmp_enabled_py, 'w') as f:
        f.write(src)


"""
This module contains various utilities for introspecting the distutils
module and the setup process.

Some of these utilities require the
`extension_helpers.setup_helpers.register_commands` function to be called first,
as it will affect introspection of setuptools command-line arguments.  Other
utilities in this module do not have that restriction.
"""

from distutils import ccompiler
from distutils.dist import Distribution
from distutils.errors import DistutilsError


__all__ = ['get_compiler']


def get_dummy_distribution():
    """
    Returns a distutils Distribution object used to instrument the setup
    environment before calling the actual setup() function.
    """

    # Pre-parse the Distutils command-line options and config files to if
    # the option is set.
    dist = Distribution({'script_name': os.path.basename(sys.argv[0]),
                         'script_args': sys.argv[1:]})

    with silence():
        try:
            dist.parse_config_files()
            dist.parse_command_line()
        except (DistutilsError, AttributeError, SystemExit):
            # Let distutils handle DistutilsErrors itself AttributeErrors can
            # get raise for ./setup.py --help SystemExit can be raised if a
            # display option was used, for example
            pass

    return dist


def get_main_package_directory(distribution):
    """
    Given a Distribution object, return the main package directory.
    """
    return min(distribution.packages, key=len).replace('.', os.sep)


def get_distutils_option(option, commands):
    """ Returns the value of the given distutils option.

    Parameters
    ----------
    option : str
        The name of the option

    commands : list of str
        The list of commands on which this option is available

    Returns
    -------
    val : str or None
        the value of the given distutils option. If the option is not set,
        returns None.
    """

    dist = get_dummy_distribution()

    for cmd in commands:
        cmd_opts = dist.command_options.get(cmd)
        if cmd_opts is not None and option in cmd_opts:
            return cmd_opts[option][1]
    else:
        return None


def get_distutils_build_option(option):
    """ Returns the value of the given distutils build option.

    Parameters
    ----------
    option : str
        The name of the option

    Returns
    -------
    val : str or None
        The value of the given distutils build option. If the option
        is not set, returns None.
    """
    return get_distutils_option(option, ['build', 'build_ext', 'build_clib'])


def get_compiler():
    """
    Determines the compiler that will be used to build extension modules.

    Returns
    -------
    compiler : str
        The compiler option specified for the build, build_ext, or build_clib
        command; or the default compiler for the platform if none was
        specified.

    """

    compiler = get_distutils_build_option('compiler')
    if compiler is None:
        return ccompiler.get_default_compiler()

    return compiler

# Licensed under a 3-clause BSD style license - see LICENSE.rst

import contextlib


__all__ = ['write_if_different']


class _DummyFile(object):
    """A noop writeable object."""

    errors = ''

    def write(self, s):
        pass

    def flush(self):
        pass


@contextlib.contextmanager
def silence():
    """A context manager that silences sys.stdout and sys.stderr."""

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = _DummyFile()
    sys.stderr = _DummyFile()
    exception_occurred = False
    try:
        yield
    except:  # noqa
        exception_occurred = True
        # Go ahead and clean up so that exception handling can work normally
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        raise

    if not exception_occurred:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
