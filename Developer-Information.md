Introduction
------------

This page lists some instructions on how to build & run Thrust's tests.  The testers have
been built & tested on Windows, Ubuntu, and Mac OS X, but these instructions assume a Debian-based OS like Ubuntu.


Prerequisites
------------

Thrust's build process depends on [Scons](http://www.scons.org ).  If you are on a
Debian-based OS, you can install Scons with 

    $ sudo apt-get install scons

The testing infrastructure isn't included in the zip file on the
website, so you'll need to checkout the source tree from the
repository with [Git](https://git-scm.com/).  If you are on a Debian-based OS, you can install Git with

    $ sudo apt-get install git

To checkout the current development version of Thrust use

    $ git clone https://github.com/thrust/thrust.git

In order to run the tests, CUDA must be installed on your system because `nvcc` is used as the compiler driver. The tests assume the following default installation locations:

**Windows:**

    bin_path = 'C:/CUDA/bin'
    lib_path = 'C:/CUDA/lib'
    inc_path = 'C:/CUDA/include'

**Posix:**

    bin_path = '/usr/local/cuda/bin'
    lib_path = '/usr/local/cuda/lib'
    inc_path = '/usr/local/cuda/include'

If your installation differs, you can use the environment variables `CUDA_BIN_PATH`, `CUDA_LIB_PATH` and `CUDA_INC_PATH` to provide the correct installation folders.

If you have installed CUDA on Linux e.g. at `/opt/cuda`, set the environment variables prior to building the tests:

    $ export CUDA_BIN_PATH=/opt/cuda/bin
    $ export CUDA_LIB_PATH=/opt/cuda/lib64
    $ export CUDA_INC_PATH=/opt/cuda/include

In order to see the available build options run 

    $ scons -h

Example Programs
----------------

You can build the suite of example programs by running the following command inside the thrust root directory:

    $ scons run_examples

In order to speed up the build process, you can enable [parallel building](http://www.scons.org/doc/production/HTML/scons-user.html#idp1416840244) through 
    
    $ scons run_examples --jobs=4

Unit Tests
------------

You can build the unit tester through running the following command inside the thrust root directory: 

    $ scons unit_tests

After building finished, you can run all tests through

    $ scons run_unit_tests