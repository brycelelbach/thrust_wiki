Introduction
------------

Here we demonstrate how to use Thrust's "backend systems". There are two basic ways to access Thrust's systems: by specifying the global "device" system associated with types like ```thrust::device_vector```, or by selecting a specific container associated with a particular system, such as ```thrust::cuda::vector```. These two approaches are complementary and may be used together within the same program.

Selecting a global device system
--------------------------------

Here, we demonstrate how to switch between the CUDA (default), OpenMP, TBB, and standard C++ "device" backend systems. This is a global setting which applies to all types associated with the device system. In the following we'll consider the [```monte_carlo```](http://thrust.googlecode.com/hg/examples/monte_carlo.cu) sample program, but any of the [example programs](http://code.google.com/p/thrust/source/browse/trunk/examples/) would also do. Note that absolutely no changes to the source code are required to switch the device system.

Using the CUDA device system
----------------------------

First, download the source code for the [```monte_carlo```](http://thrust.googlecode.com/hg/examples/monte_carlo.cu) example.

    $ wget http://thrust.googlecode.com/hg/examples/monte_carlo.cu

Now let's time the program, which estimates pi by random sampling:

    $ time ./monte_carlo
    pi is around 3.14164

    real    0m0.222s
    user    0m0.120s
    sys     0m0.100s

Enabling the OpenMP device system
---------------------------------

We can switch to the OpenMP device system with the following compiler options (no changes to the source code!)

    $ nvcc -O2 -o monte_carlo monte_carlo.cu -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp

By default OpenMP runs one thread for each of the available cores, which is 4 on this particular system. Notice that the 'real' or [wall-clock](http://en.wikipedia.org/wiki/Wall_clock_time) time is almost exactly one 1/4th the 'user' or [CPU time](http://en.wikipedia.org/wiki/System_time), suggesting that ```monte_carlo``` is completely compute-bound and scales well .

    $ time ./monte_carlo 
    pi is around 3.14163

    real    0m2.090s
    user    0m8.333s
    sys     0m0.000s

We can override OpenMP's default behavior and instruct it to only use two threads using the ```OMP_NUM_THREADS``` environment variable. Notice that the real time has doubled while the user time remains the same.

    $ export OMP_NUM_THREADS=2
    $ time ./monte_carlo 
    pi is around 3.14163

    real    0m4.168s
    user    0m8.333s
    sys     0m0.000s

When only a single thread is used the real and user times agree.

    $ export OMP_NUM_THREADS=1
    $ time ./monte_carlo 
    pi is around 3.14164

    real    0m8.333s
    user    0m8.333s
    sys     0m0.000s

Enabling the TBB device system
------------------------------

We can switch to the TBB device system with the following compiler options

    $ nvcc -O2 -o monte_carlo monte_carlo.cu -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB -ltbb
    $ time ./monte_carlo
    pi is around 3.14

    real 0m1.216s
    user 0m9.425s
    sys  0m0.040s


Because both the OpenMP and TBB systems use similar algorithm implementations to utilize the CPU, their timings are similar.

Additional Details
------------------

When using either the OpenMP or TBB systems, ```nvcc``` isn't required. In general, ```nvcc``` is only required when targeting Thrust at CUDA. For example, we could compile the previous code directly with ```g++``` with this command line:

    $ g++ -O2 -o monte_carlo monte_carlo.cpp -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp -I<path-to-thrust-headers>

Note that we've copied ```monte_carlo.cu``` to ```monte_carlo.cpp``` so that ```g++``` recognizes that it's a c++ source file. The ```-fopenmp``` command line argument instructs ```g++``` to enable OpenMP directives. Without this option, the compilation will fail. The ```-lgomp``` command line argument instructs ```g++``` to link against the OpenMP library. Without this option, linking will fail.

If necessary, we can explicitly select the CUDA backend like so:

    $ nvcc -O2 -o monte_carlo monte_carlo.cu -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA
