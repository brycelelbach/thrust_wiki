## General
  * What is Thrust?
    * Thrust is a C++ template library of parallel algorithms. Thrust allows you to program parallel architectures using an interface similar to the C++ Standard Template Library (STL).
  * What is a C++ template library?
    * C++ templates are a way to write generic algorithms and data structures. A template library is simply a cohesive collection of such algorithms and data structures in a single package.
  * Do I need to build Thrust?
    * No. Since Thrust is a template library you just `#include` the appropriate header files into your `.cu` (or `.cpp`) file and compile with `nvcc` (c++ compiler).
  * What data structures does Thrust provide?
    * Currently Thrust provides vector data structures (e.g., `thrust::device_vector`) which are analogous to `std::vector` in the STL. These vector data structures simplify memory management and data transfer.
  * What algorithms does Thrust provide?
    * sorting: `thrust::sort` and `thrust::sort_by_key`
    * transformations: `thrust::transform`
    * reductions: `thrust::reduce` and `thrust::transform_reduce`
    * scans: `thrust::inclusive_scan`, `thrust::exclusive_scan`, `thrust::transform_inclusive_scan`, etc.
    * Refer to the [documentation](https://github.com/thrust/thrust/wiki/Documentation) for a complete listing.
  * What version of CUDA do I need to develop GPU applications with Thrust?
    * The latest version of Thrust requires [CUDA 4.0](http://www.nvidia.com/object/cuda_get.html) or newer.
  * What platforms does Thrust support?
    * Thrust has been tested extensively on Linux, Windows, and OSX systems.
  * When will Thrust support OpenCL?
    * The primary barrier to OpenCL support is the lack of an OpenCL compiler and runtime with support for C++ templates (e.g., something similar to `nvcc` and the CUDA Runtime). These features are necessary to achieve close coupling of host and device codes.
  * Does Thrust depend on any other libraries?
    * No, Thrust is self-contained and requires no additional libraries.
  * Can I distribute Thrust with my application?
    * Yes! Thrust is open-source software released under liberal licensing terms.
  * What open-source license does Thrust use?
    * Thrust is licensed under the [Apache License v2.0](http://www.opensource.org/licenses/apache2.0.php).

## Functionality
  * Can I create a `thrust::device_vector` from memory I've allocated myself?
    * No. Instead, [wrap your externally allocated raw pointer](https://github.com/thrust/thrust/blob/master/examples/cuda/wrap_pointer.cu) with `thrust::device_ptr` and pass it to Thrust algorithms.
  * How do I find the array *index* of the element with the maximum/minimum value?
    * Use `thrust::max_element` or `thrust::min_element`, which are found int he file `<thrust/extrema.h>`
  * Can I call Thrust algorithms inside a CUDA kernel?
    * No, it is not currently possible to call Thrust algorithms inside a `__global__` or `__device__` function.
  * Can I call Thrust algorithms from CUDA Fortran?
    * Yes! This [example](http://cudamusing.blogspot.com/2011/06/calling-thrust-from-cuda-fortran.html) shows how to call Thrust's `sort` algorithm from Fortran.

## Troubleshooting
  * If you're targeting the CUDA backend:
    * Make sure you are using CUDA 4.0 or greater:
      * run `nvcc --version`
    * Make sure you're compiling files that `#include` Thrust with `nvcc`.
    * Make sure that files that `#include` Thrust have a `.cu` extension. Other extensions (e.g., `.cpp`) will cause `nvcc` to treat the file incorrectly and produce an error message.
  * If you're targeting the OpenMP backend:
    * Make sure you've enabled OpenMP code generation:
      * `-fopenmp` with `g++`
      * `/openmp` with `cl.exe`
  * If all else fails, send a message to [thrust-users](http://groups.google.com/group/thrust-users) and we'll do our best to assist you.