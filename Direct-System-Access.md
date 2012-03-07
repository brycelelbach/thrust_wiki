Introduction
------------

Instead of applying a global switch to the behavior of all of a program's ```device_vectors```, we can also access Thrust's systems directly. For each system, Thrust provides a ```vector``` container whose iterators are "tagged" by the system. For example, the header file [```thrust/system/tbb/vector.h```](http://code.google.com/p/thrust/source/browse/thrust/system/tbb/vector.h) defines ```thrust::tbb::vector``` whose iterators' system tag is ```thrust::tbb::tag```. When algorithms are dispatched on its iterators, they will be parallelized by TBB.

Using a system-specific vector
------------------------------

Specifying a system by its ```vector``` is easy. For example, if we wish to use OpenMP to sort a list of numbers, we can use ```thrust::omp::vector``` instead of ```thrust::device_vector```:

```c++
#include <thrust/host_vector.h>
#include <thrust/system/omp/vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <cstdlib>
#include <algorithm>

int main(void)
{
  // serially generate 1M random numbers on the host
  thrust::host_vector<int> h_vec(1 << 20);
  std::generate(h_vec.begin(), h_vec.end(), rand);

  // transfer data to OpenMP
  thrust::omp::vector<int> d_vec = h_vec;

  // sort data in parallel with OpenMP
  thrust::sort(d_vec.begin(), d_vec.end());

  // transfer data back to host
  thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

  // report the largest number
  std::cout << "Largest number is " << h_vec.back() << std::endl;

  return 0;
}
```

This code is a simple adaptation of what appears on the [frontpage](http://code.google.com/p/thrust). However, because the OpenMP system shares the same memory space as the host system, it incurs wasteful copies. To fix it, we can simply eliminate the ```host_vector```:

```c++
#include <thrust/system/omp/vector.h>
#include <thrust/sort.h>
#include <cstdlib>
#include <algorithm>
#include <iostream>

int main(void)
{
  // serially generate 1M random numbers
  thrust::omp::vector<int> vec(1 << 20);
  std::generate(vec.begin(), vec.end(), rand);

  // sort data in parallel with OpenMP
  thrust::sort(vec.begin(), vec.end());

  // no need to transfer data back to host

  // report the largest number
  std::cout << "Largest number is " << vec.back() << std::endl;

  return 0;
}
```

Because the TBB system targets the host CPU as well, it is similarly interoperable with the host CPU's memory.

Retagging an iterator
---------------------

Sometimes it can be inconvenient or wasteful to introduce a new ```vector``` simply to parallelize algorithms which operate on some existing data. We can parallelize *in situ* by "retagging" iterators.

Let's take a look at the previous example, but instead we'll show how to use a ```std::vector``` with Thrust algorithms while still providing parallelization.

```c++
#include <thrust/system/omp/memory.h>
#include <thrust/iterator/retag.h>
#include <thrust/sort.h>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <vector>

int main(void)
{
  // serially generate 1M random numbers
  std::vector<int> vec(1 << 20);
  std::generate(vec.begin(), vec.end(), rand);

  // sort data in parallel with OpenMP by retagging vec's iterators
  thrust::sort(thrust::retag<thrust::omp::tag>(vec.begin()),
               thrust::retag<thrust::omp::tag>(vec.end()));

  // report the largest number
  std::cout << "Largest number is " << vec.back() << std::endl;
}
```

In this example, Thrust knows it's okay to retag ```std::vector```'s iterator with ```omp::tag``` because

  * ```std::vector::iterator``` is implicitly tagged with ```thrust::cpp::tag``` and
  * ```thrust::omp::tag``` is related to ```thrust::cpp::tag``` by convertibility.

Thrust will refuse to retag an iterator to an unrelated tag. For example, attempts to retag ```std::vector```'s iterators with CUDA will generate a compiler error because ```thrust::cpp::tag``` is unrelated to ```thrust::cuda::tag```. This code:

```c++
// attempt to sort a std::vector's data in parallel with CUDA by retagging vec's iterators
thrust::sort(thrust::retag<thrust::cuda::tag>(vec.begin()),
             thrust::retag<thrust::cuda::tag>(vec.end()));
```
Generates an error:

    $ nvcc retag_cuda.cpp
    retag_cuda.cpp: In function ‘int main()’:
    retag_cuda.cpp:15:62: error: no matching function for call to ‘retag(std::vector<int>::iterator)’
    retag_cuda.cpp:16:60: error: no matching function for call to ‘retag(std::vector<int>::iterator)’

Reinterpreting a tag
--------------------

If we have implementation-specific knowledge of the interoperability of two systems, we can forcibly *reinterpret* an iterator's tag to some other unrelated tag using the ```reinterpret_tag``` function. For example, we can reinterpret the raw pointer returned by ```cudaMalloc``` to use the CUDA system:

```c++
#include <thrust/system/cuda/memory.h>
#include <thrust/iterator/retag.h>
#include <thrust/fill.h>
#include <cuda.h>

int main(void)
{
  size_t N = 10;

  // obtain raw pointer to device memory
  int * raw_ptr;
  cudaMalloc((void **) &raw_ptr, N * sizeof(int));

  // reinterpret the raw_ptr's tag to cuda::tag
  thrust::fill(thrust::reinterpret_tag<thrust::cuda::tag>(raw_ptr),
               thrust::reinterpret_tag<thrust::cuda::tag>(raw_ptr + N),
               (int) 0);

  // free memory
  cudaFree(raw_ptr);

  return 0;
}
```

If we need to, we can similarly reinterpret an ```omp::vector```'s iterators to use the TBB system (and vice versa):

```c++
#include <thrust/system/omp/vector.h>
#include <thrust/system/tbb/vector.h>
#include <thrust/iterator/retag.h>
#include <cstdio>

struct omp_hello
{
  void operator()(int x)
  {
    printf("Hello, world from OpenMP!\n");
  }
};

struct tbb_hello
{
  void operator()(int x)
  {
    printf("Hello, world from TBB!\n");
  }
};

int main()
{
  thrust::omp::vector<int> omp_vec(1, 7);
  thrust::tbb::vector<int> tbb_vec(1, 13);

  thrust::for_each(thrust::reinterpret_tag<thrust::tbb::tag>(omp_vec.begin()),
                   thrust::reinterpret_tag<thrust::tbb::tag>(omp_vec.end()), 
                   tbb_hello());

  thrust::for_each(thrust::reinterpret_tag<thrust::omp::tag>(tbb_vec.begin()),
                   thrust::reinterpret_tag<thrust::omp::tag>(tbb_vec.end()),
                   omp_hello());
}
```

The output:

    $ nvcc reinterpret.cu -lgomp -ltbb -Xcompiler -fopenmp -run
    Hello, world from TBB!
    Hello, world from OpenMP!

Additional info
---------------

The type of the result returned by ```retag``` and ```reinterpret_tag``` is an unspecified iterator type whose behavior is the same as the iterator parameter and whose tag is the same as the tag template parameter.
