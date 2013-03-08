Introduction
------------

Instead of applying a global switch to the behavior of all of a program's ```device_vectors```, we can also access Thrust's systems directly. For each system, Thrust provides a ```vector``` container whose iterators are "tagged" by the system. For example, the header file [```thrust/system/tbb/vector.h```](https://github.com/thrust/thrust/blob/master/thrust/system/tbb/vector.h) defines ```thrust::tbb::vector``` whose iterators' system tag is ```thrust::tbb::tag```. When algorithms are dispatched on its iterators, they will be parallelized by TBB.

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

This code is a simple adaptation of what appears on the [frontpage](http://thrust.github.com). However, because the OpenMP system shares the same memory space as the host system, it incurs wasteful copies. To fix it, we can simply eliminate the ```host_vector```:

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

Execution Policies
------------------

Sometimes it can be inconvenient or wasteful to introduce a new ```vector``` simply to parallelize
algorithms which operate on some existing data. We can parallelize *in situ* by using invoking an algorithm
with an execution policy.

Let's take a look at the previous example, but instead we'll show how to use a ```std::vector``` with Thrust
algorithms while still providing parallelization.

```c++
#include <thrust/system/omp/execution_policy.h>
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

  // sort data in parallel with OpenMP by specifying its execution policy
  thrust::sort(thrust::omp::par, vec.begin(), vec.end());

  // report the largest number
  std::cout << "Largest number is " << vec.back() << std::endl;
}
```

In this example, we've explicitly told Thrust to use OpenMP to parallelize the call to ```thrust::sort``` by
providing ```thrust::omp::par``` (```par``` for "parallel") as the first argument.

The same works with the standard C++ backend system or TBB:

```
#include <vector>
#include <algorithm>
#include <thrust/sort.h>
#include <thrust/system/cpp/execution_policy.h>
#include <thrust/system/tbb/execution_policy.h>

int main(void)
{
  // serially generate 1M random numbers
  std::vector<int> vec(1 << 20);
  std::generate(vec.begin(), vec.end(), rand);

  // sort data in parallel using the standard C++ backend by specifying its execution policy
  thrust::sort(thrust::cpp::par, vec.begin(), vec.end());

  // check that the data is actually sorted using the TBB backend
  std::cout << "vec is sorted: " << thrust::is_sorted(thrust::tbb::par, vec.begin(), vec.end());
}
```

Additional info
---------------

When using execution policies directly, it's important to make sure that the backend of interest will be
able to safely dereference the iterators provided to the algorithm.

For example, this code snippet will certainly meet with failure:

```
#include <vector>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

int main()
{
  std::vector<int> vec = ...

  // error -- CUDA kernels can't access std::vector!
  thrust::sort(thrust::cuda::par, vec.begin(), vec.end());
}
```

because CUDA isn't able to access memory in a `std::vector`.

On the other hand, it's safe to use ```thrust::cuda::par``` with raw pointers allocated by `cudaMalloc`, even when the pointer isn't wrapped by ```thrust::device_ptr```:

```
#include <thrust/tabulate.h>
#include <thrust/sort.h>
#include <iostream>

int main()
{
  int n = 13;
  int *raw_ptr = 0;
  cudaMalloc(&raw_ptr, n * sizeof(int));

  // it's OK to pass raw pointers allocated by cudaMalloc to an algorithm invoked with cuda::par
  thrust::tabulate(thrust::cuda::par, raw_ptr, raw_ptr + n, thrust::identity<int>());

  std::cout << "data is sorted: " << thrust::is_sorted(thrust::cuda::par, raw_ptr, raw_ptr + n);

  cudaFree(raw_ptr);
}
```
