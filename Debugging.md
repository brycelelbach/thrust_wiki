Introduction
------------

While Thrust is well-tested and robust, things go wrong if it is misused. This page demonstrates some techniques to diagnose problems in Thrust programs.

Catching Exceptions
-------------------

When a Thrust function detects an error condition resulting from a lower-level API, it throws an exception to communicate the error to the caller. In Thrust, exceptions are usually instances of ```thrust::system_error``` objects, which inherits from ```std::runtime_error```. You can catch these exceptions in your code and try to handle the error as appropriate:

```c++
#include <thrust/system_error.h>
#include <thrust/device_vector.h>
#include <iostream>

int main(void)
{
  thrust::device_vector<int> vec(100);

  try
  {
    // index an out of bounds location -- oops!
    vec[9001] = 13;
  }
  catch(thrust::system_error &e)
  {
    // output an error message and exit
    std::cerr << "Error accessing vector element: " << e.what() << std::endl;
    exit(-1);
  }

  return 0;
}
```

    $ nvcc error.cu -I/path/to/thrust -run
    Error accessing vector element: invalid argument


Some Thrust functions may require allocating temporary storage. If such an allocation fails, ```std::bad_alloc```, instead of ```thrust::system_error```, is returned. Catching either is easy:

```c++
#include <thrust/system_error.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <new>
#include <iostream>

int main(void)
{
  // allocate a large vector
  thrust::device_vector<int> vec;

  try
  {
    vec.resize(1 << 30);
  }
  catch(std::bad_alloc &e)
  {
    std::cerr << "Couldn't allocate vector" << std::endl;
    exit(-1);
  }

  // sort the vector
  try
  {
    thrust::sort(vec.begin(), vec.end());
  }
  catch(std::bad_alloc &e)
  {
    std::cerr << "Ran out of memory while sorting" << std::endl;
    exit(-1);
  }
  catch(thrust::system_error &e)
  {
    std::cerr << "Some other error happened during sort: " << e.what() << std::endl;
    exit(-1);
  }

  return 0;
}
```

    $ nvcc allocation_error.cu -I/path/to/thrust -run
    Couldn't allocate vector

Asynchronous Error Detection
----------------------------

For performance, many Thrust functions are [asynchronous](http://en.wikipedia.org/wiki/Asynchrony), which means the execution of the function may be deferred until later. A consequence is that errors from asynchronous functions may be reported later in the program's execution than expected. This can make it difficult to pinpoint the source of the error:

```c++
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <iostream>

int main(void)
{
  thrust::device_vector<int> x(1000);

  std::cerr << "Before transform." << std::endl;

  // transform into a bogus location
  thrust::transform(x.begin(), x.end(), thrust::device_pointer_cast<int>(0), thrust::negate<int>());

  std::cerr << "After transform." << std::endl;

  return 0;
}
```

Because ```thrust::transform``` is asynchronous, the error is not reported until the host has continued past the ```thrust::transform``` call:

    $ nvcc transform_error.cu -I/path/to/thrust -run
    Before transform.
    After transform.
    terminate called after throwing an instance of 'thrust::system::system_error'
      what():  unspecified launch failure
    Aborted

Enabling Debugging Mode
-----------------------

While debugging, we can make Thrust check for errors at the earliest opportunity by enabling a debug mode by simply defining the macro ```THRUST_DEBUG``` on the compiler's command line. This makes it easy to find the source of the error:

    $ nvcc -DTHRUST_DEBUG transform_error.cu -run
    Before transform.
    terminate called after throwing an instance of 'thrust::system::system_error'
      what():  synchronize: launch_closure_by_value: unspecified launch failure
    Aborted

Thrust reports the error as soon as possible, and the host does not continue past the ```thrust::transform``` call.

Don't forget to disable ```THRUST_DEBUG``` when building release code!