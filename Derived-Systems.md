Introduction
------------

Occasionally we may desire to customize the behavior of a Thrust algorithm. This page describes how to embed fine-grained customization into a Thrust program by modifying how an existing Thrust backend system works. We can do this by hooking into the execution policy protocol Thrust uses to dispatch its algorithms.

Simple Example
--------------

As a completely trivial example, let's suppose that we want ```thrust::for_each``` to print a message each time it gets invoked on device iterators. Otherwise, we want to preserve ```thrust::for_each```'s existing functionality.

We begin by defining a new execution policy derived from ```thrust::device_execution_policy```. This will just be an empty ```struct```:

```c++
#include <thrust/execution_policy.h>

struct my_policy : thrust::device_execution_policy<my_policy> {};
```

Note that we pass the name of our policy (```my_policy```) as a template parameter to ```thrust::device_execution_policy```. This ensures that we don't lose track of the type during algorithm dispatch.

Next, we'll provide our own version of ```for_each```. It will print a message, and then call the normal device version of ```thrust::for_each```. Let's see what that looks like:

```c++
template<typename Iterator, typename Function>
  Iterator for_each(my_policy, Iterator first, Iterator last, Function f)
{
  std::cout << "Hello, world from for_each(my_policy)!" << std::endl;

  return thrust::for_each(thrust::device, first, last, f);
}
```

The function signature of our version of ```for_each``` looks just like ```thrust::for_each```, except that we've inserted a new parameter whose type is the execution policy we created. This allows our ```for_each``` to hook into Thrust's dispatch process, and it only applies when the execution policy matches ```my_policy```.

After printing our message, we call ```thrust::for_each``` using the normal ```thrust::device``` execution policy which corresponds to Thrust's device backend.

Now let's write a program to invoke our version of ```for_each```. Whenever we want our version to be invoked, we pass an instance of our policy type as the first argument. It's as simple as that!

```c++
int main()
{
  // Create a device_vector.
  thrust::device_vector<int> vec(1);

  // Create a execution policy object.
  my_policy exec;

  // Invoke thrust::for_each with our policy object as the first parameter.
  thrust::for_each(exec, vec.begin(), vec.end(), thrust::identity<int>());

  // Invocations without an execution policy are handled normally by inspecting the system tags
  // of the iterator parameters
  thrust::for_each(vec.begin(), vec.end(), thrust::identity<int>());

  return 0;
}
```

The second call to ```thrust::for_each``` doesn't generate any message, because it is invoked using the
normal dispatch process which selects an execution policy based on the iterators' tags, which are ```device_system_tag``` in this example.

The full source code for this program is included in the [```minimal_custom_backend example```](https://github.com/thrust/thrust/blob/master/examples/minimal_custom_backend.cu).

Temporary Allocation Example
----------------------------
Some function invocations can be intercepted in this way to improve Thrust's performance. For example, Thrust uses the function ```get_temporary_buffer``` to allocate temporary memory used in the implementation of some algorithms. In the CUDA system, ```get_temporary_buffer``` calls ```cudaMalloc```, which can be a performance bottleneck. So it can be advantageous to intercept these kinds of calls if a faster allocation scheme is available.

This example demonstrates how to intercept ```get_temporary_buffer``` and ```return_temporary_buffer``` when Thrust is allocating temporary storage. We proceed in the same manner as in the last example: we introduce a special ```my_policy``` type to distinguish which calls to Thrust should be customized, and we introduce special overloads of ```get_temporary_buffer``` and ```return_temporary_buffer``` which get dispatched when ```my_policy``` is encountered.

First, we'll begin with ```my_policy```:

```c++
#include <thrust/execution_policy.h>

struct my_policy : thrust::device_execution_policy<my_policy> {};
```

Next, we'll define overloads of ```get_temporary_buffer``` and ```return_temporary_buffer```. For the purposes of illustration, they'll simply call ```thrust::device_malloc``` and ```thrust::device_free```:

```c++
template<typename T>
  thrust::pair<thrust::pointer<T,my_policy>, std::ptrdiff_t>
    get_temporary_buffer(my_policy, std::ptrdiff_t n)
{
  std::cout << "get_temporary_buffer(my_policy): calling device_malloc" << std::endl;

  // ask device_malloc for storage
  thrust::pointer<T,my_policy> result(thrust::device_malloc<T>(n).get());

  // return the pointer and the number of elements allocated
  return thrust::make_pair(result,n);
}


template<typename Pointer>
  void return_temporary_buffer(my_policy, Pointer p)
{
  std::cout << "return_temporary_buffer(my_policy): calling device_free" << std::endl;

  thrust::device_free(thrust::device_pointer_cast(p.get()));
}
```

To test whether our versions get dispatched, let's use ```my_policy``` when calling ```thrust::sort```:

```c++
int main()
{
  size_t n = 1 << 10;
  thrust::host_vector<int> h_input(n);
  thrust::generate(h_input.begin(), h_input.end(), rand);

  thrust::device_vector<int> d_input = h_input;

  // create an instance of our execution policy
  my_policy exec;

  // any temporary allocations performed by this call to sort
  // will be implemented with our special overloads of
  // get_temporary_buffer and return_temporary_buffer
  thrust::sort(exec, d_input.begin(), d_input.end());

  return 0;
}
```

This code demonstrates the basic functionality, but the more sophisticated [```custom_temporary_allocation```](https://github.com/thrust/thrust/blob/master/examples/cuda/custom_temporary_allocation.cu) example shows how to build an allocation scheme which caches calls to ```thrust::device_malloc```.

Non-Primitive Algorithms
------------------------

Does this mean that every algorithm has to have a special system-specific version? No. Most algorithms are non-primitive, which means they may be implemented with other algorithms. For example, ```thrust::copy``` may be implemented with ```thrust::transform```:

```c++
template<typename InputIterator, typename OutputIterator>
  OutputIterator copy(InputIterator first, InputIterator last, OutputIterator result)
{
  typedef typename thrust::iterator_value<InputIterator>::type T;
  return thrust::transform(first, last, result, thrust::identity<T>());
}
```

Only "primitives" like ```for_each``` require special system-specific implementations because they can't be expressed in terms of other algorithms. However, because we might wish to provide a special implementation, the entry points of these non-primitive algorithms still get dispatched just like ```thrust::for_each```. This is useful when a faster implementation of an algorithm exists, even though it is non-primitive. If no specialization is found, the generic forms of these non-primitive algorithms are dispatched instead.