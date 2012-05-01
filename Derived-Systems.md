Introduction
------------

Occasionally we may desire to customize the behavior of a Thrust algorithm. This page describes how to embed fine-grained customization into a Thrust program by modifying how an existing Thrust backend system works. We can do this by hooking into the system tag protocol Thrust uses to dispatch its algorithms.

Simple Example
--------------

As a completely trivial example, let's suppose that we want ```thrust::for_each``` to print a message each time it gets invoked on device iterators. Otherwise, we want to preserve ```thrust::for_each```'s existing functionality.

We begin by defining a new system tag derived from ```thrust::device_system_tag```. This will just be an empty ```struct```:

```c++
struct my_tag : thrust::device_system_tag {};
```

Next, we'll provide our own version of ```for_each```. It will print a message, and then call the normal version of ```thrust::for_each```. Let's see what that looks like:

```c++
template<typename Iterator, typename Function>
  Iterator for_each(my_tag, Iterator first, Iterator last, Function f)
{
  std::cout << "Hello, world from for_each(my_tag)!" << std::endl;

  thrust::for_each(thrust::retag<thrust::device_system_tag>(first),
                   thrust::retag<thrust::device_system_tag>(last),
                   f);

  return last;
}
```

The function signature of our version of ```for_each``` looks just like ```thrust::for_each```, except that we've inserted a new parameter whose type is the tag we created. This allows our ```for_each``` to hook into Thrust's dispatch process, and it only applies when an iterator's system tag matches ```my_tag```.

After printing our message, we call ```thrust::for_each```. We have to be careful to retag our iterator parameters back to ```device_system_tag``` so that Thrust won't get stuck in an infinite loop during dispatch.

Now let's write a program to invoke our version of ```for_each```. Whenever we want our version to be invoked, we ```retag``` ```thrust::for_each```'s iterator arguments with ```my_tag```. It's as simple as that!

```c++
int main()
{
  // Create a device_vector, whose iterators are tagged with device_space_tag
  thrust::device_vector<int> vec(1);

  // retag with my_tag
  thrust::for_each(thrust::retag<my_tag>(vec.begin()),
                   thrust::retag<my_tag>(vec.end()),
                   thrust::identity<int>());

  // Iterators without my_tag are handled normally.
  thrust::for_each(vec.begin(), vec.end(), thrust::identity<int>());

  return 0;
}
```

The second call to ```thrust::for_each``` doesn't generate any message, because its iterators have the normal ```device_system_tag```. The full source code for this program is included in the [```minimal_custom_backend example```](http://code.google.com/p/thrust/source/browse/examples/minimal_custom_backend.cu).

Temporary Allocation Example
----------------------------
Some of the function invocations which can be intercepted in this manner aren't part of Thrust's public interface. For example, Thrust uses the special function ```get_temporary_buffer``` to allocate temporary memory used in the implementation of some algorithms. In the CUDA system, calls to ```cudaMalloc``` can be a performance bottleneck, so it can be advantageous to intercept such calls if a faster allocation scheme is available.

This example demonstrates how to intercept ```get_temporary_buffer``` and ```return_temporary_buffer``` when Thrust is allocating temporary storage. We proceed in the same manner as in the last example: we introduce a special ```my_tag``` type to distinguish which calls to Thrust should be customized, and we introduce special overloads of ```get_temporary_buffer``` and ```return_temporary_buffer``` which get dispatched when ```my_tag``` is encountered.

First, we'll begin with ```my_tag```:

```c++
struct my_tag : thrust::device_system_tag {};
```

Next, we'll define overloads of ```get_temporary_buffer``` and ```return_temporary_buffer```. For the purposes of illustration, they'll simply call ```thrust::device_malloc``` and ```thrust::device_free```:

```c++
template<typename T>
  thrust::pair<thrust::pointer<T,my_tag>, std::ptrdiff_t>
    get_temporary_buffer(my_tag, std::ptrdiff_t n)
{
  std::cout << "get_temporary_buffer(my_tag): calling device_malloc" << std::endl;

  // ask device_malloc for storage
  thrust::pointer<T,my_tag> result(thrust::device_malloc<T>(n).get());

  // return the pointer and the number of elements allocated
  return thrust::make_pair(result,n);
}


template<typename Pointer>
  void return_temporary_buffer(my_tag, Pointer p)
{
  std::cout << "return_temporary_buffer(my_tag): calling device_free" << std::endl;

  thrust::device_free(thrust::device_pointer_cast(p.get()));
}
```

To test whether our versions get dispatched, let's ```retag``` our iterators upon a call to ```thrust::sort```:

```c++
int main()
{
  size_t n = 1 << 10;
  thrust::host_vector<int> h_input(n);
  thrust::generate(h_input.begin(), h_input.end(), rand);

  thrust::device_vector<int> d_input = h_input;

  // any temporary allocations performed by this call to sort
  // will be implemented with our special overloads of
  // get_temporary_buffer and return_temporary_buffer
  thrust::sort(thrust::retag<my_tag>(d_input.begin()),
               thrust::retag<my_tag>(d_input.end()));

  return 0;
}
```

This code demonstrates the basic functionality, but the more sophisticated ```custom_temporary_allocation``` example shows how to build an allocation scheme which caches calls to ```thrust::device_malloc```.

System Tag Dispatch
-------------------

So how does all this work? When a Thrust algorithm entry point like ```thrust::for_each``` is called, the first thing it does is inspect the iterators' system tags. This tells Thrust which version of the algorithm should be invoked.

For example, the (simplified) implementation of ```thrust::for_each``` looks something like this:

```c++
template<typename Iterator, typename Function>
  Iterator for_each(Iterator first, Iterator last, Function f)
{
  // get Iterator's system tag
  typedef typename thrust::iterator_system<Iterator>::type tag;

  // call for_each with select_system(tag)
  // forward the other arguments along
  return for_each(select_system(tag()), first, last, f);
}
```

In the first bit of code, Thrust uses ```iterator_system``` to inspect the ```Iterator```'s system tag. Next, we use the special function ```select_system``` to select the version of ```for_each``` associated with ```tag```. By default, select_system is the identity function: it simply returns a copy of ```tag```.

Each of Thrust's systems has a private version of ```for_each``` which takes an special, additional ```tag``` parameter. The type of the ```tag``` parameter is specific to the system. For example, the CUDA system's ```tag``` is of the type ```thrust::cuda::tag```. These dispatchable entry points are private -- they can't be called by the user.

Non-Primitive Algorithms
------------------------

Does this mean that every algorithm has a special system-specific version? No. Most algorithms are non-primitive, which means they may be implemented with other algorithms. For example, ```thrust::copy``` may be implemented with ```thrust::transform```:

```c++
template<typename InputIterator, typename OutputIterator>
  OutputIterator copy(InputIterator first, InputIterator last, OutputIterator result)
{
  typedef typename thrust::iterator_value<InputIterator>::type T;
  return thrust::transform(first, last, result, thrust::identity<T>());
}
```

Only "primitives" like ```for_each``` require special system-specific implementations because they can't be expressed in terms of other algorithms. However, because we might wish to provide a special implementation, the entry points of these non-primitive algorithms still get dispatched just like ```thrust::for_each```. This is useful when a faster implementation of an algorithm exists, even though it is non-primitive. If no specialization is found, the generic forms of these non-primitive algorithms are dispatched instead.
