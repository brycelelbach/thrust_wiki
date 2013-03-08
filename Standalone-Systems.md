Introduction
------------
Building a stand alone Thrust system is similar to [customizing](https://github.com/thrust/thrust/wiki/Derived-Systems) an existing one. However, because a stand alone system has no derived algorithm implementations to rely on, all primitives must be implemented. There are two major classes of primitive functionality which a Thrust system must provide: memory model and algorithms. On this page, we'll demonstrate a stand alone system similar to Thrust's standard C++ system. Our novel system will be roughly equivalent in functionality to ```thrust::cpp```, but otherwise unrelated.

Memory Model Primitives
-----------------------

The first task when building a stand alone Thrust system is to build the protocol which defines its memory model. Special primitives are required to support systems which may be implemented with distributed, or remote memory architectures. These primitives allow the programmer to manipulate logical scalar objects in memory.

Let's choose a name for our system. How about ```standalone```? Let's create its system policy tag in that namespace:

```c++
#include <thrust/execution_policy.h>

namespace standalone
{

// standalone's tag is just an empty struct derived from thrust::execution_policy.
struct tag : thrust::execution_policy<tag> {};

}
```

Next, we'll define how to allocate and deallocate the standalone system's storage. Since we're just using C++, we'll implement ```standalone::malloc``` and ```standalone::free``` with ```std::malloc``` and ```std::free```.

```c++
#include <cstdlib>
#include <thrust/memory.h>

namespace standalone
{

void *malloc(policy_tag, std::size_t n)
{
  return std::malloc(n);
}

template<typename Pointer>
void free(tag, Pointer ptr)
{
  std::free(thrust::raw_pointer_cast(ptr));
}

}
```

In our example, ```standalone::malloc```'s semantics are identical to ```std::malloc```, with the addition of the ```standalone::tag``` parameter. ```standalone::free``` needs to be a template, because it is responsible for deallocating any pointer-like thing tagged with ```standalone::tag```. We use ```thrust::raw_pointer_cast``` to get ```ptr```'s raw pointer to give to ```std::free```.

Next, we'll define the protocol for manipulating pointers tagged with ```standalone::tag```. There are three of these primitives. ```assign_value``` assigns the value pointed to by one pointer to that of another. ```get_value``` dereferences a pointer and returns a copy of its pointee's value. ```iter_swap``` exchanges the values of two pointees.

```c++
#include <thrust/memory.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/swap.h>

namespace standalone
{

template<typename Pointer1, typename Pointer2>
__host__ __device__
void assign_value(tag, Pointer1 dst, Pointer2 src)
{
  *thrust::raw_pointer_cast(dst) = *thrust::raw_pointer_cast(src);
}

template<typename Pointer>
__host__ __device__
typename thrust::iterator_value<Pointer>::type
get_value(tag, Pointer ptr)
{
  return *thrust::raw_pointer_cast(ptr);
}

template<typename Pointer1, typename Pointer2>
__host__ __device__
void iter_swap(tag, Pointer1 a, Pointer2 b)
{
  using thrust::swap;

  swap(*thrust::raw_pointer_cast(a), *thrust::raw_pointer_cast(b));
}

}
```

In this example, ```assign_value``` can simply dereference the raw pointers wrapped by ```dst``` and ```src```. However, systems such as Thrust's CUDA system must implement this primitive carefully to avoid dereferencing a remote pointer.

Likewise, ```get_value``` simply returns the result of dereferencing ```ptr```'s raw pointer. This primitive may seem superfluous and implementable with ```assign_value```, but this is not the case for heterogeneous memories like CUDA's.

Finally, ```iter_swap```` is necessary to exchange the values of two potentially remote objects without introducing a temporary. The simple standalone system can implement this primitive with ```raw_pointer_cast``` and ```swap```.

Even though these primitives are unrelated to CUDA, they require ```__host__ __device__``` decoration at this time because they may be invoked by a ```__host__ __device__``` function in Thrust's implementation.

Tagged Pointer and Reference Types
----------------------------------

Now that we've defined how to allocate and interact with objects in memory, we can wrap up the protocol so that it's less verbose to use. This is the job of ```thrust::pointer```.

We want our standalone system to be interoperable with Thrust's standard C++ system so that we can write expressions like ```*standalone_ptr = 13```. To enable this to work, we need to define what happens when the Thrust dispatch process encounters a ```standalone::tag``` paired with a ```thrust::cpp::tag```.

We do this by providing an overload of ```select_system```:

```c++
#include <thrust/system/cpp/memory.h>

namespace standalone
{

__host__ __device__
tag select_system(tag, thrust::cpp::tag)
{
  return tag();
}

}
```

Let's try it all out:

```c++
#include <thrust/memory.h>

int main()
{
  typedef thrust::pointer<int, standalone::tag> int_ptr_t;

  int_ptr_t ptr(static_cast<int*>(malloc(standalone::tag(), sizeof(int))));

  *ptr = 13;

  std::cout << "value is " << *ptr << std::endl;

  free(standalone::tag(), ptr);
}
```

    $ nvcc standalone_memory.cu -o standalone_memory
    ./standalone_memory
    value is 13

It works!

Algorithmic Primitives
----------------------

Now that we've implemented our ```standalone``` system's model primitives, we can move on to implementing the algorithmic primitives. There are several of these, and to build a fully-conforming system we'd need to implement all of them. In this example we'll demonstrate how to implement just a couple of them.

Perhaps the simplest algorithmic primitive to implement is ```for_each```. Let's build a straightforward serial implementation with a ```for``` loop:

```c++
#include <thrust/memory.h>

namespace standalone
{

template<typename Iterator, typename Function>
  Iterator for_each(tag, Iterator first, Iterator last, Function f)
{
  for(; first != last; ++first)
  {
    f(thrust::raw_reference_cast(*first));
  }

  return first;
}

}
```

Let's deconstruct what we see here. First, we've provided an overloaded form of ```for_each``` which takes ```standalone::tag``` as its first parameter. Importantly, we've provided our ```for_each``` in the *same namespace* as the ```tag```. The body of the ```for``` loop looks normal, except that we use ```thrust::raw_reference_cast``` to transform the tagged reference returned by ```*first``` into a raw reference suitable to pass to ```for_each```'s function object parameter.

Let's see if Thrust can dispatch our ```for_each``` with the pointers we built before:

```c++
#include <thrust/for_each.h>
#include <thrust/memory.h>

struct printer
{
  void operator()(int x)
  {
    std::cout << "x" << std::endl;
  }
};

int main()
{
  typedef thrust::pointer<int,standalone::tag> int_ptr_t;

  int_ptr_t ptr(static_cast<int*>(malloc(standalone::tag(), 3 * sizeof(int))));

  ptr[0] = 7;
  ptr[1] = 13;
  ptr[2] = 42;

  thrust::for_each(ptr, ptr + 3, printer());

  free(standalone::tag(), ptr);
}
```

Here's the result:

    $ nvcc standalone_for_each.cu -o standalone_for_each
    $ ./standalone_for_each
    7
    13
    42

Let's continue with ```reduce```:

```c++
#include <thrust/iterator/iterator_traits.h>
#include <thrust/memory.h>

namespace standalone
{

template<typename InputIterator, typename T, typename BinaryFunction>
  typename thrust::iterator_value<InputIterator>::type
    reduce(tag, InputIterator first, InputIterator last, T init, BinaryFunction binary_op)
{
  T result = init;

  for(; first != last; ++first)
  {
    result = binary_op(result, thrust::raw_reference_cast(*first));
  }

  return result;
}

}
```

That's one version of ```reduce```, but what about its two other overloads? Our version of ```reduce``` is its most general form; we get the other two for free. Let's try it:

```c++
#include <thrust/memory.h>
#include <thrust/functional.h>
#include <iostream>

int main()
{
  typedef thrust::pointer<int,standalone::tag> int_ptr_t;

  int_ptr_t ptr(static_cast<int*>(malloc(standalone::tag(), 3 * sizeof(int))));

  ptr[0] = 7;
  ptr[1] = 13;
  ptr[2] = 42;

  int result1 = thrust::reduce(ptr, ptr + 3);

  std::cout << "result1 is " << result1 << std::endl;

  int result2 = thrust::reduce(ptr, ptr + 3, 1);

  std::cout << "result2 is " << result2 << std::endl;

  int result3 = thrust::reduce(ptr, ptr + 3, 1, thrust::multiplies<int>());

  std::cout << "result3 is " << result3 << std::endl1;

  free(standalone::tag(), ptr);
}
```

Here's the result:

    $ nvcc standalone_reduce.cu -o standalone_reduce
    $ ./standalone_reduce
    result1 is 62
    result2 is 63
    result3 is 3822

Only the most general form of ```reduce``` is a primitive; Thrust knows how to implement the other two in terms of the more primitive form. Of course, we can always provide overloads for those other forms if we wish.

System Interoperability
-----------------------

What happens when an algorithm has many iterator parameters and their system tags differ? This is where the function ```select_system``` comes in. Remember that Thrust dispatches algorithms using the result of the special function ```select_system```. In general, ```select_system``` may receive many tag arguments during dispatch:

```c++
template<typename Iterator1, typename Iterator2, ..., typename IteratorN>
void some_algorithm(Iterator1 first1, Iterator1 last1, Iterator2 first2, ..., IteratorN firstN)
{
  // get the iterators' system tags
  typedef typename thrust::iterator_system<Iterator1>::type tag1;
  typedef typename thrust::iterator_system<Iterator2>::type tag2;
  ...
  typedef typename thrust::iterator_system<Iterator4>::type tagN;

  return some_algorithm(select_system(tag1(),tag2(), ..., tagN()), first1, first2, ..., firstN);
}
```

The default version of ```select_system``` performs a reduction over its tag arguments and returns the "minimal" tag, if it finds one. Here, the "minimal" tag is the tag whose type can be converted to from every other tag in ```select_system```'s parameter list. For example, the result of ```select_system(thrust::cuda::tag(), thrust::any_system_tag())``` is ```thrust::cuda::tag```, since ```thrust::any_system_tag``` is convertible to ```thrust::cuda::tag```. It is an error if no such system tag exists.

We might wish to customize the behavior of ```select_system```, especially when we want two systems to interoperate. A good example is when overloading Thrust's ```copy``` algorithm. Recall that our ```standalone``` system has already provided the overload ```select_system(standalone::tag, thrust::cpp::tag)```, which allows us to store values into ```standalone```-tagged pointers using expressions like ```*ptr = 13```.

To enable ```thrust::copy``` to allow copying between standard C++ and our ```standalone``` system, we need to provide two overloads of ```select_system```, each corresponding to a direction of the ```copy```:

```c++
namespace standalone
{

__host__ __device__
tag select_system(tag, thrust::cpp::tag)
{
  return tag();
}

__host__ __device-_
tag select_system(thrust::cpp::tag, tag)
{
  return tag();
}

template<typename InputIterator, typename OutputIterator>
OutputIterator copy(tag, InputIterator first, InputIterator last, OutputIterator result)
{
  for(; first != last; ++first, ++result)
  {
    thrust::raw_reference_cast(*result) = thrust::raw_reference_cast(*first);
  }

  return result;
}

}
```

Now we should be able to copy in and out of our ```standalone``` backend.

```c++
#include <thrust/copy.h>
#include <vector>
#include <iterator>

int main()
{
  typedef thrust::pointer<int,standalone::tag> int_ptr_t;

  int_ptr_t ptr(static_cast<int*>(malloc(standalone::tag(), 3 * sizeof(int))));

  std::vector<int> vec(3);
  vec[0] = 7;
  vec[1] = 13;
  vec[2] = 42;

  thrust::copy(vec.begin(), vec.end(), ptr);

  std::cout << "copied in ";
  thrust::copy(ptr, ptr + 3, std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;

  ptr[0] = 77;
  ptr[1] = 1313;
  ptr[2] = 4242;

  thrust::copy(ptr, ptr + 3, vec.begin());

  std::cout << "copied out ";
  thrust::copy(vec.begin(), vec.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;

  free(standalone::tag(), ptr);
}
```

    $ nvcc standalone_copy.cu -o standalone_copy
    $ ./standalone_copy
    copied in 7 13 42
    copied out 77 13 4242


If necessary, we can encode the directionality of the copy by introducing new tags, and specialize ```standalone```'s versions of ```select_system``` and ```copy``` further:

```c++
#include <thrust/system/cpp/memory.h>
#include <iostream>

namespace standalone
{

struct cpp_to_standalone {};

struct standalone_to_cpp {};

__host__ __device__
cpp_to_standalone select_system(thrust::cpp::tag, tag)
{
  return cpp_to_standalone();
}

__host__ __device__
standalone_to_cpp select_system(tag, thrust::cpp::tag)
{
  return standalone_to_cpp();
}

template<typename InputIterator, typename OutputIterator>
  OutputIterator copy(cpp_to_standalone, InputIterator first, InputIterator last, OutputIterator result)
{
  std::cout << "copying from cpp to standalone" << std::endl;

  ...
}

template<typename InputIterator, typename OutputIterator>
  OutputIterator copy(standalone_to_cpp, InputIterator first, InputIterator last, OutputIterator result)
{
  std::cout << "copying from standalone to cpp" << std::endl;

  ...
}

}
```

Thrust's CUDA system uses this idiom to implement copies between system and GPU memory.

In general, it is at each system's discretion how (if at all) to handle iterator parameters with heterogeneous system tags.

Further Reading
---------------

  * [Tag-based dispatching](http://www.boost.org/community/generic_programming.html#tag_dispatching)
  * [Argument dependent lookup](http://en.wikipedia.org/wiki/Argument_dependent_lookup)