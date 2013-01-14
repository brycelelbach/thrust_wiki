Like the device system, we can also control the host system which applies to untagged, or "raw" types like ```thrust::host_vector::iterator```, ```std::vector::iterator```s, and raw pointers. We can use this feature to parallelize Thrust algorithms on host types such as `thrust::host_vector` with a parallel backend like OpenMP.

The host system backend is selected via the ```THRUST_HOST_SYSTEM``` macro:

    $ nvcc -Xcompiler -fopenmp -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP my_program.cu -lgomp

By default, ```THRUST_HOST_SYSTEM``` is set to ```THRUST_HOST_SYSTEM_CPP```. It can be set to any one of

  * ```THRUST_HOST_SYSTEM_CPP```
  * ```THRUST_HOST_SYSTEM_OMP```
  * ```THRUST_HOST_SYSTEM_TBB```

Currently, the implementation of all host systems' memories are interoperable.