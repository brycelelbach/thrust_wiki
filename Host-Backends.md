Like the device system, we can also control the host system which applies to untagged, or "raw" types like ```thrust::host_vector::iterator```, ```std::vector::iterator```s, and raw pointers. The host system backend is selected via the ```THRUST_HOST_BACKEND``` macro:

    $ nvcc -Xcompiler -fopenmp -DTHRUST_HOST_BACKEND=THRUST_HOST_BACKEND_OMP my_program.cu -lgomp

By default, ```THRUST_HOST_BACKEND``` is set to ```THRUST_HOST_BACKEND_CPP```. It can be set to any one of

  * ```THRUST_HOST_BACKEND_CPP```
  * ```THRUST_HOST_BACKEND_OMP```
  * ```THRUST_HOST_BACKEND_TBB```

Currently, the implementation of all host systems' memories are interoperable.