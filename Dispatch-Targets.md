Introduction
------------

This page lists all of Thrust's targets of dispatch.

Memory Model Primitives
-----------------------

```c++
template<typename Pointer1, typename Pointer2>
__host__ __device__
void assign_value(tag, Pointer1 dst, Pointer2 src);

template<typename Pointer>
void free(tag, Pointer ptr);

template<typename Pointer>
__host__ __device__
typename thrust::iterator_value<Pointer>::type
get_value(tag, Pointer ptr);

template<typename Pointer1, typename Pointer2>
__host__ __device__
void iter_swap(tag, Pointer1 a, Pointer2 b);

template<typename Size>
void *malloc(tag, Size n);
```

Memory Model Non-Primitives
---------------------------

```c++
template<typename T>
thrust::pair<unspecified-pointer-type, unspecified-difference-type>
get_temporary_buffer(tag, unspecified-difference-type n);

template<typename Pointer>
void return_temporary_buffer(tag, Pointer p);

template<typename Tag>
__host__ __device__
unspecified-tag-type select_system(Tag);

template<typename Tag1, typename Tag2>
__host__ __device__
unspecified-tag-type select_system(Tag1,Tag2);

template<typename Tag1, typename Tag2, typename Tag3>
__host__ __device__
unspecified-tag-type select_system(Tag1,Tag2,Tag3);

template<typename Tag1, typename Tag2, typename Tag3, typename Tag4>
__host__ __device__
unspecified-tag-type select_system(Tag1,Tag2,Tag3,Tag4);
```

Primitive Algorithms
--------------------

```c++
template<typename InputIterator,
         typename OutputIterator,
         typename T,
         typename AssociativeOperator>
OutputIterator exclusive_scan(tag,
                              InputIterator first,
                              InputIterator last,
                              OutputIterator result,
                              T init,
                              AssociativeOperator binary_op);

template<typename InputIterator,
         typename Function>
InputIterator for_each(tag,
                       InputIterator first,
                       InputIterator last,
                       Function f);

template<typename InputIterator,
         typename Size,
         typename Function>
InputIterator for_each_n(tag,
                         InputIterator first,
                         Size n,
                         Function f);

template<typename InputIterator,
         typename OutputIterator,
         typename AssociativeOperator>
OutputIterator inclusive_scan(tag,
                              InputIterator first,
                              InputIterator last,
                              OutputIterator result,
                              AssociativeOperator binary_op);

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakCompare>
OutputIterator merge(tag,
                     InputIterator1 first1,
                     InputIterator1 last1,
                     InputIterator2 first2,
                     InputIterator2 last2,
                     OutputIterator result,
                     StrictWeakCompare comp);

template<typename InputIterator,
         typename T,
         typename BinaryFunction>
T reduce(tag,
         InputIterator first,
         InputIterator last,
         T init,
         BinaryFunction binary_op);

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate,
         typename BinaryFunction>
thrust::pair<OutputIterator1,OutputIterator2>
reduce_by_key(tag,
              InputIterator1 keys_first,
              InputIterator1 keys_last,
              InputIterator2 values_first,
              OutputIterator1 keys_output,
              OutputIterator2 values_output,
              BinaryPredicate binary_pred,
              BinaryFunction binary_op);

template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
void stable_sort(tag,
                 RandomAccessIterator first,
                 RandomAccessIterator last,
                 StrictWeakOrdering comp);

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
void stable_sort_by_key(tag,
                        RandomAccessIterator1 keys_first,
                        RandomAccessIterator1 keys_last,
                        RandomAccessIterator2 values_first);

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakCompare>
OutputIterator set_difference(tag,
                              InputIterator1 first1,
                              InputIterator1 last1,
                              InputIterator2 first2,
                              InputIterator2 last2,
                              OutputIterator result,
                              StrictWeakCompare comp);

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakCompare>
OutputIterator set_intersection(tag,
                                InputIterator1 first1,
                                InputIterator1 last1,
                                InputIterator2 first2,
                                InputIterator2 last2,
                                OutputIterator result,
                                StrictWeakCompare comp);

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakCompare>
OutputIterator set_symmetric_difference(tag, 
                                        InputIterator1 first1,
                                        InputIterator1 last1,
                                        InputIterator2 first2,
                                        InputIterator2 last2,
                                        OutputIterator result,
                                        StrictWeakCompare comp);

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakCompare>
OutputIterator set_union(tag, 
                         InputIterator1 first1,
                         InputIterator1 last1,
                         InputIterator2 first2,
                         InputIterator2 last2,
                         OutputIterator result,
                         StrictWeakCompare comp);
```

Note that these primitives are the fully-general forms of these algorithms.

These are the current primitives; the list may shrink or grow in the future.

Non-primitive Algorithms
------------------------

```c++
template<typename InputIterator,
         typename OutputIterator>
OutputIterator adjacent_difference(tag,
                                   InputIterator first,
                                   InputIterator last, 
                                   OutputIterator result);

template<typename InputIterator,
         typename OutputIterator,
         typename BinaryFunction>
OutputIterator adjacent_difference(tag,
                                   InputIterator first,
                                   InputIterator last,
                                   OutputIterator result,
                                   BinaryFunction binary_op);

template<typename InputIterator,
        typename Distance>
void advance(tag, InputIterator& i, Distance n);

template <typename InputIterator, typename Predicate>
bool all_of(tag, InputIterator first, InputIterator last, Predicate pred);

template <typename InputIterator, typename Predicate>
bool any_of(tag, InputIterator first, InputIterator last, Predicate pred);

template <typename ForwardIterator, typename T>
bool binary_search(tag,
                   ForwardIterator begin,
                   ForwardIterator end,
                   const T& value);

template <typename ForwardIterator, typename T, typename StrictWeakOrdering>
bool binary_search(tag,
                   ForwardIterator begin,
                   ForwardIterator end,
                   const T& value, 
                   StrictWeakOrdering comp);


template <typename ForwardIterator, typename InputIterator, typename OutputIterator>
OutputIterator binary_search(tag,
                             ForwardIterator begin, 
                             ForwardIterator end,
                             InputIterator values_begin, 
                             InputIterator values_end,
                             OutputIterator output);

template <typename ForwardIterator, typename InputIterator, typename OutputIterator, typename StrictWeakOrdering>
OutputIterator binary_search(tag,
                             ForwardIterator begin, 
                             ForwardIterator end,
                             InputIterator values_begin, 
                             InputIterator values_end,
                             OutputIterator output,
                             StrictWeakOrdering comp);

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(tag,
                      InputIterator  first,
                      InputIterator  last,
                      OutputIterator result);


template<typename InputIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator copy_n(tag,
                       InputIterator  first,
                       Size           n,
                       OutputIterator result);

template<typename InputIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator copy_if(tag,
                         InputIterator first,
                         InputIterator last,
                         OutputIterator result,
                         Predicate pred);

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
   OutputIterator copy_if(tag,
                          InputIterator1 first,
                          InputIterator1 last,
                          InputIterator2 stencil,
                          OutputIterator result,
                          Predicate pred);

template <typename InputIterator, typename EqualityComparable>
typename thrust::iterator_traits<InputIterator>::difference_type
count(tag, InputIterator first, InputIterator last, const EqualityComparable& value);

template <typename InputIterator, typename Predicate>
typename thrust::iterator_traits<InputIterator>::difference_type
count_if(tag, InputIterator first, InputIterator last, Predicate pred);

template<typename InputIterator>
inline typename thrust::iterator_traits<InputIterator>::difference_type
distance(tag, InputIterator first, InputIterator last);

template <typename InputIterator1, typename InputIterator2>
bool equal(tag, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2);

template <typename InputIterator1, typename InputIterator2, 
          typename BinaryPredicate>
bool equal(tag, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, BinaryPredicate binary_pred);

template <typename ForwardIterator, typename LessThanComparable>
thrust::pair<ForwardIterator,ForwardIterator>
equal_range(tag,
            ForwardIterator first,
            ForwardIterator last,
            const LessThanComparable &value);

template <typename ForwardIterator, typename LessThanComparable, typename StrictWeakOrdering>
thrust::pair<ForwardIterator,ForwardIterator>
equal_range(tag,
            ForwardIterator first,
            ForwardIterator last,
            const LessThanComparable &value,
            StrictWeakOrdering comp);

template<typename InputIterator,
         typename OutputIterator>
OutputIterator exclusive_scan(tag,
                              InputIterator first,
                              InputIterator last,
                              OutputIterator result);


template<typename InputIterator,
         typename OutputIterator,
         typename T>
OutputIterator exclusive_scan(tag,
                              InputIterator first,
                              InputIterator last,
                              OutputIterator result,
                              T init);

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
OutputIterator exclusive_scan_by_key(tag,
                                     InputIterator1 first1,
                                     InputIterator1 last1,
                                     InputIterator2 first2,
                                     OutputIterator result);

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T>
OutputIterator exclusive_scan_by_key(tag,
                                     InputIterator1 first1,
                                     InputIterator1 last1,
                                     InputIterator2 first2,
                                     OutputIterator result,
                                     T init);

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T,
         typename BinaryPredicate>
OutputIterator exclusive_scan_by_key(tag,
                                     InputIterator1 first1,
                                     InputIterator1 last1,
                                     InputIterator2 first2,
                                     OutputIterator result,
                                     T init,
                                     BinaryPredicate binary_pred);

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T,
         typename BinaryPredicate,
         typename AssociativeOperator>
OutputIterator exclusive_scan_by_key(tag,
                                     InputIterator1 first1,
                                     InputIterator1 last1,
                                     InputIterator2 first2,
                                     OutputIterator result,
                                     T init,
                                     BinaryPredicate binary_pred,
                                     AssociativeOperator binary_op);

template<typename ForwardIterator, typename T>
void fill(tag,
          ForwardIterator first,
          ForwardIterator last,
          const T &value);

template<typename OutputIterator, typename Size, typename T>
OutputIterator fill_n(tag,
                      OutputIterator first,
                      Size n,
                      const T &value);

template <typename InputIterator, typename T>
InputIterator find(tag,
                   InputIterator first,
                   InputIterator last,
                   const T& value);

template <typename InputIterator, typename Predicate>
InputIterator find_if(tag,
                      InputIterator first,
                      InputIterator last,
                      Predicate pred);

template <typename InputIterator, typename Predicate>
InputIterator find_if_not(tag,
                          InputIterator first,
                          InputIterator last,
                          Predicate pred);

template<typename InputIterator,
         typename RandomAccessIterator,
         typename OutputIterator>
OutputIterator gather(tag,
                      InputIterator        map_first,
                      InputIterator        map_last,
                      RandomAccessIterator input_first,
                      OutputIterator       result);

template<typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator,
         typename OutputIterator>
OutputIterator gather_if(tag,
                         InputIterator1       map_first,
                         InputIterator1       map_last,
                         InputIterator2       stencil,
                         RandomAccessIterator input_first,
                         OutputIterator       result);

template<typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator,
         typename OutputIterator,
         typename Predicate>
OutputIterator gather_if(tag,
                         InputIterator1       map_first,
                         InputIterator1       map_last,
                         InputIterator2       stencil,
                         RandomAccessIterator input_first,
                         OutputIterator       result,
                         Predicate            pred);

template<typename ForwardIterator,
         typename Generator>
void generate(tag,
              ForwardIterator first,
              ForwardIterator last,
              Generator gen);

template<typename OutputIterator,
         typename Size,
         typename Generator>
OutputIterator generate_n(tag,
                          OutputIterator first,
                          Size n,
                          Generator gen);

template<typename InputIterator,
         typename OutputIterator>
OutputIterator inclusive_scan(tag,
                              InputIterator first,
                              InputIterator last,
                              OutputIterator result);

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
OutputIterator inclusive_scan_by_key(tag,
                                     InputIterator1 first1,
                                     InputIterator1 last1,
                                     InputIterator2 first2,
                                     OutputIterator result);

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryPredicate>
OutputIterator inclusive_scan_by_key(tag,
                                     InputIterator1 first1,
                                     InputIterator1 last1,
                                     InputIterator2 first2,
                                     OutputIterator result,
                                     BinaryPredicate binary_pred);

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryPredicate,
         typename AssociativeOperator>
  OutputIterator inclusive_scan_by_key(tag,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       BinaryPredicate binary_pred,
                                       AssociativeOperator binary_op);

template<typename InputIterator1, typename InputIterator2, typename OutputType>
OutputType inner_product(tag,
                         InputIterator1 first1,
                         InputIterator1 last1,
                         InputIterator2 first2,
                         OutputType init);

template<typename InputIterator1, typename InputIterator2, typename OutputType,
         typename BinaryFunction1, typename BinaryFunction2>
OutputType inner_product(tag,
                         InputIterator1 first1,
                         InputIterator1 last1,
                         InputIterator2 first2,
                         OutputType init, 
                         BinaryFunction1 binary_op1,
                         BinaryFunction2 binary_op2);

template<typename InputIterator,
         typename Predicate>
bool is_partitioned(tag,
                    InputIterator first,
                    InputIterator last,
                    Predicate pred);

template<typename ForwardIterator>
bool is_sorted(tag,
               ForwardIterator first,
               ForwardIterator last);

template<typename ForwardIterator,
         typename Compare>
bool is_sorted(tag,
               ForwardIterator first,
               ForwardIterator last,
               Compare comp);

template<typename ForwardIterator>
ForwardIterator is_sorted_until(tag,
                                ForwardIterator first,
                                ForwardIterator last);

template<typename ForwardIterator,
         typename Compare>
ForwardIterator is_sorted_until(tag,
                                ForwardIterator first,
                                ForwardIterator last,
                                Compare comp);

template <typename ForwardIterator, typename T>
ForwardIterator lower_bound(tag, 
                            ForwardIterator begin,
                            ForwardIterator end,
                            const T& value);

template <typename ForwardIterator, typename T, typename StrictWeakOrdering>
ForwardIterator lower_bound(tag,
                            ForwardIterator begin,
                            ForwardIterator end,
                            const T& value, 
                            StrictWeakOrdering comp);

template <typename ForwardIterator, typename InputIterator, typename OutputIterator>
OutputIterator lower_bound(tag,
                           ForwardIterator begin, 
                           ForwardIterator end,
                           InputIterator values_begin, 
                           InputIterator values_end,
                           OutputIterator output);

template <typename ForwardIterator, typename InputIterator, typename OutputIterator, typename StrictWeakOrdering>
OutputIterator lower_bound(tag,
                           ForwardIterator begin, 
                           ForwardIterator end,
                           InputIterator values_begin, 
                           InputIterator values_end,
                           OutputIterator output,
                           StrictWeakOrdering comp);

template <typename ForwardIterator>
ForwardIterator max_element(tag,
                            ForwardIterator first,
                            ForwardIterator last);

template <typename ForwardIterator, typename BinaryPredicate>
ForwardIterator max_element(tag,
                            ForwardIterator first,
                            ForwardIterator last,
                            BinaryPredicate comp);

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
OutputIterator merge(tag,
                     InputIterator1 first1,
                     InputIterator1 last1,
                     InputIterator2 first2,
                     InputIterator2 last2,
                     OutputIterator result);

template <typename ForwardIterator>
ForwardIterator min_element(tag,
                            ForwardIterator first,
                            ForwardIterator last);

template <typename ForwardIterator, typename BinaryPredicate>
ForwardIterator min_element(tag,
                            ForwardIterator first,
                            ForwardIterator last,
                            BinaryPredicate comp);

template <typename ForwardIterator>
thrust::pair<ForwardIterator,ForwardIterator> minmax_element(tag,
                                                             ForwardIterator first, 
                                                             ForwardIterator last);

template <typename ForwardIterator, typename BinaryPredicate>
thrust::pair<ForwardIterator,ForwardIterator> minmax_element(tag,
                                                             ForwardIterator first, 
                                                             ForwardIterator last,
                                                             BinaryPredicate comp);

template <typename InputIterator1, typename InputIterator2>
thrust::pair<InputIterator1, InputIterator2>
mismatch(tag,
         InputIterator1 first1,
         InputIterator1 last1,
         InputIterator2 first2);

template <typename InputIterator1, typename InputIterator2, typename BinaryPredicate>
thrust::pair<InputIterator1, InputIterator2>
mismatch(tag,
         InputIterator1 first1,
         InputIterator1 last1,
         InputIterator2 first2,
         BinaryPredicate pred);

template <typename InputIterator, typename Predicate>
bool none_of(tag, InputIterator first, InputIterator last, Predicate pred);

template<typename InputIterator>
typename thrust::iterator_traits<InputIterator>::value_type
reduce(tag, InputIterator first, InputIterator last);

template<typename InputIterator, typename T>
T reduce(tag, InputIterator first, InputIterator last, T init);

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2>
thrust::pair<OutputIterator1,OutputIterator2>
reduce_by_key(tag,
              InputIterator1 keys_first, 
              InputIterator1 keys_last,
              InputIterator2 values_first,
              OutputIterator1 keys_output,
              OutputIterator2 values_output);

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate>
thrust::pair<OutputIterator1,OutputIterator2>
reduce_by_key(tag,
              InputIterator1 keys_first, 
              InputIterator1 keys_last,
              InputIterator2 values_first,
              OutputIterator1 keys_output,
              OutputIterator2 values_output,
              BinaryPredicate binary_pred);

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate,
         typename BinaryFunction>
thrust::pair<OutputIterator1,OutputIterator2>
reduce_by_key(tag,
              InputIterator1 keys_first, 
              InputIterator1 keys_last,
              InputIterator2 values_first,
              OutputIterator1 keys_output,
              OutputIterator2 values_output,
              BinaryPredicate binary_pred,
              BinaryFunction binary_op);

template<typename ForwardIterator,
         typename T>
ForwardIterator remove(tag,
                       ForwardIterator first,
                       ForwardIterator last,
                       const T &value);

template<typename InputIterator,
         typename OutputIterator,
         typename T>
OutputIterator remove_copy(tag,
                           InputIterator first,
                           InputIterator last,
                           OutputIterator result,
                           const T &value);

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
OutputIterator remove_copy_if(tag,
                              InputIterator1 first,
                              InputIterator1 last,
                              InputIterator2 stencil,
                              OutputIterator result,
                              Predicate pred);

template<typename ForwardIterator,
         typename Predicate>
ForwardIterator remove_if(tag,
                          ForwardIterator first,
                          ForwardIterator last,
                          Predicate pred);

template<typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
ForwardIterator remove_if(tag,
                          ForwardIterator first,
                          ForwardIterator last,
                          InputIterator stencil,
                          Predicate pred);

template<typename InputIterator,
         typename OutputIterator,
         typename Predicate>
OutputIterator remove_copy_if(tag,
                              InputIterator first,
                              InputIterator last,
                              OutputIterator result,
                              Predicate pred);

template<typename ForwardIterator, typename T>
void replace(tag,
             ForwardIterator first,
             ForwardIterator last,
             const T &old_value,
             const T &new_value);

template<typename InputIterator, typename OutputIterator, typename Predicate, typename T>
OutputIterator replace_copy_if(tag,
                               InputIterator first,
                               InputIterator last,
                               OutputIterator result,
                               Predicate pred,
                               const T &new_value);

template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Predicate, typename T>
OutputIterator replace_copy_if(InputIterator1 first,
                               InputIterator1 last,
                               InputIterator2 stencil,
                               OutputIterator result,
                               Predicate pred,
                               const T &new_value);

template<typename InputIterator, typename OutputIterator, typename T>
OutputIterator replace_copy(tag,
                            InputIterator first,
                            InputIterator last,
                            OutputIterator result,
                            const T &old_value,
                            const T &new_value);

template<typename ForwardIterator, typename Predicate, typename T>
void replace_if(tag,
                ForwardIterator first,
                ForwardIterator last,
                Predicate pred,
                const T &new_value);

template<typename ForwardIterator, typename InputIterator, typename Predicate, typename T>
void replace_if(tag,
                ForwardIterator first,
                ForwardIterator last,
                InputIterator stencil,
                Predicate pred,
                const T &new_value);

template<typename BidirectionalIterator>
void reverse(tag,
             BidirectionalIterator first,
             BidirectionalIterator last);

template<typename BidirectionalIterator,
         typename OutputIterator>
OutputIterator reverse_copy(tag,
                            BidirectionalIterator first,
                            BidirectionalIterator last,
                            OutputIterator result);

template<typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator>
void scatter(tag,
             InputIterator1 first,
             InputIterator1 last,
             InputIterator2 map,
             RandomAccessIterator output);

template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename RandomAccessIterator>
void scatter_if(tag,
                InputIterator1 first,
                InputIterator1 last,
                InputIterator2 map,
                InputIterator3 stencil,
                RandomAccessIterator output);

template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename RandomAccessIterator,
         typename Predicate>
void scatter_if(tag,
                InputIterator1 first,
                InputIterator1 last,
                InputIterator2 map,
                InputIterator3 stencil,
                RandomAccessIterator output,
                Predicate pred);

template<typename ForwardIterator>
void sequence(tag,
              ForwardIterator first,
              ForwardIterator last);

template<typename ForwardIterator, typename T>
void sequence(tag,
              ForwardIterator first,
              ForwardIterator last,
              T init);

template<typename ForwardIterator, typename T>
void sequence(ForwardIterator first,
              ForwardIterator last,
              T init,
              T step);

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
OutputIterator set_difference(tag,
                              InputIterator1 first1,
                              InputIterator1 last1,
                              InputIterator2 first2,
                              InputIterator2 last2,
                              OutputIterator result);

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
OutputIterator set_intersection(tag,
                                InputIterator1 first1,
                                InputIterator1 last1,
                                InputIterator2 first2,
                                InputIterator2 last2,
                                OutputIterator result);

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
OutputIterator set_symmetric_difference(tag,
                                        InputIterator1 first1,
                                        InputIterator1 last1,
                                        InputIterator2 first2,
                                        InputIterator2 last2,
                                        OutputIterator result);

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
OutputIterator set_union(tag,
                         InputIterator1 first1,
                         InputIterator1 last1,
                         InputIterator2 first2,
                         InputIterator2 last2,
                         OutputIterator result);

template<typename RandomAccessIterator>
void sort(tag,
          RandomAccessIterator first,
          RandomAccessIterator last);

template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
void sort(tag,
          RandomAccessIterator first,
          RandomAccessIterator last,
          StrictWeakOrdering comp);

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
void sort_by_key(tag,
                 RandomAccessIterator1 keys_first,
                 RandomAccessIterator1 keys_last,
                 RandomAccessIterator2 values_first);

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
void sort_by_key(tag,
                 RandomAccessIterator1 keys_first,
                 RandomAccessIterator1 keys_last,
                 RandomAccessIterator2 values_first,
                 StrictWeakOrdering comp);

template<typename RandomAccessIterator>
void stable_sort(tag,
                 RandomAccessIterator first,
                 RandomAccessIterator last);

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
void stable_sort_by_key(tag,
                        RandomAccessIterator1 keys_first,
                        RandomAccessIterator1 keys_last,
                        RandomAccessIterator2 values_first);

template<typename ForwardIterator1,
         typename ForwardIterator2>
ForwardIterator2 swap_ranges(tag,
                             ForwardIterator1 first1,
                             ForwardIterator1 last1,
                             ForwardIterator2 first2);

template<typename ForwardIterator, typename UnaryOperation>
  void tabulate(tag,
                ForwardIterator first,
                ForwardIterator last,
                UnaryOperation unary_op);

template<typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction>
OutputIterator transform(tag,
                         InputIterator first,
                         InputIterator last,
                         OutputIterator result,
                         UnaryFunction op);

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryFunction>
OutputIterator transform(tag,
                         InputIterator1 first1,
                         InputIterator1 last1,
                         InputIterator2 first2,
                         OutputIterator result,
                         BinaryFunction op);

template<typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction,
         typename T,
         typename AssociativeOperator>
OutputIterator transform_exclusive_scan(tag,
                                        InputIterator first,
                                        InputIterator last,
                                        OutputIterator result,
                                        UnaryFunction unary_op,
                                        T init,
                                        AssociativeOperator binary_op);

template<typename InputIterator,
         typename ForwardIterator,
         typename UnaryFunction,
         typename Predicate>
ForwardIterator transform_if(tag,
                             InputIterator first,
                             InputIterator last,
                             ForwardIterator result,
                             UnaryFunction unary_op,
                             Predicate pred);

template<typename InputIterator1,
         typename InputIterator2,
         typename ForwardIterator,
         typename UnaryFunction,
         typename Predicate>
ForwardIterator transform_if(tag,
                             InputIterator1 first,
                             InputIterator1 last,
                             InputIterator2 stencil,
                             ForwardIterator result,
                             UnaryFunction unary_op,
                             Predicate pred);

template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename ForwardIterator,
         typename BinaryFunction,
         typename Predicate>
ForwardIterator transform_if(tag,
                             InputIterator1 first1,
                             InputIterator1 last1,
                             InputIterator2 first2,
                             InputIterator3 stencil,
                             ForwardIterator result,
                             BinaryFunction binary_op,
                             Predicate pred);

template<typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction,
         typename BinaryFunction>
OutputIterator transform_inclusive_scan(tag,
                                        InputIterator first,
                                        InputIterator last,
                                        OutputIterator result,
                                        UnaryFunction unary_op,
                                        BinaryFunction binary_op);

template<typename InputIterator, 
         typename UnaryFunction, 
         typename OutputType,
         typename BinaryFunction>
OutputType transform_reduce(tag,
                            InputIterator first,
                            InputIterator last,
                            UnaryFunction unary_op,
                            OutputType init,
                            BinaryFunction binary_op);

template<typename InputIterator,
         typename ForwardIterator>
ForwardIterator uninitialized_copy(tag,
                                   InputIterator first,
                                   InputIterator last,
                                   ForwardIterator result);

template<typename InputIterator,
         typename Size,
         typename ForwardIterator>
ForwardIterator uninitialized_copy_n(tag,
                                     InputIterator first,
                                     Size n,
                                     ForwardIterator result);

template<typename ForwardIterator,
         typename T>
void uninitialized_fill(tag,
                        ForwardIterator first,
                        ForwardIterator last,
                        const T &x);

template<typename ForwardIterator,
         typename Size,
         typename T>
ForwardIterator uninitialized_fill_n(tag,
                                     ForwardIterator first,
                                     Size n,
                                     const T &x);

template<typename ForwardIterator>
ForwardIterator unique(tag,
                       ForwardIterator first,
                       ForwardIterator last);

template<typename ForwardIterator,
         typename BinaryPredicate>
ForwardIterator unique(tag,
                       ForwardIterator first,
                       ForwardIterator last,
                       BinaryPredicate binary_pred);

template<typename ForwardIterator1,
         typename ForwardIterator2>
thrust::pair<ForwardIterator1,ForwardIterator2>
unique_by_key(tag,
              ForwardIterator1 keys_first, 
              ForwardIterator1 keys_last,
              ForwardIterator2 values_first);

template<typename ForwardIterator1,
         typename ForwardIterator2,
         typename BinaryPredicate>
thrust::pair<ForwardIterator1,ForwardIterator2>
unique_by_key(tag,
              ForwardIterator1 keys_first, 
              ForwardIterator1 keys_last,
              ForwardIterator2 values_first,
              BinaryPredicate binary_pred);

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2>
thrust::pair<OutputIterator1,OutputIterator2>
unique_by_key_copy(tag,
                   InputIterator1 keys_first, 
                   InputIterator1 keys_last,
                   InputIterator2 values_first,
                   OutputIterator1 keys_output,
                   OutputIterator2 values_output);

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate>
thrust::pair<OutputIterator1,OutputIterator2>
unique_by_key_copy(tag,
                   InputIterator1 keys_first, 
                   InputIterator1 keys_last,
                   InputIterator2 values_first,
                   OutputIterator1 keys_output,
                   OutputIterator2 values_output,
                   BinaryPredicate binary_pred);

template<typename InputIterator,
         typename OutputIterator>
OutputIterator unique_copy(tag,
                           InputIterator first,
                           InputIterator last,
                           OutputIterator output);

template<typename InputIterator,
         typename OutputIterator,
         typename BinaryPredicate>
OutputIterator unique_copy(tag,
                           InputIterator first,
                           InputIterator last,
                           OutputIterator output,
                           BinaryPredicate binary_pred);

template <typename ForwardIterator, typename T>
ForwardIterator upper_bound(tag,
                            ForwardIterator begin,
                            ForwardIterator end,
                            const T& value);

template <typename ForwardIterator, typename T, typename StrictWeakOrdering>
ForwardIterator upper_bound(tag, 
                            ForwardIterator begin,
                            ForwardIterator end,
                            const T& value, 
                            StrictWeakOrdering comp);

template <typename ForwardIterator, typename InputIterator, typename OutputIterator>
OutputIterator upper_bound(tag,
                           ForwardIterator begin, 
                           ForwardIterator end,
                           InputIterator values_begin, 
                           InputIterator values_end,
                           OutputIterator output);

template <typename ForwardIterator, typename InputIterator, typename OutputIterator, typename StrictWeakOrdering>
OutputIterator upper_bound(tag,
                           ForwardIterator begin, 
                           ForwardIterator end,
                           InputIterator values_begin, 
                           InputIterator values_end,
                           OutputIterator output,
                           StrictWeakOrdering comp);

template<typename ForwardIterator, typename Predicate>
ForwardIterator stable_partition(tag,
                                 ForwardIterator first,
                                 ForwardIterator last,
                                 Predicate pred);

template<typename InputIterator,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
thrust::pair<OutputIterator1,OutputIterator2>
stable_partition_copy(tag,
                      InputIterator first,
                      InputIterator last,
                      OutputIterator1 out_true,
                      OutputIterator2 out_false,
                      Predicate pred);

template<typename ForwardIterator,
         typename Predicate>
ForwardIterator partition(tag,
                          ForwardIterator first,
                          ForwardIterator last,
                          Predicate pred);

template<typename InputIterator,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
thrust::pair<OutputIterator1,OutputIterator2>
partition_copy(tag,
               InputIterator first,
               InputIterator last,
               OutputIterator1 out_true,
               OutputIterator2 out_false,
               Predicate pred);

template<typename ForwardIterator,
         typename Predicate>
ForwardIterator partition_point(tag,
                                ForwardIterator first,
                                ForwardIterator last,
                                Predicate pred);
```