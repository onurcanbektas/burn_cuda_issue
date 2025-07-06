# Interacting many-body langevin systems, GPU implementation (IMBLS-GPU)
Here we use uniform spatial partitioning algorithm to implement many-body interacting particle system governed by a langevin equation.
**This work still in development.**


# TODOs
- [X] Fix `histogram` kernel: we need to calculate the offset tensor alongisde. It is just the nth occurence of a value, when incrementing the histogram, just make a save of its state of the `offset` tensor 
- [X] Write tests for the `interact` kernel
- [X] Implement periodic euclidean distance calculation in `interact_kernel` 
- [X] Implement input-in-and-out method for other kernels and benchmark memory allocation (https://nnethercote.github.io/perf-book/profiling.html)
- [X] Establish common benchmark for NP = 2^18 and NP = 2^20
- [X] PBM to neighborlist calculation can first skip zeroth bin so that every thread doesn't have to check whether it is in the zero-th bin
- [ ] Implement all the interactions terms and simulate 200 particles with the same set of parameters
- [ ] Implement the active friction term using a kernel on `Line`
- [ ] set the coefficients of the forces according to your Smarticles model
- [ ] after debugging, check the following: estimation of the velocity field is biased. The `at_n_sum` shouldn't contain +1.0. Remove and test it again once debugging is done!

## Tests

- [X] bins
- [X] cumsum
- [X] histogram
- [X] reorder
- [ ] pbm
- [X] interact

## Benchmarks

- [x] bins
- [x] cumsum
- [x] histogram
- [x] reorder
- [x] interact

## Benchmarks (todo)

# burn_cuda_issue
# burn_cuda_issue
