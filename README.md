Code conventions:

Build system:  
CMake, https://cliutils.gitlab.io/modern-cmake/

Style guide is based on JMLL

Plus
 
1) c++17 for modern std functionality (filesystem, etc)
2) pragma once for headers
3) pointers near type, not near var name 
4) Filenames: snake_case only, headers *.h; c++ are *.cpp
5) enums are forbidden, enum class instead;


Project structure:

C++:

cpp/:

Public headers: cpp/include/…
Implementation: cpp/src/…
Tests: cpp/tests/…

Python:

python/…


 
Testing framework: Google Test
Python binds: pybind11


How to build with CUDA:

NVIDIA use outdate compilers
We want host code to be c++17 
but cuda 10 still use all gcc/clang (xcode tools from 10.13 on macOs, where there is no cpp17), so building it a bit tricky

cmake .. -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-5 -DUSE_CUDA=true -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc



PYTORCH
https://pytorch.org/cppdocs/installing.html

https://github.com/pytorch/pytorch/issues/14165


# Experiments

## Cifar-10 models

| model | optimizer info | learning rate | # of epochs | training log | accuracy | comments |
| ----- | -------------- | ------------- | ----------- | ------------ | -------- | -------- |
| LeNet | SGD(momentum = 0.9) | constant, lr = 0.001 | 2 | - | 51% | - |
| LeNet | SGD(momentum = 0.9) | constant, lr = 0.001 | 50 | [log](example_training_logs/cifar10_LeNet_SGD_50.txt) | 56% | judging by log, SGD diverges |
| VGG-16 | - | - | - | - | - | takes too long on my PC to train |