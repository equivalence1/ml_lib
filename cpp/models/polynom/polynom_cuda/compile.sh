/usr/local/cuda-10.0/bin/nvcc  -c -o libsoft_polynom.so  --ptxas-options=-v --compiler-options '-fPIC' --shared soft_polynom.cu
