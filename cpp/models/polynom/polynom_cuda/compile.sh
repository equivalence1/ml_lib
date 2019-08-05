/usr/local/cuda-10.1/bin/nvcc  -c -o libsoft_polynom.so  --ptxas-options=-v --compiler-options '-fPIC' -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75  -gencode arch=compute_35,code=compute_35  -gencode arch=compute_61,code=compute_61  --shared soft_polynom.cu

