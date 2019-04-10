FROM nvidia/cuda:10.0-devel-ubuntu18.04

RUN apt-get update -q -y && apt-get install -y -q zlib1g-dev wget curl
RUN curl https://cmake.org/files/v3.13/cmake-3.13.2-Linux-x86_64.sh --output /cmake-3.13.2-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.13.2-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN ln -s /opt/cmake/bin/ctest /usr/local/bin/ctest

COPY . /app
RUN wget https://download.pytorch.org/libtorch/nightly/cu100/libtorch-shared-with-deps-latest.zip -P /
RUN apt-get install -y -q unzip
RUN unzip libtorch-shared-with-deps-latest.zip
ADD https://github.com/OPM/eigen3/archive/master.zip /eigen3.zip
RUN unzip /eigen3.zip
WORKDIR /app/build
