FROM ubuntu:latest



RUN mkdir /bonndit
WORKDIR /bonndit
ENV RDBASE=/bonndit
WORKDIR $RDBASE

RUN apt-get update
RUN apt install build-essential cmake libcerf-dev wget python3 python3-pip -y
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir scipy cython pandas dipy cvxopt mpmath psutil pynrrd plyfile

RUN apt-get install git libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev libfftw3-dev libsuitesparse-dev  libblas-dev liblapack-dev libopenblas-dev  liblapacke-dev gfortran -y
RUN wget http://ceres-solver.org/ceres-solver-2.1.0.tar.gz
RUN tar zxf ceres-solver-2.1.0.tar.gz
RUN mkdir ceres-bin
RUN cd ceres-bin && \
     cmake ../ceres-solver-2.1.0 && \
     make -j16 && \
     make install

RUN git clone --depth 1 https://github.com/MedVisBonn/bonndit.git --branch new-dev && \
    echo $(ls) && \
    mv ./bonndit/* ./ && \
    rm -rf bonndit
RUN echo $(ls) \ && exit
RUN rm -rf build
RUN mkdir build
RUN cd build && \
    cmake .. && \
   make install

RUN mkdir data
RUN groupadd -rg 1002 bonndit && useradd -ru 1002 -g bonndit -d /data tracktograph

RUN WATSON=TRUE pip install .

RUN rm -rf /bonndit

SHELL ["/bin/bash", "-c", "-l"]
USER tracktograph
