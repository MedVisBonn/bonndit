FROM ubuntu:latest

ARG watson
ARG UID
ARG GID
ARG watson
RUN if [ -z "$watson" ] ; then echo "Will build without Watson support. Set --build-arg watson=true for watson support" ; else echo "Build with watson support" ; fi

RUN mkdir /bonndit
WORKDIR /bonndit
ENV RDBASE=/bonndit
WORKDIR $RDBASE

RUN apt-get update
RUN apt install build-essential cmake libcerf-dev wget python3 python3-pip -y
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir scipy cython pandas dipy cvxopt mpmath psutil pynrrd plyfile

RUN if [ -z "$watson" ] ; then \
    apt-get install git libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev libfftw3-dev libsuitesparse-dev  liblapacke-dev -y && \
    wget http://ceres-solver.org/ceres-solver-2.1.0.tar.gz && \
    tar zxf ceres-solver-2.1.0.tar.gz && \
    mkdir ceres-bin && \
    cd ceres-bin && \
    cmake ../ceres-solver-2.1.0 && \
    make -j16 && \
    make install; \
    else \
    apt-get install git libatlas-base-dev liblapacke-dev -y; \
    fi

ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache


RUN git clone --depth 1 https://github.com/MedVisBonn/bonndit.git --branch running-time-ply && \
    echo $(ls) && \
    mv ./bonndit/* ./ && \
    rm -rf bonndit

RUN if [ -z "$watson" ]; then \
    rm -rf build && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make install; \
    fi

RUN mkdir /data
RUN groupadd -rg $GID bonndit && useradd -ru $UID -g bonndit -d /data tracktograph && \
    chown tracktograph:bonndit /data


RUN if [ -z "$watson" ]; then \
    WATSON=TRUE pip install .; \
    else \
    pip install .; \
    fi

#RUN rm -rf /bonndit
WORKDIR /data

SHELL ["/bin/bash", "-c", "-l"]
USER tracktograph
