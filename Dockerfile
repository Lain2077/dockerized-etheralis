# Use the official Python 3.11.4 image with CUDA support as the base image
FROM nvidia/cuda:12.1.1-runtime-ubuntu20.04

# Set the working directory
WORKDIR /app
ARG OPENCV_VERSION=4.7.0

# Update and upgrade packages, then install necessary dependencies
RUN apt-get update && apt-get upgrade -y && \
    # Install software-properties-common to manage PPAs
    apt-get install -y software-properties-common && \
    # Add the deadsnakes PPA for newer Python versions
    add-apt-repository ppa:deadsnakes/ppa && \
    # Update package lists after adding new PPA
    apt-get update && \
    # Install Python 3.11 and other necessary packages
    apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3.11-venv \
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    yasm \
    pkg-config \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavformat-dev \
    libpq-dev \
    libxine2-dev \
    libglew-dev \
    libtiff5-dev \
    zlib1g-dev \
    libavcodec-dev \
    libavutil-dev \
    libpostproc-dev \
    libeigen3-dev \
    libgtk2.0-dev \
    libgl1-mesa-glx \
    libavahi-common3 \
    libavahi-client-dev \
    avahi-daemon \
    && rm -rf /var/lib/apt/lists/*


RUN service avahi-daemon start

# Upgrade CMake to version 3.24 or higher
RUN wget https://cmake.org/files/v3.24/cmake-3.24.3-linux-x86_64.tar.gz \
    && tar -zxvf cmake-3.24.3-linux-x86_64.tar.gz \
    && cp -r cmake-3.24.3-linux-x86_64/bin/* /usr/local/bin/ \
    && cp -r cmake-3.24.3-linux-x86_64/share/* /usr/local/share/ \
    && rm -rf cmake-3.24.3-linux-x86_64 cmake-3.24.3-linux-x86_64.tar.gz


# Upgrade pip and install Python packages
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && pip3 install --upgrade pip \
    && pip3 install wheel

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Update alternatives to set Python 3.11 and pip3.11 as the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip3 pip3 $(which pip3.11) 1

# Copy the utils folder containing the extracted NDI SDK
COPY utils/ /app/utils/

# Install NDI SDK
RUN chmod +x /app/utils/Install_NDI_SDK_v6_Linux.sh && \
    cd /app/utils/ && \
    set -x && ./Install_NDI_SDK_v6_Linux.sh > /app/ndi_install_log.txt 2>&1 || { echo 'Installation failed'; cat /app/ndi_install_log.txt; exit 1; }

# Rename NDI SDK directory to handle spaces
RUN mv "/app/utils/NDI SDK for Linux" "/app/utils/NDI_SDK_for_Linux"

# Verify NDI SDK installation
# Set environment variables for NDI SDK
ENV NDI_SDK_DIR=/app/utils/NDI_SDK_for_Linux
ENV CMAKE_ARGS="-DNDI_SDK_DIR=$NDI_SDK_DIR"

# Clone the ndi-python repository
RUN git clone --recursive https://github.com/buresu/ndi-python.git /app/ndi-python

# Build ndi-python
RUN cd /app/ndi-python \
    && python3 setup.py build \
    && python3 setup.py bdist_wheel

# Install the built wheel
RUN pip3 install /app/ndi-python/dist/*.whl
    
# Download and install the CUDA-optimized OpenCV wheel
#RUN wget https://github.com/cudawarped/opencv-python-cuda-wheels/releases/download/4.7.0.20230527/opencv_contrib_python_rolling-4.7.0.20230527-cp36-abi3-linux_x86_64.whl -O opencv_contrib_python_rolling-4.7.0.20230527-cp36-abi3-linux_x86_64.whl && \
#    ls -lh opencv_contrib_python_rolling-4.7.0.20230527-cp36-abi3-linux_x86_64.whl && \
#    pip install opencv_contrib_python_rolling-4.7.0.20230527-cp36-abi3-linux_x86_64.whl && \
#    rm opencv_contrib_python_rolling-4.7.0.20230527-cp36-abi3-linux_x86_64.whl

RUN cd /opt/ &&\
    # Download and unzip OpenCV and opencv_contrib and delte zip files
    wget https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip &&\
    unzip $OPENCV_VERSION.zip &&\
    rm $OPENCV_VERSION.zip &&\
    wget https://github.com/opencv/opencv_contrib/archive/$OPENCV_VERSION.zip &&\
    unzip ${OPENCV_VERSION}.zip &&\
    rm ${OPENCV_VERSION}.zip &&\
    # Create build folder and switch to it
    mkdir /opt/opencv-${OPENCV_VERSION}/build && cd /opt/opencv-${OPENCV_VERSION}/build &&\
    # Cmake configure
    cmake \
        -DOPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib-${OPENCV_VERSION}/modules \
        -DWITH_CUDA=ON \
        -DCUDA_ARCH_BIN=7.5,8.0,8.6 \
        -DCMAKE_BUILD_TYPE=RELEASE \
        # Install path will be /usr/local/lib (lib is implicit)
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        .. &&\
    # Make
    make -j"$(nproc)" && \
    # Install to /usr/local/lib
    make install && \
    ldconfig &&\
    # Remove OpenCV sources and build folder
    rm -rf /opt/opencv-${OPENCV_VERSION} && rm -rf /opt/opencv_contrib-${OPENCV_VERSION}


# Copy the necessary files
COPY requirements.txt ./
COPY src/ ./src/
COPY LICENSE README.md ./

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Set environment variables for CUDA (if needed)
ENV CUDA_HOME=/usr/local/cuda



# Create and switch to non-root user
#RUN adduser -u 5678 --disabled-password --gecos "" appuser && \
#    adduser appuser video && \
#    chown -R appuser /app
#USER appuser

# Command to run the application
CMD ["python3", "src/faceRec.py"]
