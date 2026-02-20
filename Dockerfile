# Build the project on debian bookworm
FROM debian:bookworm-slim AS build

# Install standard build tools and python headers into the container
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3-dev \
    git \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
RUN cmake --build build --config Release

# Setup the runtime environment
FROM dtcooper/raspberrypi-os:bookworm AS runtime

# Install python and libpython as runtime dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 \
    libpython3.11 \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Create a user to run the image processing software
RUN groupadd -r ndt && useradd -r -g ndt -d /home/imgproc -s /sbin/nologin imgproc
RUN mkdir -p /home/imgproc/bin && chown -R imgproc:ndt /home/imgproc

# Assembly and run
FROM runtime
WORKDIR /home/imgproc

# Copy the build into the runtime environment
COPY --from=build /app/build/imgproc ./bin/imgproc

# Copy test images into the runtime environment
COPY --from=build /app/images ./images

# TODO check if this actually works on the RPi; maybe change to dynamically detect Python3
COPY scripts/ /usr/local/lib/python3.11/dist-packages/scripts

# Empty init required for making scripts packages
RUN touch /usr/local/lib/python3.11/dist-packages/scripts/__init__.py

# Switch to imgproc user and run
USER imgproc
WORKDIR /home/imgproc
ENTRYPOINT [ "bin/imgproc" ]