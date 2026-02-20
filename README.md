# Natural Gas Pipeline: NDT Analysis Using Image Processing and Computer Vision

> [!NOTE]
> Once unit tests and code coverage is tested, this section will be updated with appropriate badges.

## Overview

In order to advance towards a green and sustainable future, reusing existing infrastructure will be a key factor. Hydrogen gas has the potential to be used in natural gas pipelines in the shift to renewable energy; however, one major hurdle it can cause is [hydrogen embrittlement](https://ntrs.nasa.gov/citations/20160005654). By carefully maintaining iron pipelines using a [Pipeline Inspection Gauge (PIG)](https://www.emerson.com/en-us/automation/measurement-instrumentation/pipeline-inspection-gauge-detection), we can effectively determine the best way to avoid damage due to hydrogen embrittlement.

The NDT image analysis software aims to correctly distinguish between images containing microcracks and micro pits versus images containing no defects.

> [!NOTE]
> The current version of this project is a work-in-progress. All code is subject to change.

## Highlights

> [!NOTE]
> This section is a work-in-progress and will be updated when measured data can be acquired.

## Installation Instructions

Unfortunately, this project doesn't support builds for Windows or MacOS at the time. If you wish for this feature to be implemented for the future, submit an issue on GitHub.

### Installing on a Linux Machine

> *Prerequisites*
> - CMake
> - Unix Makefiles

```bash
cmake -B build -S . -DPython3_EXECUTABLE=$(pwd)/.venv/bin/python
cmake --build build 
```

### Installing on the Raspberry Pi Docker Image

> *Prerequisites*
> - Docker
> - QEMU

```bash
# on boot, run this command before anything else (only required once)
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes

# build for the arm64 architecture
docker build --platform=linux/arm64 -t imgproc-pi .
```

## Usage Instructions

### Running the Image Processing Software

```bash
# run the C++ implementation
./build/imgproc

# run the Python implementation
./build/imgproc -p
```

### Emulating the Raspberry Pi using Docker

```bash
# run the container with arm64 (will emulate using qemu)
docker run --rm --platform=linux/arm64 imgproc-pi
```

### Running the Research Scripts

> *Prerequisites*
> - Python
> - Numpy
> - Matplotlib
> - OpenCV

```bash
# run the binary mask, max-median filter, morphological hough transform algorithm
python3.12 research/binary-close-open-hough-transform.py
```