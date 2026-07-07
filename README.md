# Natural Gas Pipeline: NDT Analysis Using Image Processing and Computer Vision

## Overview

In order to advance towards a green and sustainable future, reusing existing infrastructure will be a key factor. Hydrogen gas has the potential to be used in natural gas pipelines in the shift to renewable energy; however, one major problem it can cause is [hydrogen embrittlement](https://www.twi-global.com/technical-knowledge/faqs/what-is-hydrogen-embrittlement). By carefully maintaining iron pipelines using a [Pipeline Inspection Gauge (PIG)](https://en.wikipedia.org/wiki/Pigging), we can effectively determine the best way to avoid damage due to hydrogen embrittlement.

The NDT image analysis software aims to correctly distinguish between images containing microcracks and micro pits versus images containing no defects for use within a PIG.

> [!NOTE]
> The current version of this project is a work-in-progress. Naive solutions have been implemented mainly in Python. Future implementations may work on developing the same or similar methods in C++.

## Highlights

Our most notable accomplishment is discovering a promising technique based on a matched filter approach documented by [Zhen-liang et al](https://www.nature.com/articles/s41598-025-08280-z). Several variations of this technique are implemented in the [`matched_filter_pyprocessor.py`](scripts/matched_filter_pyprocessor.py) module located within the `scripts/` directory.

The idea is simple: take a "template" of a crack across various orientations and calculate a score corresponding to the likelihood of each pixel representing a part of the crack. Pixels with higher scores "respond" better to the template, which hopefully also means they're more likely to be a crack.

### Technical Details

The image processing pipeline consists of the following steps:

1. Preprocessing: prepare the image for template matching by first applying a blur with a small kernel size. Box and Gaussian blurs were chosen between for preprocessing; we found no major difference between the two, but preprocessing in general did prove more effective over no preprocessing.

2. Matching: this step can be split up into three sub-steps:
   
   a. Generate templates by using a 2D Gaussian curve mapped across a square kernel at different rotations. In our case, we set the size of the kernel based on the scale (sigma variable) of the Gaussian and oriented the template at 12 different angles (15 degrees apart).
   
   b. Calculate the response of each template using cross-correlation (or convolution, no difference since the template is symmetrical).
   
   c. Keep the maximum response out of *all* templates. This value will help us determine which help us distinguish which pixels are part of the crack.

3. Correcting: normalize the floating-point values generated in the previous step to the range [0.0, 1.0]. Some special correctors also perform the following:
   - Apply gamma correction by raising values in the effective range to some power. 
   - Apply sigmoid stretching using a midpoint and a certain steepness.

4. Thresholding: based on the values calculated in the previous step, determine which pixels are considered part of the crack. Two different strategies for thresholding were tested:
   - Keep only values strictly above a certain threshold. This is also known as "binary thresholding".
   - Definitely keep values above a "high" threshold while discarding values below a "low" threshold. Anything in between is only kept if directly adjacent to values above the "high" threshold. This is also known as "hysteresis thresholding".

5. Denoising: now that a binary image is produced, discard small regions of "connected" pixels if they fall below a minimum area. In our implementation, some percentage (depending on the method) of the area of the image is considered. For example, the baseline matched filter approach used 0.2% of the area of the image as the threshold. See [`matched_filter_pyprocessor.py`](scripts/matched_filter_pyprocessor.py) for the exact values used in the other methods.

### Results

Using a variation of matched filter coined the "pyramid matched filter", we were able to detect microcracks under ideal conditions. This method was heavily inspired by [SIFT](https://vincmazet.github.io/bip/detection/sift.html)'s technique for scale invariance. In our implementation, we use a three-level Gaussian pyramid for matching; the idea is to take the max response not only out of all templates, but at different scales of the image. You can check out the comparisons of our results in [`segmentation_metrics_benchmarking.ipynb`](research/segmentation_metrics_benchmarking.ipynb).

> [!IMPORTANT]
> Although our results are promising, this comes with a few caveats.
> 1. Matched filter and pyramid matched filter operate best when lighting produces a stark contrast between the crack and the surface of the metal. In certain conditions, cracks may blend in with the background; this often occurs due to improper alignment of the light source. If the light source shines directly into the crack, it may even render the crack unnoticeable by humans as well. In the development of the PIG, we suspect this might be the biggest area of concern.
> 2. Although pyramid matched filter performs better than matched filter due to detecting cracks at different scales, there still seems to be a range it performs best at. Heuristics is an area we explored during our research; however, it was not the primary concern for our project. Future teams may consider rigorously testing the range of microcracks and under what configuration values pyramid matched filter excels at.
> 3. Machine learning may be a viable field to further investigate crack detection. This comes at the cost of computational efficiency as well as hardware support. We briefly explored the [Segment Anything Model](https://github.com/facebookresearch/segment-anything) ([web version](https://huggingface.co/spaces/Xenova/segment-anything-web)) for microcrack detection; future teams may further investigate if they have the resources and technical knowledge on the subject.  

## Installation Instructions

This project was developed for the Raspberry Pi 5. Development environments within Windows and MacOS can be configured using the [devcontainer](.devcontainer/Dockerfile) directory provided. 

### Building on a Linux Machine

> *Prerequisites*
> - CMake
> - Ninja
> - Clang

```bash
cmake --preset linux-x64-debug
cmake --build --preset linux-x64-debug
```

### Building the Docker Image

> *Prerequisites*
> - Docker

> [!NOTE]
> With certain IDEs and editors, such as VSCode, Visual Studio, and CLion, you can rely on a devcontainer extension for Docker. Note that this project has only been tested with debug symbols via VSCode, so do not expect other IDEs to work seamlessly with the configuration files.

```bash
# run compose from project root (or cd into the .devcontainer/ directory instead)
docker compose -f ./.devcontainer/compose.yml up -d

# drop into the interactive shell to configure and build
# replace <container-name> with the name of the container
# this should be dev-1 if configured using the container tools for VSCode
docker exec -it <container-name> /bin/bash

cmake --preset linux-x64-debug
cmake --build --preset linux-x64-debug
```

## Usage Instructions

### Running the Image Processing Software

```bash
# run the C++ implementation
./bin/Debug/imgproc

# run the Python implementation
./bin/Debug/imgproc -p

# help command
./bin/Debug/imgproc -h
```

### Emulating using Docker

You can run the above commands from the interactive shell (`docker exec -it <container-name> /bin/bash`), or you can run a command through docker:

```bash
# run the C++ implementation within the Docker container
docker exec <container-name> /app/bin/Debug/imgproc

# run the Python implementation within the Docker container
docker exec <container-name> /app/bin/Debug/imgproc

# help command
docker exec <container-name> /app/bin/Debug/imgproc -h
```

### Running the Research Scripts

> *Prerequisites*
> - Python
> - Python dependencies (see [`pyproject.toml`](pyproject.toml))

> [!NOTE]
> Please ensure you have a virtual environment (`.venv/`) set up before running these scripts.

```bash
# run the binary mask, max-median filter, morphological hough transform algorithm
python3 research/binary-close-open-hough-transform.py
```
