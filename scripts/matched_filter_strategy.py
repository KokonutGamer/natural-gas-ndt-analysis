from abc import abstractmethod
from . import correction
import cv2
import numpy as np
from skimage.filters import apply_hysteresis_threshold
from typing import Protocol, TypedDict


class PreProcessor(Protocol):
    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Matcher(Protocol):
    @abstractmethod
    def filter(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Corrector(Protocol):
    @abstractmethod
    def correct(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Thresholder(Protocol):
    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Denoiser(Protocol):
    @abstractmethod
    def denoise(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class BoxBlur(PreProcessor):
    def __init__(self, ksize: int = 11) -> None:
        self._ksize = ksize

    def process(self, image: np.ndarray) -> np.ndarray:
        return cv2.blur(image, (self._ksize, self._ksize)).astype(np.float32)


class GaussianBlur(PreProcessor):
    def __init__(self, ksize: int = 11, sigma: float = 3.0) -> None:
        self._ksize = ksize
        self._sigma = sigma

    def process(self, image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(image, (self._ksize, self._ksize), self._sigma).astype(
            np.float32
        )


class GaussianMatcher(Matcher):
    def __init__(self, sigma: float = 44.0, L: int = 54, angle_step: int = 15) -> None:
        self.templates = []

        h_x = int(np.ceil(3 * sigma))
        h_y = int(np.ceil(L / 2))
        size = max(h_x, h_y) + 2
        # kernels SHOULD BE SQUARE, so the max size is defined either by 3 * sigma or L
        # / 2 according to the original research paper

        # we use a meshgrid so we can efficiently slice existing coordinates from -size
        # to size + 1; THIS STILL RESULTS IN AN ODD BY ODD MATRIX!
        y, x = np.mgrid[-size : size + 1, -size : size + 1]

        for angle in range(0, 180, angle_step):
            theta = np.deg2rad(angle)

            # notice that the rotations for x and y are swapped
            # this is because we are using rows as y and cols as x
            x_rot = x * np.cos(theta) - y * np.sin(theta)
            y_rot = x * np.sin(theta) + y * np.cos(theta)

            # mask should only contain values within the 3 * sigma range (for x) and L /
            # 2 range (for y)
            mask = (np.abs(x_rot) <= 3 * sigma) & (np.abs(y_rot) <= L / 2)

            kernel = np.zeros_like(x, dtype=np.float32)

            # 1D NEGATIVE Gaussian in x (trough / valley)
            kernel[mask] = -np.exp(-(x_rot[mask] ** 2) / (2 * sigma**2))

            mean_val = np.mean(kernel[mask])
            kernel[mask] = kernel[mask] - mean_val  # IMPORTANT for zero-mean
            # zero-mean Gaussians are Gaussians whose total value adds up to zero
            # this is because mean * number of elements = sum of elements in kernel
            # zero-mean filters are important to maintain gray-value intensity

            self.templates.append(kernel)

    def filter(self, image: np.ndarray) -> np.ndarray:
        max_response = np.zeros_like(image)

        for kernel in self.templates:
            response = cv2.filter2D(image, cv2.CV_32F, kernel)
            max_response = np.maximum(max_response, response)

        return max_response


class Normalizer(Corrector):
    def correct(self, image: np.ndarray) -> np.ndarray:
        normalized_response = np.zeros_like(image)
        cv2.normalize(image, normalized_response, 0.0, 1.0, cv2.NORM_MINMAX)
        return normalized_response


class GammaCorrector(Corrector):
    def __init__(self, gamma: float = 2.0) -> None:
        self._gamma = gamma

    def correct(self, image: np.ndarray) -> np.ndarray:
        gamma_corrected = correction.apply_gamma_correction(image, gamma=self._gamma)
        cv2.normalize(gamma_corrected, gamma_corrected, 0.0, 1.0, cv2.NORM_MINMAX)
        return gamma_corrected


class SigmoidStretcher(Corrector):
    def __init__(self, percent: float = 0.5, steepness: float = 10.0) -> None:
        self._percent = percent
        self._steepness = steepness

    def correct(self, image: np.ndarray) -> np.ndarray:
        midpoint = (np.max(image) + np.min(image)) * self._percent
        sigmoid_corrected = correction.apply_sigmoid_stretch(
            image, midpoint, steepness=self._steepness
        )
        cv2.normalize(sigmoid_corrected, sigmoid_corrected, 0.0, 1.0, cv2.NORM_MINMAX)
        return sigmoid_corrected


class BinaryThresholder(Thresholder):
    def __init__(self, threshold: float = 0.5) -> None:
        self._threshold = threshold

    def apply(self, image: np.ndarray) -> np.ndarray:
        _, binary = cv2.threshold(image, self._threshold, 1.0, cv2.THRESH_BINARY)
        mask = (binary * 255).astype(np.uint8)
        return mask


class HysteresisThresholder(Thresholder):
    def __init__(self, low: float = 0.45, high: float = 0.7) -> None:
        self._low = low
        self._high = high

    def apply(self, image: np.ndarray) -> np.ndarray:
        binary = apply_hysteresis_threshold(image, low=self._low, high=self._high)
        mask = (binary * 255).astype(np.uint8)
        return mask


class ConnectedComponentDenoiser(Denoiser):
    def __init__(self, percent: float = 0.002) -> None:
        self._percent = percent

    def denoise(self, image: np.ndarray) -> np.ndarray:
        min_size = int(self._percent * image.size)
        num_labels, labels = cv2.connectedComponents(image)
        cleaned = np.zeros_like(image)

        for label in range(1, num_labels):
            mask = labels == label
            if np.sum(mask) < min_size:
                continue
            cleaned[mask] = 255

        return cleaned.astype(np.uint8)


class PipelineComponents(TypedDict):
    preprocessor: PreProcessor
    matcher: Matcher
    thresholder: Thresholder
    denoiser: Denoiser
    corrector: Corrector
