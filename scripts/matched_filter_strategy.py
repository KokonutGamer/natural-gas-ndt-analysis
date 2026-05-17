from abc import abstractmethod
from . import correction
import cv2
import numpy as np
from skimage.filters import apply_hysteresis_threshold
from typing import Protocol, TypedDict


class PreProcessor(Protocol):
    """
    Protocol defining the interface for image pre-processing strategies.
    """

    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Applies a pre-processing operation to the image.

        Args:
            image (np.ndarray): The input OpenCV image matrix.

        Returns:
            np.ndarray: The pre-processed image matrix.
        """
        raise NotImplementedError


class Matcher(Protocol):
    """
    Protocol defining the interface for matched filtering strategies.
    """

    @abstractmethod
    def filter(self, image: np.ndarray) -> np.ndarray:
        """
        Applies a filtering/matching operation to highlight specific features.

        Args:
            image (np.ndarray): The pre-processed input image matrix.

        Returns:
            np.ndarray: The filtered image matrix containing maximum responses.
        """
        raise NotImplementedError


class Corrector(Protocol):
    """
    Protocol defining the interface for image correction and normalization strategies.
    """

    @abstractmethod
    def correct(self, image: np.ndarray) -> np.ndarray:
        """
        Applies mathematical corrections or normalization to the filtered image.

        Args:
            image (np.ndarray): The filtered image matrix.

        Returns:
            np.ndarray: The corrected image matrix.
        """
        raise NotImplementedError


class Thresholder(Protocol):
    """
    Protocol defining the interface for image thresholding strategies.
    """

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Applies a thresholding operation to convert the image into a mask.

        Args:
            image (np.ndarray): The corrected input image matrix.

        Returns:
            np.ndarray: A binary image mask representing the thresholded features.
        """
        raise NotImplementedError


class Denoiser(Protocol):
    """
    Protocol defining the interface for image denoising strategies.
    """

    @abstractmethod
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Removes noise and artifacts from the image mask.

        Args:
            image (np.ndarray): The thresholded binary image matrix.

        Returns:
            np.ndarray: The cleaned binary image matrix.
        """
        raise NotImplementedError


class BoxBlur(PreProcessor):
    """
    Pre-processing strategy that applies a simple box blur.
    """

    def __init__(self, ksize: int = 11) -> None:
        """
        Initializes the BoxBlur strategy.

        Args:
            ksize (int, optional): The kernel size for the blur. Defaults to 11.
        """
        self._ksize = ksize

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Blurs the input image using a normalized box filter.

        Args:
            image (np.ndarray): The input image matrix.

        Returns:
            np.ndarray: The blurred image matrix as float32.
        """
        return cv2.blur(image, (self._ksize, self._ksize)).astype(np.float32)


class GaussianBlur(PreProcessor):
    """
    Pre-processing strategy that applies a Gaussian blur.
    """

    def __init__(self, ksize: int = 11, sigma: float = 3.0) -> None:
        """
        Initializes the GaussianBlur strategy.

        Args:
            ksize (int, optional): The kernel size for the blur. Defaults to 11.
            sigma (float, optional): Gaussian kernel standard deviation. Defaults to
                3.0.
        """
        self._ksize = ksize
        self._sigma = sigma

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Blurs the input image using a Gaussian filter.

        Args:
            image (np.ndarray): The input image matrix.

        Returns:
            np.ndarray: The Gaussian blurred image matrix as float32.
        """
        return cv2.GaussianBlur(image, (self._ksize, self._ksize), self._sigma).astype(
            np.float32
        )


class GaussianMatcher(Matcher):
    """
    Matching strategy using a bank of rotated, zero-mean Gaussian kernels.
    """

    def __init__(self, sigma: float = 44.0, L: int = 54, angle_step: int = 15) -> None:
        """
        Initializes the GaussianMatcher and generates the filter templates.

        Args:
            sigma (float, optional): Standard deviation of the Gaussian trough. Defaults
                to 44.0.
            L (int, optional): Length of the neighborhood in the y-direction. Defaults
                to 54.
            angle_step (int, optional): Angle increment in degrees for kernel rotations.
                Defaults to 15.
        """
        self.templates = []

        h_x = int(np.ceil(3 * sigma))
        h_y = int(np.ceil(L / 2))
        size = max(h_x, h_y) + 2

        y, x = np.mgrid[-size : size + 1, -size : size + 1]

        for angle in range(0, 180, angle_step):
            theta = np.deg2rad(angle)

            x_rot = x * np.cos(theta) - y * np.sin(theta)
            y_rot = x * np.sin(theta) + y * np.cos(theta)

            mask = (np.abs(x_rot) <= 3 * sigma) & (np.abs(y_rot) <= L / 2)

            kernel = np.zeros_like(x, dtype=np.float32)
            kernel[mask] = -np.exp(-(x_rot[mask] ** 2) / (2 * sigma**2))

            mean_val = np.mean(kernel[mask])
            kernel[mask] = kernel[mask] - mean_val

            self.templates.append(kernel)

    def filter(self, image: np.ndarray) -> np.ndarray:
        """
        Filters the image against all generated templates and computes the maximum
            response.

        Args:
            image (np.ndarray): The pre-processed input image matrix.

        Returns:
            np.ndarray: The maximum response matrix across all template convolutions.
        """
        max_response = np.zeros_like(image)

        for kernel in self.templates:
            response = cv2.filter2D(image, cv2.CV_32F, kernel)
            max_response = np.maximum(max_response, response)

        return max_response


class PyramidGaussianMatcher(GaussianMatcher):
    """
    Matching strategy utilizing an image pyramid combined with Gaussian matching.
    """

    def __init__(
        self, sigma: float = 8.0, L: int = 48, angle_step: int = 15, levels: int = 3
    ) -> None:
        """
        Initializes the PyramidGaussianMatcher.

        Args:
            sigma (float, optional): Gaussian standard deviation for templates. Defaults
                to 8.0.
            L (int, optional): Neighborhood length. Defaults to 48.
            angle_step (int, optional): Degrees between template rotations. Defaults to
                15.
            levels (int, optional): Number of pyramid down-sampling levels. Defaults to
                3.
        """
        self._levels = levels
        super().__init__(sigma, L, angle_step)

    def filter(self, image: np.ndarray) -> np.ndarray:
        """
        Performs Gaussian matching across multiple scales using a Gaussian pyramid.

        Args:
            image (np.ndarray): The pre-processed input image matrix.

        Returns:
            np.ndarray: The aggregated maximum response across all pyramid levels.
        """
        shape = image.shape
        global_response = np.zeros(shape, dtype=np.float32)
        curr = image

        for level in range(self._levels):
            response = super().filter(curr)

            if level > 0:
                upsampled = cv2.resize(
                    response, shape[::-1], interpolation=cv2.INTER_LINEAR
                )
            else:
                upsampled = response

            global_response = np.maximum(global_response, upsampled)
            curr = cv2.pyrDown(curr)

        return global_response


class Normalizer(Corrector):
    """
    Correction strategy that linearly maps image values to the [0.0, 1.0] range.
    """

    def correct(self, image: np.ndarray) -> np.ndarray:
        """
        Normalizes the image intensity using min-max scaling.

        Args:
            image (np.ndarray): The filtered input image matrix.

        Returns:
            np.ndarray: The min-max normalized image matrix.
        """
        normalized_response = np.zeros_like(image)
        cv2.normalize(image, normalized_response, 0.0, 1.0, cv2.NORM_MINMAX)
        return normalized_response


class GammaCorrector(Corrector):
    """
    Correction strategy that applies gamma correction followed by normalization.
    """

    def __init__(self, gamma: float = 2.0) -> None:
        """
        Initializes the GammaCorrector.

        Args:
            gamma (float, optional): The gamma exponent parameter. Defaults to 2.0.
        """
        self._gamma = gamma

    def correct(self, image: np.ndarray) -> np.ndarray:
        """
        Applies a non-linear gamma curve to correct image intensities.

        Args:
            image (np.ndarray): The filtered input image matrix.

        Returns:
            np.ndarray: The gamma-corrected and normalized image matrix.
        """
        gamma_corrected = correction.apply_gamma_correction(image, gamma=self._gamma)
        cv2.normalize(gamma_corrected, gamma_corrected, 0.0, 1.0, cv2.NORM_MINMAX)
        return gamma_corrected


class SigmoidStretcher(Corrector):
    """
    Correction strategy that applies a non-linear sigmoid stretch to the image.
    """

    def __init__(self, percent: float = 0.5, steepness: float = 10.0) -> None:
        """
        Initializes the SigmoidStretcher.

        Args:
            percent (float, optional): Multiplier for calculating the midpoint. Defaults
                to 0.5.
            steepness (float, optional): The steepness/slope of the sigmoid curve.
                Defaults to 10.0.
        """
        self._percent = percent
        self._steepness = steepness

    def correct(self, image: np.ndarray) -> np.ndarray:
        """
        Applies an S-curve transform to increase contrast around a calculated midpoint.

        Args:
            image (np.ndarray): The filtered input image matrix.

        Returns:
            np.ndarray: The sigmoid-stretched and normalized image matrix.
        """
        midpoint = (np.max(image) + np.min(image)) * self._percent
        sigmoid_corrected = correction.apply_sigmoid_stretch(
            image, midpoint, steepness=self._steepness
        )
        cv2.normalize(sigmoid_corrected, sigmoid_corrected, 0.0, 1.0, cv2.NORM_MINMAX)
        return sigmoid_corrected


class BinaryThresholder(Thresholder):
    """
    Thresholding strategy utilizing basic global thresholding.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """
        Initializes the BinaryThresholder.

        Args:
            threshold (float, optional): The fixed threshold limit. Defaults to 0.5.
        """
        self._threshold = threshold

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Converts the image to a binary mask based on a fixed scalar threshold.

        Args:
            image (np.ndarray): The corrected input image matrix.

        Returns:
            np.ndarray: The 8-bit unsigned integer mask (0 or 255).
        """
        _, binary = cv2.threshold(image, self._threshold, 1.0, cv2.THRESH_BINARY)
        mask = (binary * 255).astype(np.uint8)
        return mask


class HysteresisThresholder(Thresholder):
    """
    Thresholding strategy utilizing two thresholds to maintain component connectivity.
    """

    def __init__(self, low: float = 0.45, high: float = 0.7) -> None:
        """
        Initializes the HysteresisThresholder.

        Args:
            low (float, optional): The lower threshold boundary. Defaults to 0.45.
            high (float, optional): The upper threshold boundary. Defaults to 0.7.
        """
        self._low = low
        self._high = high

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Converts the image to a binary mask using hysteresis thresholding.

        Args:
            image (np.ndarray): The corrected input image matrix.

        Returns:
            np.ndarray: The 8-bit unsigned integer mask (0 or 255).
        """
        binary = apply_hysteresis_threshold(image, low=self._low, high=self._high)
        mask = (binary * 255).astype(np.uint8)
        return mask


class ConnectedComponentDenoiser(Denoiser):
    """
    Denoising strategy that removes small isolated blobs by analyzing connected
        components.
    """

    def __init__(self, percent: float = 0.002) -> None:
        """
        Initializes the ConnectedComponentDenoiser.

        Args:
            percent (float, optional): Minimum size percentage (relative to total image
                size) a component must be to remain. Defaults to 0.002.
        """
        self._percent = percent

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Removes connected components smaller than a dynamically calculated minimum size.

        Args:
            image (np.ndarray): The binary image matrix.

        Returns:
            np.ndarray: The cleaned binary image matrix.
        """
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
    """
    Type definition dictating the shape of matched filter pipeline dependencies.
    """

    preprocessor: PreProcessor
    matcher: Matcher
    thresholder: Thresholder
    denoiser: Denoiser
    corrector: Corrector
