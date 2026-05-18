import numpy as np
from .abstract_pyprocessor import PyProcessor
from abc import ABC
from . import matched_filter_strategy as strat


class BaseMatchedFilter(PyProcessor, ABC, register=False):
    """
    Abstract base class providing the skeleton for matched filter processing pipelines.

    Subclasses rely on the Builder pattern to define the exact strategies
    used for each stage of execution.
    """

    def __init__(
        self,
        preprocessor: strat.PreProcessor,
        matcher: strat.Matcher,
        corrector: strat.Corrector,
        thresholder: strat.Thresholder,
        denoiser: strat.Denoiser,
    ) -> None:
        """
        Initializes the matched filter pipeline with specified component strategies.

        Args:
            preprocessor (strat.PreProcessor): The pre-processing strategy.
            matcher (strat.Matcher): The matched filtering strategy.
            corrector (strat.Corrector): The correction/normalization strategy.
            thresholder (strat.Thresholder): The image thresholding strategy.
            denoiser (strat.Denoiser): The noise-removal strategy.
        """
        self.preprocessor = preprocessor if preprocessor else strat.BoxBlur()
        self.matcher = matcher if matcher else strat.GaussianMatcher()
        self.corrector = corrector if corrector else strat.Normalizer()
        self.thresholder = thresholder if thresholder else strat.BinaryThresholder()
        self.denoiser = denoiser if denoiser else strat.ConnectedComponentDenoiser()

    def execute(self, image: np.ndarray) -> None:
        """
        Executes the matched filter pipeline sequentially on the provided image.

        The process updates the provided image in-place.

        Args:
            image (np.ndarray): The input OpenCV image matrix to be processed.
        """
        processed = self.preprocessor.process(image)
        filtered = self.matcher.filter(processed)
        corrected = self.corrector.correct(filtered)
        thresholded = self.thresholder.apply(corrected)
        image[:] = self.denoiser.denoise(thresholded)

    @staticmethod
    def builder():
        """
        Instantiates a builder to aid in constructing customized matched filters.

        Returns:
            MatchedFilterBuilder: A new builder instance pre-loaded with default
                components.
        """
        return MatchedFilterBuilder()


class MatchedFilterBuilder:
    """
    Fluent builder class for assembling pipeline components for BaseMatchedFilter
    implementations.
    """

    def __init__(self) -> None:
        """
        Initializes the builder with a default set of standard pipeline components.
        """
        self.components: strat.PipelineComponents = {
            "preprocessor": strat.BoxBlur(),
            "matcher": strat.GaussianMatcher(),
            "corrector": strat.Normalizer(),
            "thresholder": strat.BinaryThresholder(),
            "denoiser": strat.ConnectedComponentDenoiser(),
        }

    def preprocessor(self, strategy: strat.PreProcessor):
        """
        Sets the pre-processing strategy.

        Args:
            strategy (strat.PreProcessor): The strategy to apply.

        Returns:
            MatchedFilterBuilder: The current builder instance for chaining.
        """
        self.components["preprocessor"] = strategy
        return self

    def matcher(self, strategy: strat.Matcher):
        """
        Sets the matched filter strategy.

        Args:
            strategy (strat.Matcher): The strategy to apply.

        Returns:
            MatchedFilterBuilder: The current builder instance for chaining.
        """
        self.components["matcher"] = strategy
        return self

    def thresholder(self, strategy: strat.Thresholder):
        """
        Sets the thresholding strategy.

        Args:
            strategy (strat.Thresholder): The strategy to apply.

        Returns:
            MatchedFilterBuilder: The current builder instance for chaining.
        """
        self.components["thresholder"] = strategy
        return self

    def denoiser(self, strategy: strat.Denoiser):
        """
        Sets the denoising strategy.

        Args:
            strategy (strat.Denoiser): The strategy to apply.

        Returns:
            MatchedFilterBuilder: The current builder instance for chaining.
        """
        self.components["denoiser"] = strategy
        return self

    def corrector(self, strategy: strat.Corrector):
        """
        Sets the correction strategy.

        Args:
            strategy (strat.Corrector): The strategy to apply.

        Returns:
            MatchedFilterBuilder: The current builder instance for chaining.
        """
        self.components["corrector"] = strategy
        return self


class MatchedFilterPyProcessor(BaseMatchedFilter, key="mfccd"):
    """
    Standard implementation of a Matched Filter image processor utilizing default
    pipeline components.
    """

    def __init__(self) -> None:
        """Initializes processor with default strategies."""
        components: strat.PipelineComponents = BaseMatchedFilter.builder().components
        super().__init__(**components)

    def get_name(self) -> str:
        """Retrieves the descriptive name of the image processor."""
        return "Matched Filter Python image processor"


class SmallMFPyProcessor(BaseMatchedFilter, key="smf"):
    """
    Implementation of a Matched Filter tailored with custom parameters suited for
    smaller images/features.
    """

    def __init__(self) -> None:
        """Initializes processor with customized parameters for smaller features."""
        components: strat.PipelineComponents = (
            BaseMatchedFilter.builder()
            .preprocessor(strat.GaussianBlur())
            .matcher(strat.GaussianMatcher(sigma=4.0, L=24))
            .corrector(strat.SigmoidStretcher(steepness=2.0))
            .thresholder(strat.HysteresisThresholder())
            .denoiser(strat.ConnectedComponentDenoiser(0.01))
            .components
        )
        super().__init__(**components)

    def get_name(self) -> str:
        """Retrieves the descriptive name of the image processor."""
        return "Small Matched Filter Python image processor"


class MFGammaPyProcessor(BaseMatchedFilter, key="mfgamma"):
    """
    Implementation of a Matched Filter that employs Gamma Correction during the
    pipeline.
    """

    def __init__(self) -> None:
        """Initializes processor with a Gamma corrector and custom thresholding."""
        components: strat.PipelineComponents = (
            BaseMatchedFilter.builder()
            .corrector(strat.GammaCorrector())
            .thresholder(strat.BinaryThresholder(threshold=0.25))
            .components
        )
        super().__init__(**components)

    def get_name(self) -> str:
        """Retrieves the descriptive name of the image processor."""
        return "Matched Filter (Gamma Corrected) Python image processor"


class MFSigmoidPyProcessor(BaseMatchedFilter, key="mfsigmoid"):
    """
    Implementation of a Matched Filter that utilizes a Sigmoid stretch correction.
    """

    def __init__(self) -> None:
        """Initializes processor with a Sigmoid Stretcher strategy."""
        components: strat.PipelineComponents = (
            BaseMatchedFilter.builder().corrector(strat.SigmoidStretcher()).components
        )
        super().__init__(**components)

    def get_name(self) -> str:
        """Retrieves the descriptive name of the image processor."""
        return "Matched Filter (Sigmoid Stretched) Python image processor"


class MFHysteresisPyProcessor(BaseMatchedFilter, key="mfhyst"):
    """
    Implementation of a Matched Filter using Hysteresis logic for image thresholding.
    """

    def __init__(self) -> None:
        """Initializes processor with a Hysteresis Thresholder strategy."""
        components: strat.PipelineComponents = (
            BaseMatchedFilter.builder()
            .thresholder(strat.HysteresisThresholder(low=0.4, high=0.7))
            .components
        )
        super().__init__(**components)

    def get_name(self) -> str:
        """Retrieves the descriptive name of the image processor."""
        return "Matched Filter Hysteresis Python image processor"


class MFHystGammaPyProcessor(BaseMatchedFilter, key="mfhgamma"):
    """
    Implementation of a Matched Filter combining Hysteresis thresholding and Gamma
    correction.
    """

    def __init__(self) -> None:
        """
        Initializes processor with both Gamma correction and Hysteresis thresholding.
        """
        components: strat.PipelineComponents = (
            BaseMatchedFilter.builder()
            .corrector(strat.GammaCorrector())
            .thresholder(strat.HysteresisThresholder(low=0.4, high=0.7))
            .components
        )
        super().__init__(**components)

    def get_name(self) -> str:
        """Retrieves the descriptive name of the image processor."""
        return "Matched Filter Hysteresis (Gamma Corrected) Python image processor"


class MFHystSigmoidPyProcessor(BaseMatchedFilter, key="mfhsigmoid"):
    """
    Implementation of a Matched Filter combining Hysteresis thresholding and Sigmoid
    stretching.
    """

    def __init__(self) -> None:
        """
        Initializes processor with both Sigmoid Stretching and Hysteresis thresholding.
        """
        components: strat.PipelineComponents = (
            BaseMatchedFilter.builder()
            .corrector(strat.SigmoidStretcher())
            .thresholder(strat.HysteresisThresholder(low=0.4, high=0.7))
            .components
        )
        super().__init__(**components)

    def get_name(self) -> str:
        """Retrieves the descriptive name of the image processor."""
        return "Matched Filter Hysteresis (Sigmoid Stretched) Python image processor"


class PyramidMFPyProcessor(BaseMatchedFilter, key="pmf"):
    """
    Implementation of a Matched Filter that builds a multi-scale Gaussian Pyramid.
    """

    def __init__(self) -> None:
        """Initializes processor with a Pyramid Gaussian Matcher strategy."""
        components: strat.PipelineComponents = (
            BaseMatchedFilter.builder()
            .matcher(strat.PyramidGaussianMatcher())
            .components
        )
        super().__init__(**components)

    def get_name(self) -> str:
        """Retrieves the descriptive name of the image processor."""
        return "Pyramid Matched Filter Python image processor"
