import numpy as np
from .abstract_pyprocessor import PyProcessor
from abc import ABC
from . import matched_filter_strategy as strat


class BaseMatchedFilter(PyProcessor, ABC, register=False):
    def __init__(
        self,
        preprocessor: strat.PreProcessor,
        matcher: strat.Matcher,
        corrector: strat.Corrector,
        thresholder: strat.Thresholder,
        denoiser: strat.Denoiser,
    ) -> None:
        self.preprocessor = preprocessor if preprocessor else strat.BoxBlur()
        self.matcher = matcher if matcher else strat.GaussianMatcher()
        self.corrector = corrector if corrector else strat.Normalizer()
        self.thresholder = thresholder if thresholder else strat.BinaryThresholder()
        self.denoiser = denoiser if denoiser else strat.ConnectedComponentDenoiser()

    def execute(self, image: np.ndarray) -> None:
        processed = self.preprocessor.process(image)
        filtered = self.matcher.filter(processed)
        corrected = self.corrector.correct(filtered)
        thresholded = self.thresholder.apply(corrected)
        image[:] = self.denoiser.denoise(thresholded)

    @staticmethod
    def builder():
        return MatchedFilterBuilder()


class MatchedFilterBuilder:
    def __init__(self) -> None:
        self.components: strat.PipelineComponents = {
            "preprocessor": strat.BoxBlur(),
            "matcher": strat.GaussianMatcher(),
            "corrector": strat.Normalizer(),
            "thresholder": strat.BinaryThresholder(),
            "denoiser": strat.ConnectedComponentDenoiser(),
        }

    def preprocessor(self, strategy: strat.PreProcessor):
        self.components["preprocessor"] = strategy
        return self

    def matcher(self, strategy: strat.Matcher):
        self.components["matcher"] = strategy
        return self

    def thresholder(self, strategy: strat.Thresholder):
        self.components["thresholder"] = strategy
        return self

    def denoiser(self, strategy: strat.Denoiser):
        self.components["denoiser"] = strategy
        return self

    def corrector(self, strategy: strat.Corrector):
        self.components["corrector"] = strategy
        return self


class MatchedFilterPyProcessor(BaseMatchedFilter, key="mfccd"):
    def __init__(self) -> None:
        components: strat.PipelineComponents = BaseMatchedFilter.builder().components
        super().__init__(**components)

    def get_name(self) -> str:
        return "Matched Filter Python image processor"


class SmallMFPyProcessor(BaseMatchedFilter, key="smf"):
    def __init__(self) -> None:
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
        return "Small Matched Filter Python image processor"

class MFGammaPyProcessor(BaseMatchedFilter, key="mfgamma"):
    def __init__(self) -> None:
        components: strat.PipelineComponents = (
            BaseMatchedFilter.builder()
            .corrector(strat.GammaCorrector())
            .thresholder(strat.BinaryThresholder(threshold=0.25))
            .components
        )
        super().__init__(**components)

    def get_name(self) -> str:
        return "Matched Filter (Gamma Corrected) Python image processor"


class MFSigmoidPyProcessor(BaseMatchedFilter, key="mfsigmoid"):
    def __init__(self) -> None:
        components: strat.PipelineComponents = (
            BaseMatchedFilter.builder().corrector(strat.SigmoidStretcher()).components
        )
        super().__init__(**components)

    def get_name(self) -> str:
        return "Matched Filter (Sigmoid Stretched) Python image processor"


class MFHysteresisPyProcessor(BaseMatchedFilter, key="mfhyst"):
    def __init__(self) -> None:
        components: strat.PipelineComponents = (
            BaseMatchedFilter.builder()
            .thresholder(strat.HysteresisThresholder(low=0.4, high=0.7))
            .components
        )
        super().__init__(**components)

    def get_name(self) -> str:
        return "Matched Filter Hysteresis Python image processor"


class MFHystGammaPyProcessor(BaseMatchedFilter, key="mfhgamma"):
    def __init__(self) -> None:
        components: strat.PipelineComponents = (
            BaseMatchedFilter.builder()
            .corrector(strat.GammaCorrector())
            .thresholder(strat.HysteresisThresholder(low=0.4, high=0.7))
            .components
        )
        super().__init__(**components)

    def get_name(self) -> str:
        return "Matched Filter Hysteresis (Gamma Corrected) Python image processor"


class MFHystSigmoidPyProcessor(BaseMatchedFilter, key="mfhsigmoid"):
    def __init__(self) -> None:
        components: strat.PipelineComponents = (
            BaseMatchedFilter.builder()
            .corrector(strat.SigmoidStretcher())
            .thresholder(strat.HysteresisThresholder(low=0.4, high=0.7))
            .components
        )
        super().__init__(**components)

    def get_name(self) -> str:
        return "Matched Filter Hysteresis (Sigmoid Stretched) Python image processor"
