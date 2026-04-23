import torch
import torch.nn as nn
import torch.nn.functional as F

"""
A neural network preprocessor replicating the series of dilations and blurs
used in the original FMMorph algorithm.
"""


class PreProcessingModel(nn.Module):
    """
    TODO document constructor
    """

    def __init__(self) -> None:
        super().__init__()
        self.n_filters: int = 4
        self.ksize: int = 3
        self.pad: int = self.ksize // 2

    """
    TODO document forward method
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.n_filters):
            # dilation
            x = F.max_pool2d(x, kernel_size=self.ksize, stride=1, padding=self.pad)

            # blur (average pool instead of median)
            x = F.avg_pool2d(x, kernel_size=self.ksize, stride=1, padding=self.pad)
        return x


"""
A neural network morphology model replicating the series of morphological
openings and closings used in the original FMMorph algorithm.
"""


class MorphologyModel(nn.Module):
    """
    TODO document constructor
    """

    def __init__(self) -> None:
        super().__init__()
        self.n_morphs: int = 4
        self.ksize: int = 5
        self.pad: int = self.ksize // 2

    """
    TODO document erode method
    """

    def erode(self, x: torch.Tensor) -> torch.Tensor:
        return -F.max_pool2d(-x, kernel_size=self.ksize, stride=1, padding=self.pad)

    """
    TODO document dilate method
    """

    def dilate(self, x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(x, kernel_size=self.ksize, stride=1, padding=self.pad)

    """
    TODO document forward method
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.n_morphs):
            # morphological open
            x = self.erode(x)
            x = self.dilate(x)

            # morphological close
            x = self.dilate(x)
            x = self.erode(x)
        return x


"""
TODO document export_models function
"""


def export_models() -> None:
    shape: torch.Tensor = torch.randn(1, 1, 480, 640)

    print("Exporting Pre-processing Model...")
    model_pre = PreProcessingModel()
    model_pre.eval()
    torch.onnx.export(
        model_pre,
        (shape,),
        "preprocess_model.onnx",
        opset_version=18,
        input_names=["input_image"],
        output_names=["processed_image"],
    )

    print("Exporting Morphology Model...")
    model_morph = MorphologyModel()
    model_morph.eval()
    torch.onnx.export(
        model_morph,
        (shape,),
        "morphology_model.onnx",
        opset_version=18,
        input_names=["binary_image"],
        output_names=["final_image"],
    )

    print("Exporting finished.")


if __name__ == "__main__":
    export_models()
