import numpy as np
from abc import ABC, abstractmethod

"""
TODO document PyProcessor abstract base class
"""
class PyProcessor(ABC):
    """
    TODO document class registry
    """
    registry = {}

    """
    TODO document __init_subclass__ dunder method
    """
    def __init_subclass__(cls, key=None, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if key is not None:
            cls.registry[key] = cls
        else:
            cls.registry[cls.__name__] = cls
    
    """
    TODO document abstract execute method
    """
    @abstractmethod
    def execute(self, image: np.ndarray) -> None:
        pass
    
    """
    TODO document abstract name method
    """
    @abstractmethod
    def get_name(self) -> str:
        return "Abstract Python image processor"

    """
    TODO document get_registry class method
    """
    @classmethod
    def get_registry(cls) -> dict:
        return dict(cls.registry)
