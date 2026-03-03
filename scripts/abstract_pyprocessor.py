import numpy as np
from abc import ABC, abstractmethod

"""
TODO document PyProcessor abstract base class
"""
class PyProcessor(ABC):
    """
    TODO document class registry
    """
    _registry = {}

    """
    TODO document __init_subclass__ dunder method
    """
    def __init_subclass__(cls, key=None, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if key is not None:
            cls._registry[key] = cls
        else:
            cls._registry[cls.__name__] = cls
    
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
        return dict(cls._registry)
    
    """
    TODO document dispatch name class method
    """
    @classmethod
    def dispatch_execute(cls, key: str, image: np.ndarray) -> None:
        if key in cls._registry:
            cls._registry[key].execute(cls._registry[key], image)
        else:
            raise ValueError(f'Processor with key "{key}" does not exist.')
 
    """
    TODO document dispatch name class method
    """
    @classmethod
    def dispatch_name(cls, key: str) -> str:
        if key in cls._registry:
            return cls._registry[key].get_name(cls._registry[key])
        else:
            raise ValueError(f'Processor with key "{key}" does not exist.')