import numpy as np
from abc import ABC, abstractmethod


class PyProcessor(ABC):
    """
    Abstract base class for Python-based image processing algorithms.

    Provides a framework for registering and dispatching different image
    processing routines dynamically. Subclasses must implement the `execute`
    and `get_name` methods.
    """

    _registry = {}
    """
    dict: Internal registry mapping string keys to concrete subclass types.
    Populated automatically via the `__init_subclass__` hook.
    """

    def __init_subclass__(cls, key=None, **kwargs) -> None:
        """
        Hook that is called whenever a subclass is created.

        Registers the subclass in the internal `_registry` dictionary using
        either the provided `key` argument or the class name as a fallback.

        Args:
            key (str, optional): The unique identifier for the subclass.
                                 Defaults to None (uses class name).
        """
        super().__init_subclass__(**kwargs)
        if key is not None:
            cls._registry[key] = cls
        else:
            cls._registry[cls.__name__] = cls

    @abstractmethod
    def execute(self, image: np.ndarray) -> None:
        """
        Executes the image processing algorithm.

        Must be overridden by subclasses to provide specific image
        manipulation logic. Processing should generally be done in-place.

        Args:
            image (np.ndarray): The input OpenCV image matrix to be processed.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Retrieves the descriptive name of the image processor.

        Returns:
            str: The name of the specific processing algorithm.
        """
        return "Abstract Python image processor"

    @classmethod
    def get_registry(cls) -> dict:
        """
        Retrieves a copy of the current processor registry.

        Returns:
            dict: A dictionary of registered processor keys and their class types.
        """
        return dict(cls._registry)

    @classmethod
    def dispatch_execute(cls, key: str, image: np.ndarray) -> None:
        """
        Routes an execution request to the appropriate registered subclass.

        Args:
            key (str): The unique identifier of the target processor.
            image (np.ndarray): The image to be processed.

        Raises:
            ValueError: If the requested key does not exist in the registry.
        """
        if key in cls._registry:
            cls._registry[key].execute(cls._registry[key], image)
        else:
            raise ValueError(f'Processor with key "{key}" does not exist.')

    @classmethod
    def dispatch_name(cls, key: str) -> str:
        """
        Routes a name retrieval request to the appropriate registered subclass.

        Args:
            key (str): The unique identifier of the target processor.

        Returns:
            str: The descriptive name of the requested processor.

        Raises:
            ValueError: If the requested key does not exist in the registry.
        """
        if key in cls._registry:
            return cls._registry[key].get_name(cls._registry[key])
        else:
            raise ValueError(f'Processor with key "{key}" does not exist.')
