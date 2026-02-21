# src/detectors/base.py
"""
Abstract base class for drift detectors.

Current detectors (commission, omission, structural) use module-level
functions for backwards compatibility. This ABC defines the target
interface for future detector implementations that need stateful
configuration (e.g., custom thresholds from taxonomy.yaml).

Usage:
    class CustomDetector(BaseDetector):
        def detect(self, transcript):
            # access self.config for thresholds/weights
            ...
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Type

class BaseDetector(ABC):
    """
    Abstract base class for drift detectors.

    Accepts a config dict (typically loaded from config/taxonomy.yaml)
    and enforces a consistent detect() interface.
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def detect(self, *args, **kwargs) -> List[Any]:
        """
        Analyze a transcript and return a list of detected drift events.
        """
        pass

class DetectorRegistry:
    """Registry for all active detectors."""
    _window_detectors: List[Type[BaseDetector]] = []
    _full_detectors: List[Type[BaseDetector]] = []

    @classmethod
    def register_window_detector(cls, detector_class: Type[BaseDetector]):
        cls._window_detectors.append(detector_class)
        return detector_class

    @classmethod
    def register_full_detector(cls, detector_class: Type[BaseDetector]):
        cls._full_detectors.append(detector_class)
        return detector_class

    @classmethod
    def get_window_detectors(cls, config: Dict[str, Any] | None = None) -> List[BaseDetector]:
        return [detector(config) for detector in cls._window_detectors]

    @classmethod
    def get_full_detectors(cls, config: Dict[str, Any] | None = None) -> List[BaseDetector]:
        return [detector(config) for detector in cls._full_detectors]


