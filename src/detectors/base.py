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
from typing import List, Dict, Any


class BaseDetector(ABC):
    """
    Abstract base class for drift detectors.

    Accepts a config dict (typically loaded from config/taxonomy.yaml)
    and enforces a consistent detect() interface.
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def detect(self, transcript: List[Dict]) -> List[Dict]:
        """
        Analyze a transcript and return a list of detected drift events.

        Args:
            transcript: List of turn dicts with 'role', 'content', 'turn' keys.

        Returns:
            List of drift event dicts (format varies by detector type).
        """
        pass
