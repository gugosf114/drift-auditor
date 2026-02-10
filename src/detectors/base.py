# src/detectors/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseDetector(ABC):
    """
    Abstract base class for drift detectors.
    Enforces a consistent interface for the audit pipeline.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def detect(self, transcript: List[Dict]) -> List[Dict]:
        """
        Analyzes a transcript and returns a list of detected drift events.
        """
        pass
