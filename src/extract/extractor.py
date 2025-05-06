from abc import ABC, abstractmethod
from typing import Any

def Extractor(ABC):
    """
    Extracts data from a data source. One extractor per data source.
    """
    @abstractmethod
    def extract(self) -> Any:
        """
        Extracts data from the data source.
        :return: Data extracted from the data source.
        """
        pass
