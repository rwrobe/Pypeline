from abc import ABC, abstractmethod
from typing import Any

def Transformer(ABC):
    """
    Modifies data in the pipeline.
    """
    @abstractmethod
    def transform(self, data: Any) -> Any:
        """
        Transforms data.
        :param data: Data to be transformed
        :return: Transformed data
        """
        pass