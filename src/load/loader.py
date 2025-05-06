from abc import ABC, abstractmethod
from typing import Any

class Loader(ABC):
    """
    Loads data into a data sink. One loader per data sink.
    """
    @abstractmethod
    def load(self, data: Any) -> Any:
        """
        Loads data into the data sink.
        :param data: Data to be loaded into the data sink.
        :return: None
        """
        pass