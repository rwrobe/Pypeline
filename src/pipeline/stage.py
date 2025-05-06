from abc import ABC, abstractmethod
from typing import Union

from src.model import DTO, SkipStageError, SkipPipelineError


def Stage(ABC):
    """
    Abstract base class for all stages in the pipeline.

    A stage can extract, transform or load data. For each stage, we instantiate the class and call the `accept` method
    to check if the stage should run. If it should, we call the `run` method to execute the stage.
    """
    @abstractmethod
    def accept(self, dto: DTO) -> Union[SkipStageError, SkipPipelineError]:
        """
        Checks whether pre-conditions are met for the stage to run.
        :return: None
        """
        pass

    @abstractmethod
    def run(self, dto: DTO) -> DTO:
        """
        Runs the stage.
        :param dto: Data transfer object (DTO) for the data pipeline.
        :return: DTO
        """
        pass