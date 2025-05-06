import enum
from typing import Any, Union

import tensorflow_datasets as tfds

from src.model import DTO, SkipStageError, SkipPipelineError
from src.pipeline.stage import Stage


class DatasetSplit(enum.Enum):
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"


class ExtractFromTensorFlow(Stage):
    def __init__(self, name: str, split: DatasetSplit = DatasetSplit.TRAIN, with_info: bool = False,
                 as_supervised: bool = False):
        self.split = split
        self.name = name
        self.with_info = with_info
        self.as_supervised = as_supervised

    def accept(self, dto: DTO) -> Union[None, SkipStageError, SkipPipelineError]:
        """
        TFLoader has no preconditions to run.
        :param dto:
        :return:
        """
        return None

    def run(self, dto: DTO) -> DTO:
        dataset, info = tfds.load(self.name, split=self.split, with_info=self.with_info,
                                  as_supervised=self.as_supervised)
        dto.raw_data = dataset
        dto.class_names = info.features['label'].names

        return dto
