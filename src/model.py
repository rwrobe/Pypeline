import uuid
from enum import Enum
from typing import Any, Dict, Optional
import tensorflow as tf
import keras
from tensorflow.python.types.data import DatasetV2


class SplitEnum(Enum):
    """
    Enum for the dataset split.
    """
    TEST = "test"
    TRAIN = "train"
    VALIDATION = "validation"

class DTO:
    """
    Data transfer object (DTO) for the data pipeline.

    This is the best indicator of expected inputs and outputs for the pipeline and can be invaluable for blackbox
    testing.

    Attributes:
        run_id (uuid.UUID): Unique identifier for the pipeline run.
        raw_data (Any): Raw data pulled from the extractor.
        class_names (Dict[str, Any]): The classification names for the dataset.
        split_data (Dict[SplitEnum, DatasetV2]): The split data for the dataset.
        keras_base_model (tf.keras.Model): The base Keras model to be used for training.
        keras_inputs (tf.keras.Input): The inputs for the Keras model.
        processed_data (Any): Data to be loaded into the data sink.
    """

    def __init__(self,
                 uuid: uuid.UUID,
                 raw_data: tf.data.Dataset = None,
                 ):
        """
        :param uuid: Each pipeline run has a unique identifier, helps with tracking and debugging.
        :param raw_data: Raw data pulled from the extractor.
        """
        self.run_id = uuid
        self.raw_data = raw_data
        self.class_names: Optional[Dict[str, Any]] = None
        self.split_data: Optional[Dict[SplitEnum, DatasetV2]] = None
        self.keras_base_model: Optional[keras.Model] = None
        self.keras_inputs: Optional[keras.Input] = None
        self.keras_model: Optional[keras.Model] = None
        self.processed_data = None


class SkipStageError(Exception):
    """
    Exception to skip the current stage run.
    """

    def __init__(self, message: str):
        super().__init__(message)


class SkipPipelineError(Exception):
    """
    Exception to skip the current pipeline run.
    """

    def __init__(self, message: str):
        super().__init__(message)
