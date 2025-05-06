from typing import Union, List

import keras
import tensorflow as tf

from src.model import DTO, SkipStageError, SkipPipelineError, SplitEnum
from src.pipeline.stage import Stage


class KerasConfig:
    def __init__(self,
                 epochs: int = 5,
                 loss: str = 'sparse_categorical_crossentropy',
                 metrics: List[str] = ['accuracy'],
                 optimizer: str = 'adam',
                 ):
        """
        Configuration for the Keras model.
        :param epochs: Number of epochs to train the model.
        :param loss: Loss function for the Keras model.
        :param metrics: Metrics for the Keras model.
        :param optimizer: Optimizer for the Keras model.
        """
        self.epochs = epochs
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer

class ApplyKerasSequential(Stage):
    def __init__(self, config: KerasConfig, layers: List[keras.Layer]):
        """
        :param config: Configuration for the Keras model.
        :param layers: List of preprocessing steps to apply.
        """
        self.config = config
        self.layers = layers

    def accept(self, dto: DTO) -> Union[None, SkipStageError, SkipPipelineError]:
        """
        Check if the DTO has a TensorFlow Dataset for training and validation, and we have a valid base model.
        :param dto:
        :return:
        """
        if dto.keras_inputs is None:
            raise SkipPipelineError("No Keras inputs available for preprocessing.")

        if dto.split_data is None or SplitEnum.TRAIN not in dto.split_data:
            raise SkipPipelineError("No data split available for training.")

        if not isinstance(dto.split_data[SplitEnum.TRAIN], tf.data.Dataset):
            raise SkipPipelineError("Training data must be a TensorFlow Dataset to train with TrainWithKeras.")

        return None

    def run(self, dto: DTO) -> DTO:
        model = tf.keras.Sequential(*self.layers)

        model.compile(optimizer=self.config.optimizer,
                      loss=self.config.loss,
                      metrics=self.config.metrics)

        # Train the model
        model.fit(dto.split_data.get(SplitEnum.TRAIN), validation_data=dto.split_data.get(SplitEnum.VALIDATION), epochs=self.config.epochs)

        return dto