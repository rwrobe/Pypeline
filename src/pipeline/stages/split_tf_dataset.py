from typing import Union

from src.model import SkipPipelineError, SkipStageError, DTO, SplitEnum
import tensorflow as tf

from src.pipeline.stage import Stage


class SplitConfig:
    def __init__(self,
                 batch=32,
                 count=20000,
                 shuffle=1024,
                 size=20000,
                 train_ratio=0.8,
                 valid_ratio=0.2,
                 ):
        """
        :param batch: Batch size for the dataset.
        :param shuffle: Buffer size for shuffling the dataset.
        :param size: Number of samples to take from the dataset.
        :param train_ratio: Ratio of training data.
        :param valid_ratio: Ratio of validation data.
        """
        self.batch = batch
        self.shuffle = shuffle
        self.size = size
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio

    def get(self, key, default=None):
        """
        Get the value of a configuration key.
        :param key: The key to retrieve.
        :param default: Default value if the key is not found.
        :return: The value of the key.
        """
        return getattr(self, key, default)


class SplitTFDataset(Stage):
    def __init__(self, config: SplitConfig):
        """
        :param config: Configuration for the split dataset stage.
        """
        self.config = config

    def accept(self, dto: DTO) -> Union[None, SkipStageError, SkipPipelineError]:
        """
        Check if raw_data is available in the DTO and if the split ratio is valid.
        :param dto:
        :return:
        """

        if dto.raw_data is None:
            raise SkipPipelineError("No raw data available for splitting.")

        if not isinstance(dto.raw_data, tf.data.Dataset):
            raise SkipPipelineError("Raw data must be a TensorFlow Dataset to split with SplitTFDataset.")

        if not (0.0 <= self.config.train_ratio <= 1.0) or not (0.0 <= self.config.valid_ratio <= 1.0):
            raise SkipPipelineError("Split ratio must be between 0 and 1.")

        return None

    def run(self, dto: DTO) -> DTO:
        """
        Split the TensorFlow dataset into training and validation sets.
        :param dto:
        :return:
        """
        dataset = dto.raw_data

        # Config values.
        batch = self.config.get("batch")
        shuffle = self.config.get("shuffle")
        size = self.config.get("size")
        train_ratio = self.config.get("train_ratio")
        valid_ratio = self.config.get("valid_ratio")

        train_ds = (dataset
                    .take(size)
                    .shuffle(shuffle)
                    .batch(batch)
                    .prefetch(tf.data.AUTOTUNE))

        val_ds = (dataset
                  .skip(size)
                  .batch(batch)
                  .prefetch(tf.data.AUTOTUNE)
                  )

        dto.split_data = {
            SplitEnum.TRAIN.value: train_ds,
            SplitEnum.VALIDATION.value: val_ds,
        }

        return dto
