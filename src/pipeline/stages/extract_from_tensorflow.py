import enum
import tensorflow_datasets as tfds


class DatasetSplit(enum.Enum):
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"

class TFLoader:
    def __init__(self, name: str, split: DatasetSplit = DatasetSplit.TRAIN, with_info: bool = False,
                 as_supervised: bool = False):
        self.split = split
        self.name = name
        self.with_info = with_info
        self.as_supervised = as_supervised

    def load(self) -> Any:
        dataset, info = tfds.load(self.name, split=self.split, with_info=self.with_info, as_supervised=self.as_supervised)
