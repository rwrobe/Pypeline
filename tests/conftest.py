import pytest
import uuid
import tensorflow as tf
from src.model import DTO, SplitEnum

@pytest.fixture
def dummy_dto():
    """Create a basic DTO with minimal initialization for testing."""
    return DTO(uuid=uuid.uuid4())

@pytest.fixture
def mock_tf_dataset():
    """Mock TensorFlow dataset"""
    # Create a simple dataset with 10 elements
    ds = tf.data.Dataset.from_tensor_slices(
        (tf.random.normal([10, 4]), tf.random.uniform([10], minval=0, maxval=3, dtype=tf.int32))
    )
    return ds

@pytest.fixture
def dto_with_raw_data(dummy_dto, mock_tf_dataset):
    """DTO with raw_data"""
    dummy_dto.raw_data = mock_tf_dataset
    dummy_dto.class_names = ["class1", "class2", "class3"]
    return dummy_dto

@pytest.fixture
def dto_with_split_data(dto_with_raw_data, mock_tf_dataset):
    """DTO with split_data"""
    train_ds = mock_tf_dataset.batch(2)
    val_ds = mock_tf_dataset.batch(2)
    
    dto_with_raw_data.split_data = {
        SplitEnum.TRAIN.value: train_ds,
        SplitEnum.VALIDATION.value: val_ds
    }
    return dto_with_raw_data

@pytest.fixture
def dto_with_keras_inputs(dto_with_split_data):
    """DTO with keras_inputs"""
    dto_with_split_data.keras_inputs = tf.keras.Input(shape=(4,))
    return dto_with_split_data