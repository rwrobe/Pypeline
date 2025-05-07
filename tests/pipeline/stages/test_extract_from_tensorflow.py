import pytest
from unittest.mock import patch, MagicMock
import tensorflow as tf

from src.model import DTO
from src.pipeline.stages.extract_from_tensorflow import ExtractFromTensorFlow, DatasetSplit


@pytest.fixture
def mock_info():
    """Create a mock dataset info object."""
    mock = MagicMock()
    mock.features = {"label": MagicMock()}
    mock.features["label"].names = ["class1", "class2", "class3"]
    return mock


class TestExtractFromTensorFlow:
    
    def test_accept_always_returns_none(self, dummy_dto):
        """Test that accept always returns None (no preconditions)."""
        stage = ExtractFromTensorFlow(name="mnist")
        assert stage.accept(dummy_dto) is None
    
    @patch("tensorflow_datasets.load")
    def test_run_sets_raw_data_and_class_names(self, mock_load, dummy_dto, mock_info):
        """Test that run sets raw_data and class_names in the DTO."""
        # Setup mock return value
        mock_dataset = tf.data.Dataset.range(5)
        mock_load.return_value = (mock_dataset, mock_info)
        
        # Create and run stage
        stage = ExtractFromTensorFlow(name="mnist", with_info=True, as_supervised=True)
        result_dto = stage.run(dummy_dto)
        
        # Assert
        mock_load.assert_called_once_with("mnist", split=DatasetSplit.TRAIN, with_info=True, as_supervised=True)
        assert result_dto.raw_data is mock_dataset
        assert result_dto.class_names == ["class1", "class2", "class3"]
        
    @patch("tensorflow_datasets.load")
    def test_run_with_custom_split(self, mock_load, dummy_dto, mock_info):
        """Test that run uses the specified dataset split."""
        # Setup mock return value
        mock_dataset = tf.data.Dataset.range(5)
        mock_load.return_value = (mock_dataset, mock_info)
        
        # Create and run stage with test split
        stage = ExtractFromTensorFlow(name="mnist", split=DatasetSplit.TEST, with_info=True)
        stage.run(dummy_dto)
        
        # Assert correct split was used
        mock_load.assert_called_once_with("mnist", split=DatasetSplit.TEST, with_info=True, as_supervised=False)