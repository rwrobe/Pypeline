import pytest
import tensorflow as tf

from src.model import DTO, SplitEnum, SkipPipelineError
from src.pipeline.stages.split_tf_dataset import SplitTFDataset, SplitConfig


class TestSplitTFDataset:
    
    def test_accept_with_valid_dto(self, dto_with_raw_data):
        """Test that accept returns None when the DTO has valid raw_data."""
        config = SplitConfig(train_ratio=0.8, valid_ratio=0.2)
        stage = SplitTFDataset(config)
        
        result = stage.accept(dto_with_raw_data)
        assert result is None
    
    def test_accept_with_missing_raw_data(self, dummy_dto):
        """Test that accept raises SkipPipelineError when raw_data is missing."""
        config = SplitConfig()
        stage = SplitTFDataset(config)
        
        with pytest.raises(SkipPipelineError, match="No raw data available for splitting."):
            stage.accept(dummy_dto)
    
    def test_accept_with_non_tf_dataset(self, dummy_dto):
        """Test that accept raises SkipPipelineError when raw_data is not a TensorFlow Dataset."""
        dummy_dto.raw_data = [1, 2, 3, 4, 5]  # Not a TensorFlow Dataset
        
        config = SplitConfig()
        stage = SplitTFDataset(config)
        
        with pytest.raises(SkipPipelineError, match="Raw data must be a TensorFlow Dataset"):
            stage.accept(dummy_dto)
    
    def test_accept_with_invalid_split_ratio(self, dto_with_raw_data):
        """Test that accept raises SkipPipelineError when split ratio is invalid."""
        config = SplitConfig(train_ratio=1.5, valid_ratio=0.2)  # Invalid train_ratio
        stage = SplitTFDataset(config)
        
        with pytest.raises(SkipPipelineError, match="Split ratio must be between 0 and 1"):
            stage.accept(dto_with_raw_data)
    
    def test_run_creates_train_and_validation_splits(self, dto_with_raw_data):
        """Test that run creates training and validation datasets in the DTO."""
        # Create a simple config with predictable values
        config = SplitConfig(
            batch=2,
            shuffle=10,
            size=5,
            train_ratio=0.8,
            valid_ratio=0.2
        )
        stage = SplitTFDataset(config)
        
        # Run the stage
        result_dto = stage.run(dto_with_raw_data)
        
        # Assert that split_data contains train and validation datasets
        assert SplitEnum.TRAIN.value in result_dto.split_data
        assert SplitEnum.VALIDATION.value in result_dto.split_data
        
        # Check that the datasets have the correct type
        assert isinstance(result_dto.split_data[SplitEnum.TRAIN.value], tf.data.Dataset)
        assert isinstance(result_dto.split_data[SplitEnum.VALIDATION.value], tf.data.Dataset)
        
        # Verify some properties of the created datasets
        train_ds = result_dto.split_data[SplitEnum.TRAIN.value]
        val_ds = result_dto.split_data[SplitEnum.VALIDATION.value]
        
        # Check batch size is applied (this may need adjustments based on actual implementation)
        for batch in train_ds.take(1):
            # Assuming a tuple of (features, labels)
            features, _ = batch
            assert features.shape[0] <= 2  # Batch size should be 2 or less