import pytest
from unittest.mock import patch, MagicMock
import tensorflow as tf

from src.model import DTO, SplitEnum, SkipPipelineError
from src.pipeline.stages.apply_keras_sequential import ApplyKerasSequential, KerasConfig


class TestApplyKerasSequential:
    
    def test_accept_with_valid_dto(self, dto_with_keras_inputs):
        """Test that accept returns None when the DTO has all required components."""
        config = KerasConfig()
        layers = [tf.keras.layers.Dense(10), tf.keras.layers.Dense(3)]
        stage = ApplyKerasSequential(config, layers)
        
        result = stage.accept(dto_with_keras_inputs)
        assert result is None
    
    def test_accept_without_keras_inputs(self, dto_with_split_data):
        """Test that accept raises SkipPipelineError when keras_inputs is missing."""
        config = KerasConfig()
        layers = [tf.keras.layers.Dense(10)]
        stage = ApplyKerasSequential(config, layers)
        
        with pytest.raises(SkipPipelineError, match="No Keras inputs available"):
            stage.accept(dto_with_split_data)
    
    def test_accept_without_split_data(self, dummy_dto):
        """Test that accept raises SkipPipelineError when split_data is missing."""
        dummy_dto.keras_inputs = tf.keras.Input(shape=(4,))
        
        config = KerasConfig()
        layers = [tf.keras.layers.Dense(10)]
        stage = ApplyKerasSequential(config, layers)
        
        with pytest.raises(SkipPipelineError, match="No data split available"):
            stage.accept(dummy_dto)
    
    def test_accept_with_non_tf_dataset_train_data(self, dto_with_keras_inputs):
        """Test that accept raises SkipPipelineError when train data is not a TF Dataset."""
        # Replace train data with something that's not a TF Dataset
        dto_with_keras_inputs.split_data[SplitEnum.TRAIN.value] = [1, 2, 3, 4, 5]
        
        config = KerasConfig()
        layers = [tf.keras.layers.Dense(10)]
        stage = ApplyKerasSequential(config, layers)
        
        with pytest.raises(SkipPipelineError, match="Training data must be a TensorFlow Dataset"):
            stage.accept(dto_with_keras_inputs)
    
    @patch.object(tf.keras.Sequential, 'fit')
    @patch.object(tf.keras.Sequential, 'compile')
    def test_run_creates_and_trains_model(self, mock_compile, mock_fit, dto_with_keras_inputs):
        """Test that run creates, compiles, and trains a Keras model."""
        # Setup
        config = KerasConfig(
            epochs=3,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
            optimizer='adam'
        )
        layers = [
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ]
        stage = ApplyKerasSequential(config, layers)
        
        # Run the stage
        result_dto = stage.run(dto_with_keras_inputs)
        
        # Assert compile was called with the right parameters
        mock_compile.assert_called_once_with(
            optimizer=config.optimizer,
            loss=config.loss,
            metrics=config.metrics
        )
        
        # Assert fit was called with the right parameters
        mock_fit.assert_called_once_with(
            dto_with_keras_inputs.split_data.get(SplitEnum.TRAIN.value),
            validation_data=dto_with_keras_inputs.split_data.get(SplitEnum.VALIDATION.value),
            epochs=config.epochs
        )