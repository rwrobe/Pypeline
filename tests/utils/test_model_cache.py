import os
import tempfile
import pytest
import tensorflow as tf
from unittest.mock import patch, MagicMock

from src.utils.model_cache import ModelCache


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for the model cache."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def simple_model():
    """Create a simple Keras model for testing."""
    inputs = tf.keras.Input(shape=(10,))
    x = tf.keras.layers.Dense(5, activation='relu')(inputs)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


@pytest.fixture
def model_config():
    """Create a model configuration for testing."""
    return {
        "optimizer": "adam",
        "loss": "binary_crossentropy",
        "metrics": ["accuracy"],
        "epochs": 5
    }


@pytest.fixture
def layers_config():
    """Create a layers configuration for testing."""
    return [
        {
            "class_name": "Dense",
            "config": {
                "units": 5,
                "activation": "relu"
            }
        },
        {
            "class_name": "Dense",
            "config": {
                "units": 1,
                "activation": "sigmoid"
            }
        }
    ]


@pytest.fixture
def dataset_info():
    """Create dataset information for testing."""
    return {
        "class_count": 2,
        "input_shape": [10]
    }


class TestModelCache:
    
    def test_init_creates_cache_dir(self, temp_cache_dir):
        """Test that the cache directory is created during initialization."""
        cache_dir = os.path.join(temp_cache_dir, "model_cache")
        ModelCache(cache_dir)
        assert os.path.exists(cache_dir)
    
    def test_get_model_hash_consistency(self, model_config, layers_config, dataset_info):
        """Test that the same configuration always produces the same hash."""
        cache = ModelCache()
        hash1 = cache.get_model_hash(layers_config, model_config, dataset_info)
        hash2 = cache.get_model_hash(layers_config, model_config, dataset_info)
        assert hash1 == hash2
    
    def test_get_model_hash_differentiates_configs(self, model_config, layers_config, dataset_info):
        """Test that different configurations produce different hashes."""
        cache = ModelCache()
        hash1 = cache.get_model_hash(layers_config, model_config, dataset_info)
        
        # Modify the dataset info
        modified_dataset = dataset_info.copy()
        modified_dataset["class_count"] = 3
        hash2 = cache.get_model_hash(layers_config, model_config, modified_dataset)
        
        assert hash1 != hash2
        
        # Modify the model config
        modified_model_config = model_config.copy()
        modified_model_config["epochs"] = 10
        hash3 = cache.get_model_hash(layers_config, modified_model_config, dataset_info)
        
        assert hash1 != hash3
        assert hash2 != hash3
    
    def test_save_and_get_model(self, temp_cache_dir, simple_model, model_config, layers_config, dataset_info):
        """Test saving and retrieving a model from the cache."""
        cache = ModelCache(temp_cache_dir)
        
        # Save the model
        model_hash = cache.save_model(simple_model, layers_config, model_config, dataset_info)
        
        # Verify metadata was updated
        assert model_hash in cache.metadata
        
        # Get the model
        loaded_model = cache.get_model(model_hash)
        assert loaded_model is not None
        
        # Check that the model architecture is the same
        assert len(loaded_model.layers) == len(simple_model.layers)
    
    def test_get_nonexistent_model(self, temp_cache_dir):
        """Test that getting a non-existent model returns None."""
        cache = ModelCache(temp_cache_dir)
        assert cache.get_model("nonexistent_hash") is None
    
    @patch("tensorflow.keras.models.load_model")
    def test_get_model_handles_errors(self, mock_load_model, temp_cache_dir, simple_model, 
                                     model_config, layers_config, dataset_info):
        """Test that errors during model loading are handled gracefully."""
        mock_load_model.side_effect = Exception("Test error")
        
        cache = ModelCache(temp_cache_dir)
        model_hash = cache.save_model(simple_model, layers_config, model_config, dataset_info)
        
        # Try to load the model, which should fail but not crash
        with patch("logging.warning") as mock_warning:
            loaded_model = cache.get_model(model_hash)
            assert loaded_model is None
            mock_warning.assert_called_once()
    
    def test_clear_cache(self, temp_cache_dir, simple_model, model_config, layers_config, dataset_info):
        """Test clearing the cache."""
        cache = ModelCache(temp_cache_dir)
        
        # Save a model
        cache.save_model(simple_model, layers_config, model_config, dataset_info)
        
        # Verify metadata contains the model
        assert len(cache.metadata) > 0
        
        # Clear the cache
        cache.clear_cache()
        
        # Verify metadata is empty
        assert len(cache.metadata) == 0