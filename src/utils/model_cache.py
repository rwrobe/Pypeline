import os
import json
import hashlib
import tensorflow as tf
from typing import Dict, Any, Optional, Tuple, List
import logging


class ModelCache:
    def __init__(self, cache_dir: str = ".model_cache"):
        """
        Initialize the model cache.
        
        Args:
            cache_dir: Directory to store cached models
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.metadata_file = os.path.join(cache_dir, "metadata.json")
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata about cached models."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logging.warning(f"Failed to load cache metadata: {e}")
                return {}
        return {}
    
    def _save_metadata(self) -> None:
        """Save metadata about cached models."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f)
        except IOError as e:
            logging.warning(f"Failed to save cache metadata: {e}")
    
    def get_model_hash(self, 
                       layers_config: List[Dict[str, Any]], 
                       model_config: Dict[str, Any], 
                       dataset_info: Dict[str, Any]) -> str:
        """
        Generate a unique hash for the model based on its configuration and dataset.
        
        Args:
            layers_config: Layer configuration details
            model_config: Model compilation and training configuration
            dataset_info: Information about the dataset used for training
            
        Returns:
            A hash string representing the model configuration
        """
        # Combine configs into a single dictionary for hashing
        config_dict = {
            "layers": layers_config,
            "model_config": model_config,
            "dataset_info": dataset_info
        }
        
        # Convert to a stable string representation and hash it
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def get_model_path(self, model_hash: str) -> str:
        """Get the file path for a cached model."""
        return os.path.join(self.cache_dir, model_hash)
    
    def get_model(self, model_hash: str) -> Optional[tf.keras.Model]:
        """
        Load a model from the cache based on its hash.
        
        Args:
            model_hash: The hash string for the model
            
        Returns:
            The loaded model if found, None otherwise
        """
        model_path = self.get_model_path(model_hash)
        if os.path.exists(model_path):
            try:
                return tf.keras.models.load_model(model_path)
            except Exception as e:
                logging.warning(f"Failed to load cached model: {e}")
        return None
    
    def save_model(self, 
                   model: tf.keras.Model, 
                   layers_config: List[Dict[str, Any]],
                   model_config: Dict[str, Any], 
                   dataset_info: Dict[str, Any]) -> str:
        """
        Save a model to the cache.
        
        Args:
            model: The Keras model to save
            layers_config: Layer configuration details
            model_config: Model compilation and training configuration
            dataset_info: Information about the dataset used for training
            
        Returns:
            The hash string for the saved model
        """
        model_hash = self.get_model_hash(layers_config, model_config, dataset_info)
        model_path = self.get_model_path(model_hash)
        
        # Save the model
        model.save(model_path)
        
        # Update metadata
        self.metadata[model_hash] = {
            "layers": layers_config,
            "model_config": model_config,
            "dataset_info": dataset_info,
            "path": model_path,
            "created_at": str(tf.timestamp().numpy())
        }
        
        self._save_metadata()
        return model_hash
    
    def clear_cache(self) -> None:
        """Clear all cached models."""
        for model_hash in list(self.metadata.keys()):
            model_path = self.get_model_path(model_hash)
            if os.path.exists(model_path):
                try:
                    tf.io.gfile.rmtree(model_path)
                except Exception as e:
                    logging.warning(f"Failed to delete cached model {model_hash}: {e}")
        
        self.metadata = {}
        self._save_metadata()