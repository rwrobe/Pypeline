import os
import json
from typing import Dict, Any, List, Optional
import logging

from src.utils.model_cache import ModelCache


def list_cached_models(cache_dir: str = ".model_cache") -> List[Dict[str, Any]]:
    """
    List all cached models with their metadata.
    
    Args:
        cache_dir: Directory where models are cached
        
    Returns:
        List of dictionaries with model metadata
    """
    cache = ModelCache(cache_dir)
    return [
        {
            "hash": model_hash,
            "created_at": metadata.get("created_at", "unknown"),
            "dataset": metadata.get("dataset_info", {}),
            "config": metadata.get("model_config", {}),
            "path": metadata.get("path", "")
        }
        for model_hash, metadata in cache.metadata.items()
    ]


def print_cache_summary(cache_dir: str = ".model_cache") -> None:
    """
    Print a summary of all cached models.
    
    Args:
        cache_dir: Directory where models are cached
    """
    models = list_cached_models(cache_dir)
    
    if not models:
        print("No cached models found.")
        return
    
    print(f"Found {len(models)} cached models:")
    for i, model in enumerate(models, 1):
        hash_short = model["hash"][:8] + "..." if len(model["hash"]) > 8 else model["hash"]
        print(f"{i}. Model {hash_short}")
        print(f"   Created: {model['created_at']}")
        print(f"   Dataset: {len(model['dataset'])} properties")
        print(f"   Config: {len(model['config'])} parameters")
        print(f"   Path: {model['path']}")
        print()


def delete_model_from_cache(model_hash: str, cache_dir: str = ".model_cache") -> bool:
    """
    Delete a specific model from the cache.
    
    Args:
        model_hash: Hash of the model to delete
        cache_dir: Directory where models are cached
        
    Returns:
        True if the model was deleted, False otherwise
    """
    cache = ModelCache(cache_dir)
    
    if model_hash not in cache.metadata:
        logging.warning(f"Model {model_hash} not found in cache")
        return False
    
    model_path = cache.get_model_path(model_hash)
    if os.path.exists(model_path):
        try:
            import tensorflow as tf
            tf.io.gfile.rmtree(model_path)
        except Exception as e:
            logging.error(f"Failed to delete model directory: {e}")
            return False
    
    # Remove from metadata
    del cache.metadata[model_hash]
    cache._save_metadata()
    
    return True