from typing import Union, List, Dict, Any, Optional
import os

import keras
import tensorflow as tf
import json

from src.model import DTO, SkipStageError, SkipPipelineError, SplitEnum
from src.pipeline.stage import Stage
from src.utils.model_cache import ModelCache


class KerasConfig:
    def __init__(self,
                 epochs: int = 5,
                 loss: str = 'sparse_categorical_crossentropy',
                 metrics: List[str] = ['accuracy'],
                 optimizer: str = 'adam',
                 use_cache: bool = True,
                 ):
        """
        Configuration for the Keras model.
        :param epochs: Number of epochs to train the model.
        :param loss: Loss function for the Keras model.
        :param metrics: Metrics for the Keras model.
        :param optimizer: Optimizer for the Keras model.
        :param use_cache: Whether to use model caching.
        """
        self.epochs = epochs
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.use_cache = use_cache
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to a dictionary for caching"""
        return {
            "epochs": self.epochs,
            "loss": self.loss,
            "metrics": self.metrics,
            "optimizer": self.optimizer
        }


class ApplyKerasSequential(Stage):
    def __init__(self, config: KerasConfig, layers: List[keras.Layer], cache_dir: Optional[str] = None):
        """
        :param config: Configuration for the Keras model.
        :param layers: List of layers for the Sequential model.
        :param cache_dir: Optional directory for model caching.
        """
        self.config = config
        self.layers = layers
        self.model_cache = ModelCache(cache_dir) if cache_dir else ModelCache()

    def get_dataset_info(self, dto: DTO) -> Dict[str, Any]:
        """
        Extract dataset information for caching purposes.
        
        :param dto: Data transfer object.
        :return: Dictionary with dataset information.
        """
        info = {
            "class_count": len(dto.class_names) if dto.class_names else 0,
            "input_shape": None
        }
        
        # Try to determine the input shape from the dataset
        if dto.split_data and SplitEnum.TRAIN.value in dto.split_data:
            train_data = dto.split_data[SplitEnum.TRAIN.value]
            # Get the shape from the first batch
            for x, _ in train_data.take(1):
                if hasattr(x, 'shape'):
                    info["input_shape"] = x.shape[1:].as_list()  # Exclude batch dimension
                break
                
        return info

    def get_layers_config(self) -> List[Dict[str, Any]]:
        """
        Extract layer configuration for caching purposes.
        
        :return: List of layer configurations.
        """
        # Convert layers to a serializable format for caching
        layers_config = []
        for layer in self.layers:
            if hasattr(layer, 'get_config'):
                # For standard Keras layers
                layer_config = {
                    "class_name": layer.__class__.__name__,
                    "config": layer.get_config()
                }
                layers_config.append(layer_config)
            elif hasattr(layer, 'name') and hasattr(layer, 'trainable'):
                # For pretrained models like MobileNetV2
                layer_config = {
                    "class_name": layer.__class__.__name__,
                    "name": layer.name,
                    "trainable": layer.trainable
                }
                if hasattr(layer, 'input_shape'):
                    layer_config["input_shape"] = layer.input_shape
                layers_config.append(layer_config)
            else:
                # Generic fallback
                layers_config.append({
                    "class_name": layer.__class__.__name__,
                })
        return layers_config

    def accept(self, dto: DTO) -> Union[None, SkipStageError, SkipPipelineError]:
        """
        Check if the DTO has a TensorFlow Dataset for training and validation, and we have a valid base model.
        :param dto:
        :return:
        """
        if dto.keras_inputs is None:
            raise SkipPipelineError("No Keras inputs available for preprocessing.")

        if dto.split_data is None or SplitEnum.TRAIN.value not in dto.split_data:
            raise SkipPipelineError("No data split available for training.")

        if not isinstance(dto.split_data[SplitEnum.TRAIN.value], tf.data.Dataset):
            raise SkipPipelineError("Training data must be a TensorFlow Dataset to train with TrainWithKeras.")

        return None

    def run(self, dto: DTO) -> DTO:
        # Extract configuration for caching
        layers_config = self.get_layers_config()
        dataset_info = self.get_dataset_info(dto)
        
        # If caching is enabled, try to load the model from cache
        if self.config.use_cache:
            model_hash = self.model_cache.get_model_hash(
                layers_config, 
                self.config.to_dict(), 
                dataset_info
            )
            cached_model = self.model_cache.get_model(model_hash)
            
            if cached_model is not None:
                print(f"Using cached model (hash: {model_hash[:8]}...)")
                dto.keras_model = cached_model
                return dto
        
        # If no cached model is found or caching is disabled, train a new model
        print("Training new model...")
        model = tf.keras.Sequential(self.layers)

        model.compile(optimizer=self.config.optimizer,
                      loss=self.config.loss,
                      metrics=self.config.metrics)

        # Train the model
        model.fit(
            dto.split_data.get(SplitEnum.TRAIN.value), 
            validation_data=dto.split_data.get(SplitEnum.VALIDATION.value), 
            epochs=self.config.epochs
        )
        
        # Store the model in the DTO
        dto.keras_model = model
        
        # Cache the model if caching is enabled
        if self.config.use_cache:
            self.model_cache.save_model(
                model, 
                layers_config, 
                self.config.to_dict(), 
                dataset_info
            )
            
        return dto