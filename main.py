import uuid

import folium
import tensorflow as tf
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import random

from src.model import DTO
from src.pipeline.stages.apply_keras_sequential import ApplyKerasSequential, KerasConfig
from src.pipeline.stages.extract_from_tensorflow import ExtractFromTensorFlow, DatasetSplit
from src.pipeline.pipeline import Pipeline
from src.pipeline.stages.split_tf_dataset import SplitTFDataset, SplitConfig


def main():
    # Begin by defining the DTO we will hydrate in our pipeline.
    dto = DTO(
        uuid=uuid.uuid4(),
    )

    # Define our base model. Could be part of the DTO, but we curry it in, so not necessary.
    # Using MobileNetV2 for transfer learning because it's small and pretty good at classification. I could use
    # something like ResNet50, but that would be overkill for this task, or VGG16 which would be underkill. Got here
    # through experimentation.
    base_model = tf.keras.applications.MobileNetV2(input_shape=(64, 64, 3),
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False

    # Define an ETL pipeline for image classification with EuroSAT dataset
    es_pipe = Pipeline(
        stages=[
            ExtractFromTensorFlow(name="eurosat/rgb", split=DatasetSplit.TRAIN, with_info=True, as_supervised=True),
            # Split the dataset into training and validation sets
            SplitTFDataset(
                config=SplitConfig(
                    batch=32,
                    count=20000,
                    shuffle=1024,
                    size=20000,
                    train_ratio=0.8,
                    valid_ratio=0.2
                )
            ),
            # Train the model with Keras
            ApplyKerasSequential(
                config=KerasConfig(
                    epochs=5,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'],
                    optimizer='adam'
                ),
                layers=[
                    tf.keras.layers.Rescaling(1. / 255),
                    tf.keras.layers.Resizing(64, 64),
                    base_model,
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dense(len(dto.class_names), activation='softmax')
                ]
            ),
        ]
    )

    # Run the pipeline.
    try:
        dto = es_pipe.run(dto)
    except Exception as e:
        print(f"Pipeline failed: {e}")
        return


if __name__ == "__main__":
    main()
