import folium
import tensorflow as tf
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import random

from src.extract.tf import TFLoader, DatasetSplit
from src.pipeline.pipeline import Pipeline


def main():

    # Define an ETL pipeline for image classification with EuroSAT dataset
    es_pipe = Pipeline(
        extractor=TFLoader(name="eurosat/rgb", split=DatasetSplit.TRAIN, with_info=True, as_supervised=True),
        transformers=[
            # Add any transformers you need here
        ],
        loader=GeoJSONLoader()
    )

    # Load EuroSAT dataset from TensorFlow Datasets
    dataset, info = tfds.load("eurosat/rgb", split='train', with_info=True, as_supervised=True)
    class_names = info.features['label'].names

    # Split dataset
    train_ds = dataset.take(20000).shuffle(1024).batch(32).prefetch(tf.data.AUTOTUNE)
    val_ds = dataset.skip(20000).batch(32).prefetch(tf.data.AUTOTUNE)

    # Build a transfer learning model using MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(input_shape=(64, 64, 3),
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Resizing(64, 64),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(train_ds, validation_data=val_ds, epochs=5)

    # Simulate geo-locations and assign class predictions
    geo_data = []

    for image, label in dataset.take(500):
        lat = random.uniform(45.0, 55.0)   # Simulated latitude (Europe range)
        lon = random.uniform(5.0, 15.0)    # Simulated longitude
        geo_data.append({
            "geometry": Point(lon, lat),
            "label": class_names[label.numpy()]
        })

    # Create GeoDataFrame and plot
    gdf = gpd.GeoDataFrame(geo_data, crs="EPSG:4326")
    gdf.plot(column="label", legend=True, figsize=(10, 6))
    plt.title("Simulated Geo-located EuroSAT Predictions")
    plt.show()

    # Optional: export to GeoJSON
    gdf.to_file("output.geojson", driver="GeoJSON")

    # Load the GeoJSON file created earlier
    gdf = gpd.read_file("output.geojson")

    # Create a base map centered at an arbitrary point (e.g., Europe)
    m = folium.Map(location=[50, 10], zoom_start=5)  # You can change coordinates

    # Plot each point in the GeoDataFrame
    for _, row in gdf.iterrows():
        lat, lon = row['geometry'].y, row['geometry'].x
        folium.Marker(
            location=[lat, lon],
            popup=row['label'],
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(m)

    # Save the map as an HTML file
    m.save("satellite_map.html")


if __name__ == "__main__":
    main()