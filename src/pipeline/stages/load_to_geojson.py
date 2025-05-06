import os
from typing import Any
import geopandas as gpd

from src.pipeline.stage import Stage


class LoadToGeoJSON(Stage):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def accept(self):
        """Check if the file path is valid and if the data is in GeoJSON format."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File path {self.file_path} does not exist.")
        if not self.file_path.endswith('.geojson'):
            raise ValueError("File path must end with .geojson")
        return True

    def run(self, data: Any) -> None:
        """Load GeoJSON data from the specified file path."""
        try:
            gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
            gdf.to_file(os.path.Join(self.file_path, "output.geojson"), driver="GeoJSON")
        except Exception as e:
            print(f"Error loading GeoJSON data: {e}")
