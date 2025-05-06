import os
from typing import Any
import geopandas as gpd


class GeoJSONLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load(self, data: Any) -> None:
        """Load GeoJSON data from the specified file path."""
        try:
            gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
            gdf.to_file(os.path.Join(self.file_path, "output.geojson"), driver="GeoJSON")
        except Exception as e:
            print(f"Error loading GeoJSON data: {e}")
