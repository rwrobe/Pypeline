import os
from typing import Any, Union
import geopandas as gpd

from src.model import DTO, SkipStageError, SkipPipelineError
from src.pipeline.stage import Stage


class LoadToGeoJSON(Stage):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def accept(self, dto: DTO = None) -> Union[None, SkipStageError, SkipPipelineError]:
        """Check if the file path is valid and if the data is in GeoJSON format."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File path {self.file_path} does not exist.")
        if not self.file_path.endswith('.geojson'):
            raise ValueError("File path must end with .geojson")
        return True

    def run(self, dto: DTO) -> DTO:
        """Load GeoJSON data from the specified file path."""
        try:
            gdf = gpd.GeoDataFrame(dto.processed_data, crs="EPSG:4326")
            gdf.to_file(os.path.join(self.file_path, "output.geojson"), driver="GeoJSON")
            return dto
        except Exception as e:
            print(f"Error loading GeoJSON data: {e}")
            return dto
