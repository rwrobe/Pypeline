import pytest
from unittest.mock import patch, MagicMock
import os

from src.pipeline.stages.load_to_geojson import LoadToGeoJSON


class TestLoadToGeoJSON:
    
    def test_accept_with_valid_path(self, monkeypatch):
        """Test that accept returns True when the file path is valid."""
        # Mock os.path.exists to always return True
        monkeypatch.setattr(os.path, "exists", lambda x: True)
        
        stage = LoadToGeoJSON("/valid/path/file.geojson")
        assert stage.accept() is True
    
    def test_accept_with_nonexistent_path(self, monkeypatch):
        """Test that accept raises FileNotFoundError when the file path doesn't exist."""
        # Mock os.path.exists to always return False
        monkeypatch.setattr(os.path, "exists", lambda x: False)
        
        stage = LoadToGeoJSON("/nonexistent/path/file.geojson")
        with pytest.raises(FileNotFoundError):
            stage.accept()
    
    def test_accept_with_invalid_extension(self, monkeypatch):
        """Test that accept raises ValueError when the file doesn't end with .geojson."""
        # Mock os.path.exists to always return True
        monkeypatch.setattr(os.path, "exists", lambda x: True)
        
        stage = LoadToGeoJSON("/valid/path/file.txt")
        with pytest.raises(ValueError, match="File path must end with .geojson"):
            stage.accept()
    
    @patch("geopandas.GeoDataFrame")
    def test_run_saves_geojson_file(self, mock_geodataframe, monkeypatch):
        """Test that run successfully saves data to a GeoJSON file."""
        # Setup mocks
        mock_gdf = MagicMock()
        mock_geodataframe.return_value = mock_gdf
        
        # Create and run stage
        stage = LoadToGeoJSON("/valid/path")
        
        # Create a DTO with processed_data
        test_dto = MagicMock()
        test_dto.processed_data = {"column1": [1, 2, 3], "geometry": ["POINT(0 0)", "POINT(1 1)", "POINT(2 2)"]}
        
        result = stage.run(test_dto)
        
        # Assert GeoDataFrame was created with correct parameters
        mock_geodataframe.assert_called_once_with(test_dto.processed_data, crs="EPSG:4326")
        
        # Assert to_file was called with correct parameters
        mock_gdf.to_file.assert_called_once_with("/valid/path/output.geojson", driver="GeoJSON")
        
        # Assert that the function returns the input DTO
        assert result is test_dto
    
    @patch("geopandas.GeoDataFrame")
    @patch("builtins.print")
    def test_run_handles_exceptions(self, mock_print, mock_geodataframe, monkeypatch):
        """Test that run handles exceptions appropriately."""
        # Setup mock to raise an exception
        mock_geodataframe.side_effect = Exception("Test error")
        
        # Create and run stage
        stage = LoadToGeoJSON("/valid/path")
        
        # Create a DTO with processed_data
        test_dto = MagicMock()
        test_dto.processed_data = {"column1": [1, 2, 3]}
        
        result = stage.run(test_dto)
        
        # Assert that the error was printed
        mock_print.assert_called_once_with("Error loading GeoJSON data: Test error")
        
        # Assert that the function returns the input DTO
        assert result is test_dto