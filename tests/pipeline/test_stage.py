import pytest
from abc import ABC

from src.model import DTO
from src.pipeline.stage import Stage


class TestStage:
    
    def test_stage_is_abstract_base_class(self):
        """Test that Stage is an abstract base class."""
        assert issubclass(Stage, ABC)
    
    def test_accept_is_abstract_method(self):
        """Test that accept is an abstract method."""
        with pytest.raises(TypeError):
            Stage().accept(DTO(uuid=None))
    
    def test_run_is_abstract_method(self):
        """Test that run is an abstract method."""
        with pytest.raises(TypeError):
            Stage().run(DTO(uuid=None))