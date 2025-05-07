import pytest
import logging
from unittest.mock import MagicMock, patch

from src.model import DTO, SkipStageError, SkipPipelineError
from src.pipeline.pipeline import Pipeline, with_logger


class TestPipeline:
    
    def test_init_sets_stages(self):
        """Test that Pipeline.__init__ sets stages."""
        # Create mock stages
        stage1 = MagicMock()
        stage2 = MagicMock()
        
        # Create pipeline
        pipeline = Pipeline([stage1, stage2])
        
        # Assert stages are set
        assert pipeline.stages == [stage1, stage2]
        assert pipeline.logger is None
    
    def test_run_calls_stage_run_for_each_stage(self):
        """Test that Pipeline.run calls run on each stage."""
        # Create mock stages
        stage1 = MagicMock()
        stage2 = MagicMock()
        
        # Configure stage1.run to return the DTO
        stage1.run.side_effect = lambda dto: dto
        stage2.run.side_effect = lambda dto: dto
        
        # Create pipeline and mock DTO
        pipeline = Pipeline([stage1, stage2])
        dto = MagicMock()
        
        # Run the pipeline
        result = pipeline.run(dto)
        
        # Assert run was called on each stage
        stage1.run.assert_called_once_with(dto)
        stage2.run.assert_called_once_with(dto)
        assert result is dto
    
    def test_run_handles_skip_stage_error(self):
        """Test that Pipeline.run handles SkipStageError and continues to the next stage."""
        # Create mock stages
        stage1 = MagicMock()
        stage2 = MagicMock()
        
        # Configure stage1.run to raise SkipStageError
        stage1.run.side_effect = SkipStageError("Skip stage")
        stage2.run.side_effect = lambda dto: dto
        
        # Create pipeline and mock DTO
        pipeline = Pipeline([stage1, stage2])
        pipeline.log = MagicMock()  # Mock logger
        dto = MagicMock()
        
        # Run the pipeline
        result = pipeline.run(dto)
        
        # Assert log was called for the skipped stage
        pipeline.log.assert_called_once()
        
        # Assert the second stage was still run
        stage2.run.assert_called_once_with(dto)
        assert result is dto
    
    def test_run_propagates_skip_pipeline_error(self):
        """Test that Pipeline.run propagates SkipPipelineError."""
        # Create mock stage
        stage1 = MagicMock()
        
        # Configure stage1.run to raise SkipPipelineError
        stage1.run.side_effect = SkipPipelineError("Skip pipeline")
        
        # Create pipeline and mock DTO
        pipeline = Pipeline([stage1])
        pipeline.log = MagicMock()  # Mock logger
        dto = MagicMock()
        
        # Run the pipeline and expect exception
        with pytest.raises(SkipPipelineError, match="Skip pipeline"):
            pipeline.run(dto)
        
        # Assert log was called for the skipped pipeline
        pipeline.log.assert_called_once()
    
    def test_log_with_logger(self):
        """Test that Pipeline.log uses logger when available."""
        # Create mock logger
        mock_logger = MagicMock()
        
        # Create pipeline with logger
        pipeline = Pipeline([])
        pipeline.logger = mock_logger
        
        # Log a message
        pipeline.log("Test message", logging.INFO)
        
        # Assert logger.log was called
        mock_logger.log.assert_called_once_with(logging.INFO, "Test message")
    
    def test_log_without_logger(self, capsys):
        """Test that Pipeline.log prints message when logger not available."""
        # Create pipeline without logger
        pipeline = Pipeline([])
        
        # Log a message
        pipeline.log("Test message")
        
        # Capture stdout and assert message was printed
        captured = capsys.readouterr()
        assert "Test message" in captured.out
    
    def test_with_logger_option(self):
        """Test that with_logger option sets logger on pipeline."""
        # Create mock logger
        mock_logger = MagicMock()
        
        # Create pipeline and apply the option function
        pipeline = Pipeline([])
        with_logger(mock_logger)(pipeline)
        
        # Assert logger was set
        assert pipeline.logger is mock_logger
    
    def test_with_logger_sets_handler_if_none_exist(self):
        """Test that with_logger adds a handler if none exists."""
        # Create logger without handlers
        logger = logging.getLogger("test_logger")
        logger.handlers = []
        
        # Create the option function
        option = with_logger(logger)
        
        # Apply option to pipeline
        pipeline = Pipeline([])
        option(pipeline)
        
        # Assert handler was added
        assert len(logger.handlers) == 1
        
        # Cleanup
        logger.handlers = []