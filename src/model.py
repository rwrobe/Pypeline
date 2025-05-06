import uuid
from typing import Any


class DTO:
    """
    Data transfer object (DTO) for the data pipeline.

    This is the best indicator of expected inputs and outputs for the pipeline and can be invaluable for blackbox
    testing.

    Attributes:
        run_id (uuid.UUID): Unique identifier for the pipeline run.
        raw_data (Any): Raw data pulled from the extractor.
        load_data (Any): Data to be loaded into the data sink.
    """
    def __init__(self,
                 uuid: uuid.UUID,
                 extract_data: Any,
                 load_data: Any = None,
                 ):
        """
        :param uuid: Each pipeline run has a unique identifier, helps with tracking and debugging.
        :param extract_data: Raw data pulled from the extractor.
        :param load_data: Data to be loaded into the data sink.
        """
        self.run_id = uuid
        self.raw_data = extract_data

class SkipStageError(Exception):
    """
    Exception to skip the current stage run.
    """
    def __init__(self, message: str):
        super().__init__(message)

class SkipPipelineError(Exception):
    """
    Exception to skip the current pipeline run.
    """
    def __init__(self, message: str):
        super().__init__(message)