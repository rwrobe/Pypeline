from typing import Any, List, Dict, TypeVar, Callable, Optional
import logging

from src.model import DTO, SkipPipelineError, SkipStageError
from src.pipeline.stage import Stage

# Functional option pattern in Python!
T = TypeVar("T", bound="Pipeline")

Option = Callable[[T], None]

class Pipeline:
    def __init__(self, dto: DTO, stages: List[Stage], *options:Option):
        self.dto = dto
        self.stages = stages
        self.logger: Optional[logging.Logger] = None

    def run(self) -> None:
        """
        Runs the pipeline.
        :return: None
        """
        # For each stage in the pipline
        for s in self.stages:
            try:
                # Note that we replace the DTO at each pipeline stage. We can use this for playback by persisting the
                # DTO at each stage along with the stage name.
                self.dto = s.run(self.dto)
            except SkipStageError as e:
                self.log(f"Hiccup, skipping stage {s.__class__.__name__}: {e}")
                continue
            except SkipPipelineError as e:
                self.log(f"Show stopper! Skipping pipeline: {e}")
                return


    def log(self, msg: str, lvl: int = logging.INFO) -> None:
        """
        Logs a message.
        :param msg: Message to log.
        :param lvl: Logging level.
        :return: None
        """
        if self.logger is not None:
            self.logger.log(lvl, msg)
        else:
            print(msg)

LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# with_logger could be a kwarg or a decorator, but sometimes you do something because you want to know if you can
def with_logger(logger: logging.Logger) -> Option:
    def option(instance: Pipeline) -> None:
        logger.setLevel(logging.INFO)

        # Add a stdout handler if not already attached
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        instance.logger = logger
    return option