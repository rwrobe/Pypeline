from typing import Any, List, Dict, TypeVar, Callable

from src.extract.extractor import Extractor
from src.transform.transformer import Transformer
from src.load.loader import Loader


# Functional option pattern in Python!
T = TypeVar("T", bound="Pipeline")

Option = Callable[[T], None]

class Pipeline:
    def __init__(self, extractor: Extractor, transformers: List[Transformer], loader: Loader, *options:Option):
        self.extractor = extractor
        self.transformers = transformers
        self.loader = loader

    def run(self) -> None:
        """
        Runs the pipeline.
        :return: None
        """
        # Extract data
        try:
            data = self.extractor.extract()
        except Exception as e:
            # TODO: Need different extraction errors.
            print(f"Error during extraction: {e}")
            return

        # Transform data
        for transformer in self.transformers:
            data = transformer.transform(data)

        # Load data
        try:
            self.loader.load(data)
        except Exception as e:
            log.error(f"Error during loading: {e}")
