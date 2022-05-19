from transformers import PreTrainedTokenizer

from .dataset import *
from ..configs import SketchModelConfig


def build_dataset(
        config: SketchModelConfig,
        index_path: str,
        tokenizer: PreTrainedTokenizer,
) -> SketchDataset:
    return SketchDataset(config, index_path, tokenizer)
