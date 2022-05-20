from transformers import PreTrainedTokenizer

from .dataset import *


def build_dataset(index_path: str, data_folder: str, tokenizer: PreTrainedTokenizer) -> SketchDataset:
    return SketchDataset(index_path, data_folder, tokenizer)
