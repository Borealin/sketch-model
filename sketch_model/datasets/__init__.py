from .dataset import *


def build_dataset(index_path, tokenizer) -> SketchDataset:
    return SketchDataset(index_path, tokenizer)
