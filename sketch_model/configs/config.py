import enum
from dataclasses import dataclass
from typing import Optional

from sketch_model.datasets.dataset import LAYER_CLASS_MAP
from fastclasses_json import dataclass_json, JSONMixin

class Aggregation(enum.Enum):
    CONCAT = 0
    SUM = 1


class PosPattern(enum.Enum):
    """
    Enum class for the possible position patterns.
    """
    ONE = 0
    FOUR = 1
    TWO = 2


class SentenceMethod(enum.Enum):
    SUM = 0
    MAX = 1
    MEAN = 2


@dataclass
class TransformerConfig:
    enc_layers: int = 6
    dim_feedforward: int = 2048
    hidden_dim: int = 256
    dropout: float = 0.1
    nheads: int = 8
    pre_norm: bool = True
    use_mask: bool = True


@dataclass
class DatasetConfig:
    train_index_json: str = '/home/borealin/sketch_transformer_dataset/index_train.json'
    test_index_json: str = '/home/borealin/sketch_transformer_dataset/index_test.json'


@dataclass
class SaveConfig:
    task_name: str = 'sketch_transformer'
    output_dir: str = './work_dir'
    resume: Optional[str] = None


@dataclass
class DeviceConfig:
    device: str = 'cuda'
    num_workers: int = 4


@dataclass
class InitConfig:
    seed: int = 42
    start_epoch: int = 0
    evaluate: bool = False


@dataclass
class LRConfig:
    lr: float = 1e-4
    lr_drop: int = 100


@dataclass
class DefaultConfig:
    batch_size: int = 8
    weight_decay: float = 1e-4
    epochs: int = 300
    clip_max_norm: float = 0.1


@dataclass_json
@dataclass
class SketchModelConfig(
    JSONMixin,
    TransformerConfig,
    DatasetConfig,
    SaveConfig,
    DeviceConfig,
    InitConfig,
    LRConfig,
    DefaultConfig
):
    tokenizer_name: str = 'bert-base-chinese'
    max_name_length: int = 32
    num_classes: int = 4
    max_seq_length: int = 200

    class_types: int = len(LAYER_CLASS_MAP)
    pos_freq: int = 64
    pos_pattern: PosPattern = PosPattern.ONE
    sentence_method: SentenceMethod = SentenceMethod.SUM
    aggregation: Aggregation = Aggregation.CONCAT

    use_image: bool = True
    use_name: bool = True
    use_color: bool = True
    use_class: bool = True

    vocab_size: int = 21128
    pad_token_id: int = 0

    def save(self, path: str):
        open(path, 'w').write(self.to_json())