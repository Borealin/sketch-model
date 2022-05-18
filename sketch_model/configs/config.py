from dataclasses import dataclass
from typing import Optional


@dataclass
class SketchModelConfig:
    lr: float = 1e-4
    lr_backbone: float = 1e-5
    batch_size: int = 8
    weight_decay: float = 1e-4
    epochs: int = 300
    lr_drop: int = 200
    clip_max_norm: float = 0.1

    # Model parameters
    frozen_weights: Optional[str] = None
    position_embedding: str = 'sine'

    # * Transformer
    enc_layers: int = 6
    dim_feedforward: int = 2048
    hidden_dim: int = 256
    dropout: float = 0.1
    nheads: int = 8
    num_queries: int = 100
    pre_norm: bool = False

    # Loss
    aux_loss: bool = True
    # * Matcher
    set_cost_class: float = 1
    set_cost_bbox: float = 5
    set_cost_giou: float = 2
    # * Loss coefficients
    mask_loss_coef: float = 1
    dice_loss_coef: float = 1
    bbox_loss_coef: float = 5
    giou_loss_coef: float = 2
    eos_coef: float = 0.1
    # dataset parameters
    train_index_json: str = 'C:\\nozomisharediskc\\sketch_transformer_dataset\\index_train.json'
    test_index_json: str = 'C:\\nozomisharediskc\\sketch_transformer_dataset\\index_test.json'

    output_dir: str = './work_dir'
    device: str = 'cpu'
    seed: int = 42
    resume: Optional[bool] = None
    start_epoch: int = 0
    evaluate: bool = False
    num_workers: int = 2

    tokenizer_name: str = 'bert-base-chinese'
    max_name_length: int = 32
