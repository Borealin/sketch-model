from typing import Tuple

import torch
from torch import nn
from torch.nn.modules.loss import _Loss as Loss

from sketch_model.configs import SketchModelConfig
from sketch_model.layers import LayerStructureEmbedding, build_transformer, SketchTransformer
from sketch_model.utils import NestedTensor


class SketchLayerClassifierModel(nn.Module):
    def __init__(
            self,
            config: SketchModelConfig,
            transformer: SketchTransformer,
    ):
        super().__init__()
        self.config = config
        self.transformer: SketchTransformer = transformer
        self.hidden_dim = transformer.d_model
        self.structure_embed = LayerStructureEmbedding(
            config,
        )
        self.class_embed = nn.Linear(self.hidden_dim, config.num_classes)

    def forward(
            self,
            batch_img: NestedTensor,
            batch_name: NestedTensor,
            batch_bbox: NestedTensor,
            batch_color: NestedTensor,
            batch_class: NestedTensor
    ):
        x, pos_embed = self.structure_embed(batch_img, batch_name, batch_bbox, batch_color, batch_class)
        if self.config.use_mask:
            mask = batch_class.mask
        else:
            mask = None
        x = self.transformer(x, mask, pos_embed)
        class_embed = self.class_embed(x)
        return class_embed.softmax(dim=-1)


def build(config: SketchModelConfig) -> Tuple[SketchLayerClassifierModel, Loss]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer = build_transformer(config)
    model = SketchLayerClassifierModel(
        config,
        transformer,
    )
    criterion = nn.CrossEntropyLoss(reduction='sum')
    criterion.to(device)
    return model, criterion
