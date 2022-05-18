import torch
from torch import nn
from transformers import PreTrainedTokenizer

from sketch_model.configs import SketchModelConfig
from sketch_model.layers.embedding import LayerStructureEmbedding
from sketch_model.layers.transformer import build_transformer, SketchTransformer
from sketch_model.utils import NestedTensor


class SketchLayerClassifierModel(nn.Module):
    def __init__(self, transformer, num_classes, tokenizer):
        super().__init__()
        self.transformer: SketchTransformer = transformer
        self.hidden_dim = transformer.d_model
        self.structure_embed = LayerStructureEmbedding(
            self.hidden_dim,
            tokenizer
        )
        self.class_embed = nn.Linear(self.hidden_dim, num_classes)

    def forward(
            self,
            batch_img: NestedTensor,
            batch_name: NestedTensor,
            batch_bbox: NestedTensor,
            batch_color: NestedTensor,
            batch_class: NestedTensor
    ):
        x, pos_embed = self.structure_embed(batch_img, batch_name, batch_bbox, batch_color, batch_class)
        x = self.transformer(x, None, pos_embed)
        return self.class_embed(x)


def build(config: SketchModelConfig, tokenizer: PreTrainedTokenizer):
    num_classes = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer = build_transformer(config)
    model = SketchLayerClassifierModel(
        transformer,
        num_classes,
        tokenizer
    )
    criterion = nn.CrossEntropyLoss(reduction='sum')
    criterion.to(device)
    return model, criterion
