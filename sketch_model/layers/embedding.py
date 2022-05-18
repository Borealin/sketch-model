import enum
from typing import Tuple

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase

from sketch_model.datasets.dataset import LAYER_CLASS_MAP
from sketch_model.utils import NestedTensor


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


class TextEmbedding(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            tokenizer: PreTrainedTokenizerBase,
            dropout_rate: float,
            sentence_method: SentenceMethod = SentenceMethod.SUM,
            layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(len(tokenizer.get_vocab()), embedding_dim,
                                            padding_idx=tokenizer.pad_token_id)
        self.layer_norm = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_rate)
        self.sentence_method = sentence_method

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        if self.sentence_method == SentenceMethod.SUM:
            return embeddings.sum(dim=2)
        elif self.sentence_method == SentenceMethod.MAX:
            return embeddings.max(dim=2)[0]
        elif self.sentence_method == SentenceMethod.MEAN:
            return embeddings.mean(dim=2)


class ImageEmbedding(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            patch_size: int,
            image_size: int = 64,
            num_channels: int = 3
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.image_size = (image_size, image_size)
        self.num_channels = num_channels
        self.projection = nn.Sequential(
            nn.Conv2d(num_channels, embedding_dim, kernel_size=patch_size, stride=patch_size),
            nn.ReLU(),
            nn.Flatten(2),
            nn.Linear((image_size // self.patch_size) ** 2, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # batch_size, seq_len, channels, height, width = images.shape
        flat_images = images.view(-1, self.num_channels, *self.image_size)
        features = self.projection(flat_images)
        expand_features = features.view(*images.shape[:2], self.embedding_dim)
        return expand_features


class LayerStructureEmbedding(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            tokenizer: PreTrainedTokenizerBase,
            dropout_rate=0.2,
            concat_image=True,
            pos_pattern=PosPattern.ONE,
            freq_depth=64,
            sentence_method: SentenceMethod = SentenceMethod.SUM,
            aggregation=Aggregation.SUM,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.concat_image = concat_image

        self.pos_pattern = pos_pattern
        if self.pos_pattern == PosPattern.ONE:
            self.num_groups = 4
        elif self.pos_pattern == PosPattern.FOUR:
            self.num_groups = 1
        elif self.pos_pattern == PosPattern.TWO:
            self.num_groups = 2
        else:
            raise ValueError("Unknown pos pattern")
        self.freq_depth = freq_depth
        self.freq_fc = nn.Linear(4 // self.num_groups, self.freq_depth)
        self.coord_embeder = nn.Sequential(
            nn.Linear(freq_depth * 2, self.embedding_dim // self.num_groups),
            nn.ReLU(),
            nn.Linear(self.embedding_dim // self.num_groups, self.embedding_dim // self.num_groups),
        )

        self.sentence_method = sentence_method
        self.token_embeder = TextEmbedding(embedding_dim, tokenizer, self.dropout_rate, self.sentence_method)

        if self.concat_image:
            self.image_embeder = ImageEmbedding(embedding_dim, 4)  # Layer image resized to 64x64
        else:
            self.image_embeder = None

        self.color_embeder = nn.Linear(4, self.embedding_dim)  # RGBA

        self.class_embeder = nn.Embedding(len(LAYER_CLASS_MAP), self.embedding_dim)  # Layer class

        self.concat_embeder = nn.Linear(self.embedding_dim * (4 if self.concat_image else 3), self.embedding_dim)

    def forward(
            self,
            images: NestedTensor,
            names: NestedTensor,
            boxes: NestedTensor,
            colors: NestedTensor,
            classes: NestedTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        name_embeds = self.token_embeder(names.tensors)
        pos_embeds = self.embed_pos(boxes.tensors, boxes.mask)
        color_embeds = self.color_embeder(colors.tensors)
        class_embeds = self.class_embeder(classes.tensors)
        if self.concat_image:
            image_embeds = self.image_embeder(images.tensors)
            embeds = torch.cat([image_embeds, name_embeds, color_embeds, class_embeds], dim=-1)
        else:
            embeds = torch.cat([name_embeds, color_embeds, class_embeds], dim=-1)
        embeds = self.concat_embeder(embeds)
        pos_embeds = nn.Dropout(self.dropout_rate)(pos_embeds)
        embeds = nn.Dropout(self.dropout_rate)(embeds)

        return embeds, pos_embeds

    def embed_pos(self, boxes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.pos_pattern == PosPattern.ONE:
            boxes = torch.unsqueeze(boxes, dim=3)
        elif self.pos_pattern == PosPattern.FOUR:
            boxes = torch.unsqueeze(boxes, dim=2)
        elif self.pos_pattern == PosPattern.TWO:
            boxes = torch.reshape(boxes, boxes.shape[:2] + (2, 2))
        else:
            raise ValueError('Unknown pos pattern')
        freqs = self.freq_fc(boxes)
        features = torch.concat([torch.cos(freqs), torch.sin(freqs)], dim=-1)
        coord_embeds: torch.Tensor = self.coord_embeder(features)
        coord_embeds = torch.reshape(coord_embeds, coord_embeds.shape[:2] + (-1,))
        return coord_embeds
